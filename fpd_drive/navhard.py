"""Official NAVSIM v2 two-stage NAVHARD scoring."""
from __future__ import annotations
import json
import math
import multiprocessing
import os
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

_WORKER_STATE: Dict[str, Any] = {}

def _configure_navsim_v2(navsim_v2_root: Path, openscene_data_root: Path) -> None:
    root = navsim_v2_root.resolve()
    if not (root / "navsim" / "planning" / "script" / "run_pdm_score.py").is_file():
        raise FileNotFoundError(f"Invalid NAVSIM-v2 root: {root}")
    os.environ["NAVSIM_DEVKIT_ROOT"] = str(root)
    os.environ.setdefault("OPENSCENE_DATA_ROOT", str(openscene_data_root.resolve()))
    os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
    os.environ.setdefault("NUPLAN_MAPS_ROOT", str(openscene_data_root.resolve() / "maps"))
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    root_string = str(root)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)



def _normalize_trajectories(trajectories: Mapping[str, Any]) -> Dict[str, Any]:
    import numpy as np
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.common.dataclasses import Trajectory
    from tqdm import tqdm

    normalized: Dict[str, Any] = {}
    for token, trajectory in tqdm(trajectories.items(), desc="Validate NavHard trajectories"):
        if not hasattr(trajectory, "poses") or not hasattr(trajectory, "trajectory_sampling"):
            raise TypeError(f"token={token}: object is not a NAVSIM Trajectory")
        poses = np.asarray(trajectory.poses, dtype=np.float32)
        if poses.ndim != 2 or poses.shape[1] != 3:
            raise ValueError(f"token={token}: expected poses [N, 3], got {poses.shape}")
        if not np.isfinite(poses).all():
            raise ValueError(f"token={token}: trajectory contains NaN or Inf")
        interval_length = float(trajectory.trajectory_sampling.interval_length)
        num_poses = int(trajectory.trajectory_sampling.num_poses)
        if num_poses != poses.shape[0]:
            raise ValueError(f"token={token}: sampling poses={num_poses}, array poses={poses.shape[0]}")
        if not math.isclose(num_poses * interval_length, 4.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(f"token={token}: NavHard requires a 4.0 s trajectory")
        normalized[str(token)] = Trajectory(
            poses=poses,
            trajectory_sampling=TrajectorySampling(num_poses=num_poses, interval_length=interval_length),
        )
    return normalized



def _format_tokens(tokens: Iterable[str], limit: int = 10) -> str:
    values = sorted(map(str, tokens))
    return ", ".join(values[:limit]) + (" ..." if len(values) > limit else "")



def _validate_metric_cache_schema(metric_cache: Any) -> None:
    required = {
        "log_name",
        "timepoint",
        "scene_type",
        "human_trajectory",
        "past_human_trajectory",
        "past_detections_tracks",
        "current_tracked_objects",
        "future_tracked_objects",
        "map_parameters",
    }
    missing = sorted(name for name in required if not hasattr(metric_cache, name))
    if missing:
        raise RuntimeError(f"Metric cache is not NAVSIM-v2 compatible; missing fields: {missing}")



def _initialize_worker(metric_cache_path: str, traffic_agents_config_path: str) -> None:
    from hydra.utils import instantiate
    from nuplan.common.actor_state.state_representation import StateSE2
    from nuplan.common.geometry.convert import relative_to_absolute_poses
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from omegaconf import OmegaConf

    from navsim.common.dataloader import MetricCacheLoader
    from navsim.evaluate.pdm_score import pdm_score
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (
        PDMScorer,
        PDMScorerConfig,
    )
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator

    sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
    scorer_config = PDMScorerConfig(
        progress_weight=5.0,
        ttc_weight=5.0,
        lane_keeping_weight=2.0,
        history_comfort_weight=2.0,
        two_frame_extended_comfort_weight=2.0,
        driving_direction_horizon=1.0,
        driving_direction_compliance_threshold=2.0,
        driving_direction_violation_threshold=6.0,
        stopped_speed_threshold=5e-3,
        future_collision_horizon_window=1.0,
        progress_distance_threshold=5.0,
        lane_keeping_deviation_limit=0.5,
        lane_keeping_horizon_window=2.0,
        human_penalty_filter=True,
    )
    traffic_cfg = OmegaConf.load(traffic_agents_config_path)
    _WORKER_STATE.clear()
    _WORKER_STATE.update(
        metric_cache_loader=MetricCacheLoader(Path(metric_cache_path)),
        sampling=sampling,
        simulator=PDMSimulator(sampling),
        scorer=PDMScorer(sampling, scorer_config),
        traffic_agents_policy=instantiate(traffic_cfg.reactive, sampling),
        pdm_score=pdm_score,
        state_se2_cls=StateSE2,
        relative_to_absolute_poses=relative_to_absolute_poses,
    )



def _score_token(item: tuple[str, Any]) -> Any:
    token, trajectory = item
    try:
        state = _WORKER_STATE
        metric_cache = state["metric_cache_loader"].get_from_token(token)
        score_row, ego_simulated_states = state["pdm_score"](
            metric_cache=metric_cache,
            model_trajectory=trajectory,
            future_sampling=state["sampling"],
            simulator=state["simulator"],
            scorer=state["scorer"],
            traffic_agents_policy=state["traffic_agents_policy"],
        )
        score_row["valid"] = True
        score_row["log_name"] = metric_cache.log_name
        score_row["frame_type"] = metric_cache.scene_type
        score_row["start_time"] = metric_cache.timepoint.time_s
        end_pose = state["state_se2_cls"](*trajectory.poses[-1])
        absolute_endpoint = state["relative_to_absolute_poses"](
            metric_cache.ego_state.rear_axle, [end_pose]
        )[0]
        score_row["endpoint_x"] = absolute_endpoint.x
        score_row["endpoint_y"] = absolute_endpoint.y
        score_row["start_point_x"] = metric_cache.ego_state.rear_axle.x
        score_row["start_point_y"] = metric_cache.ego_state.rear_axle.y
        score_row["ego_simulated_states"] = [ego_simulated_states]
        score_row["token"] = token
        return score_row
    except Exception as error:
        raise RuntimeError(f"NAVSIM-v2 NavHard scoring failed for token={token}") from error



def _mapping_tokens(mappings: Mapping[tuple[str, str], list[tuple[str, str]]]) -> set[str]:
    tokens: set[str] = set()
    for (current, previous), pairs in mappings.items():
        tokens.update((current, previous))
        for pair in pairs:
            tokens.update(pair)
    return tokens



def evaluate_navhard_trajectories(
    trajectories: Mapping[str, Any],
    output_dir: Path,
    metric_cache_path: Path,
    navsim_v2_root: Path,
    openscene_data_root: Path,
    synthetic_scenes_path: Path,
    num_workers: int = min(32, os.cpu_count() or 1),
    worker_chunksize: int = 1,
    mapping_limit: Optional[int] = None,
) -> Path:
    if num_workers <= 0 or worker_chunksize <= 0:
        raise ValueError("num_workers and worker_chunksize must be positive")
    if mapping_limit is not None and mapping_limit <= 0:
        raise ValueError("mapping_limit must be positive")
    _configure_navsim_v2(navsim_v2_root, openscene_data_root)

    import numpy as np
    import pandas as pd
    from hydra.utils import instantiate
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from omegaconf import OmegaConf
    from tqdm import tqdm

    from navsim.common.dataclasses import PDMResults, SensorConfig
    from navsim.common.dataloader import MetricCacheLoader, SceneLoader
    from navsim.common.enums import SceneFrameType
    from navsim.planning.script.run_pdm_score import (
        calculate_individual_mapping_scores,
        compute_final_scores,
        create_scene_aggregators,
    )

    root = navsim_v2_root.resolve()
    split_path = root / "navsim/planning/script/config/common/train_test_split/navhard_two_stage.yaml"
    scene_filter_path = root / "navsim/planning/script/config/common/train_test_split/scene_filter/navhard_two_stage.yaml"
    traffic_path = root / "navsim/planning/script/config/common/traffic_agents_policy/navsim_IDM_traffic_agents.yaml"
    split_cfg = OmegaConf.load(split_path)
    scene_loader = SceneLoader(
        data_path=openscene_data_root.resolve() / "navsim_logs" / str(split_cfg.data_split),
        original_sensor_path=None,
        synthetic_sensor_path=None,
        synthetic_scenes_path=synthetic_scenes_path.resolve(),
        scene_filter=instantiate(OmegaConf.load(scene_filter_path)),
        sensor_config=SensorConfig.build_no_sensors(),
    )
    metric_cache_loader = MetricCacheLoader(metric_cache_path.resolve())
    normalized = _normalize_trajectories(trajectories)

    all_mappings: Dict[tuple[str, str], list[tuple[str, str]]] = {}
    loader_tokens = set(map(str, scene_loader.tokens))
    for original, previous, pairs in split_cfg.reactive_all_mapping:
        if str(previous) in loader_tokens or str(original) in loader_tokens:
            all_mappings[(str(original), str(previous))] = [tuple(map(str, pair)) for pair in pairs]
    if mapping_limit is not None:
        all_mappings = dict(list(all_mappings.items())[:mapping_limit])

    official_score_tokens = set(map(str, scene_loader.tokens_stage_one)) | set(
        map(str, scene_loader.reactive_tokens_stage_two)
    )
    score_tokens = _mapping_tokens(all_mappings) if mapping_limit is not None else official_score_tokens
    cache_tokens = set(map(str, metric_cache_loader.tokens))
    trajectory_tokens = set(normalized)
    missing_cache = score_tokens - cache_tokens
    missing_trajectories = score_tokens - trajectory_tokens
    missing_mapping_tokens = _mapping_tokens(all_mappings) - score_tokens
    if missing_cache or missing_trajectories or missing_mapping_tokens:
        details = []
        if missing_cache:
            details.append(f"missing cache={len(missing_cache)} ({_format_tokens(missing_cache)})")
        if missing_trajectories:
            details.append(
                f"missing trajectories={len(missing_trajectories)} ({_format_tokens(missing_trajectories)})"
            )
        if missing_mapping_tokens:
            details.append(f"missing mapping tokens={len(missing_mapping_tokens)}")
        raise RuntimeError("; ".join(details))
    first_cache = metric_cache_loader.get_from_token(next(iter(score_tokens)))
    _validate_metric_cache_schema(first_cache)
    print(
        "[NAVSIM-v2 NavHard] "
        f"score_tokens={len(score_tokens)} mappings={len(all_mappings)} "
        f"cache_tokens={len(cache_tokens)} trajectory_tokens={len(trajectory_tokens)} "
        f"workers={num_workers}"
    )

    work_items = ((token, normalized[token]) for token in sorted(score_tokens))
    score_rows = []
    if num_workers == 1:
        _initialize_worker(str(metric_cache_path.resolve()), str(traffic_path))
        score_rows.extend(
            tqdm(map(_score_token, work_items), total=len(score_tokens), desc="Score NavHard EPDMS")
        )
    else:
        context = multiprocessing.get_context("fork")
        with context.Pool(
            processes=num_workers,
            initializer=_initialize_worker,
            initargs=(str(metric_cache_path.resolve()), str(traffic_path)),
        ) as pool:
            iterator = pool.imap_unordered(_score_token, work_items, chunksize=worker_chunksize)
            score_rows.extend(tqdm(iterator, total=len(score_tokens), desc="Score NavHard EPDMS"))

    sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
    score_df = pd.concat(score_rows, ignore_index=True)
    score_df = create_scene_aggregators(all_mappings, score_df, sampling)
    score_df = compute_final_scores(score_df)
    score_columns = [
        column
        for column in score_df.columns
        if (
            any(result.name in column for result in fields(PDMResults))
            or column == "two_frame_extended_comfort"
            or column == "score"
        )
        and column != "pdm_score"
    ]
    combined, stage_one, stage_two = calculate_individual_mapping_scores(
        score_df[score_columns + ["token", "weight"]], all_mappings
    )

    for column in score_columns:
        stage_one_mask = score_df["frame_type"] == SceneFrameType.ORIGINAL
        stage_two_mask = score_df["frame_type"] == SceneFrameType.SYNTHETIC
        score_df.loc[stage_one_mask, f"{column}_stage_one"] = score_df.loc[stage_one_mask, column]
        score_df.loc[stage_two_mask, f"{column}_stage_two"] = score_df.loc[stage_two_mask, column]
    score_df.drop(columns=score_columns, inplace=True)
    score_df["score"] = score_df["score_stage_one"].combine_first(score_df["score_stage_two"])
    score_df.drop(columns=["score_stage_one", "score_stage_two"], inplace=True)
    stage_one_columns = [f"{column}_stage_one" for column in score_columns if column != "score"]
    stage_two_columns = [f"{column}_stage_two" for column in score_columns if column != "score"]
    score_df = score_df[["token", "valid", *stage_one_columns, *stage_two_columns, "score"]]

    summary_rows = []
    for token, values, suffix in (
        ("extended_pdm_score_stage_one", stage_one, "stage_one"),
        ("extended_pdm_score_stage_two", stage_two, "stage_two"),
    ):
        row = pd.Series(index=score_df.columns, dtype=object)
        row["token"] = token
        row["valid"] = True
        row["score"] = values.get("score", np.nan)
        for column in values.index:
            if column not in {"token", "valid", "score"}:
                row[f"{column}_{suffix}"] = values[column]
        summary_rows.append(row)
    combined_row = pd.Series(index=score_df.columns, dtype=object)
    combined_row["token"] = "extended_pdm_score_combined"
    combined_row["valid"] = True
    combined_row["score"] = combined["score"]
    for column in stage_one.index:
        if column not in {"token", "valid", "score"}:
            combined_row[f"{column}_stage_one"] = stage_one[column]
    for column in stage_two.index:
        if column not in {"token", "valid", "score"}:
            combined_row[f"{column}_stage_two"] = stage_two[column]
    summary_rows.append(combined_row)
    score_df = pd.concat([score_df, pd.DataFrame(summary_rows)], ignore_index=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "navhard_epdms.csv"
    score_df.to_csv(csv_path, index=False)
    summary = {
        "benchmark_comparable": mapping_limit is None,
        "metric_cache_path": str(metric_cache_path.resolve()),
        "synthetic_scenes_path": str(synthetic_scenes_path.resolve()),
        "num_scored_tokens": len(score_tokens),
        "num_mappings": len(all_mappings),
        "num_workers": num_workers,
        "stage_one_score": None if pd.isna(stage_one.get("score")) else float(stage_one["score"]),
        "stage_two_score": None if pd.isna(stage_two.get("score")) else float(stage_two["score"]),
        "combined_epdms": None if pd.isna(combined.get("score")) else float(combined["score"]),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[NAVSIM-v2 NavHard] combined_epdms={summary['combined_epdms']}")
    print(f"[NAVSIM-v2 NavHard] csv={csv_path}")
    return csv_path

