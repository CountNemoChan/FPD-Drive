"""NAVSIM scene loading, checkpoint I/O, and official NAVTEST v1 scoring."""
from __future__ import annotations
import datetime
import inspect
import lzma
import multiprocessing as mp
import os
import pickle
import sys
import traceback
import zipfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

_PDM_WORKER_SIMULATOR: Any = None
_PDM_WORKER_SCORER: Any = None

def ensure_navsim_root(cfg: DictConfig) -> None:
    env_cfg = cfg.evaluation.get("env", {})
    for key, value in env_cfg.items():
        if value not in {None, "", "null", "None"}:
            os.environ[str(key)] = str(value)

    navsim_root = cfg.evaluation.get("navsim_root", None)
    if navsim_root not in {None, "", "null", "None"}:
        navsim_root = str(navsim_root)
        os.environ.setdefault("NAVSIM_DEVKIT_ROOT", navsim_root)
        if navsim_root not in sys.path:
            sys.path.insert(0, navsim_root)



def _parse_torch_dtype(dtype: str) -> torch.dtype:
    value = str(dtype).lower().strip()
    if value in {"fp32", "float32", "float"}:
        return torch.float32
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16", "half"}:
        return torch.float16
    raise ValueError(f"Unsupported dtype: {dtype}")



def _move_features(features: Mapping[str, Any], device: torch.device, dtype: torch.dtype) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in features.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device=device, dtype=dtype) if torch.is_floating_point(value) else value.to(device=device)
        else:
            moved[key] = value
    return moved



def _load_checkpoint_state_dict(checkpoint_path: str | Path) -> tuple[Dict[str, Any], str]:
    path = Path(str(checkpoint_path))
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint_format = "torch_file"
    load_path = path
    if path.is_dir():
        latest_path = path / "latest"
        if latest_path.is_file():
            tag = latest_path.read_text().strip()
            load_path = path / tag / "mp_rank_00_model_states.pt"
            checkpoint_format = "deepspeed_dir"
        elif (path / "mp_rank_00_model_states.pt").is_file():
            load_path = path / "mp_rank_00_model_states.pt"
            checkpoint_format = "deepspeed_tag_dir"
        else:
            raise FileNotFoundError(
                f"Unsupported checkpoint directory: {path}. Expected `latest` or `mp_rank_00_model_states.pt`."
            )

    if not load_path.is_file():
        raise FileNotFoundError(f"Checkpoint state file not found: {load_path}")

    # Lightning/DeepSpeed containers include metadata beyond tensor weights.
    load_options = dict(map_location="cpu", weights_only=False)
    # Full joint checkpoints can contain tens of GB of unused video weights.
    # Map ZIP-format storage lazily so only the planning tensors are paged in.
    if 'mmap' in inspect.signature(torch.load).parameters and zipfile.is_zipfile(load_path):
        load_options['mmap'] = True
    checkpoint = torch.load(load_path, **load_options)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint format at {load_path}")

    if "module" in checkpoint and isinstance(checkpoint["module"], dict):
        return checkpoint["module"], checkpoint_format
    if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        return checkpoint["state_dict"], checkpoint_format
    return checkpoint, checkpoint_format



def _load_msgpack(path: str | Path) -> Dict[str, Any]:
    import msgpack

    with open(path, "rb") as f:
        return msgpack.unpackb(f.read(), raw=False)



def _scene_shard_tokens(tokens, *, count, rank, limit=None):
    """Deterministic, unpadded scene partition; limit applies globally."""
    if count < 1 or not 0 <= rank < count:
        raise ValueError("Invalid evaluation token shard rank/count.")
    tokens = sorted(tokens)
    if limit is not None:
        if int(limit) < 1:
            raise ValueError("evaluation.limit must be positive or null.")
        tokens = tokens[:int(limit)]
    return tokens[rank::count]



def build_scene_loader_dataloader(cfg: DictConfig) -> DataLoader:
    # Keep the external NAVSIM API names; project code uses FPD feature aliases.
    try:
        from hydra.utils import instantiate
        from navsim.agents.drivoR.drivor_features import (
            DrivoRFeatureBuilder as FPDFeatureBuilder,
        )
        from navsim.common.dataloader import SceneFilter, SceneLoader
        from navsim.common.dataclasses import SensorConfig
    except Exception as exc:
        raise ImportError("SceneLoader evaluation requires NAVSIM, NuPlan, Hydra, and raw NAVSIM data dependencies.") from exc

    required = ["sensor_blobs_path", "navsim_log_path", "scene_filter_path"]
    missing = [key for key in required if cfg.evaluation.scene_loader.get(key, None) in {None, ""}]
    if missing:
        raise ValueError(f"SceneLoader evaluation requires evaluation.scene_loader fields: {missing}")

    scene_filter_path = Path(cfg.evaluation.scene_loader.scene_filter_path)
    scene_filter_cfg = (OmegaConf.load(scene_filter_path)
                        if scene_filter_path.suffix.lower() in {'.yaml', '.yml'}
                        else _load_msgpack(scene_filter_path))
    scene_filter: SceneFilter = instantiate(scene_filter_cfg)
    scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.evaluation.scene_loader.sensor_blobs_path),
        data_path=Path(cfg.evaluation.scene_loader.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=SensorConfig(
            cam_f0=list(cfg.model.config.cam_f0),
            cam_l0=list(cfg.model.config.cam_l0),
            cam_l1=list(cfg.model.config.cam_l1),
            cam_l2=list(cfg.model.config.cam_l2),
            cam_r0=list(cfg.model.config.cam_r0),
            cam_r1=list(cfg.model.config.cam_r1),
            cam_r2=list(cfg.model.config.cam_r2),
            cam_b0=list(cfg.model.config.cam_b0),
            lidar_pc=list(cfg.model.config.lidar_pc),
        ),
    )
    if "token_shard_count" in cfg.evaluation:
        # Sort before applying the global limit and sharding. DistributedSampler
        # pads uneven datasets, which would duplicate scenes in the PDM mean.
        tokens = _scene_shard_tokens(
            scene_loader.tokens,
            count=int(cfg.evaluation.token_shard_count),
            rank=int(cfg.evaluation.get("token_shard_rank", 0)),
            limit=cfg.evaluation.get("limit", None),
        )
        scene_loader.scene_frames_dicts = {
            token: scene_loader.scene_frames_dicts[token]
            for token in tokens
        }
    return DataLoader(
        _InferenceDataset(scene_loader, FPDFeatureBuilder(cfg.model.config)),
        batch_size=int(cfg.evaluation.batch_size),
        shuffle=False,
        num_workers=int(cfg.evaluation.get("num_workers", 0)),
        pin_memory=bool(cfg.evaluation.get("pin_memory", True)),
        drop_last=False,
    )



def _extract_batch(batch: Any) -> tuple[Dict[str, Any], Iterable[str]]:
    features, tokens = batch
    return features, [str(token) for token in tokens]


class _InferenceDataset(torch.utils.data.Dataset):
    """Build only observed inputs and scene tokens; no ground-truth targets."""

    def __init__(self, scene_loader: Any, feature_builder: Any) -> None:
        self.scene_loader = scene_loader
        self.feature_builder = feature_builder
        self.tokens = list(scene_loader.tokens)

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, index: int) -> tuple[Dict[str, Any], str]:
        token = self.tokens[index]
        agent_input = self.scene_loader.get_agent_input_from_token(token)
        return self.feature_builder.compute_features(agent_input), str(token)


def build_navhard_scene_loader_dataloader(cfg: DictConfig) -> DataLoader:
    """Build FPDplanner inputs from NAVSIM-v2 original/synthetic NavHard scenes."""
    from hydra.utils import instantiate
    from navsim.common.dataclasses import SensorConfig
    from navsim.common.dataloader import SceneLoader
    import navsim.agents as navsim_agents

    feature_navsim_root = Path(
        str(cfg.evaluation.get("feature_navsim_root", "external/navsim-v1"))
    )
    feature_agents_path = str(feature_navsim_root / "navsim" / "agents")
    if feature_agents_path not in navsim_agents.__path__:
        navsim_agents.__path__.append(feature_agents_path)
    from navsim.agents.drivoR.drivor_features import DrivoRFeatureBuilder as FPDFeatureBuilder

    scene_loader_cfg = cfg.evaluation.scene_loader
    required = [
        "original_sensor_path",
        "synthetic_sensor_path",
        "navsim_log_path",
        "synthetic_scenes_path",
        "scene_filter_path",
    ]
    missing = [key for key in required if scene_loader_cfg.get(key, None) in {None, ""}]
    if missing:
        raise ValueError(f"FPDplanner NavHard scene_loader evaluation requires fields: {missing}")

    scene_filter_path = Path(str(scene_loader_cfg.scene_filter_path))
    scene_filter_cfg = (
        OmegaConf.load(scene_filter_path)
        if scene_filter_path.suffix.lower() in {".yaml", ".yml"}
        else _load_msgpack(scene_filter_path)
    )
    scene_filter = instantiate(scene_filter_cfg)
    model_cfg = cfg.model.config
    scene_loader = SceneLoader(
        original_sensor_path=Path(str(scene_loader_cfg.original_sensor_path)),
        synthetic_sensor_path=Path(str(scene_loader_cfg.synthetic_sensor_path)),
        data_path=Path(str(scene_loader_cfg.navsim_log_path)),
        synthetic_scenes_path=Path(str(scene_loader_cfg.synthetic_scenes_path)),
        scene_filter=scene_filter,
        sensor_config=SensorConfig(
            cam_f0=list(model_cfg.cam_f0),
            cam_l0=list(model_cfg.cam_l0),
            cam_l1=list(model_cfg.cam_l1),
            cam_l2=list(model_cfg.cam_l2),
            cam_r0=list(model_cfg.cam_r0),
            cam_r1=list(model_cfg.cam_r1),
            cam_r2=list(model_cfg.cam_r2),
            cam_b0=list(model_cfg.cam_b0),
            lidar_pc=list(model_cfg.lidar_pc),
        ),
    )
    return DataLoader(
        _InferenceDataset(scene_loader, FPDFeatureBuilder(model_cfg)),
        batch_size=int(cfg.evaluation.get("batch_size", 1)),
        shuffle=False,
        num_workers=int(cfg.evaluation.get("num_workers", 0)),
        pin_memory=bool(cfg.evaluation.get("pin_memory", True)),
        drop_last=False,
    )



def _build_pdm_objects_from_sampling(sampling_cfg: Mapping[str, Any]):
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    sampling = TrajectorySampling(
        num_poses=int(sampling_cfg.get("num_poses", 40)),
        interval_length=float(sampling_cfg.get("interval_length", 0.1)),
    )
    simulator = PDMSimulator(sampling)
    scorer = PDMScorer(sampling, PDMScorerConfig())
    return simulator, scorer



def build_pdm_objects(cfg: DictConfig):
    return _build_pdm_objects_from_sampling(cfg.evaluation.get("pdm_sampling", {}))



def _initialize_pdm_worker(
    sampling_cfg: Mapping[str, Any],
    env_cfg: Mapping[str, Any],
    navsim_root: Optional[str],
    torch_threads: int,
) -> None:
    """Initialize NAVSIM/PDM state once in each spawned scoring process."""
    for key, value in env_cfg.items():
        if value not in {None, "", "null", "None"}:
            os.environ[str(key)] = str(value)
    if navsim_root not in {None, "", "null", "None"}:
        root = str(navsim_root)
        os.environ.setdefault("NAVSIM_DEVKIT_ROOT", root)
        if root not in sys.path:
            sys.path.insert(0, root)
    if torch_threads > 0:
        torch.set_num_threads(torch_threads)

    global _PDM_WORKER_SIMULATOR, _PDM_WORKER_SCORER
    _PDM_WORKER_SIMULATOR, _PDM_WORKER_SCORER = _build_pdm_objects_from_sampling(sampling_cfg)



def _load_metric_cache(metric_cache_path: Optional[str]) -> Any:
    if metric_cache_path is None:
        raise KeyError("Metric cache path is missing")
    with lzma.open(metric_cache_path, "rb") as f:
        return pickle.load(f)



def _score_pdm_item(
    token: str,
    metric_cache_path: Optional[str],
    trajectory: Any,
    simulator: Any,
    scorer: Any,
) -> Dict[str, Any]:
    from navsim.evaluate.pdm_score import pdm_score

    if trajectory is None:
        raise KeyError(f"Trajectory is missing for token={token}")

    metric_cache = _load_metric_cache(metric_cache_path)
    if not hasattr(metric_cache, "trajectory"):
        raise ValueError("NAVTEST scoring requires an official V1 evaluation metric cache.")
    if simulator is None or scorer is None:
        raise RuntimeError("PDM simulator/scorer is not initialized")
    pdm_result = pdm_score(
        metric_cache=metric_cache,
        model_trajectory=trajectory,
        future_sampling=simulator.proposal_sampling,
        simulator=simulator,
        scorer=scorer,
    )
    return asdict(pdm_result)



def _safe_score_pdm_item(
    token: str,
    metric_cache_path: Optional[str],
    trajectory: Any,
    simulator: Any,
    scorer: Any,
) -> tuple[Dict[str, Any], Optional[str]]:
    row: Dict[str, Any] = {"token": token, "valid": True}
    try:
        row.update(_score_pdm_item(token, metric_cache_path, trajectory, simulator, scorer))
        return row, None
    except Exception:
        row["valid"] = False
        return row, traceback.format_exc()



def _score_pdm_worker(task: tuple[int, str, Optional[str], Any]):
    index, token, metric_cache_path, trajectory = task
    row, error = _safe_score_pdm_item(
        token,
        metric_cache_path,
        trajectory,
        _PDM_WORKER_SIMULATOR,
        _PDM_WORKER_SCORER,
    )
    return index, row, error



def _to_plain_mapping(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if OmegaConf.is_config(value):
        return dict(OmegaConf.to_container(value, resolve=True))
    return dict(value)



def score_trajectories(cfg: DictConfig, trajectories: Dict[str, Any]) -> pd.DataFrame:
    from navsim.common.dataloader import MetricCacheLoader

    metric_cache_loader = MetricCacheLoader(Path(cfg.evaluation.metric_cache_path))
    old_prefix = cfg.evaluation.get("metric_cache_old_prefix", None)
    if old_prefix not in {None, "", "null", "None"}:
        new_prefix = cfg.evaluation.get("metric_cache_new_prefix", None)
        if new_prefix in {None, "", "null", "None"}:
            new_prefix = str(Path(cfg.evaluation.metric_cache_path)) + "/"
        metric_cache_loader.metric_cache_paths = {
            token: path.replace(str(old_prefix), str(new_prefix))
            for token, path in metric_cache_loader.metric_cache_paths.items()
        }
    tokens = list(trajectories.keys())
    if bool(cfg.evaluation.get("score_all_metric_cache_tokens", False)):
        tokens = list(metric_cache_loader.tokens)
    limit = cfg.evaluation.get("limit", None)
    if limit is not None:
        tokens = tokens[: int(limit)]

    tasks = [
        (
            index,
            token,
            metric_cache_loader.metric_cache_paths.get(token),
            trajectories.get(token),
        )
        for index, token in enumerate(tokens)
    ]
    rows: list[Optional[Dict[str, Any]]] = [None] * len(tasks)

    def record_result(index: int, row: Dict[str, Any], error: Optional[str]) -> None:
        rows[index] = row
        if error is not None:
            print(f'[FPD-Drive Eval] failed token={row["token"]}', file=sys.stderr)
            print(error, file=sys.stderr, end="" if error.endswith("\n") else "\n")

    pdm_num_workers = max(1, int(cfg.evaluation.get("pdm_num_workers", 1)))
    if pdm_num_workers == 1 or len(tasks) <= 1:
        print(f"[FPD-Drive Eval] PDM scoring mode=serial tokens={len(tasks)}")
        simulator, scorer = build_pdm_objects(cfg)
        for index, token, metric_cache_path, trajectory in tqdm(
            tasks, desc="Compute PDM score"
        ):
            row, error = _safe_score_pdm_item(
                token, metric_cache_path, trajectory, simulator, scorer
            )
            record_result(index, row, error)
    else:
        pdm_num_workers = min(pdm_num_workers, len(tasks))
        chunksize = max(1, int(cfg.evaluation.get("pdm_chunksize", 8)))
        start_method = str(cfg.evaluation.get("pdm_mp_start_method", "spawn"))
        if start_method not in mp.get_all_start_methods():
            raise ValueError(
                f"Unsupported evaluation.pdm_mp_start_method={start_method!r}; "
                f"available={mp.get_all_start_methods()}"
            )
        torch_threads = max(1, int(cfg.evaluation.get("pdm_worker_torch_threads", 1)))
        print(
            f"[FPD-Drive Eval] PDM scoring mode=parallel workers={pdm_num_workers} "
            f"chunksize={chunksize} start_method={start_method} tokens={len(tasks)}"
        )
        sampling_cfg = _to_plain_mapping(cfg.evaluation.get("pdm_sampling", {}))
        env_cfg = _to_plain_mapping(cfg.evaluation.get("env", {}))
        navsim_root = cfg.evaluation.get("navsim_root", None)
        with ProcessPoolExecutor(
            max_workers=pdm_num_workers,
            mp_context=mp.get_context(start_method),
            initializer=_initialize_pdm_worker,
            initargs=(sampling_cfg, env_cfg, navsim_root, torch_threads),
        ) as executor:
            results = executor.map(_score_pdm_worker, tasks, chunksize=chunksize)
            for index, row, error in tqdm(results, total=len(tasks), desc="Compute PDM score"):
                record_result(index, row, error)

    completed_rows = [row for row in rows if row is not None]
    if len(completed_rows) != len(tasks):
        raise RuntimeError(f"Only {len(completed_rows)}/{len(tasks)} PDM rows were produced")

    df = pd.DataFrame(completed_rows)
    if len(df) == 0:
        raise RuntimeError("No PDM score rows were produced.")
    average_row = df.drop(columns=["token", "valid"]).mean(skipna=True)
    average_row["token"] = "average"
    average_row["valid"] = bool(df["valid"].all())
    df.loc[len(df)] = average_row
    return df



def write_score_csv(df: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    csv_path = output_dir / f"{timestamp}.csv"
    df.to_csv(csv_path)
    header = df.head(0).to_csv().splitlines()[0]
    avg = df[df["token"] == "average"].to_csv(header=False).splitlines()[0]
    print(f"[FPD-Drive Eval] CSV header: {header}")
    print(f"[FPD-Drive Eval] Average row: {avg}")
    print(f"[FPD-Drive Eval] Results stored in: {csv_path}")
    return csv_path

