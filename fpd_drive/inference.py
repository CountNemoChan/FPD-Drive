"""One model-generation and evaluation path for Base/Pro on NAVSIM v1/v2."""
import hashlib
import json
import pickle
from datetime import datetime
from pathlib import Path

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from . import navsim
from .model import FPDDriveModel, load_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def generate_trajectories(cfg, output_dir):
    from navsim.common.dataclasses import Trajectory
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    build_loader = (navsim.build_scene_loader_dataloader if cfg.evaluation.source == 'scene_loader'
                    else navsim.build_navhard_scene_loader_dataloader)
    loader = build_loader(cfg)
    device = torch.device(cfg.evaluation.device)
    dtype = navsim._parse_torch_dtype(cfg.evaluation.dtype)
    if device.type == 'cpu':
        dtype = torch.float32
    model = FPDDriveModel(cfg)
    load_checkpoint(model, cfg.evaluation.checkpoint_path)
    model.to(device=device, dtype=dtype).eval()
    sampling = TrajectorySampling(time_horizon=4.0, interval_length=0.5)
    trajectories = {}
    total = 0
    with torch.inference_mode():
        for batch in tqdm(loader, desc='Generate FPD-Drive trajectories'):
            features, tokens = navsim._extract_batch(batch)
            features = navsim._move_features(features, device=device, dtype=dtype)
            with torch.amp.autocast(device_type='cuda', dtype=dtype,
                                    enabled=device.type == 'cuda' and dtype != torch.float32):
                poses = model(features, scenario_tokens=tokens)
            for token, pose in zip(tokens, poses.detach().float().cpu().numpy()):
                trajectories[str(token)] = Trajectory(poses=pose.astype('float32'), trajectory_sampling=sampling)
            total += len(tokens)
            if cfg.evaluation.get('limit') is not None and total >= int(cfg.evaluation.limit):
                break
    with (output_dir / 'trajectories.pkl').open('wb') as handle:
        pickle.dump({'predictions': [trajectories]}, handle)
    print(f'[FPD-Drive] generated trajectories={len(trajectories)}', flush=True)
    return trajectories


def evaluate(cfg, inference_only=False):
    navsim.ensure_navsim_root(cfg)
    output_dir = Path(cfg.output.path) / 'evaluations' / f'{cfg.evaluation.run_name}_{datetime.now():%Y%m%d_%H%M%S}'
    output_dir.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(cfg, output_dir / 'config.yaml')
    trajectories = generate_trajectories(cfg, output_dir)
    is_v1 = cfg.evaluation.score_protocol == 'navsim_v1_pdms'
    policy = cfg.evaluation.get('selection_method', 'top1' if is_v1 else 'top3_blend')
    if not inference_only:
        if is_v1:
            table = navsim.score_trajectories(cfg, trajectories)
            csv_path = navsim.write_score_csv(table, output_dir)
            cases = table[table.token != 'average']
            if not cases.valid.astype(str).str.lower().eq('true').all() or cases.score.isna().any():
                raise RuntimeError(f'Evaluation contains invalid scenes; inspect {csv_path}')
            score = float(cases.score.mean())
            summary = dict(checkpoint=str(cfg.evaluation.checkpoint_path), protocol='navsim_v1_pdms',
                           scenes=len(cases), full_split=cfg.evaluation.get('limit') is None,
                           pdms=score, pdms_times_100=score * 100, csv=str(csv_path),
                           trajectories=str(output_dir / 'trajectories.pkl'),
                           ranking_precision=str(cfg.evaluation.ranking_precision),
                           ranking_weights=OmegaConf.to_container(cfg.evaluation.ranking_weights),
                           metrics={key: float(cases[key].mean()) for key in (
                               'no_at_fault_collisions', 'drivable_area_compliance', 'ego_progress',
                               'time_to_collision_within_bound', 'comfort', 'driving_direction_compliance', 'score')})
            print(f'[NAVTEST] PDMS={score * 100:.6f}; scenes={len(cases)}', flush=True)
        else:
            from .navhard import evaluate_navhard_trajectories
            csv_path = evaluate_navhard_trajectories(
                trajectories=trajectories, output_dir=output_dir,
                metric_cache_path=Path(cfg.evaluation.metric_cache_path),
                navsim_v2_root=Path(cfg.evaluation.navsim_root),
                openscene_data_root=Path(cfg.evaluation.env.OPENSCENE_DATA_ROOT),
                synthetic_scenes_path=Path(cfg.evaluation.scene_loader.synthetic_scenes_path),
                num_workers=int(cfg.evaluation.score_num_workers),
                worker_chunksize=int(cfg.evaluation.get('score_worker_chunksize', 1)),
            )
            summary = json.loads((output_dir / 'summary.json').read_text())
        summary.update(checkpoint=str(cfg.evaluation.checkpoint_path), selection_method=policy,
                       model_class='FPDDriveModel', batch_size=int(cfg.evaluation.batch_size),
                       ddim_steps=int(cfg.joint.refiner.num_inference_steps))
        (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')

    adaptive = policy == 'adaptive_top8'
    ensemble = cfg.evaluation.get('trajectory_ensemble_weights') if not adaptive else None
    sources = [ROOT / 'run_inference.py', *sorted((ROOT / 'fpd_drive').glob('*.py')),
               *sorted((ROOT / 'diffsynth').rglob('*.py'))]
    protocol = dict(
        checkpoint=str(cfg.evaluation.checkpoint_path), mode='from_model', model_class='FPDDriveModel',
        selection_method=policy, dtype=str(cfg.evaluation.dtype), batch_size=int(cfg.evaluation.batch_size),
        candidates=64, ddim_steps=int(cfg.joint.refiner.num_inference_steps), seed=27,
        ranking_precision=str(cfg.evaluation.get('adaptive_ranking_precision') if adaptive else cfg.evaluation.ranking_precision),
        ranking_weights=OmegaConf.to_container(cfg.evaluation.get('stage_ranking_weights') if not is_v1 else cfg.evaluation.ranking_weights),
        fusion=adaptive or ensemble is not None,
        trajectory_ensemble_weights=list(ensemble) if ensemble is not None else None,
        adaptive_top_k=8 if adaptive else None, adaptive_temperature=1.0 if adaptive else None,
        scoring_performed=not inference_only,
        source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
    )
    (output_dir / 'inference_protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    return output_dir / 'trajectories.pkl' if inference_only else csv_path
