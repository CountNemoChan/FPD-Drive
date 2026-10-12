#!/usr/bin/env python
"""Run FPD-Drive Base/Pro inference and NAVSIM evaluation."""

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_config(args):
    from omegaconf import OmegaConf

    path = args.config or ROOT / 'configs' / 'inference' / f'{args.variant}_{args.benchmark}.yaml'
    cfg = OmegaConf.load(path)
    # Environment defaults < dotlist overrides < explicit command-line flags.
    defaults = []
    for variable, key in (
        ('BATCH_SIZE', 'batch_size'), ('NUM_WORKERS', 'num_workers'),
        ('PDM_WORKERS', 'score_num_workers' if args.benchmark == 'v2' else 'pdm_num_workers'),
        ('RUN_NAME', 'run_name'),
    ):
        if variable in os.environ:
            defaults.append(f'evaluation.{key}={os.environ[variable]}')
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(defaults), OmegaConf.from_dotlist(args.overrides))
    if args.checkpoint_path is not None:
        cfg.evaluation.checkpoint_path = str(Path(args.checkpoint_path).expanduser().resolve())
    if args.denoise_steps is not None:
        cfg.joint.refiner.num_inference_steps = args.denoise_steps
    OmegaConf.resolve(cfg)
    expected = 'navsim_v1_pdms' if args.benchmark == 'v1' else 'navsim_v2_navhard'
    if cfg.evaluation.score_protocol != expected:
        raise ValueError(f'{args.benchmark} requires evaluation.score_protocol={expected}')
    expected_source = 'scene_loader' if args.benchmark == 'v1' else 'navhard_scene_loader'
    if cfg.evaluation.source != expected_source:
        raise ValueError(f'{args.benchmark} requires evaluation.source={expected_source}')
    if str(cfg.evaluation.dtype) not in ('bf16', 'bfloat16'):
        raise ValueError('The release presets require BF16 inference')
    if cfg.evaluation.ranking_precision != 'bf16_sigmoid_numpy_fp64':
        raise ValueError('Unsupported evaluation.ranking_precision')
    steps = int(cfg.joint.refiner.num_inference_steps)
    if not 1 <= steps <= int(cfg.joint.refiner.ddim_num_train_timesteps):
        raise ValueError('DDIM steps must be between 1 and ddim_num_train_timesteps')
    if args.benchmark == 'v2' and steps != 12:
        raise ValueError('The NAVSIM V2 reference policy requires 12 DDIM steps')
    if int(cfg.evaluation.batch_size) < 1 or int(cfg.evaluation.num_workers) < 0:
        raise ValueError('Batch size must be positive and num_workers nonnegative')
    if args.variant == 'pro' and args.benchmark == 'v2':
        ev = cfg.evaluation
        if ev.get('selection_method') != 'adaptive_top8':
            raise ValueError('Pro V2 requires selection_method=adaptive_top8')
        if ev.get('adaptive_ranking_precision') != 'fp32_sigmoid_numpy_fp64':
            raise ValueError('Adaptive Top8 requires FP32 sigmoid and NumPy FP64 ranking')
        if ev.get('adaptive_top_k') != 8 or ev.get('adaptive_temperature') != 1.0:
            raise ValueError('The released adaptive Top8 policy requires k=8 and temperature=1')
    if cfg.evaluation.get('trajectory_pickle_path'):
        raise ValueError('The inference command generates fresh trajectories from the checkpoint')
    if args.benchmark == 'v1' and cfg.evaluation.get('trajectory_ensemble_weights') is not None:
        raise ValueError('The V1 presets use Top1 selection; trajectory blending is disabled')
    if args.benchmark == 'v2' and not args.inference_only:
        if cfg.evaluation.get('limit') is not None or cfg.evaluation.get('score_mapping_limit') is not None:
            raise ValueError('NAVHARD benchmark evaluation requires the full split')
    for name in ('scorer_future', 'candidate_densification', 'future_tokens'):
        if cfg.joint.get(name, {}).get('enable', False):
            raise ValueError(f'The student inference presets do not support joint.{name}')
    if cfg.evaluation.get('teacher_scorer_checkpoint'):
        raise ValueError('The release uses the checkpoint-native student scorer')
    if cfg.model.config.image_backbone.get('finetune', False):
        raise ValueError('This package supports inference only; fine-tuning is unavailable')
    if cfg.evaluation.get('score_backend', 'official') != 'official':
        raise ValueError('Only official evaluation metric caches are supported')
    if cfg.evaluation.scene_loader.get('cache_path') is not None:
        raise ValueError('Inference reads scene inputs directly; feature/target caches are unsupported')
    return cfg


def check_inputs(cfg, inference_only=False):
    """Report missing assets before importing the GPU or simulator dependencies."""
    paths = {
        'checkpoint': cfg.evaluation.checkpoint_path,
        'image backbone': cfg.model.config.image_backbone.model_weights,
        'NAVSIM devkit': cfg.evaluation.navsim_root,
    }
    for key, value in cfg.evaluation.scene_loader.items():
        if key.endswith('_path') and key != 'cache_path' and value:
            paths[f'scene_loader.{key}'] = value
    if not inference_only:
        paths['metric cache'] = cfg.evaluation.metric_cache_path
        paths['maps'] = cfg.evaluation.env.NUPLAN_MAPS_ROOT
    feature_root = cfg.evaluation.get('feature_navsim_root', cfg.evaluation.navsim_root)
    paths['NAVSIM feature builder'] = str(Path(feature_root) / 'navsim/agents/drivoR/drivor_features.py')
    if cfg.evaluation.score_protocol == 'navsim_v2_navhard':
        paths['NAVHARD stage metadata'] = str(Path(cfg.evaluation.navsim_root) / 'navsim/planning/script/config/common/train_test_split/navhard_two_stage.yaml')
    missing = [f'  {label}: {path}' for label, path in paths.items() if not path or not Path(path).exists()]
    checkpoint = Path(cfg.evaluation.checkpoint_path)
    if checkpoint.is_dir():
        latest = checkpoint / 'latest'
        state = (checkpoint / latest.read_text().strip() if latest.is_file() else checkpoint) / 'mp_rank_00_model_states.pt'
        if not state.is_file():
            missing.append(f'  DeepSpeed model state: {state}')
    if missing:
        raise FileNotFoundError('Missing inference assets (see README.md for path settings):\n' + '\n'.join(missing))


def run(cfg, args):
    # Model and simulator imports are delayed until the configured environment is ready.
    import sys
    for key, value in cfg.evaluation.env.items():
        os.environ[str(key)] = str(value)
    sys.path.insert(0, str(Path(cfg.evaluation.navsim_root).resolve()))
    from fpd_drive.inference import evaluate
    return evaluate(cfg, inference_only=args.inference_only)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=('base', 'pro'), required=True)
    parser.add_argument('--benchmark', choices=('v1', 'v2'), required=True)
    parser.add_argument('--config', type=Path, default=os.environ.get('CONFIG_PATH'), help='Replace the selected preset YAML')
    parser.add_argument('--checkpoint-path', '--checkpoint_path', dest='checkpoint_path')
    parser.add_argument('--gpus', help='Single nonnegative GPU ID')
    parser.add_argument('--denoise-steps', type=int, help='Override DDIM steps (NAVSIM v1 only)')
    parser.add_argument('--print-config', action='store_true', help='Print resolved YAML without loading model dependencies')
    parser.add_argument('--check-inputs', action='store_true', help='Check required input paths and exit')
    parser.add_argument('--inference-only', action='store_true', help='Write trajectories without official metric scoring')
    parser.add_argument('overrides', nargs='*', help='OmegaConf key=value overrides')
    args = parser.parse_args(argv)
    if args.gpus is not None:
        if not args.gpus.isdigit():
            parser.error('--gpus expects a single nonnegative GPU ID')
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    if args.config is not None:
        args.config = args.config.expanduser().resolve()
    if args.checkpoint_path is not None:
        args.checkpoint_path = str(Path(args.checkpoint_path).expanduser().resolve())
    # All preset-relative paths are anchored to this checkout.
    os.chdir(ROOT)
    for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ.setdefault(variable, '1')
    try:
        cfg = load_config(args)
        if args.print_config:
            from omegaconf import OmegaConf
            print(OmegaConf.to_yaml(cfg, resolve=True))
            return 0
        check_inputs(cfg, args.inference_only)
        if args.check_inputs:
            print('All required input paths exist; model execution has not been tested.')
            return 0
        print(f'Output: {run(cfg, args)}')
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
