from __future__ import annotations

import argparse
import json

from .service import evaluate_run


def add_run_evaluation_arguments(
    parser: argparse.ArgumentParser,
    *,
    run_required: bool = True,
    include_source_paths: bool = True,
) -> None:
    parser.add_argument("--run", required=run_required, help="Saved run directory")
    parser.add_argument("--metrics", default=None, help="Comma-separated psnr,ssim,lpips,decode_time,memory")
    parser.add_argument("--timesteps", default="all", help="all or comma-separated inclusive selections")
    parser.add_argument("--targets", default="all", help="all or comma-separated target names")
    parser.add_argument("--source", choices=("auto", "checkpoint", "prediction"), default=None)
    if include_source_paths:
        parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--prediction", default=None)
    parser.add_argument("--render", action="store_true", help="Render selected predictions")
    parser.add_argument("--eval-config", default=None, help="Render/evaluation profile YAML")
    parser.add_argument("--overwrite", action="store_true", help="Bypass compatible cached evaluations")
    parser.add_argument("--device", default=None)


def execute_run_evaluation(args: argparse.Namespace) -> dict:
    result = evaluate_run(
        args.run,
        metrics=args.metrics,
        timesteps=args.timesteps,
        targets=args.targets,
        source=args.source,
        checkpoint=getattr(args, "checkpoint", None),
        prediction=getattr(args, "prediction", None),
        render=bool(args.render),
        render_profile=args.eval_config,
        overwrite=bool(args.overwrite),
        device=args.device,
    )
    print(json.dumps({"output_dir": str(result["output_dir"])}, ensure_ascii=False))
    return result
