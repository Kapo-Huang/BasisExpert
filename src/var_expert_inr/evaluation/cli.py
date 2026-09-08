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
    parser.add_argument(
        "--metrics",
        default=None,
        help="Comma-separated psnr,ssim,lpips,error,pearson_error,mi_error,decode_time,training_time,inference_time,memory",
    )
    parser.add_argument(
        "--timesteps",
        default="all",
        help="all, uniform:N, or comma-separated inclusive selections",
    )
    parser.add_argument("--targets", default="all", help="all or comma-separated target names")
    parser.add_argument("--source", choices=("auto", "checkpoint", "prediction"), default=None)
    if include_source_paths:
        parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--prediction", default=None)
    parser.add_argument("--render", action="store_true", help="Render selected predictions")
    parser.add_argument("--eval-config", default=None, help="Render/evaluation profile YAML")
    parser.add_argument("--overwrite", action="store_true", help="Bypass compatible cached evaluations")
    parser.add_argument("--device", default=None)
    parser.add_argument("--error-vmin", type=float, default=None, help="Fixed error-render minimum in percent")
    parser.add_argument("--error-vmax", type=float, default=None, help="Fixed error-render maximum in percent")
    parser.add_argument("--result-root", default="EvalResult", help="Schema-v2 result root")
    parser.add_argument("--evaluation-id", default="default", help="Evaluation namespace")
    parser.add_argument("--training-probe-samples", type=int, default=72_000_000)
    parser.add_argument("--training-total-samples", type=int, default=14_400_000_000)
    parser.add_argument("--inference-fraction", type=float, default=0.1)


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
        result_root=args.result_root,
        evaluation_id=args.evaluation_id,
        error_vmin=args.error_vmin,
        error_vmax=args.error_vmax,
        training_probe_samples=args.training_probe_samples,
        training_total_samples=args.training_total_samples,
        inference_fraction=args.inference_fraction,
    )
    print(json.dumps({"output_dir": str(result["output_dir"])}, ensure_ascii=False))
    return result
