import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


HELPER = Path(__file__).resolve().parents[1] / "scripts" / "lib" / "batch_runner.sh"
RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "main" / "run_all.sh"
MOE_RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "main" / "run_moe_non_ionization.sh"
MOE_LIST = Path(__file__).resolve().parents[1] / "scripts" / "main" / "moe_non_ionization.list"
COMBINED_RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "main" / "run_neural_expert_non_ionization.sh"
COMBINED_LIST = Path(__file__).resolve().parents[1] / "scripts" / "main" / "neural_expert_non_ionization.list"
V4_RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "ablation" / "run_depth_and_regularization.sh"
V5_RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "exploration" / "run_optimizer_tuning.sh"
V6_RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "exploration" / "run_ecnr_tuning.sh"
COORDNET_MVNET_STSR_RUNNER = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "main" / "run_selected_datasets.sh"
)
COMBUSTION_FV_APMG_INSTANTVNR_RUNNER = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "main" / "run_combustion_fv_apmg_instantvnr.sh"
)
COMBUSTION_STSR_MVNET_RUNNER = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "main" / "run_combustion_stsr_mvnet.sh"
)
COMBUSTION_MINER_ECNR_RUNNER = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "main" / "run_combustion_miner_ecnr.sh"
)


class BatchRunnerTestCase(unittest.TestCase):
    def _bash(self):
        candidates = []
        if os.name == "nt":
            candidates.extend([r"D:\Git\bin\bash.exe", r"C:\Program Files\Git\bin\bash.exe"])
        candidates.append(shutil.which("bash"))
        return next((path for path in candidates if path and Path(path).exists()), None)

    def test_four_column_resume_last_status_and_dry_run(self):
        bash = self._bash()
        if bash is None:
            self.skipTest("Bash is not installed")
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "configs").mkdir()
            for name in ("a.yaml", "b.yaml", "c.yaml", "missing.yaml"):
                (root / "configs" / name).write_text("exp_id: test\n", encoding="utf-8")
            log_root = root / "batch"
            log_root.mkdir()
            status = log_root / "status.tsv"
            with status.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(
                    "config\tstatus\texit_code\tlog\n"
                    "configs/a.yaml\trunning\t\ta.attempt-1.log\n"
                    "configs/a.yaml\tfailed\t7\ta.attempt-1.log\n"
                    "configs/b.yaml\trunning\t\tb.attempt-1.log\n"
                    "configs/b.yaml\tok\t0\tb.attempt-1.log\n"
                    "configs/c.yaml\tunexpected\t\tc.log\n"
                )
            script = root / "exercise.sh"
            script.write_text(
                """#!/usr/bin/env bash
set -uo pipefail
if command -v cygpath >/dev/null 2>&1; then
    REPO_ROOT="$(cygpath -u "$1")"
    LOG_ROOT="$(cygpath -u "$2")"
    HELPER="$(cygpath -u "$3")"
else
    REPO_ROOT="$1"
    LOG_ROOT="$2"
    HELPER="$3"
fi
STATUS_FILE="${LOG_ROOT}/status.tsv"
FAILURE_FILE="${LOG_ROOT}/failed.txt"
CONDA_ENV=compression
DRY_RUN=1
source "${HELPER}"
batch_init_status
printf 'latest=%s,%s,%s,%s\n' "$(batch_latest_status configs/a.yaml)" "$(batch_latest_status configs/b.yaml)" "$(batch_latest_status configs/c.yaml)" "$(batch_latest_status configs/missing.yaml)"
printf 'attempt=%s,%s,%s,%s\n' "$(batch_attempt_number configs/a.yaml)" "$(batch_attempt_number configs/b.yaml)" "$(batch_attempt_number configs/c.yaml)" "$(batch_attempt_number configs/missing.yaml)"
before="$(wc -l < "${STATUS_FILE}")"
batch_run_one_config "${REPO_ROOT}/configs/b.yaml" 1 2
batch_run_one_config "${REPO_ROOT}/configs/missing.yaml" 2 2
after="$(wc -l < "${STATUS_FILE}")"
printf 'lines=%s,%s\n' "${before}" "${after}"
batch_rebuild_failures
printf 'failed=%s\n' "$(tr '\n' ',' < "${FAILURE_FILE}")"
batch_append_status configs/a.yaml running '' a.attempt-2.log
batch_append_status configs/a.yaml ok 0 a.attempt-2.log
batch_rebuild_failures
printf 'after_append=%s,%s,%s\n' "$(batch_latest_status configs/a.yaml)" "$(batch_attempt_number configs/a.yaml)" "$(tr '\n' ',' < "${FAILURE_FILE}")"
""",
                encoding="utf-8",
                newline="\n",
            )
            environment = os.environ.copy()
            if os.name == "nt":
                git_root = Path(bash).parents[1]
                environment["PATH"] = os.pathsep.join(
                    [str(git_root / "usr" / "bin"), str(git_root / "bin"), environment.get("PATH", "")]
                )
            completed = subprocess.run(
                [bash, script.as_posix(), root.as_posix(), log_root.as_posix(), HELPER.as_posix()],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            )
            output = completed.stdout
            self.assertIn("latest=failed,ok,unexpected,", output, completed.stderr)
            self.assertIn("attempt=2,2,1,1", output)
            self.assertIn("SKIP ok: configs/b.yaml", output)
            self.assertIn("RUN attempt=1 previous=missing: configs/missing.yaml", output)
            self.assertIn("lines=6,6", output)
            self.assertIn("failed=configs/a.yaml,", output)
            self.assertIn("after_append=ok,3,", output)
            self.assertEqual(status.read_text(encoding="utf-8").count("\n"), 8)

    def test_formal_runner_uses_config_list_subset_in_matrix_order(self):
        bash = self._bash()
        if bash is None:
            self.skipTest("Bash is not installed")
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            selection = root / "selected.list"
            selection.write_text(
                "# Only these two configs should run.\n"
                "configs/rd_curve/CoordNet/Size082/ionization__GT.yaml\n"
                "configs/main/VarExpert/combustion_40NH3_1.yaml\n",
                encoding="utf-8",
                newline="\n",
            )
            environment = os.environ.copy()
            environment.update(
                {
                    "DRY_RUN": "1",
                    "MAX_PARALLEL_JOBS": "1",
                    "CONFIG_LIST_FILE": selection.as_posix(),
                    "BATCH_LOG_ROOT": (root / "batch").as_posix(),
                }
            )
            if os.name == "nt":
                git_root = Path(bash).parents[1]
                environment["PATH"] = os.pathsep.join(
                    [str(git_root / "usr" / "bin"), str(git_root / "bin"), environment.get("PATH", "")]
                )
            completed = subprocess.run(
                [bash, RUNNER.as_posix()],
                cwd=RUNNER.parents[1],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            )
            output = completed.stdout

            self.assertIn("Selected 2 of 355 configs", output)
            main_position = output.index("configs/main/VarExpert/combustion_40NH3_1.yaml")
            size_position = output.index("configs/rd_curve/CoordNet/Size082/ionization__GT.yaml")
            self.assertLess(main_position, size_position)
            self.assertIn("Completed 2 configs; failures=0", output)

    def test_moe_runner_dry_run_selects_only_22_main_configs(self):
        bash = self._bash()
        if bash is None:
            self.skipTest("Bash is not installed")
        with tempfile.TemporaryDirectory() as tmpdir:
            environment = os.environ.copy()
            environment.update(
                {
                    "DRY_RUN": "1",
                    "BATCH_LOG_ROOT": (Path(tmpdir) / "batch").as_posix(),
                }
            )
            if os.name == "nt":
                git_root = Path(bash).parents[1]
                environment["PATH"] = os.pathsep.join(
                    [str(git_root / "usr" / "bin"), str(git_root / "bin"), environment.get("PATH", "")]
                )
            completed = subprocess.run(
                [bash, MOE_RUNNER.as_posix()],
                cwd=MOE_RUNNER.parents[1],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            )
            output = completed.stdout

            self.assertIn("Selected 22 of 355 configs", output)
            self.assertEqual(output.count("DRY_RUN:"), 22)
            bathymetry = output.index("main:MoE-INR:bathymetry:all")
            combustion = output.index("main:MoE-INR:combustion_40NH3_1:all")
            katrina = output.index("main:MoE-INR:katrina:all")
            self.assertLess(bathymetry, combustion)
            self.assertLess(combustion, katrina)
            self.assertNotIn("main:MoE-INR:ionization", output)
            self.assertNotIn("size:MoE-INR", output)

    def test_combined_runner_dry_run_selects_exact_scope_and_default_parallelism(self):
        bash = self._bash()
        if bash is None:
            self.skipTest("Bash is not installed")
        with tempfile.TemporaryDirectory() as tmpdir:
            environment = os.environ.copy()
            environment.update(
                {
                    "DRY_RUN": "1",
                    "BATCH_LOG_ROOT": (Path(tmpdir) / "batch").as_posix(),
                }
            )
            environment.pop("MAX_PARALLEL_JOBS", None)
            if os.name == "nt":
                git_root = Path(bash).parents[1]
                environment["PATH"] = os.pathsep.join(
                    [str(git_root / "usr" / "bin"), str(git_root / "bin"), environment.get("PATH", "")]
                )
            completed = subprocess.run(
                [bash, COMBINED_RUNNER.as_posix()],
                cwd=COMBINED_RUNNER.parents[1],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            )
            output = completed.stdout

            self.assertIn("SIREN + NeuralExpert non-Ionization matrix: 45 configs, max_parallel=5", output)
            self.assertIn("Selected 45 of 355 configs", output)
            self.assertEqual(output.count("DRY_RUN:"), 45)
            siren = output.index("main:SIREN:combustion_40NH3_1:all")
            bathymetry_manager = output.index("main:NeuralExpert:bathymetry:manager")
            combustion_manager = output.index("main:NeuralExpert:combustion_40NH3_1:manager")
            bathymetry_main = output.index("main:NeuralExpert:bathymetry:main")
            combustion_main = output.index("main:NeuralExpert:combustion_40NH3_1:main")
            self.assertLess(siren, bathymetry_manager)
            self.assertLess(bathymetry_manager, combustion_manager)
            self.assertLess(combustion_manager, bathymetry_main)
            self.assertLess(bathymetry_main, combustion_main)
            self.assertNotIn("main:NeuralExpert:katrina", output)
            self.assertNotIn("main:NeuralExpert:ionization", output)
            self.assertNotIn("size:NeuralExpert", output)

    def test_combined_runner_validates_training_artifacts_and_psnr_results(self):
        bash = self._bash()
        if bash is None:
            self.skipTest("Bash is not installed")
        selected = [
            line.split("#", 1)[0].strip()
            for line in COMBINED_LIST.read_text(encoding="utf-8").splitlines()
            if line.split("#", 1)[0].strip()
        ]
        cases = (
            ("ok", None, 0, "Validated SIREN=13/13"),
            ("training_failure", "failed_status", 1, "INVALID status=failed"),
            ("ok", "missing_siren_psnr", 1, "INVALID missing or non-finite final SIREN PSNR"),
            ("ok", "missing_manager_export", 1, "INVALID missing manager checkpoint export"),
            ("nonfinite", None, 1, "INVALID missing or malformed NeuralExpert PSNR result"),
            ("failure", None, 1, "INVALID NeuralExpert PSNR evaluation failed"),
        )

        for evaluation_mode, corruption, expected_code, expected_message in cases:
            with self.subTest(evaluation_mode=evaluation_mode, corruption=corruption), tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                log_root = root / "batch"
                logs = log_root / "logs"
                fake_bin = root / "bin"
                logs.mkdir(parents=True)
                fake_bin.mkdir()
                status_path = log_root / "status.tsv"
                log_paths = {}
                with status_path.open("w", encoding="utf-8", newline="\n") as handle:
                    handle.write("config\tstatus\texit_code\tlog\n")
                    for index, relative in enumerate(selected):
                        log_path = logs / f"config-{index}.log"
                        log_paths[relative] = log_path
                        if relative.startswith("configs/main/SIREN/"):
                            content = "PSNR epoch 600/600: aggregate=42.00 time=1.0s\n"
                        elif relative.endswith("__managerpretrain.yaml"):
                            content = "Exported manager pretrain checkpoint to /fake/manager.pth\n"
                        else:
                            content = "Saved final state dict to /fake/trained_models/model_final.pth\n"
                        log_path.write_text(content, encoding="utf-8")
                        status = "failed" if corruption == "failed_status" and index == 0 else "ok"
                        exit_code = "7" if status == "failed" else "0"
                        handle.write(f"{relative}\t{status}\t{exit_code}\t{log_path.as_posix()}\n")

                if corruption == "missing_siren_psnr":
                    relative = next(path for path in selected if path.startswith("configs/main/SIREN/"))
                    log_paths[relative].write_text("training ended without PSNR\n", encoding="utf-8")
                if corruption == "missing_manager_export":
                    relative = next(path for path in selected if path.endswith("__managerpretrain.yaml"))
                    log_paths[relative].write_text("manager training ended without export\n", encoding="utf-8")

                fake_conda = fake_bin / "conda"
                fake_conda.write_text(
                    """#!/usr/bin/env bash
mode="${FAKE_EVAL_MODE:-ok}"
config=""
is_train=0
while [[ "$#" -gt 0 ]]; do
    if [[ "$1" == "train" ]]; then
        is_train=1
    fi
    if [[ "$1" == "--config" ]]; then
        shift
        config="$1"
    fi
    shift
done
if [[ "${mode}" == "training_failure" && "${is_train}" == "1" ]]; then
    exit 7
fi
if [[ "${mode}" == "failure" ]]; then
    exit 7
fi
filename="${config##*/}"
target="${filename#*__}"
target="${target%.yaml}"
psnr="42.5"
if [[ "${mode}" == "nonfinite" ]]; then
    psnr="nan"
fi
printf 'NEURAL_EXPERT_PSNR\\t%s\\t%s\\t/fake/run\\t/fake/metrics.json\\n' "${target}" "${psnr}"
""",
                    encoding="utf-8",
                    newline="\n",
                )
                fake_conda.chmod(0o755)

                environment = os.environ.copy()
                environment.update(
                    {
                        "BATCH_LOG_ROOT": log_root.as_posix(),
                        "FAKE_EVAL_MODE": evaluation_mode,
                    }
                )
                environment.pop("DRY_RUN", None)
                path_parts = [str(fake_bin)]
                if os.name == "nt":
                    git_root = Path(bash).parents[1]
                    path_parts.extend([str(git_root / "usr" / "bin"), str(git_root / "bin")])
                path_parts.append(environment.get("PATH", ""))
                environment["PATH"] = os.pathsep.join(path_parts)

                completed = subprocess.run(
                    [bash, COMBINED_RUNNER.as_posix()],
                    cwd=COMBINED_RUNNER.parents[1],
                    check=False,
                    capture_output=True,
                    text=True,
                    env=environment,
                )
                self.assertEqual(completed.returncode, expected_code, completed.stdout + completed.stderr)
                combined_output = completed.stdout + completed.stderr
                self.assertIn(expected_message, combined_output)
                if expected_code == 0:
                    summary = (log_root / "experiment_psnr.tsv").read_text(encoding="utf-8").splitlines()
                    self.assertEqual(len(summary), 30)
                    self.assertEqual(sum("\tSIREN\t" in f"\t{line}\t" for line in summary), 13)
                    self.assertEqual(sum("\tNeuralExpert\t" in f"\t{line}\t" for line in summary), 16)

    def test_exploration_v4_dry_run_has_exact_matrix_and_default_parallelism(self):
        bash = self._bash()
        if bash is None:
            self.skipTest("Bash is not installed")
        with tempfile.TemporaryDirectory() as tmpdir:
            environment = os.environ.copy()
            environment.update(
                {
                    "DRY_RUN": "1",
                    "BATCH_LOG_ROOT": (Path(tmpdir) / "batch").as_posix(),
                    "RUN_TOKEN": "test-v4",
                }
            )
            environment.pop("MAX_PARALLEL_JOBS", None)
            if os.name == "nt":
                git_root = Path(bash).parents[1]
                environment["PATH"] = os.pathsep.join(
                    [str(git_root / "usr" / "bin"), str(git_root / "bin"), environment.get("PATH", "")]
                )
            completed = subprocess.run(
                [bash, V4_RUNNER.as_posix()],
                cwd=V4_RUNNER.parents[1],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            )
            output = completed.stdout
            self.assertEqual(output.count("DRY_RUN:"), 30)
            self.assertIn("CoordNet equal-budget depth (30 configs, max_parallel=5)", output)
            self.assertIn("Completed 30 exploration-v4 configs; failures=0", output)

    def test_exploration_v5_dry_run_has_exact_matrix_and_method_dispatch(self):
        bash = self._bash()
        if bash is None:
            self.skipTest("Bash is not installed")
        syntax = subprocess.run(
            [bash, "-n", V5_RUNNER.as_posix()],
            cwd=V5_RUNNER.parents[1],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(syntax.returncode, 0, syntax.stdout + syntax.stderr)
        with tempfile.TemporaryDirectory() as tmpdir:
            environment = os.environ.copy()
            environment.update(
                {
                    "DRY_RUN": "1",
                    "BATCH_LOG_ROOT": (Path(tmpdir) / "batch").as_posix(),
                    "RUN_TOKEN": "test-v5",
                }
            )
            environment.pop("MAX_PARALLEL_JOBS", None)
            environment.pop("DEVICE", None)
            if os.name == "nt":
                git_root = Path(bash).parents[1]
                environment["PATH"] = os.pathsep.join(
                    [
                        str(git_root / "usr" / "bin"),
                        str(git_root / "bin"),
                        environment.get("PATH", ""),
                    ]
                )
            completed = subprocess.run(
                [bash, V5_RUNNER.as_posix()],
                cwd=V5_RUNNER.parents[1],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            )
            output = completed.stdout
            self.assertIn("Selected device: cuda:0 (CUDA_VISIBLE_DEVICES=0)", output)
            self.assertEqual(output.count("DRY_RUN:"), 42)
            self.assertEqual(output.count("var_expert_inr.methods.fv_srn.cli"), 24)
            self.assertEqual(output.count("var_expert_inr.cli"), 18)
            self.assertIn(
                "fV-SRN structure and optimizer sweep (24 configs, max_parallel=5)",
                output,
            )
            self.assertIn("InstantVNR optimizer sweep (18 configs, max_parallel=5)", output)
            self.assertIn("Completed 42 exploration-v5 configs; failures=0", output)

    def test_exploration_v5_device_selects_physical_cuda_index(self):
        bash = self._bash()
        if bash is None:
            self.skipTest("Bash is not installed")
        with tempfile.TemporaryDirectory() as tmpdir:
            environment = os.environ.copy()
            environment.update(
                {
                    "DEVICE": "cuda:1",
                    "DRY_RUN": "1",
                    "BATCH_LOG_ROOT": (Path(tmpdir) / "batch").as_posix(),
                    "RUN_TOKEN": "test-v5-device",
                }
            )
            if os.name == "nt":
                git_root = Path(bash).parents[1]
                environment["PATH"] = os.pathsep.join(
                    [
                        str(git_root / "usr" / "bin"),
                        str(git_root / "bin"),
                        environment.get("PATH", ""),
                    ]
                )
            completed = subprocess.run(
                [bash, V5_RUNNER.as_posix()],
                cwd=V5_RUNNER.parents[1],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertIn(
                "Selected device: cuda:1 (CUDA_VISIBLE_DEVICES=1)",
                completed.stdout,
            )

            environment["DEVICE"] = "cpu"
            invalid = subprocess.run(
                [bash, V5_RUNNER.as_posix()],
                cwd=V5_RUNNER.parents[1],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertEqual(invalid.returncode, 2)
            self.assertIn("DEVICE must use cuda:N form", invalid.stderr)

    def test_exploration_v6_dry_run_has_exact_ecnr_matrix(self):
        bash = self._bash()
        if bash is None:
            self.skipTest("Bash is not installed")
        syntax = subprocess.run(
            [bash, "-n", V6_RUNNER.as_posix()],
            cwd=V6_RUNNER.parents[1],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(syntax.returncode, 0, syntax.stdout + syntax.stderr)
        with tempfile.TemporaryDirectory() as tmpdir:
            environment = os.environ.copy()
            environment.update(
                {
                    "DRY_RUN": "1",
                    "BATCH_LOG_ROOT": (Path(tmpdir) / "batch").as_posix(),
                    "RUN_TOKEN": "test-v6",
                }
            )
            environment.pop("MAX_PARALLEL_JOBS", None)
            environment.pop("DEVICE", None)
            if os.name == "nt":
                git_root = Path(bash).parents[1]
                environment["PATH"] = os.pathsep.join(
                    [
                        str(git_root / "usr" / "bin"),
                        str(git_root / "bin"),
                        environment.get("PATH", ""),
                    ]
                )
            completed = subprocess.run(
                [bash, V6_RUNNER.as_posix()],
                cwd=V6_RUNNER.parents[1],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            )
            output = completed.stdout
            self.assertEqual(output.count("DRY_RUN:"), 18)
            self.assertEqual(output.count("var_expert_inr.cli"), 18)
            self.assertIn(
                "ECNR main-training sweep (18 configs, max_parallel=1)",
                output,
            )
            self.assertIn("Completed 18 exploration-v6 configs; failures=0", output)

    def test_coordnet_mvnet_stsr_dry_run_has_exact_stages(self):
        bash = self._bash()
        if bash is None:
            self.skipTest("Bash is not installed")
        with tempfile.TemporaryDirectory() as tmpdir:
            environment = os.environ.copy()
            environment.update(
                {
                    "DRY_RUN": "1",
                    "BATCH_LOG_ROOT": (Path(tmpdir) / "batch").as_posix(),
                    "RUN_TOKEN": "test-coordnet-mvnet-stsr",
                }
            )
            environment.pop("MAX_PARALLEL_JOBS", None)
            if os.name == "nt":
                git_root = Path(bash).parents[1]
                environment["PATH"] = os.pathsep.join(
                    [str(git_root / "usr" / "bin"), str(git_root / "bin"), environment.get("PATH", "")]
                )
            completed = subprocess.run(
                [bash, COORDNET_MVNET_STSR_RUNNER.as_posix()],
                cwd=COORDNET_MVNET_STSR_RUNNER.parents[1],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            )
            output = completed.stdout
            self.assertEqual(output.count("DRY_RUN:"), 15)
            self.assertIn("CoordNet Combustion (13 configs, max_parallel=5)", output)
            self.assertIn("MVNet Katrina (1 config, max_parallel=5)", output)
            self.assertIn("STSR-INR RedSea (1 config, max_parallel=5)", output)
            self.assertIn("Completed 15 configs; failures=0", output)

    def test_combustion_group_runners_have_exact_dry_run_stages(self):
        bash = self._bash()
        if bash is None:
            self.skipTest("Bash is not installed")
        with tempfile.TemporaryDirectory() as tmpdir:
            environment = os.environ.copy()
            environment.update(
                {
                    "DRY_RUN": "1",
                    "BATCH_LOG_ROOT": (Path(tmpdir) / "batch").as_posix(),
                }
            )
            environment.pop("MAX_PARALLEL_JOBS", None)
            environment.pop("CONFIG_LIST_FILE", None)
            if os.name == "nt":
                git_root = Path(bash).parents[1]
                environment["PATH"] = os.pathsep.join(
                    [str(git_root / "usr" / "bin"), str(git_root / "bin"), environment.get("PATH", "")]
                )

            results = {}
            for runner in (
                COMBUSTION_FV_APMG_INSTANTVNR_RUNNER,
                COMBUSTION_STSR_MVNET_RUNNER,
                COMBUSTION_MINER_ECNR_RUNNER,
            ):
                syntax = subprocess.run(
                    [bash, "-n", runner.as_posix()],
                    cwd=runner.parents[1],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(syntax.returncode, 0, syntax.stdout + syntax.stderr)
                environment["RUN_TOKEN"] = f"test-{runner.stem}"
                completed = subprocess.run(
                    [bash, runner.as_posix()],
                    cwd=runner.parents[1],
                    check=True,
                    capture_output=True,
                    text=True,
                    env=environment,
                )
                results[runner] = completed.stdout

            first = results[COMBUSTION_FV_APMG_INSTANTVNR_RUNNER]
            self.assertEqual(first.count("DRY_RUN:"), 36)
            self.assertEqual(first.count("var_expert_inr.methods.fv_srn.cli"), 12)
            self.assertEqual(first.count("var_expert_inr.methods.apmgsrn.cli"), 12)
            self.assertEqual(first.count("python -m var_expert_inr.cli"), 12)
            self.assertLess(first.index("== fV-SRN Combustion"), first.index("== APMGSRN Combustion"))
            self.assertLess(first.index("== APMGSRN Combustion"), first.index("== InstantVNR Combustion"))

            joint = results[COMBUSTION_STSR_MVNET_RUNNER]
            self.assertEqual(joint.count("DRY_RUN:"), 2)
            self.assertEqual(joint.count("python -m var_expert_inr.cli"), 2)
            self.assertLess(joint.index("== STSR-INR Combustion"), joint.index("== MVNet Combustion"))

            adaptive = results[COMBUSTION_MINER_ECNR_RUNNER]
            self.assertEqual(adaptive.count("DRY_RUN:"), 24)
            self.assertEqual(adaptive.count("python -m var_expert_inr.cli"), 24)
            self.assertLess(adaptive.index("== MINER Combustion"), adaptive.index("== ECNR Combustion"))

    def test_combustion_group_runner_rejects_invalid_lists_before_training(self):
        bash = self._bash()
        if bash is None:
            self.skipTest("Bash is not installed")
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            environment = os.environ.copy()
            environment.update(
                {
                    "DRY_RUN": "1",
                    "BATCH_LOG_ROOT": (root / "batch").as_posix(),
                    "RUN_TOKEN": "test-invalid-combustion-list",
                }
            )
            if os.name == "nt":
                git_root = Path(bash).parents[1]
                environment["PATH"] = os.pathsep.join(
                    [str(git_root / "usr" / "bin"), str(git_root / "bin"), environment.get("PATH", "")]
                )

            stsr = "configs/main/STSR-INR/combustion_40NH3_1.yaml"
            cases = {
                "duplicate": (f"{stsr}\n{stsr}\n", "Duplicate Combustion config"),
                "out-of-scope": (
                    "configs/main/SIREN/combustion_40NH3_1__Temperature.yaml\n",
                    "Out-of-scope Combustion config",
                ),
                "missing": ("configs/main/STSR-INR/not-present.yaml\n", "Config not found"),
                "wrong-count": (f"{stsr}\n", "Expected MVNet Combustion=1 configs"),
            }
            for name, (content, expected_error) in cases.items():
                config_list = root / f"{name}.list"
                config_list.write_text(content, encoding="utf-8", newline="\n")
                environment["CONFIG_LIST_FILE"] = config_list.as_posix()
                completed = subprocess.run(
                    [bash, COMBUSTION_STSR_MVNET_RUNNER.as_posix()],
                    cwd=COMBUSTION_STSR_MVNET_RUNNER.parents[1],
                    check=False,
                    capture_output=True,
                    text=True,
                    env=environment,
                )
                self.assertEqual(completed.returncode, 2, (name, completed.stdout, completed.stderr))
                self.assertIn(expected_error, completed.stderr, name)
                self.assertNotIn("DRY_RUN:", completed.stdout, name)

    def test_moe_runner_validates_terminal_status_and_final_psnr(self):
        bash = self._bash()
        if bash is None:
            self.skipTest("Bash is not installed")
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            log_root = root / "batch"
            logs = log_root / "logs"
            logs.mkdir(parents=True)
            selected = [
                line.split("#", 1)[0].strip()
                for line in MOE_LIST.read_text(encoding="utf-8").splitlines()
                if line.split("#", 1)[0].strip()
            ]
            status = log_root / "status.tsv"
            with status.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write("config\tstatus\texit_code\tlog\n")
                for index, config in enumerate(selected):
                    log_path = logs / f"config-{index}.log"
                    log_path.write_text(
                        "2026-08-12 | INFO | PSNR epoch 600/600: aggregate=42.00 time=1.0s\n",
                        encoding="utf-8",
                    )
                    handle.write(f"{config}\tok\t0\t{log_path.as_posix()}\n")

            environment = os.environ.copy()
            environment["BATCH_LOG_ROOT"] = log_root.as_posix()
            if os.name == "nt":
                git_root = Path(bash).parents[1]
                environment["PATH"] = os.pathsep.join(
                    [str(git_root / "usr" / "bin"), str(git_root / "bin"), environment.get("PATH", "")]
                )
            completed = subprocess.run(
                [bash, MOE_RUNNER.as_posix()],
                cwd=MOE_RUNNER.parents[1],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertIn("Validated 22/22", completed.stdout)

            (logs / "config-0.log").write_text("training ended without PSNR\n", encoding="utf-8")
            failed = subprocess.run(
                [bash, MOE_RUNNER.as_posix()],
                cwd=MOE_RUNNER.parents[1],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertEqual(failed.returncode, 1)
            self.assertIn("INVALID missing final PSNR", failed.stderr)


if __name__ == "__main__":
    unittest.main()
