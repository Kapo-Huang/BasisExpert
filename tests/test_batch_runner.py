import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


HELPER = Path(__file__).resolve().parents[1] / "scripts" / "lib" / "batch_runner.sh"
RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "run_all_configs.sh"


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
                "configs/SIREN/Size082/ionization__GT.yaml\n"
                "configs/VarExpert/combustion_40NH3_1.yaml\n",
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

            self.assertIn("Selected 2 of 356 configs", output)
            main_position = output.index("configs/VarExpert/combustion_40NH3_1.yaml")
            size_position = output.index("configs/SIREN/Size082/ionization__GT.yaml")
            self.assertLess(main_position, size_position)
            self.assertIn("Completed 2 configs; failures=0", output)


if __name__ == "__main__":
    unittest.main()
