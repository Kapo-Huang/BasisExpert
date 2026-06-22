import os
import unittest
from unittest import mock

from var_expert_inr.utils.runtime import configure_thread_env


class RuntimeThreadEnvTestCase(unittest.TestCase):
    def test_configure_thread_env_sets_safe_defaults(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            intra_threads, interop_threads = configure_thread_env()

            self.assertEqual(intra_threads, 64)
            self.assertEqual(interop_threads, 64)
            self.assertEqual(os.environ["OPENBLAS_NUM_THREADS"], "64")
            self.assertEqual(os.environ["OMP_NUM_THREADS"], "64")
            self.assertEqual(os.environ["MKL_NUM_THREADS"], "64")
            self.assertEqual(os.environ["NUMEXPR_NUM_THREADS"], "64")

    def test_configure_thread_env_respects_var_expert_override(self):
        with mock.patch.dict(
            os.environ,
            {
                "VAR_EXPERT_INR_NUM_THREADS": "8",
                "VAR_EXPERT_INR_NUM_INTEROP_THREADS": "4",
            },
            clear=True,
        ):
            intra_threads, interop_threads = configure_thread_env()

            self.assertEqual(intra_threads, 8)
            self.assertEqual(interop_threads, 4)
            self.assertEqual(os.environ["OPENBLAS_NUM_THREADS"], "8")
            self.assertEqual(os.environ["OMP_NUM_THREADS"], "8")
            self.assertEqual(os.environ["MKL_NUM_THREADS"], "8")
            self.assertEqual(os.environ["NUMEXPR_NUM_THREADS"], "8")


if __name__ == "__main__":
    unittest.main()


