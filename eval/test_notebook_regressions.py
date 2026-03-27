import os
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

os.environ.setdefault("XDG_CACHE_HOME", "/tmp/pp_fno_pinn_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/pp_fno_pinn_cache/matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from experiment_runner import run_inverse_problem_case, run_learning_pde_solution_case


class NotebookRegressionTests(unittest.TestCase):
    def test_learning_pde_solution_fno_rmse(self):
        metrics = run_learning_pde_solution_case(
            "fno",
            seed=0,
            n_obs=40,
            use_gpu=False,
        )
        self.assertLess(metrics.rmse, 5e-2)

    def test_learning_pde_solution_pinn_rmse(self):
        metrics = run_learning_pde_solution_case(
            "pinn",
            seed=0,
            n_obs=40,
            use_gpu=False,
        )
        self.assertLess(metrics.rmse, 2e-2)

    def test_inverse_problem_pinn_parameter_error(self):
        metrics = run_inverse_problem_case(seed=0, n_obs=40, use_gpu=False)
        self.assertLess(metrics.parameter_error, 1e-1)


if __name__ == "__main__":
    unittest.main()
