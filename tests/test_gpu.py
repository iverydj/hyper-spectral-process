"""GPU tests; skipped when CUDA is unavailable."""
import numpy as np
import pytest
import torch

from hyper_spectral_process.fit import FitConfig, MapFitOptions, PeakSpec, StageSpec, fit_map
from hyper_spectral_process.synthetic import SyntheticPeak, synthetic_map

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
STAGES = (StageSpec("coarse", 1500, 1e-2), StageSpec("fine", 1000, 2e-3))


def problem():
    ax = np.linspace(340, 440, 300)
    sm = synthetic_map(ax, [SyntheticPeak("lorentzian", 385, 4, 1.0, pos_shift=1.0),
                            SyntheticPeak("pseudovoigt", 405, 4, 1.2, width_g=3.0)], shape=(6, 6), noise_std=0.01)
    cfg = FitConfig((PeakSpec("lorentzian", 384, 5, 0.9, 3, 0, 3, 0, 0.3, 0),
                     PeakSpec("pseudovoigt", 406, 5, 1.0, 3, 0, 3, 0, 0.3, 0, extra={"width_g": 4.0})), STAGES,
                    fit_baseline=True)
    return sm, cfg


def test_gpu_matches_cpu():
    sm, cfg = problem()
    cpu = fit_map(sm.ssm, cfg, MapFitOptions(devices=("cpu",))).result
    gpu = fit_map(sm.ssm, cfg, MapFitOptions(devices=("cuda:0",))).result
    np.testing.assert_allclose(gpu.pos, cpu.pos, atol=5e-3)
    np.testing.assert_allclose(gpu.height, cpu.height, rtol=5e-3)


def test_several_workers_give_the_same_result():
    sm, cfg = problem()
    one = fit_map(sm.ssm, cfg, MapFitOptions(devices=("cuda:0",), batch_size=8)).result
    two = fit_map(sm.ssm, cfg, MapFitOptions(devices=("cuda:0", "cpu"), batch_size=8)).result
    np.testing.assert_allclose(one.pos, two.pos, atol=5e-3)
