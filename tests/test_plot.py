import numpy as np

from hyper_spectral_process.fit import FitConfig, MapFitOptions, PeakSpec, StageSpec, fit_map, save_fit_outputs
from hyper_spectral_process.plot import band_intensity, plot_band_intensity
from hyper_spectral_process.synthetic import SyntheticPeak, synthetic_map


def test_outputs_and_summary_plots(tmp_path):
    sm = synthetic_map(np.linspace(340, 440, 200), [SyntheticPeak("lorentzian", 385, 4, 1.0),
                                                     SyntheticPeak("gaussian", 410, 6, 0.5)], shape=(4, 4))
    cfg = FitConfig((PeakSpec("lorentzian", 384, 5, 0.9, 3, 0, 3, 0, 0.3, 0, name="a"),
                     PeakSpec("gaussian", 409, 5, 0.5, 3, 0, 3, 0, 0.3, 0, name="b")),
                    (StageSpec("s", 300, 1e-2),))
    fit = fit_map(sm.ssm, cfg, MapFitOptions())
    npz = save_fit_outputs(fit, tmp_path, "demo")
    assert npz.exists() and (tmp_path / "summary" / "parameter_maps.png").exists()
    assert (tmp_path / "params_demo.csv").read_text().splitlines()[0].startswith("idx,valid")
    np.testing.assert_allclose(band_intensity(sm.ssm, 380, 390, "max"), sm.ssm.spectra[:, (sm.ssm.axis >= 380)
                                                                                        & (sm.ssm.axis <= 390)].max(1))
    plot_band_intensity(sm.ssm, 380, 390)
