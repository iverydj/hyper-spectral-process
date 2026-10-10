import numpy as np
import pytest
from scipy.integrate import trapezoid

from hyper_spectral_process.io import (HC_EV_NM, StandardSpectrumMap, frames_to_xy, raman_shift_to_wavelength,
                                       read_horiba_txt, read_nanobase_csv, read_ntmdt_txt, symmetric_raster, to_image,
                                       wavelength_to_energy, wavelength_to_raman_shift, write_horiba_txt)


def _map(n=6, m=40, unit="cm^-1", seed=0):
    rng = np.random.default_rng(seed)
    xy = np.column_stack([np.tile(np.arange(3), n // 3), np.repeat(np.arange(n // 3), 3)]).astype(float)
    return StandardSpectrumMap(rng.normal(size=(n, m)), xy, np.linspace(100, 200, m), unit)


def test_validation():
    s = _map()
    with pytest.raises(ValueError, match="2-D"):
        StandardSpectrumMap(s.spectra[0], s.xy, s.axis)
    with pytest.raises(ValueError, match="xy"):
        StandardSpectrumMap(s.spectra, s.xy[:-1], s.axis)
    with pytest.raises(ValueError, match="strictly increasing"):
        StandardSpectrumMap(s.spectra, s.xy, s.axis[::-1])
    bad = s.spectra.copy()
    bad[1, 2] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        StandardSpectrumMap(bad, s.xy, s.axis)


def test_from_unsorted_sorts_and_merges_duplicates():
    y = np.array([[1.0, 2.0, 4.0, 8.0]])
    m = StandardSpectrumMap.from_unsorted(y, [[0.0, 0.0]], [3.0, 1.0, 1.0, 2.0])
    np.testing.assert_array_equal(m.axis, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(m.spectra, [[3.0, 8.0, 1.0]])   # the two values at axis 1 are averaged


def test_npz_roundtrip(tmp_path):
    s = _map()
    s.save(tmp_path / "m.npz")
    r = StandardSpectrumMap.load(tmp_path / "m.npz")
    np.testing.assert_array_equal(r.spectra, s.spectra)
    assert r.unit == "cm^-1"
    np.savez(tmp_path / "minimal.npz", spectra=s.spectra, xy=s.xy, axis=s.axis)
    assert StandardSpectrumMap.load(tmp_path / "minimal.npz").unit == ""
    np.savez(tmp_path / "broken.npz", spectra=s.spectra, axis=s.axis)
    with pytest.raises(ValueError, match="missing"):
        StandardSpectrumMap.load(tmp_path / "broken.npz")


def test_to_image_allows_missing_nodes_and_rejects_duplicates():
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    img, xs, ys = to_image(np.array([1.0, 2.0, 3.0]), xy)
    assert img.shape == (2, 2) and np.isnan(img[1, 1]) and img[1, 0] == 3.0
    with pytest.raises(ValueError, match="duplicate"):
        to_image(np.zeros(2), np.zeros((2, 2)))


def test_horiba_roundtrip(tmp_path):
    s = _map()
    write_horiba_txt(s, tmp_path / "m.txt")
    r = read_horiba_txt(tmp_path / "m.txt", unit="cm^-1")
    np.testing.assert_allclose(r.spectra, s.spectra, atol=1e-7)
    np.testing.assert_allclose(r.axis, s.axis, atol=1e-7)


def test_horiba_rejects_ragged_rows(tmp_path):
    (tmp_path / "bad.txt").write_text("\t\t1\t2\t3\n0\t0\t1\t2\t3\n1\t0\t1\t2\n")
    with pytest.raises(ValueError, match="expected 5 values"):
        read_horiba_txt(tmp_path / "bad.txt")


def test_nanobase_csv(tmp_path):
    text = ("FILETYPE=RAMAN SPECTRUM (MAPPING)\nAXISDATA=\nX,Y,Z,XINDEX,YINDEX,ZINDEX,300.0,301.0,302.0\n"
            "-1,2,0,0,0,0,5,6,7\n1,2,0,1,0,0,8,9,10\n")
    (tmp_path / "n.csv").write_text(text)
    m = read_nanobase_csv(tmp_path / "n.csv")              # default: the original orientation (swap, negate x)
    np.testing.assert_array_equal(m.axis, [300, 301, 302])
    np.testing.assert_array_equal(m.xy, [[-2, -1], [-2, 1]])
    raw = read_nanobase_csv(tmp_path / "n.csv", swap_xy=False, negate_x=False)
    np.testing.assert_array_equal(raw.xy, [[-1, 2], [1, 2]])


def test_ntmdt(tmp_path):
    (tmp_path / "t.txt").write_text("0,0,500,1,501,2,502,3\n1,0,500,4,501,5,502,6\n")
    m = read_ntmdt_txt(tmp_path / "t.txt")
    np.testing.assert_array_equal(m.spectra, [[1, 2, 3], [4, 5, 6]])
    (tmp_path / "u.txt").write_text("0,0,500,1,501,2,502,3\n1,0,500,4,501.5,5,502,6\n")
    with pytest.raises(ValueError, match="axis differs"):
        read_ntmdt_txt(tmp_path / "u.txt")


def test_sif_frame_order():
    xs, ys = symmetric_raster(1.0, 2.0, 2, 3)
    np.testing.assert_array_equal(xs, [-1, 1])
    xy_x = frames_to_xy(6, xs, ys, "x_fast")
    xy_y = frames_to_xy(6, xs, ys, "y_fast")
    np.testing.assert_array_equal(xy_x[:3], [[-1, -2], [1, -2], [-1, 0]])
    np.testing.assert_array_equal(xy_y[:3], [[-1, -2], [-1, 0], [-1, 2]])
    with pytest.raises(ValueError, match="frames"):
        frames_to_xy(5, xs, ys)


def test_wavelength_jacobian_conserves_area():
    lam = np.linspace(550, 750, 4001)
    e0 = 1.85
    energy = HC_EV_NM / lam
    band_e = 1.0 / (1.0 + 4.0 * ((energy - e0) / 0.05) ** 2)            # Lorentzian in energy
    band_lam = band_e * HC_EV_NM / lam**2                                   # the same band per unit wavelength
    m = StandardSpectrumMap(band_lam[None, :], [[0.0, 0.0]], lam, "nm")
    conv = wavelength_to_energy(m, "eV", jacobian=True)
    np.testing.assert_allclose(conv.spectra[0], 1.0 / (1.0 + 4.0 * ((conv.axis - e0) / 0.05) ** 2), rtol=1e-9)
    assert trapezoid(conv.spectra[0], conv.axis) == pytest.approx(trapezoid(band_lam, lam), rel=1e-4)
    rel = wavelength_to_energy(m, "eV", jacobian=True, reference_nm=650.0)
    np.testing.assert_allclose(rel.spectra / conv.spectra, HC_EV_NM / 650.0**2, rtol=1e-12)
    with pytest.raises(ValueError, match="'nm'"):
        wavelength_to_energy(m.replace(unit="cm^-1"), "eV", jacobian=False)


def test_raman_shift_wavelength_roundtrip():
    m = StandardSpectrumMap(np.ones((1, 50)), [[0.0, 0.0]], np.linspace(100, 3000, 50), "cm^-1")
    lam = raman_shift_to_wavelength(m, 532.0, jacobian=False)
    assert lam.unit == "nm" and lam.axis[0] > 532.0
    back = wavelength_to_raman_shift(lam, 532.0, jacobian=False)
    np.testing.assert_allclose(back.axis, m.axis, rtol=1e-10)
