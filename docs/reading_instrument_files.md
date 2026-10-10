# Reading instrument files

Module: `hyper_spectral_process.io`. Every reader returns a
`StandardSpectrumMap` and raises `ValueError` when the file does not match
the documented layout (no rows are skipped or padded). Exports do not record
the axis unit; pass it as `unit`.

| function | instrument / software | layout |
|---|---|---|
| `read_horiba_txt(path, unit="cm^-1")` | HORIBA LabSpec map export | first row: the `M` axis values (after two empty cells); every other row: `x  y  I_1 ... I_M`; tabs, spaces, commas or semicolons; `#` and `//` lines skipped |
| `write_horiba_txt(ssm, path, frame_order="x_fast")` | the same layout, rows in raster order | needs a complete raster |
| `read_nanobase_csv(path, unit, x_column="X", y_column="Y", swap_xy=False, negate_x=False, negate_y=False)` | Nanobase mapping CSV | metadata lines, an `AXISDATA=` marker, a header `X, Y, Z, XINDEX, YINDEX, ZINDEX, <axis...>`, one spectrum per row |
| `read_ntmdt_txt(path, unit="nm", axis_atol=1e-6)` | NT-MDT export | every row `x, y, a_0, I_0, a_1, I_1, ...`; the axis repeated per row must agree within `axis_atol` |
| `read_andor_sif(path, x_coords, y_coords, frame_order="x_fast", unit="nm")` | Andor Solis `.sif` kinetic series of full-vertical-binning frames | needs the optional package `sif_parser`; the wavelength calibration comes from the file |

```python
from hyper_spectral_process.io import read_horiba_txt, read_andor_sif, symmetric_raster

ssm = read_horiba_txt("map.txt", unit="cm^-1")
ssm.save("map.npz")

xs, ys = symmetric_raster(12.0, 12.0, 120, 120)          # linspace(-12, 12, 120) in x and y
sif = read_andor_sif("scan.sif", xs, ys, frame_order="x_fast")
```

**Coordinates.** Readers keep the coordinates as exported. Nanobase exports
of some setups need a swap and/or a sign change to look right
(`swap_xy`, `negate_x`, `negate_y`; the original converter applied
`swap_xy=True, negate_x=True`). SIF files carry no coordinates: frame `k` is
placed at `(x[k % nx], y[k // nx])` for `frame_order="x_fast"` (x changes
fastest, i.e. y-major scanning) or `(x[k // ny], y[k % ny])` for `"y_fast"`
(`frames_to_xy`); pass a reversed coordinate array for a scan that runs
towards negative x or y.

**Axis conversions** (`axis.py`). A spectrum recorded by detector pixels of
constant width in wavelength is a density per unit wavelength. Per unit
energy `E = K / lambda` it is multiplied by `|d lambda / d E| = lambda^2 / K`
(`K = 1239.841984 eV nm`, or `1e7` for cm^-1), which conserves the
integrated intensity. The Jacobian is a required argument everywhere:

| function | Jacobian factor when `jacobian=True` |
|---|---|
| `wavelength_to_energy(ssm, "eV" or "cm^-1", jacobian, reference_nm=None)` | `lambda^2 / K` (or `(lambda / reference_nm)^2`, same shape, scale kept) |
| `raman_shift_to_wavelength(ssm, laser_nm, jacobian)` | `1e7 / lambda^2` |
| `wavelength_to_raman_shift(ssm, laser_nm, jacobian)` | `lambda^2 / 1e7` |

Use `jacobian=False` when the values are counts per detector pixel and only
the axis labels change (for example a PL map exported on a Raman-shift axis:
convert to wavelength with `jacobian=False`, then to energy with
`jacobian=True`, as in the MoS2 example). Check that the pixels are uniform
in wavelength first (`np.diff(ssm.axis)`). For Raman spectra the Jacobian
changes ratios of bands far apart, and published ratios are usually computed
without it, so state the choice.
