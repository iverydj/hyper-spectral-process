# Example data: Raman and photoluminescence maps of monolayer MoS2 on a wrinkled substrate

Measured by Dong-Joon Yi. Used by `examples/05_mos2_raman_pl_analysis.py`.

## Sample and acquisition

| item | value |
|---|---|
| sample | monolayer MoS2 on a wrinkled substrate |
| substrate material | not recorded |
| instrument | HORIBA LabRAM HR Evolution |
| excitation | 532 nm |
| objective | 100x, NA 0.9 |
| grating | 1800 gr/mm (Raman map), 150 gr/mm (PL map) |
| laser power | not recorded |
| acquisition time | not recorded |
| map | 88 x 81 points, 0.30 um step in x and y |
| detector values | as exported by LabSpec (integers in steps of 2 for the Raman map and 10 for the PL map, apart from the interpolated column noted below), i.e. scaled rather than raw ADC counts |

## Files

| file | content |
|---|---|
| `raman_map.npz` | Raman map, 7128 spectra x 1024 points, Raman shift 139.9-651.5 cm^-1 |
| `pl_map.npz` | PL map, 7128 spectra x 1024 points, stored on a Raman-shift axis 285.6-5628.4 cm^-1 (540-759 nm at 532 nm) |

Both files follow the StandardSpectrumMap NPZ layout (`docs/data_format.md`)
and load with `StandardSpectrumMap.load(path)`. The spectra are the exported
values, stored as float32 (which represents them exactly); nothing was
cropped, resampled or otherwise processed. They were read from the LabSpec
map exports with `read_horiba_txt(path, unit="cm^-1")`:

| original export | sha256 |
|---|---|
| Raman map (LabSpec text export) | `31c8f90d5936a1c4d2c78bba1ab1f884069eb8a3e9340081a9bd27772054b33c` |
| PL map (LabSpec text export) | `c8021421b74093caaf65063fbb1aca2954ad43e3d95cf6f574e8ddb8c8914827` |

## Notes for analysis

- The stage coordinates of the PL export are offset from those of the Raman
  export by about (-0.39, -5.89) um in (x, y), although both maps cover the
  same area; register the images before comparing pixels.
- Index 747 of the exported axis (517.5 cm^-1 in the Raman map, 4518.9 cm^-1
  shift or 1.770 eV in the PL map) is the exact mean of its two neighbours in
  every spectrum, i.e. interpolated by the export (presumably a bad-pixel
  correction), not measured.
- Some spectra contain cosmic-ray spikes. One off-flake Raman spectrum has a
  single-sample spike to 64,933 next to the only negative value in the data
  (-602).
- The PL map also contains Raman lines below 900 cm^-1 shift, among them the
  MoS2 E2g and A1g lines.
- Off the flake the spectra show the substrate background: in the Raman map a
  straight line between 300 and 505 cm^-1, in the PL map a broad emission band
  with its maximum near 2.17 eV whose intensity follows the wrinkle stripes.
- The detector pixel width of the PL map is constant in wavelength
  (0.214 nm) to within 1 %, so its values are proportional to a density per
  unit wavelength.

## License

The data in this directory are licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (attribution:
Dong-Joon Yi). The code of the repository is MIT-licensed.
