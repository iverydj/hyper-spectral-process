"""Convert instrument map exports to StandardSpectrumMap NPZ files.

Each entry of ``AppConfig.inputs`` names a file, its reader and the axis unit
to store. To make the example self-contained, the script first writes a
LabSpec-format text export (a crop of the bundled Raman map) with
``write_horiba_txt`` and then converts it back.

Readers: ``read_horiba_txt`` (HORIBA LabSpec), ``read_nanobase_csv``
(Nanobase), ``read_ntmdt_txt`` (NT-MDT), ``read_andor_sif`` (Andor Solis,
needs the optional package ``sif_parser`` and the raster coordinates).
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hyper_spectral_process.io import (StandardSpectrumMap, read_horiba_txt, read_nanobase_csv, read_ntmdt_txt,
                                       write_horiba_txt)

HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class AppConfig:
    out_dir: Path = Path("example_output") / "01_convert"
    make_demo_export: bool = True                       # write demo_labspec_export.txt first
    demo_source: Path = HERE / "data" / "mos2_wrinkled" / "raman_map.npz"
    # (path, reader, unit); reader is "horiba", "nanobase" or "ntmdt"
    inputs: tuple = ((Path("example_output") / "01_convert" / "demo_labspec_export.txt", "horiba", "cm^-1"),)


CONFIG = AppConfig()
READERS = {"horiba": read_horiba_txt, "nanobase": read_nanobase_csv, "ntmdt": read_ntmdt_txt}


def main(cfg: AppConfig) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    if cfg.make_demo_export:
        src = StandardSpectrumMap.load(cfg.demo_source)
        x, y = src.xy[:, 0], src.xy[:, 1]
        corner = (x < np.unique(x)[10]) & (y < np.unique(y)[8])          # a 10 x 8 corner of the map
        write_horiba_txt(src.select(corner), cfg.out_dir / "demo_labspec_export.txt")
    for path, reader, unit in cfg.inputs:
        ssm = READERS[reader](path, unit=unit)
        out = cfg.out_dir / (Path(path).stem + ".npz")
        ssm.save(out)
        print(f"{Path(path).name} -> {out}: {ssm.n_spectra} spectra x {ssm.n_points} points, "
              f"axis {ssm.axis[0]:.1f}-{ssm.axis[-1]:.1f} {ssm.unit}")


if __name__ == "__main__":
    main(CONFIG)
