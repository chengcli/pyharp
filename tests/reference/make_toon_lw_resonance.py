"""Write toon_lw_resonance.json: thick layers at and near the longwave resonance lambda mu = 1 of each Gauss angle,
with the reference fluxes of toon_lw_reference.py (250 digits, no clip). The band Planck values are those pyharp
computes in each precision, so the table holds exactly the inputs the solver sees.

Run from the repository root: python tests/reference/make_toon_lw_resonance.py"""

import json
import math
from pathlib import Path

import numpy as np
import torch

import pyharp
from toon_lw_reference import UARR, toon_lw_reference

HERE = Path(__file__).resolve().parent


def be_of(temf, dtype):
    t = torch.tensor(temf, dtype=dtype).view(1, -1)
    lo, hi = torch.tensor([1.0], dtype=dtype).view(1, 1, 1), torch.tensor([1.0e5], dtype=dtype).view(1, 1, 1)
    return [float(v) for v in pyharp.bbflux_wavenumber(lo, hi, t).flatten()]


def cases():
    out = []
    for m in (2, 3, 4):                                    # angles with a resonance at 0 < w < 1 (g = 0)
        w_res = 1.0 - 1.0 / (4.0 * UARR[m] ** 2)
        offsets = [0.0, 1e-12, -1e-12, 1e-7, -1e-7, 1e-3, -1e-3] + ([1e-2, -1e-2] if m == 2 else [])
        for dtau in (20.0, 40.0, 80.0, 150.0):
            for d in offsets:
                out.append(dict(rows=[[dtau, w_res * (1 + d), 0.0]], temf=[600.0, 300.0], albedo=0.2,
                                hard_surface=True, top=0, label=f"mu{m} dtau {dtau:g} w(1{d:+.0e})"))
    for dtau in (40.0, 80.0):
        out.append(dict(rows=[[dtau, 0.5 * (1.0 - 1.0 / (4.0 * UARR[2] ** 2)), 0.0]], temf=[600.0, 300.0],
                        albedo=0.2, hard_surface=True, top=0, label=f"thick, half the resonant w, dtau {dtau:g}"))
    w2 = 1.0 - 1.0 / (4.0 * UARR[2] ** 2)
    out.append(dict(rows=[[0.5, 0.3, 0.2], [40.0, w2, 0.0], [1.0, 0.6, 0.4]], temf=[320.0, 290.0, 250.0, 210.0],
                    albedo=0.3, hard_surface=False, top=1, label="column, resonant thick middle layer"))
    out.append(dict(rows=[[1.0, w2, 0.0]], temf=[600.0, 300.0], albedo=0.2, hard_surface=True, top=0,
                    label="thin resonant layer"))
    return out


def main():
    table = []
    for dtype, name in ((torch.float64, "float64"), (torch.float32, "float32")):
        for c in cases():
            rows = [[float(np.array(v, dtype=np.float32 if name == "float32" else np.float64)) for v in r]
                    for r in c["rows"]]
            be = be_of(c["temf"], dtype)
            ref = toon_lw_reference(rows, be, c["albedo"], hard_surface=c["hard_surface"], top_emission_flag=c["top"])
            table.append(dict(c, dtype=name, rows=rows, ref=[[float(v) for v in r] for r in ref]))
    (HERE / "toon_lw_resonance.json").write_text(json.dumps(table, indent=0))
    print(f"{len(table)} cases")


if __name__ == "__main__":
    main()
