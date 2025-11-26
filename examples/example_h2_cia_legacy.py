#!/usr/bin/env python

import torch
import numpy as np
from pyharp import h2_cia_legacy, find_resource
from pyharp.opacity import OpacityOptions, WaveTemp

def setup_h2_cia_legacy_opacity():
    op = OpacityOptions().type("wavetemp")
    op.opacity_files(["H2-H2-eq.xiz.pt", "H2-He-eq.xiz.pt"])
    op.fractions([0.9, 0.1])
    op.species_ids([0])
    return WaveTemp(op)

def run_forward():
    full_path = find_resource("H2-H2-eq.xiz.pt")
    data = torch.jit.load(full_path)
    cia = setup_h2_cia_legacy_opacity()
    wave = torch.logspace(np.log10(10), np.log10(10000), 10)
    atm = {
        'temp': torch.tensor([100.0, 300.0, 600.0]).unsqueeze(-1),
        'wavenumber': wave,
    }

    conc = torch.ones_like(atm['temp']).unsqueeze(-1)
    result = cia.forward(conc, atm).squeeze()
    print(result.shape)

if __name__ == "__main__":
    run_forward()
