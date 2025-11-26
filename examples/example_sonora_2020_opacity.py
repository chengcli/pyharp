#!/usr/bin/env python

import torch
import numpy as np
from pyharp import constants as const
from pyharp import calc_dz_hypsometric
from pyharp.sonora import load_sonora_data
from pyharp.opacity import MultiBand, OpacityOptions
from matplotlib import pyplot as plt

def setup_sonora_opacity():
    fname = "sonora_2020_feh+000_co_100.data.196.pt"
    op = OpacityOptions().type("multiband-ck")
    op.opacity_files([fname])
    op.species_ids([0])
    ab = MultiBand(op)
    sonora = torch.jit.load(fname)
    return ab, sonora

def setup_atm_grid():
    temp = torch.tensor([100.0, 300.0, 600.0])
    pres = torch.tensor([0.1e5, 1.0e5, 10.0e5, 100.e5])
    X, Y = torch.meshgrid(temp, pres, indexing='ij')
    return {'temp': X, 'pres': Y}

def run_forward(case: str='kcross'):
    ab, sonora = setup_sonora_opacity()
    atm = setup_atm_grid()

    if case == 'kcross':
        conc = torch.ones_like(atm['pres']).unsqueeze(-1)
        # m^2/mol -> cm^2/molecule
        result = ab.forward(conc, atm).squeeze() * 1.e4 / const.Avogadro
    elif case == 'kcoeff':
        conc = (atm['pres'] / (const.Rgas * atm['temp'])).unsqueeze(-1)
        # 1/m
        result = ab.forward(conc, atm).squeeze()
    elif case == 'tau':
        conc = (atm['pres'] / (const.Rgas * atm['temp'])).unsqueeze(-1)
        mu_grav = 2.2e-3 * 24.8
        # m
        dz = calc_dz_hypsometric(atm["pres"], atm["temp"],
                                 torch.tensor(-mu_grav / const.Rgas))
        result = ab.forward(conc, atm).squeeze() * dz.unsqueeze(0)

    # load sonora ck table info
    nwave, ncol, nlyr = result.shape
    ng = len(sonora.gauss_pts)
    wave_um = 1.e4 / (0.5 * (sonora.wmin + sonora.wmax))

    # reshape to (band, ng, ncol, nlyr)
    result = result.reshape((nwave // ng, ng, ncol, nlyr))
    result = (result* sonora.gauss_wts[None, :, None, None]).sum(dim=1)
    return wave_um, result, atm['temp'][:,0], atm['pres'][0,:] / 1.e5

def plot_opacity(case: str='kcross'):
    print('running case = ', case)
    wave_um, result, temp, pres = run_forward(case=case)

    fig, axs = plt.subplots(figsize=(10, 6),
                            nrows=len(temp), sharex=True)
    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0.2, 0.9, result.shape[-1]))

    for i in range(len(temp)):
        ax = axs[i]
        for j in range(len(pres)):
            ax.plot(wave_um, result[:, i, j],
                    label='{:.1f} bar'.format(pres[j]),
                    color=colors[j], lw=2)
        if i == 0:
            if case == 'kcross':
                ax.set_title("Absorption Cross Section (cm$^2$ molecule$^{-1}$)")
            elif case == 'kcoeff':
                ax.set_title("Absorption Coefficient (m$^{-1}$)")
            elif case == 'tau':
                ax.set_title("Optical Depth")
            else:
                raise ValueError(f"Unknown case: {case}")

        if i == len(temp) - 1:
            ax.legend(frameon=False)
            ax.set_xlabel("Wavelength (um)")
        ax.set(xscale="log", yscale="log", xlim=(0.25, 15),
               ylabel=f"{temp[i].item():.0f} K")

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.0)
    #plt.show()
    if case == 'kcross':
        plt.savefig("sonora_2020_absorption_xsection.png", dpi=300)
    elif case == 'kcoeff':
        plt.savefig("sonora_2020_attenuation_coefficient.png", dpi=300)
    elif case == 'tau':
        plt.savefig("sonora_2020_optical_depth.png", dpi=300)
    else:
        raise ValueError(f"Unknown case: {case}")

if __name__ == "__main__":
    plot_opacity(case='kcross')
    plot_opacity(case='kcoeff')
    plot_opacity(case='tau')
