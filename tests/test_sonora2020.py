#!/usr/bin/env python

import torch
import numpy as np
from pyharp import constants as const
from pyharp import calc_dz_hypsometric
from pyharp.sonora import (
        load_sonora_data,
        load_sonora_atm,
        )
from pyharp.opacity import MultiBand, AttenuatorOptions
from matplotlib import pyplot as plt

def setup_sonora_opacity():
    fname = "sonora_2020_feh+000_co_100.data.196.pt"
    op = AttenuatorOptions().type("multiband-ck")
    op.opacity_files([fname])
    op.species_ids([0])
    ab = MultiBand(op)
    sonora = torch.jit.load(fname)
    atm = load_sonora_atm()
    print('kpres = ', ab.klnp)
    print('ktemp = ', ab.ktemp)
    print('pres = ', sonora.pres)
    print('temp = ', sonora.temp)
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
        print(dz)
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

def test_sonora2020():
    fname = "sonora_2020_feh+000_co_100.data.196.pt"
    op = AttenuatorOptions().type("multiband-ck")
    op.opacity_files([fname])
    op.species_ids([0])

    ab = MultiBand(op)
    print('p = ', ab.klnp[12].exp())
    print('t = ', ab.ktemp[22])

    sonora = torch.jit.load(fname)
    wmin = sonora.wmin
    wmax = sonora.wmax

    data = load_sonora_data("sonora_2020_feh+000_co_100.data.196")

    atm = {
        'pres': torch.tensor([[1.732e5]]),
        'temp': torch.tensor([[302.5]]),
    }
    #conc = atm['pres'] / (const.Rgas * atm['temp'])
    print('kwave = ', ab.kwave)
    print('klnp = ', ab.klnp)
    print('ktemp = ', ab.ktemp)
    print('kwave = ', ab.kwave[256])
    print("op1 = ", ab.kdata[256, 12, 22, 0].exp())
    print("op2 = ", sonora.kappa[256, 13, 22].exp())
    print("op3 = ", np.exp(data['kappa'][32, 0, 12, 22]))
    print("op4 = ", np.exp(data['kappa'][32, 0, 13, 22]))

    atm1 = {
        'pres': torch.tensor([[1.0e5]]),
        'temp': torch.tensor([[300.0]]),
    }
    atm2 = {
        'pres': torch.tensor([[3.0e5]]),
        'temp': torch.tensor([[300.0]]),
    }
    atm3 = {
        'pres': torch.tensor([[1.0e5]]),
        'temp': torch.tensor([[310.0]]),
    }
    atm4 = {
        'pres': torch.tensor([[3.0e5]]),
        'temp': torch.tensor([[310.0]]),
    }

    conc = torch.ones_like(atm1['pres']).unsqueeze(-1)
    kcoeff1 = ab.forward(conc, atm1).squeeze()
    kcoeff1 *= 1.e4 / const.Avogadro
    kcoeff2 = ab.forward(conc, atm2).squeeze()
    kcoeff2 *= 1.e4 / const.Avogadro
    kcoeff3 = ab.forward(conc, atm3).squeeze()
    kcoeff3 *= 1.e4 / const.Avogadro
    kcoeff4 = ab.forward(conc, atm4).squeeze()
    kcoeff4 *= 1.e4 / const.Avogadro

    kcoeff = ab.forward(conc, atm).squeeze()
    kcoeff *= 1.e4 / const.Avogadro
    print('kcoeff1 = ', kcoeff1[256])
    print('kcoeff2 = ', kcoeff2[256])
    print('kcoeff3 = ', kcoeff3[256])
    print('kcoeff4 = ', kcoeff4[256])
    print('kcoeff = ', (3./8. * (kcoeff1.log() + kcoeff2.log())
                      + 1./8. * (kcoeff3.log() + kcoeff4.log())).exp()[256])
    print('kcoeff = ', kcoeff[256])

    exit()

    fig, ax = plt.subplots()
    ax.plot(ab.kwave, kcoeff)
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Attenuation Coefficient (m$^{-1}$)")
    ax.set_xscale("log")

    plt.show()


if __name__ == "__main__":
    torch.set_printoptions(precision=12)
    test_sonora2020()
    #plot_opacity(case='kcross')
    #plot_opacity(case='kcoeff')
    #plot_opacity(case='tau')
