#!/usr/bin/env python

import torch
import numpy as np
from pyharp.opacity import MultiBand, AttenuatorOptions
from pyharp import constants as const
from pyharp.sonora import load_sonora_data
from matplotlib import pyplot as plt

def run_kcross():
    fname = "sonora_2020_feh+000_co_100.data.196.pt"
    op = AttenuatorOptions().type("multiband-ck")
    op.opacity_files([fname])
    op.species_ids([0])
    ab = MultiBand(op)

    temp = torch.tensor([100.0, 300.0, 600.0])
    pres = torch.tensor([0.1e5, 1.0e5, 10.0e5, 100.e5])
    X, Y = torch.meshgrid(temp, pres, indexing='ij')
    atm = {'temp': X, 'pres': Y}

    conc = torch.ones_like(atm['pres']).unsqueeze_(-1)
    # m^2/mol -> cm^2/molecule
    kcross = ab.forward(conc, atm).squeeze() * 1.e4 / const.Avogadro

    # load sonora ck table info
    sonora = torch.jit.load(fname)
    nwave, ncol, nlyr = kcross.shape
    ng = len(sonora.gauss_pts)
    wave_um = 1.e4 / (0.5 * (sonora.wmin + sonora.wmax))

    # reshape to (band, ng, ncol, nlyr)
    kcross = kcross.reshape((nwave // ng, ng, ncol, nlyr))
    kcross = (kcross * sonora.gauss_wts[None, :, None, None]).sum(dim=1)
    return wave_um, kcross, temp, pres

def plot_kcross():
    wave_um, kcross, temp, pres = run_kcross()

    fig, axs = plt.subplots(figsize=(10, 6),
                            nrows=len(temp), sharex=True)
    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0.2, 0.9, kcross.shape[-1]))

    for i in range(len(temp)):
        ax = axs[i]
        for j in range(len(pres)):
            ax.plot(wave_um, kcross[:, i, j],
                    label='{:.1f} bar'.format(pres[j] / 1.e5),
                    color=colors[j], lw=2)
        if i == 0:
            ax.set_title("Absorption Cross Section (cm$^2$ molecule$^{-1}$)")

        if i == len(temp) - 1:
            ax.legend(frameon=False)
            ax.set_xlabel("Wavelength (um)")
        ax.set(xscale="log", yscale="log", xlim=(0.25, 15),
               ylabel=f"{temp[i].item():.0f} K")
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.0)
    #plt.show()
    plt.savefig("sonora_2020_absorption_xsection.png", dpi=300)

def test_sonora2020():
    fname = "sonora_2020_feh+000_co_100.data.196.pt"
    op = AttenuatorOptions().type("multiband-ck")
    op.opacity_files([fname])
    op.species_ids([0])

    ab = MultiBand(op)
    print(ab.kdata.is_contiguous())
    print(ab.kwave)
    print(ab.klnp)
    print(ab.ktemp)
    print(ab.weights)
    print('p = ', ab.klnp[12].exp())
    print('t = ', ab.ktemp[22])

    sonora = torch.jit.load(fname)
    wmin = sonora.wmin
    wmax = sonora.wmax
    print('kappa = ', sonora.kappa.is_contiguous())

    data = load_sonora_data("sonora_2020_feh+000_co_100.data.196")

    atm = {
        'pres': torch.tensor([[1.0e5]]),
        'temp': torch.tensor([[305.0]]),
    }
    conc = atm['pres'] / (const.Rgas * atm['temp'])
    print(conc)
    print("op1 = ", ab.kdata[1, 12, 22, 0].exp() * 6.022e23 * conc)
    print("kappa shape = ", data['kappa'].shape)
    print("kappa = ", np.exp(data['kappa'][0, 1, 12, 22]) * 6.022e23 * conc)

    exit()
    kcoeff = ab.forward(conc, atm).squeeze()
    print('kcoeff1 = {:.10f}'.format(kcoeff[1]))

    exit()

    fig, ax = plt.subplots()
    ax.plot(ab.kwave, kcoeff)
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Attenuation Coefficient (m$^{-1}$)")
    ax.set_xscale("log")

    plt.show()


if __name__ == "__main__":
    #test_sonora2020()
    plot_kcross()
