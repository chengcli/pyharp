import torch
import os
import pyharp
import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from pyharp.sonora import (
        load_sonora_data,
        load_sonora_window,
        save_sonora_multiband,
        )
from pyharp import (
        constants,
        RadiationOptions,
        Radiation,
        calc_dz_hypsometric,
        disort_config,
        )

def preprocess_sonora(fname: str):
    # dictionary of data
    data = load_sonora_data(fname)

    # print all keys
    print("Keys in data:", data.keys())
    print(data['nwno'])

    # pressure grid
    pres = data['press']
    print("Pressure grid:", pres)

    # temperature grid
    temp = data['temps']
    print("Temperature grid:", temp)

    # wavenumber
    print("wavenumber:", data['wno'])

    save_sonora_multiband(fname, data, clean=False)

    # load it back!
    sonora = torch.jit.load(fname + ".pt")

def construct_atm(pmax: float, pmin: float,
                  ncol: int = 1,
                  nlyr: int = 100) -> dict[str, torch.Tensor]:
    p1bar, T1bar, Tmin = 1.e5, 169., 135.
    pres = torch.logspace(np.log10(pmax), np.log10(pmin), nlyr + 1, dtype=torch.float64)
    temp = T1bar * torch.pow(pres / p1bar, 2. / 7.)
    temp.clip_(min=Tmin)
    atm = {
        'pres' : (pres[1:] * pres[:-1]).sqrt().unsqueeze(0).expand(ncol, nlyr),
        'temp' : (temp[1:] * temp[:-1]).sqrt().unsqueeze(0).expand(ncol, nlyr),
        'btemp0' : temp[0].unsqueeze(0).expand(ncol),
        'ttemp0' : temp[-1].unsqueeze(0).expand(ncol),
    }
    print("atm pres = ", atm['pres'])
    print("atm temp = ", atm['temp'])
    return atm

def configure_bands(config_file: str,
                    ncol: int = 1,
                    nlyr: int = 100,
                    nstr: int = 4) -> Radiation:
    rad_op = RadiationOptions.from_yaml(config_file)
    wmin, wmax = load_sonora_window()

    for [name, band] in rad_op.bands().items():
        if name == "sonora196":
            band.ww(band.query_weights())
            nwave = len(band.ww())
            ng = int(nwave / len(wmin))

            band.disort().accur(1.0e-4)
            disort_config(band.disort(), nstr, nlyr, ncol, nwave)

            data = [wmin] * ng
            band.disort().wave_lower([x for col in zip(*data) for x in col])

            data = [wmax] * ng
            band.disort().wave_upper([x for col in zip(*data) for x in col])
        else:
            raise ValueError(f"Unknown band: {name}")

    return Radiation(rad_op)

def run_rt(rad: Radiation, conc: torch.Tensor, dz: torch.Tensor,
           atm: dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ncol = conc.shape[0]
    bc = {}
    for [name, band] in rad.options.bands().items():
        nwave = len(band.ww())
        bc[name + "/albedo"] = torch.ones((nwave, ncol), dtype=torch.float64)
        bc[name + "/temis"] = torch.zeros((nwave, ncol), dtype=torch.float64)

    bc["btemp"] = torch.zeros((ncol), dtype=torch.float64)
    bc["ttemp"] = torch.zeros((ncol), dtype=torch.float64)

    return rad.forward(conc, dz, bc, atm)

def plot_optical_depth(fname: str,
                       rad: Radiation,
                       conc: torch.Tensor,
                       atm: dict[str, torch.Tensor],
                       dz: torch.Tensor):
    ab = rad.get_module("sonora196").get_module("H2")
    tauc = ab.forward(conc, atm).squeeze(-1) * dz.unsqueeze(0)

    # load sonora ck table info
    sonora = torch.jit.load(fname + ".pt")
    nwave, ncol, nlyr = tauc.shape
    ng = len(sonora.gauss_pts)
    wave_um = 1.e4 / (0.5 * (sonora.wmin + sonora.wmax))

    # reshape to (band, ng, ncol, nlyr)
    tauc = tauc.reshape((nwave // ng, ng, ncol, nlyr))
    #tauc = (tauc * sonora.gauss_wts[None, :, None, None]).sum(dim=1)
    tauc.squeeze_()
    print('tauc = ', tauc.shape)

    with open('saved_dictionary.pkl', 'rb') as f:
        df = pickle.load(f)
    tauc2 = torch.tensor(df['full_output']['taugas'])
    #tauc2 = (tauc2 * sonora.gauss_wts[None, None, :]).sum(dim=-1)
    print('tauc2 = ', tauc2.shape)

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0.2, 0.9, len(sonora.gauss_pts)))

    for ck in [0, 7]:
        ax.plot(wave_um, tauc[:, ck, -2], lw=2, ls='-',
                color=colors[ck], label='pyharp, ck = {}'.format(ck))
        ax.plot(wave_um, tauc2[0, :, ck], lw=2, ls='--',
                color=colors[ck], label='picaso, ck = {}'.format(ck))

    ax.set(xscale='log', yscale='log', xlim=(0.25, 15),
           xlabel='Wavelength (um)',
           ylabel='Optical Thickness')
    ax.legend(frameon=False)

    #plt.show()

if __name__ == "__main__":
    # prepare sonora2020 opacity data
    fname = "sonora_2020_feh+000_co_100.data.196"
    if not os.path.exists(fname + ".pt"):
        preprocess_sonora(fname)

    # configure atmosphere model
    atm = construct_atm(100.e5, 10., ncol=1, nlyr=100)

    # configure radiation model
    config_file = "example_sonora_2020.yaml"
    rad = configure_bands(config_file, ncol=1,
                          nlyr=atm['pres'].shape[-1], nstr=8)
    print(rad.options)

    # calculate concentration and layer thickness
    mean_mol_weight = pyharp.species_weights()[0]
    print("mean mol weight = ", mean_mol_weight)
    grav = 24.8 # m/s^2
    dz = calc_dz_hypsometric(atm["pres"], atm["temp"],
                             torch.tensor(mean_mol_weight * grav / constants.Rgas)
                             )
    print("dz = ", dz)
    conc = atm["pres"] / (atm["temp"] * constants.Rgas)
    conc.unsqueeze_(-1)

    # run rt
    netflux, dnflux, upflux = run_rt(rad, conc, dz, atm)
    print("netflux = ", netflux)
    print("surface flux = ", dnflux)
    print("toa flux = ", upflux)

    # plot optical depth
    plot_optical_depth(fname, rad, conc, atm, dz)
    plt.tight_layout()
    plt.savefig("sonora_2020_optical_depth.png", dpi=300)
