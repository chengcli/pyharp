import torch
import os
import pyharp
import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from pyharp import h2_cia_legacy
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
                  ncol: int=1,
                  nlyr: int=100) -> dict[str, torch.Tensor]:
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
    #print("atm pres = ", atm['pres'])
    #print("atm temp = ", atm['temp'])
    return atm

def configure_bands(config_file: str,
                    ncol: int = 1,
                    nlyr: int = 100,
                    nstr: int = 4) -> Radiation:
    rad_op = RadiationOptions.from_yaml(config_file)
    wmin, wmax = load_sonora_window()

    for [name, band] in rad_op.bands().items():
        if name == "sonora196":
            band.ww(band.query_weights("H2-molecule"))
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
    ab_mol = rad.get_module("sonora196").get_module("H2-molecule")
    tauc_mol = ab_mol.forward(conc, atm).squeeze(-1) * dz.unsqueeze(0)

    ab_cia = rad.get_module("sonora196").get_module("H2-continuum")
    tauc_cia = ab_cia.forward(conc, atm).squeeze(-1) * dz.unsqueeze(0)

    sonora = torch.jit.load(fname + ".pt")
    nwave, ncol, nlyr = tauc_mol.shape
    ng = len(sonora.gauss_pts)
    wave_um = 1.e4 / (0.5 * (sonora.wmin + sonora.wmax))

    # reshape to (band, ng, ncol, nlyr)
    tauc_mol = tauc_mol.reshape((nwave // ng, ng, ncol, nlyr))
    tauc_cia = tauc_cia.reshape((nwave // ng, ng, ncol, nlyr))

    # average over gauss points
    tauc_mol = (tauc_mol * sonora.gauss_wts[None, :, None, None]).sum(dim=1)
    tauc_mol.squeeze_()
    print('tauc_mol = ', tauc_mol.shape)

    tauc_cia = (tauc_cia * sonora.gauss_wts[None, :, None, None]).sum(dim=1)
    tauc_cia.squeeze_()
    print('tauc_cia = ', tauc_cia.shape)

    with open('saved_dictionary.pkl', 'rb') as f:
        df = pickle.load(f)
    tauc_tot = torch.tensor(df['full_output']['taugas'])
    tauc_tot = (tauc_tot * sonora.gauss_wts[None, None, :]).sum(dim=-1)
    print('tauc_tot = ', tauc_tot.shape)

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0.2, 0.9, len(sonora.gauss_pts)))

    ax.plot(wave_um, tauc_mol[:, 1], lw=2, ls='-',
            color='b', label='pyharp-molecule')
    ax.plot(wave_um, tauc_mol[:, 1] + tauc_cia[:, 1], lw=2, ls='-',
            color='g', label='pyharp-continuum')
    ax.plot(wave_um, tauc_tot[-1, :], lw=2, ls='--',
            color='g', label='picaso')

    ax.set(xscale='log', yscale='log', xlim=(0.25, 15),
           xlabel='Wavelength (um)',
           ylabel='Optical Thickness')
    ax.legend(frameon=False)

def plot_flux(atm, netflux):
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot(netflux[0,1:], atm['pres'][0,:] / 1.e5, lw=2, ls='-',
            color='b')
    ax.set(xscale='linear', yscale='log',
           ylim=(1.e3, 1.e-4), xlim=(0.5, 25.),
           ylabel='Pressure (bar)',
           xlabel='Net Flux (W/m2)')

if __name__ == "__main__":
    # prepare sonora2020 opacity data
    fname = "sonora_2020_feh+000_co_100.data.196"
    if not os.path.exists(fname + ".pt"):
        preprocess_sonora(fname)

    # configure atmosphere model
    atm = construct_atm(1000.e5, 10., ncol=1, nlyr=100)

    # configure radiation model
    config_file = "example_sonora_2020.yaml"
    rad = configure_bands(config_file, ncol=1,
                          nlyr=atm['pres'].shape[-1], nstr=8)

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
    wmin = rad.get_module("sonora196").options.disort().wave_lower()
    wmax = rad.get_module("sonora196").options.disort().wave_upper()
    atm['wavenumber'] = 0.5 * (torch.tensor(wmin) + torch.tensor(wmax))

    netflux, dnflux, upflux = run_rt(rad, conc, dz, atm)
    print("netflux = ", netflux)
    print("surface flux = ", dnflux)
    print("toa flux = ", upflux)

    # plot optical depth
    #plot_optical_depth(fname, rad, conc, atm, dz)
    #plt.tight_layout()
    #plt.savefig("sonora_2020_optical_depth.png", dpi=300)
    plot_flux(atm, netflux)
    plt.tight_layout()
    plt.savefig("sonora_2020_flux.png", dpi=300)
