#! /user/bin/env python3
import torch
import numpy as np
import pyharp as harp
from torch import tensor, logspace, zeros, ones
from pyharp import (
    interpn,
    constants,
    calc_dz_hypsometric,
    bbflux_wavenumber,
    RadiationOptions,
    Radiation,
    disort_config,
    read_rfm_atm,
)

if __name__ == "__main__":
    ncol = 1
    nlyr = 80
    nstr = 4

    surf_sw_albedo = 0.3
    sr_sun = 2.92842e-5
    btemp0 = 210
    ttemp0 = 100
    solar_temp = 5772
    lum_scale = 0.7
    grav = 3.711
    mean_mol_weight = 0.044

    pres = logspace(np.log10(500.0), np.log10(0.01), nlyr)
    pres *= 100.0
    temp = 200.0 * ones((nlyr,), dtype=torch.float64)

    # mole fractions
    xfrac = zeros((ncol, nlyr, 5), dtype=torch.float64)

    # molecules
    rfm_atm = read_rfm_atm("rfm.atm")
    rfm_pre = rfm_atm["PRE"] * 100.0
    rfm_tem = rfm_atm["TEM"]

    xfrac[:, :, 0] = interpn(
        [pres.log()], [rfm_pre.log()], rfm_atm["CO2"].unsqueeze(-1) * 1.0e-6
    ).squeeze(-1)
    xfrac[:, :, 1] = interpn(
        [pres.log()], [rfm_pre.log()], rfm_atm["H2O"].unsqueeze(-1) * 1.0e-6
    ).squeeze(-1)
    xfrac[:, :, 2] = interpn(
        [pres.log()], [rfm_pre.log()], rfm_atm["SO2"].unsqueeze(-1) * 1.0e-6
    ).squeeze(-1)

    # aerosols
    aero_ptx = tensor(np.genfromtxt("aerosol_output_data.txt"))
    aero_p = aero_ptx[:, 0] * 1.0e5
    aero_t = aero_ptx[:, 1]
    aero_x = aero_ptx[:, 2:]

    xfrac[:, :, 3:] = interpn([pres.log()], [aero_p.log()], aero_x)
    atm = {
        "pres": pres.unsqueeze(0).expand(ncol, nlyr),
        "temp": temp.unsqueeze(0).expand(ncol, nlyr),
    }
    bc = {}

    # layer thickness
    dz = calc_dz_hypsometric(
        pres, temp, tensor(mean_mol_weight * grav / constants.Rgas)
    )

    rad_op = RadiationOptions.from_yaml("amars-ck.yaml")

    # configure bands
    for [name, band] in rad_op.bands().items():
        band.ww(band.query_weights())
        nwave = len(band.ww()) if name != "SW" else 200

        wmin = band.disort().wave_lower()[0]
        wmax = band.disort().wave_upper()[0]

        band.disort().accur(1.0e-12)
        disort_config(band.disort(), nstr, nlyr, ncol, nwave)

        if name == "SW":  # shortwave
            band.ww(np.linspace(wmin, wmax, nwave))
            wave = tensor(band.ww(), dtype=torch.float64)
            bc[name + "/fbeam"] = (
                lum_scale * sr_sun * bbflux_wavenumber(wave, solar_temp)
            )
            bc[name + "/albedo"] = surf_sw_albedo * ones(
                (nwave, ncol), dtype=torch.float64
            )
            bc[name + "/umu0"] = 0.707 * ones((ncol,), dtype=torch.float64)
        else:  # longwave
            band.disort().wave_lower([wmin] * nwave)
            band.disort().wave_upper([wmax] * nwave)
            bc[name + "/albedo"] = zeros((nwave, ncol), dtype=torch.float64)
            bc[name + "/temis"] = ones((nwave, ncol), dtype=torch.float64)

    bc["btemp"] = btemp0 * ones((ncol,), dtype=torch.float64)
    bc["ttemp"] = ttemp0 * ones((ncol,), dtype=torch.float64)

    # construct radiation model
    print("radiation options:\n", rad_op)
    rad = Radiation(rad_op)

    # run RT
    conc = xfrac.clone()
    conc *= atm["pres"].unsqueeze(-1) / (constants.Rgas * atm["temp"].unsqueeze(-1))
    netflux = rad.forward(conc, dz, bc, atm)

    downwarf_flux = harp.shared()["radiation/downward_flux"]
    upward_flux = harp.shared()["radiation/upward_flux"]

    print("net flux = ", netflux)
    print("downward flux = ", downwarf_flux)
    print("upward flux = ", upward_flux)
