from typing import Any

import yaml
import torch
import pyharp
from pyharp import (
        RadiationOptions,
        Radiation,
        )

class GreyOpacity(torch.nn.Module):
    """grey opacity module for pyharp JIT loading."""

    def __init__(
        self,
        species_weights: list[float],
        kappa_a: float,
        kappa_b: float,
        kappa_cut: float,
        w0: float = 0.0,
        g: float = 0.0,
        nwave: int = 1,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "species_weights",
            torch.tensor(species_weights, dtype=torch.float64),
            persistent=True,
        )
        self.kappa_a = float(kappa_a)
        self.kappa_b = float(kappa_b)
        self.kappa_cut = float(kappa_cut)
        self.w0 = float(w0)
        self.g = float(g)
        self.nwave = int(nwave)
        self.nprop = 3  # (extinction, single scattering albedo, g)

    def forward(self, conc: torch.Tensor, pres: torch.Tensor, temp: torch.Tensor) -> torch.Tensor:
        # conc: (ncol, nlyr, nspecies) [mol/m^3]
        # pres: (ncol, nlyr) [Pa]
        # temp: (ncol, nlyr) [K]
        # mw: (nspecies) [kg/mol]

        ncol = conc.shape[0]
        nlyr = conc.shape[1]

        # extinction = rho * kappa(pres)
        mw = self.species_weights.to(device=conc.device, dtype=conc.dtype)
        rho = (conc * mw.view(1, 1, -1)).sum(dim=-1)

        kappa = self.kappa_a * torch.pow(pres, self.kappa_b)
        kappa = torch.clamp(kappa, min=self.kappa_cut)
        extinction = rho * kappa  # [1/m]

        out = torch.zeros(
            (self.nwave, ncol, nlyr, self.nprop),
            dtype=conc.dtype,
            device=conc.device,
        )

        out[..., 0] = extinction.unsqueeze(0)
        out[..., 1] = self.w0
        out[..., 2] = self.g

        return out

def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

@dataclass
class RadiativeTransferConfig:
    update_dt: float
    sw_surface_albedo: float
    lw_surface_albedo: float
    stellar_flux_nadir: float

@dataclass
class RadiativeTransferState:
    rad: Radiation
    cfg: RadiativeTransferConfig
    dz: torch.Tensor  # (nlyr,)
    il: int
    iu: int
    last_heating: torch.Tensor  # (ny, nx, nlyr), W/m^3 == Pa/s
    next_update_time: float
    sw_band_weight_sum: float

def run_rt(rad: Radiation, 
           conc: torch.Tensor, 
           dz: torch.Tensor,
           atm: dict[str, torch.Tensor],
           config: dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ncol = conc.shape[0]
    bc = {}
    for n, band in enumerate(rad.options.bands()):
        nwave = band.nwave()
        albedo = config["bands"][n]["surface_albedo"]
        bc[band.name() + "/albedo"] = torch.ones((nwave, ncol)) * albedo

    return rad.forward(conc, dz, bc, atm)

if __name__ == "__main__":
    config_file = "example_grey.yaml"
    config = load_config(config_file)

    rt_cfg = RadiativeTransferConfig(
        update_dt=float(config["problem"]["update_dt"]),
        sw_surface_albedo=float(config["problem"]["sw_surface_albedo"]),
        lw_surface_albedo=float(config["problem"]["lw_surface_albedo"]),
        stellar_flux_nadir=float(config["problem"]["stellar_flux_nadir"]),
    )

    # initialize the radiation options from config file
    rad_op = RadiationOptions.from_yaml(config_file)

    species_names = pyharp.species_names()
    print("species_names = ", species_names)

    species_weights = pyharp.species_weights()
    print("species_weights = ", species_weights)

    opacities = config["opacities"]

    # compile the grey-sw opacity model to file
    params = opacities["grey-sw"]["parameters"]
    grey_sw = GreyOpacity(
            species_weights=species_weights,
            kappa_a=float(params["kappa_a"]),
            kappa_b=float(params["kappa_b"]),
            kappa_cut=float(params["kappa_cut"]),
            w0=float(params["w0"]),
            g=float(params["g"]),
        )
    pyharp.compile(grey_sw, "grey_sw.pt")

    # compile the grey-lw opacity model to file
    params = opacities["grey-lw"]["parameters"]
    grey_lw = GreyOpacity(
            species_weights=species_weights,
            kappa_a=float(params["kappa_a"]),
            kappa_b=float(params["kappa_b"]),
            kappa_cut=float(params["kappa_cut"]),
            w0=float(params["w0"]),
            g=float(params["g"]),
        )
    pyharp.compile(grey_lw, "grey_lw.pt")

    # create radiation model
    for band in rad_op.bands():
        band.weight([1.])

    rad = Radiation(rad_op)

    ncol = 3
    nlyr = 5
    nspecies = len(species_weights)

    conc = torch.ones((ncol, nlyr, nspecies), dtype=torch.float64) * 1e5  # mol/m^3
    pres = torch.ones((ncol, nlyr), dtype=torch.float64) * 1e5  # Pa
    temp = torch.ones((ncol, nlyr), dtype=torch.float64) * 300.0  # K

    opacity_output = opacity_model(conc, pres, temp)
    print(opacity_output)
