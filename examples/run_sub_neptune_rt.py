#!/usr/bin/env python3
"""Run a tidally locked Sub-Neptune GCM with pyharp Toon radiative transfer."""

from __future__ import annotations

import argparse
import glob
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
import snapy
from snapy import MeshBlock, MeshBlockOptions, kIV1, kICY, kIPR
from kintera import Kinetics, KineticsOptions, ThermoX
from paddle import evolve_kinetics, setup_profile

try:
    import pyharp
    from pyharp import Radiation, RadiationOptions
except Exception as exc:  # pragma: no cover - runtime dependency check
    pyharp = None
    Radiation = None
    RadiationOptions = None
    _PYHARP_IMPORT_ERROR = exc
else:
    _PYHARP_IMPORT_ERROR = None


SECONDS_PER_DAY = 86400.0


@dataclass
class RadiativeTransferConfig:
    update_dt: float
    shortwave_band: str
    longwave_band: str
    sw_surface_albedo: float
    lw_surface_albedo: float
    stellar_flux_nadir: float


@dataclass
class RadiativeTransferState:
    rad: Radiation
    cfg: RadiativeTransferConfig
    cos_zenith_dayside: torch.Tensor  # (ny, nx)
    dz: torch.Tensor  # (nlyr,)
    area1: torch.Tensor  # (ncol, nlyr + 1)
    vol: torch.Tensor  # (ncol, nlyr)
    il: int
    iu: int
    last_heating: torch.Tensor  # (ny, nx, nlyr), W/m^3 == Pa/s
    next_update_time: float
    sw_nwave: int
    lw_nwave: int
    sw_band_weight_sum: float


class SimpleGreyJITOpacity(torch.nn.Module):
    """TorchScriptable double-grey opacity module for pyharp JIT loading."""

    def __init__(
        self,
        species_weights: list[float],
        kappa_a: float,
        kappa_b: float,
        kappa_cut: float,
        nwave: int = 1,
        nmom: int = 1,
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
        self.nwave = int(nwave)
        self.nprop = 2 + int(nmom)

    def forward(self, conc: torch.Tensor, pres: torch.Tensor, temp: torch.Tensor) -> torch.Tensor:
        # conc: (ncol, nlyr, nspecies) [mol/m^3]
        # pres/temp: (ncol, nlyr)
        ncol = conc.shape[0]
        nlyr = conc.shape[1]

        # Match the C++ simple_grey.cpp behavior: extinction = rho * kappa(p)
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
        # pure absorption: single scattering albedo = 0, g = 0
        return out


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def select_device(block: MeshBlock) -> torch.device:
    if torch.cuda.is_available() and block.options.layout().backend() == "nccl":
        return torch.device(block.device())
    return torch.device("cpu")


def create_models(config_file: str, output_dir: str | None = None):
    op = MeshBlockOptions.from_yaml(config_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        op.output_dir(output_dir)

    block = MeshBlock(op)
    device = select_device(block)
    block.to(device)

    thermo_y = block.module("hydro.eos.thermo")
    thermo_x = ThermoX(thermo_y.options)
    thermo_x.to(device)

    op_kin = KineticsOptions.from_yaml(config_file)
    kinet = Kinetics(op_kin)
    kinet.to(device)

    eos = block.module("hydro.eos")
    return block, eos, thermo_y, thermo_x, kinet, device


def initialize_isothermal(block: MeshBlock, config: dict[str, Any]) -> tuple[dict[str, torch.Tensor], float]:
    grav = -float(config["forcing"]["const-gravity"]["grav1"])
    problem = config["problem"]

    param: dict[str, float] = {
        "Ts": float(problem["Ts"]),
        "Ps": float(problem["Ps"]),
        "Tmin": float(problem.get("Tmin", problem["Ts"])),
        "grav": grav,
    }

    thermo_y = block.module("hydro.eos.thermo")
    for name in thermo_y.options.species():
        param[f"x{name}"] = float(problem.get(f"x{name}", 0.0))

    hydro_w = setup_profile(block, param, method="isothermal")
    hydro_w[kIV1] += 1e-6 * torch.randn_like(hydro_w[kIV1])
    return block.initialize({"hydro_w": hydro_w})


def _resolve_local_face_name(block: MeshBlock) -> str:
    layout = snapy.distributed.get_layout()
    rank = int(snapy.distributed.get_rank())
    loc = layout.loc_of(rank)
    face_id = int(loc[2])
    return snapy.coord.get_cs_face_name(face_id)


def _build_local_cos_zenith(block: MeshBlock, config: dict[str, Any], device: torch.device) -> torch.Tensor:
    coord = block.module("coord")
    x2v = coord.buffer("x2v")
    x3v = coord.buffer("x3v")

    beta, alpha = torch.meshgrid(x3v, x2v, indexing="ij")
    face_name = _resolve_local_face_name(block)
    lon, lat = snapy.coord.cs_ab_to_lonlat(face_name, alpha, beta)

    problem = config["problem"]
    lon0 = math.radians(float(problem.get("substellar_lon_deg", 0.0)))
    lat0 = math.radians(float(problem.get("substellar_lat_deg", 0.0)))

    cos_zenith = (
        torch.sin(lat) * math.sin(lat0)
        + torch.cos(lat) * math.cos(lat0) * torch.cos(lon - lon0)
    )
    return torch.clamp(cos_zenith, min=0.0).to(device)


def _extract_species_weights_from_config(config: dict[str, Any]) -> list[float]:
    # Atomic masses [kg/mol] for elements used in current RT species definitions.
    atomic_mass = {
        "H": 1.00784e-3,
        "He": 4.002602e-3,
        "C": 12.0107e-3,
        "N": 14.0067e-3,
        "O": 15.999e-3,
        "S": 32.065e-3,
    }
    out: list[float] = []
    for sp in config["species"]:
        mw = 0.0
        for el, stoich in sp["composition"].items():
            if el not in atomic_mass:
                raise KeyError(f"Unsupported element '{el}' in species '{sp['name']}' for JIT opacity.")
            mw += float(stoich) * atomic_mass[el]
        out.append(mw)
    return out


def ensure_jit_opacity_files(config: dict[str, Any], config_path: Path, rebuild: bool = False) -> None:
    if pyharp is None:
        raise RuntimeError(f"pyharp import failed: {_PYHARP_IMPORT_ERROR}")

    opacities = config.get("opacities", {})
    if not opacities:
        return

    species_weights = _extract_species_weights_from_config(config)

    for name, spec in opacities.items():
        if spec.get("type") != "jit":
            continue
        if str(spec.get("model", "")).lower() not in {"simple-grey", "simple_grey"}:
            continue

        files = spec.get("data", [])
        if not files:
            raise ValueError(f"Opacity '{name}' is type=jit but has no data files.")
        out_file = (config_path.parent / files[0]).resolve()
        out_file.parent.mkdir(parents=True, exist_ok=True)
        if out_file.exists() and not rebuild:
            continue

        nmom = int(spec.get("nmom", 1))
        params = spec.get("parameters", {})
        model = SimpleGreyJITOpacity(
            species_weights=species_weights,
            kappa_a=float(params["kappa_a"]),
            kappa_b=float(params["kappa_b"]),
            kappa_cut=float(params["kappa_cut"]),
            nwave=1,
            nmom=nmom,
        )
        scripted = torch.jit.script(model)
        scripted.save(str(out_file))
        print(f"Saved JIT opacity for '{name}' -> {out_file}")


def _band_config_map(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(b["name"]): b for b in config.get("bands", [])}


def _configure_runtime_band_grids(rad_op: RadiationOptions, config: dict[str, Any], ncol: int, nlyr: int) -> tuple[int, int, float]:
    band_cfgs = _band_config_map(config)
    sw_weight_sum = 1.0
    sw_nwave = 1
    lw_nwave = 1

    for band in rad_op.bands():
        name = band.name()
        cfg = band_cfgs[name]
        wn0, wn1 = [float(x) for x in cfg["range"]]
        nwave = int(cfg.get("nwave", 1))

        if nwave <= 1:
            wavenumber = [0.5 * (wn0 + wn1)]
            weight = [wn1 - wn0]
        else:
            grid = torch.linspace(wn0, wn1, nwave, dtype=torch.float64).tolist()
            wavenumber = [float(x) for x in grid]
            dwn = (wn1 - wn0) / max(1, nwave - 1)
            # Rectangle weights are acceptable for this grey use-case and avoid the
            # internal wavenumber/weight parser mismatch in current pyharp.
            weight = [dwn] * nwave

        band.ncol(ncol)
        band.nlyr(nlyr)
        band.wavenumber(wavenumber)
        band.weight(weight)

        lname = name.lower()
        if "sw" in lname or "vis" in lname or "short" in lname:
            sw_nwave = nwave
            sw_weight_sum = float(sum(weight))
        if "lw" in lname or "ir" in lname or "long" in lname:
            lw_nwave = nwave

    return sw_nwave, lw_nwave, sw_weight_sum


def build_rt_state(
    block: MeshBlock,
    config: dict[str, Any],
    config_file: str,
    device: torch.device,
    rebuild_jit: bool = False,
) -> RadiativeTransferState:
    if pyharp is None:
        raise RuntimeError(f"pyharp import failed: {_PYHARP_IMPORT_ERROR}")

    rt_cfg_raw = config.get("radiative-transfer", {})
    cfg = RadiativeTransferConfig(
        update_dt=float(rt_cfg_raw.get("update_dt", 3600.0)),
        shortwave_band=str(rt_cfg_raw.get("shortwave_band", "sw")),
        longwave_band=str(rt_cfg_raw.get("longwave_band", "lw")),
        sw_surface_albedo=float(rt_cfg_raw.get("sw_surface_albedo", 0.0)),
        lw_surface_albedo=float(rt_cfg_raw.get("lw_surface_albedo", 0.0)),
        stellar_flux_nadir=float(config["problem"]["stellar_flux_nadir"]),
    )

    cfg_path = Path(config_file).resolve()
    ensure_jit_opacity_files(config, cfg_path, rebuild=rebuild_jit)
    pyharp.add_resource_directory(str(cfg_path.parent), prepend=True)
    pyharp.add_resource_directory(str((cfg_path.parent / "rt").resolve()), prepend=True)

    coord = block.module("coord")
    il, iu = coord.il(), coord.iu()
    nlyr = iu - il + 1
    ny = int(coord.buffer("x3v").numel())
    nx = int(coord.buffer("x2v").numel())
    ncol = ny * nx

    rad_op = RadiationOptions.from_yaml(str(cfg_path))
    sw_nwave, lw_nwave, sw_weight_sum = _configure_runtime_band_grids(rad_op, config, ncol=ncol, nlyr=nlyr)
    rad = Radiation(rad_op)
    if hasattr(rad, "to"):
        rad.to(device)

    dz = coord.buffer("dx1f")[il : iu + 1].to(device=device, dtype=torch.float64)

    area1 = coord.face_area1()[..., il : iu + 2]
    vol = coord.cell_volume()[..., il : iu + 1]
    area1 = area1.reshape(ncol, nlyr + 1).to(device=device, dtype=torch.float64)
    vol = vol.reshape(ncol, nlyr).to(device=device, dtype=torch.float64)

    cosz = _build_local_cos_zenith(block, config, device).to(dtype=torch.float64)
    last_heating = torch.zeros((ny, nx, nlyr), dtype=torch.float64, device=device)

    return RadiativeTransferState(
        rad=rad,
        cfg=cfg,
        cos_zenith_dayside=cosz,
        dz=dz,
        area1=area1,
        vol=vol,
        il=il,
        iu=iu,
        last_heating=last_heating,
        next_update_time=0.0,
        sw_nwave=sw_nwave,
        lw_nwave=lw_nwave,
        sw_band_weight_sum=sw_weight_sum,
    )


def _compute_rt_heating(
    block: MeshBlock,
    eos,
    thermo_y,
    thermo_x: ThermoX,
    block_vars: dict[str, torch.Tensor],
    rt_state: RadiativeTransferState,
) -> torch.Tensor:
    hydro_w = block_vars["hydro_w"]

    temp = eos.compute("W->T", (hydro_w,))
    pres = hydro_w[kIPR]
    xfrac = thermo_y.compute("Y->X", (hydro_w[kICY:],))
    conc = thermo_x.compute("TPX->V", (temp, pres, xfrac))

    il, iu = rt_state.il, rt_state.iu
    nlyr = iu - il + 1
    ny, nx = rt_state.cos_zenith_dayside.shape
    ncol = ny * nx

    temp_i = temp[..., il : iu + 1].reshape(ncol, nlyr).to(dtype=torch.float64)
    pres_i = pres[..., il : iu + 1].reshape(ncol, nlyr).to(dtype=torch.float64)
    conc_i = conc[..., il : iu + 1, :].reshape(ncol, nlyr, conc.shape[-1]).to(dtype=torch.float64)

    sw_flux_density = rt_state.cfg.stellar_flux_nadir / max(rt_state.sw_band_weight_sum, 1.0)
    bc: dict[str, torch.Tensor] = {
        f"{rt_state.cfg.shortwave_band}/fbeam": (
            sw_flux_density
            * torch.ones((rt_state.sw_nwave, ncol), dtype=torch.float64, device=temp_i.device)
        ),
        f"{rt_state.cfg.shortwave_band}/umu0": rt_state.cos_zenith_dayside.reshape(ncol),
        f"{rt_state.cfg.shortwave_band}/albedo": (
            rt_state.cfg.sw_surface_albedo
            * torch.ones((rt_state.sw_nwave, ncol), dtype=torch.float64, device=temp_i.device)
        ),
        f"{rt_state.cfg.longwave_band}/albedo": (
            rt_state.cfg.lw_surface_albedo
            * torch.ones((rt_state.lw_nwave, ncol), dtype=torch.float64, device=temp_i.device)
        ),
    }
    atm = {"pres": pres_i, "temp": temp_i}

    net_flux, _, _ = rt_state.rad.forward(conc_i, rt_state.dz, bc, atm)  # (ncol, nlyr+1)

    # Volumetric heating [W/m^3] = -div(F) on a spherical shell using face areas.
    div_f = (
        rt_state.area1[:, 1:] * net_flux[:, 1:] - rt_state.area1[:, :-1] * net_flux[:, :-1]
    ) / rt_state.vol
    heating = -div_f

    return heating.reshape(ny, nx, nlyr)


def update_rt_tendency_if_needed(
    block: MeshBlock,
    eos,
    thermo_y,
    thermo_x: ThermoX,
    block_vars: dict[str, torch.Tensor],
    current_time: float,
    rt_state: RadiativeTransferState,
) -> None:
    if current_time + 1.0e-12 < rt_state.next_update_time:
        return

    rt_state.last_heating = _compute_rt_heating(block, eos, thermo_y, thermo_x, block_vars, rt_state)
    rt_state.next_update_time = current_time + max(rt_state.cfg.update_dt, 0.0)

    print(
        "RT update:",
        f"t={current_time / SECONDS_PER_DAY:.3f} d,",
        f"heating[min,max]=({rt_state.last_heating.min().item():.3e}, {rt_state.last_heating.max().item():.3e}) Pa/s",
    )


def apply_rt_forcing(block_vars: dict[str, torch.Tensor], rt_state: RadiativeTransferState, dt: float) -> None:
    block_vars["hydro_u"][kIPR, ..., rt_state.il : rt_state.iu + 1] += rt_state.last_heating * dt


def write_restart_manifest(
    checkpoint_dir: Path,
    checkpoint_day: int,
    current_time: float,
    config_file: str,
    output_dir: str,
    basename: str,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    restart_candidates = sorted(glob.glob(str(Path(output_dir) / f"{basename}.*.restart")))
    restart_file = restart_candidates[-1] if restart_candidates else None

    payload = {
        "checkpoint_day": checkpoint_day,
        "simulation_time_seconds": float(current_time),
        "simulation_time_days": float(current_time / SECONDS_PER_DAY),
        "config_file": str(Path(config_file).resolve()),
        "output_dir": str(Path(output_dir).resolve()),
        "latest_restart_archive": restart_file,
        "resume_hint": {
            "command": (
                "python sub_neptune_rt/run_sub_neptune_rt.py "
                f"-c {config_file} --output-dir {output_dir} --restart-name "
                + (Path(restart_file).name if restart_file else "<restart-file-name>")
            )
        },
    }

    manifest_file = checkpoint_dir / f"checkpoint_day_{checkpoint_day:04d}.yaml"
    with open(manifest_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def run_simulation(
    block: MeshBlock,
    eos,
    thermo_y,
    thermo_x: ThermoX,
    kinet: Kinetics,
    block_vars: dict[str, torch.Tensor],
    current_time: float,
    tlim: float,
    rt_state: RadiativeTransferState,
    config_file: str,
    output_dir: str,
    basename: str,
) -> tuple[dict[str, torch.Tensor], float]:
    block.options.intg().tlim(tlim)

    next_checkpoint_day = int(current_time // (10.0 * SECONDS_PER_DAY)) * 10 + 10
    checkpoint_dir = Path(output_dir) / "restart_checkpoints"

    update_rt_tendency_if_needed(block, eos, thermo_y, thermo_x, block_vars, current_time, rt_state)
    block.make_outputs(block_vars, current_time)

    while not block.intg.stop(block.inc_cycle(), current_time):
        dt = block.max_time_step(block_vars)
        block.print_cycle_info(block_vars, current_time, dt)

        for stage in range(len(block.intg.stages)):
            block.forward(block_vars, dt, stage)
            update_rt_tendency_if_needed(block, eos, thermo_y, thermo_x, block_vars, current_time, rt_state)
            apply_rt_forcing(block_vars, rt_state, dt)

        err = block.check_redo(block_vars)
        if err > 0:
            continue
        if err < 0:
            break

        del_rho = evolve_kinetics(block_vars["hydro_w"], eos, thermo_x, thermo_y, kinet, dt)
        block_vars["hydro_u"][kICY:] += del_rho

        current_time += dt
        block.make_outputs(block_vars, current_time)

        while current_time >= next_checkpoint_day * SECONDS_PER_DAY:
            write_restart_manifest(
                checkpoint_dir=checkpoint_dir,
                checkpoint_day=next_checkpoint_day,
                current_time=current_time,
                config_file=config_file,
                output_dir=output_dir,
                basename=basename,
            )
            next_checkpoint_day += 10

    return block_vars, current_time


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Sub-Neptune simulation with Toon radiative transfer.")
    p.add_argument("-c", "--config", required=True, help="YAML configuration file")
    p.add_argument("--output-dir", default="output", help="Output directory")
    p.add_argument(
        "--restart-name",
        default="",
        help=(
            "Restart archive filename inside output dir (e.g. "
            "sub_neptune_rt_doublegrey.00005.restart or sub_neptune_rt_doublegrey.final.restart)"
        ),
    )
    p.add_argument(
        "--rebuild-jit",
        action="store_true",
        help="Rebuild JIT opacity .pt files even if they already exist.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    block, eos, thermo_y, thermo_x, kinet, device = create_models(args.config, args.output_dir)

    if args.restart_name:
        block_vars, current_time = block.initialize_from_restart(args.restart_name)
    else:
        block_vars, current_time = initialize_isothermal(block, config)

    for key, data in block_vars.items():
        if isinstance(data, torch.Tensor):
            print(f"{key}: shape={tuple(data.shape)} dtype={data.dtype} device={data.device}")

    rt_state = build_rt_state(
        block=block,
        config=config,
        config_file=args.config,
        device=device,
        rebuild_jit=args.rebuild_jit,
    )
    print(
        "RT forcing summary:",
        f"stellar_flux_nadir={rt_state.cfg.stellar_flux_nadir:.3f} W/m^2,",
        f"rt_update_dt={rt_state.cfg.update_dt:.1f} s,",
        f"sw_band={rt_state.cfg.shortwave_band}, lw_band={rt_state.cfg.longwave_band}",
    )

    tlim = float(config["integration"]["tlim"])
    basename = Path(args.config).stem
    block_vars, current_time = run_simulation(
        block=block,
        eos=eos,
        thermo_y=thermo_y,
        thermo_x=thermo_x,
        kinet=kinet,
        block_vars=block_vars,
        current_time=current_time,
        tlim=tlim,
        rt_state=rt_state,
        config_file=args.config,
        output_dir=args.output_dir,
        basename=basename,
    )

    block.finalize(block_vars, current_time)


if __name__ == "__main__":
    main()
