import torch
import pyharp
from pyharp import (
        h2_cia_legacy,
        RadiationOptions,
        Radiation,
        disort_config
        )
import numpy as np
from pyharp.opacity import AttenuatorOptions, JITOpacity

torch.set_default_dtype(torch.float64)

class GreyOpacity(torch.nn.Module):
    """
    GreyOpacity class for handling grey opacity calculations.

    The class signature and variable dimensions are strict.
    The class must have a forward method that takes a concentration tensor
    and returns an opacity tensor.

    The concentration vector is 3D with dimensions (ncol, nlyr, nspecies),
    where ncol is the number of columns, nlyr is the number of layers,
    and nspecies is the number of species.

    The opacity tensor is 4D with dimensions (nwave, ncol, nlyr, nprop),
    where nwave is the number of wavelengths, ncol is the number of columns,
    nlyr is the number of layers, and nprop is the number of optical properties.

    The first optical property is the total extinction cross-section [m^2/mol].
    The second optical property is the single scattering albedo.
    Starting from the third, the optical properties are phase function moments
    (excluding the zero-th moment).

    This class is later compiled to a TorchScript file using the `pyharp.compile` function.
    """
    def __init__(self, nwave: int, nprop: int):
        super().__init__()
        self.nwave = nwave
        self.nprop = nprop

    def forward(self, conc, temp) -> torch.Tensor:
        """
        Args:
            conc: Tensor of shape (ncol, nlyr, nspecies) representing concentrations.
        Returns:
            Tensor of shape (nwave, ncol, nlyr, nprop) representing optical properties.
        """
        ncol, nlyr = conc.shape[0], conc.shape[1]
        return 0.1 * torch.ones((self.nwave, ncol, nlyr, self.nprop), dtype=torch.float64)

# compile a jit model to file
nlyr = 4
nwave = 10
ncol = 1
nprop = 6
nstr = 4


model = GreyOpacity(nwave, nprop)
scripted = torch.jit.script(model)
scripted.save(f"grey-opacity.pt")

model = GreyOpacity(nwave, nprop)
scripted = torch.jit.script(model)
scripted.save(f"grey-opacity2.pt")

conc = torch.ones(ncol, nlyr, 6) * 0.2

temp = 300.0 * torch.ones(nlyr).unsqueeze(0)
print(temp)

rad_op = RadiationOptions.from_yaml('amars-ck.yaml')

dz = torch.ones(nlyr)*1000.0

for name, band in rad_op.bands().items():
    wmin = band.disort().wave_lower()[0]
    wmax = band.disort().wave_upper()[0]

    band.disort().accur(1.0e-12)
    disort_config(band.disort(), nstr, nlyr, ncol, nwave)
    band.ww(np.linspace(wmin, wmax, nwave))
    wave = torch.tensor(band.ww(), dtype=torch.float64)
    print(band.ww())
    bc = {}
    bc[name + "/fbeam"] = torch.tensor(55.).expand(nwave,ncol)
    bc[name + "/albedo"] = 0.3 * torch.ones((nwave,ncol))
    bc[name + "/umu0"] = torch.ones((ncol,))

print(rad_op)
print(conc.shape)
#exit()

atm = {}
atm['pres'] = torch.tensor([50000.,20000.,10000.,5000.]).unsqueeze(0)
atm['temp'] = temp

rad = Radiation(rad_op)

#just hacking these to be the right dtype
#conc = conc.float()
#dz = dz.float()
#for key in bc:
#    if isinstance(bc[key], torch.Tensor):
#        bc[key] = bc[key].float()
#    elif isinstance(bc[key], np.ndarray):
#        bc[key] = torch.from_numpy(bc[key]).float()
#for key in atm:
#    if isinstance(atm[key], torch.Tensor):
#        atm[key] = atm[key].float()
#    elif isinstance(atm[key], np.ndarray):
#        atm[key] = torch.from_numpy(atm[key]).float()

print("dz = ", dz)
print("atm = ", atm)
print("bc = ", bc)

netflux, downward_flux, upward_flux = rad.forward(conc, dz, bc, atm)
print(netflux)
