import torch
import pyharp
from pyharp import (
        h2_cia_legacy,
        RadiationOptions,
        Radiation,
        )
from pyharp.opacity import AttenuatorOptions, JITOpacity

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

    def forward(self, conc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            conc: Tensor of shape (ncol, nlyr, nspecies) representing concentrations.
        Returns:
            Tensor of shape (nwave, ncol, nlyr, nprop) representing optical properties.
        """
        ncol, nlyr = conc.shape[0], conc.shape[1]
        return 0.1 * torch.ones((self.nwave, ncol, nlyr, self.nprop), dtype=torch.float64)

# compile a jit model to file
pyharp.compile(GreyOpacity(1,1), "grey_opacity.pt")

# user it later
op = AttenuatorOptions().type("jit")
op.opacity_files(["grey_opacity.pt"])

ab = JITOpacity(op)
conc = torch.ones(3, 5)
print(ab.forward(conc, {}))
