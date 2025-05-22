import torch
from pyharp import (
        h2_cia_legacy,
        RadiationOptions,
        Radiation,
        )
from pyharp.opacity import AttenuatorOptions, JITOpacity

class GreyOpacity(torch.nn.Module):
    """
    GreyOpacity class for handling grey opacity calculations.
    """
    def forward(self, conc: torch.Tensor) -> torch.Tensor:
        ncol, nlyr = conc.shape[0], conc.shape[1]
        return 0.1 * torch.ones((ncol, nlyr, 1), dtype=torch.float64)

# save a jit model to file
model = GreyOpacity()
scripted = torch.jit.script(model)
scripted.save("grey_opacity.pt")

op = AttenuatorOptions().type("jit")
op.opacity_files(["grey_opacity.pt"])

ab = JITOpacity(op)
conc = torch.ones(3, 5)
print(ab.forward(conc, {}))
