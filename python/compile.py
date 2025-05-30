import torch

def compile(model, filename):
    """
    Compile a PyTorch model to a TorchScript file.

    Args:
        model: The PyTorch model to compile.
        filename: The name of the file to save the compiled model.

    Returns:
        A compiled version of the model.
    """
    scripted = torch.jit.script(model)
    scripted.save(filename)
    return torch.compile(model)
