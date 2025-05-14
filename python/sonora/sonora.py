import os
import torch
import numpy as np
import tarfile
from importlib import resources
from typing import Tuple, List
from .get_legacy_data_1460 import _get_legacy_data_1460

def load_sonora_atm() -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the atmospheric pressure and temperature from the Sonora 2020 database.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Atmospheric pressure (Pa) and temperature (K).
    """
    with resources.files('pyharp.sonora').joinpath('sonora2020_1460_layer_list.txt').open('r') as f:
        data = np.genfromtxt(f, skip_header=2)
    return data[:, 2] * 1.e5, data[:, 1]

def load_sonora_window() -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the Sonora 2020 spectral window (start, end) in nm.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Start and end wavelengths (nm).
    """
    with resources.files('pyharp.sonora').joinpath('sonora2020_196_windows.txt').open('r') as f:
        lines = f.readlines()

    data = {'lambda1': [], 'lambda2': []}
    current_key = None

    for line in lines:
        line = line.strip()
        if line.startswith('lambda1'):
            current_key = 'lambda1'
            line = line.replace('lambda1 =', '').strip()
        elif line.startswith('lambda2'):
            current_key = 'lambda2'
            line = line.replace('lambda2 =', '').strip()

        if current_key and line:
            values = [float(x) for x in line.split()]
            data[current_key].extend(values)

    return np.array(data['lambda1']), np.array(data['lambda2'])

def load_sonora_abundances(filename: str) -> Tuple[List[str], np.ndarray]:
    """
    Returns the abundances from the Sonora 2020 database.

    Args:
        filename (str): Path to the abundances file.
    Returns:
        Tuple[List[str], np.ndarray]: List of species and their abundances.
    """

    species = np.genfromtxt(filename, dtype=str, max_rows=1)
    abundances = np.genfromtxt(filename, skip_header=1)

    return species.tolist(), abundances

def load_sonora_data(ck_name: str) -> dict:
    """
    This functions calls the get_legacy_data_1460

    Args:
        ck_name (str): The name of the ck file without the .tar.gz extension.
    Returns:
        dict: A dictionary containing the loaded data.
    """

    # create a dummy class to hold result
    class Dummy:
        full_abunds = {}

    with tarfile.open(ck_name + ".tar.gz", "r:gz") as tar:
        # Access the file inside without extracting to disk
        member = tar.getmember(ck_name + '/ascii_data')
        op = Dummy()
        op.ck_file = tar.extractfile(member)
        _get_legacy_data_1460(op)

    # kappa was (pres, temp, band, wave)
    # reshape kappa to (band, wave, pres, temp)
    op.kappa = np.transpose(op.kappa, (2, 3, 0, 1))

    # bar -> pa
    op.press = list(op.pressures[:op.max_pc] * 1.e5)
    del op.pressures
    return vars(op)

def save_sonora_multiband(ck_name: str, data: dict, clean: bool=True) -> None:
    """
    Save the Sonora 2020 data to a .pt file.

    Args:
        ck_name (str): The name of the ck file.
        data (dict): The data to save.
        clean (bool): Whether to clean up the original tar.gz file.

    Returns:
        None
    """
    wmin, wmax = load_sonora_window()

    class Container(torch.nn.Module):
        def __init__(self, values: dict):
            super().__init__()
            for key in values:
                setattr(self, key, values[key])

    out = {
        'pres': torch.tensor(data['press'], dtype=torch.float64),
        'temp': torch.tensor(data['temps'], dtype=torch.float64),
        'wmin': torch.tensor(wmin, dtype=torch.float64),
        'wmax': torch.tensor(wmax, dtype=torch.float64),
        'gauss_pts': torch.tensor(data['gauss_pts'], dtype=torch.float64),
        'gauss_wts': torch.tensor(data['gauss_wts'], dtype=torch.float64),
        'kappa': torch.tensor(data['kappa'], dtype=torch.float64),
    }

    wmin = out['wmin'][:, None]
    wmax = out['wmax'][:, None]
    pt = out['gauss_pts'][None, :]
    out['wavenumber'] = (wmin * (1. - pt) + wmax * pt).flatten()
    out['weights'] = out['gauss_wts'].repeat(data['nwno'])

    # (nband, ng, ...) -> (nband * ng, ...)
    out['kappa'] = out['kappa'].reshape(-1, *out['kappa'].shape[2:])

    container = torch.jit.script(Container(out))
    container.save(f'{ck_name}.pt')

    # Clean up the original tar.gz file
    if clean:
        os.remove(f"{ck_name}.tar.gz")
    print(f"Saved {ck_name}.pt")
