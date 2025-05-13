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

    Returns:
        Tuple[List[str], np.ndarray]: List of species and their abundances.
    """

    species = np.genfromtxt(filename, dtype=str, max_rows=1)
    abundances = np.genfromtxt(filename, skip_header=1)

    return species.tolist(), abundances

def load_sonora_data(ck_name: str):
    """
    This functions calls the get_legacy_data_1460
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

    return vars(op)
