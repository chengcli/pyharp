import numpy as np
from importlib import resources
from typing import Tuple

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
