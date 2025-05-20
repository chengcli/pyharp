import torch
import numpy as np

def load_xiz_legacy_data(fname: str) -> dict:
    data = np.genfromtxt(fname)
    return {
            'wavenumber': data[1:,0],
            'temp': data[0,1:],
            'kappa': -data[1:, 1:],
        }

def load_orton_legacy_data(fname: str) -> dict:
    data = np.genfromtxt(fname)
    return {
            'wavenumber': data[1:,0],
            'temp': data[0,1:],
            'kappa': data[1:, 1:],
        }

def save_cia_legacy_wave_temp(fname: str, data: dict) -> None:
    out = {
        'wavenumber': torch.tensor(data['wavenumber'], dtype=torch.float64),
        'temp': torch.tensor(data['temp'], dtype=torch.float64),
        'kappa': torch.tensor(data['kappa'], dtype=torch.float64),
    }

    class Container(torch.nn.Module):
        def __init__(self, values: dict):
            super().__init__()
            for key in values:
                setattr(self, key, values[key])

    container = torch.jit.script(Container(out))
    container.save(f'{fname}.pt')

if __name__ == '__main__':
    datafiles = [
        'H2-H2-eq.orton.txt',
        'H2-H2-eq.xiz.txt',
        'H2-He-eq.orton.txt',
        'H2-He-eq.xiz.txt',
        'H2-H2-nm.orton.txt',
        'H2-H2-nm.xiz.txt',
        'H2-He-nm.orton.txt',
        'H2-He-nm.xiz.txt',
        ]

    for fname in datafiles:
        if fname.endswith('.xiz.txt'):
            data = load_xiz_legacy_data(fname)
        else:
            data = load_orton_legacy_data(fname)
        save_cia_legacy_wave_temp(fname[:-4], data)
