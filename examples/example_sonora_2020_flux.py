import torch
import os
from pyharp.sonora import (
        load_sonora_data,
        save_sonora_multiband
        )

def preprocess_sonora(fname: str):
    # dictionary of data
    data = load_sonora_data(fname)

    # print all keys
    print("Keys in data:", data.keys())
    print(data['nwno'])

    # pressure grid
    pres = data['press']
    print("Pressure grid:", pres)

    # temperature grid
    temp = data['temps']
    print("Temperature grid:", temp)

    # wavenumber
    print("wavenumber:", data['wno'])

    save_sonora_multiband(fname, data, clean=False)

if __name__ == "__main__":
    # Example usage
    fname = "sonora_2020_feh+000_co_100.data.196"

    if not os.path.exists(fname + ".pt"):
        preprocess_sonora(fname)

    container = torch.jit.load(fname + ".pt")
    print(container.gauss_pts)
    print(container.gauss_wts)
    print(container.wavenumber[:10], len(container.wavenumber))
    print(container.weights[:10], len(container.weights))
    print(container.kappa.shape)
    # load sonora data
    # load_sonora_window(fname)
    # load_sonora_atm(fname)
