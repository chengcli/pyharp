import torch
import os
from pyharp.sonora import (
        load_sonora_data,
        save_sonora_multiband_ck
        )

def preprocess_sonora(fname: str):
    # dictionary of data
    data = load_sonora_data(fname)

    # pressure grid
    pres = data['press']
    print("Pressure grid:", pres)

    # temperature grid
    temp = data['temps']
    print("Temperature grid:", temp)

    save_sonora_multiband_ck(fname, data, clean=False)

if __name__ == "__main__":
    # Example usage
    fname = "sonora_2020_feh+000_co_100.data.196"
    if not os.path.exists(fname + ".pt"):
        preprocess_sonora(fname)

    container = torch.jit.load(fname + ".pt")
    print(container.points)
    print(container.weights)
    # load sonora data
    # load_sonora_window(fname)
    # load_sonora_atm(fname)
