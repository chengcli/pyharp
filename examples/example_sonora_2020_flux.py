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

    # temperature grid
    temp = data['temps']

    save_sonora_multiband_ck(fname, data)

if __name__ == "__main__":
    # Example usage
    fname = "sonora_2020_feh+000_co_100.data.196"
    if not os.path.exists(fname + ".pt"):
        preprocess_sonora(fname)

    data = torch.load(fname + ".pt")
    # load sonora data
    # load_sonora_window(fname)
    # load_sonora_atm(fname)
