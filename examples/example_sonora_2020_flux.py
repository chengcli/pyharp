from pyharp.sonora import (
        load_sonora_atm,
        load_sonora_window,
        load_sonora_data,
        )

pres, temp = load_sonora_atm()
wmin, wmax = load_sonora_window()

data = load_sonora_data("sonora_2020_feh+000_co_100.data.196")
print(data.keys())
print(data["kappa"].shape)
