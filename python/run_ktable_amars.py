#! /usr/bin/env python3
import sys

sys.path.append("../python")
sys.path.append(".")

import canoe
from canoe import load_configure, find_resource
from canoe.harp import radiation_band
from atm_profile_utils import read_atm_profile
from multiprocessing import Pool, cpu_count
from typing import Tuple
from rfmlib import *




if __name__ == "__main__":
    canoe.start()

    atm_profile = "amarsw.atm"
    opacity_config = "amarsw-op.yaml"
    opacity_output = "amarsw-lbl"
    tem_grid = (5, -50, 50)
    max_threads = cpu_count()

    hitran_file = find_resource("HITRAN2020.par")
    atm = read_atm_profile(atm_profile)
    config = load_configure(opacity_config)

    pool = Pool(max_threads)
    bnames = list(map(str, config["bands"]))
    pool.map(run_ktable_one_band, bnames)
