#! /usr/bin/env python3
import sys

sys.path.append("../python")
sys.path.append(".")

import canoe
from canoe import load_configure
from canoe.harp import radiation_band
from netCDF4 import Dataset
from scipy.interpolate import interp1d
from multiprocessing import Pool, cpu_count
from typing import List
import numpy as np


if __name__ == "__main__":
    canoe.start()

    opacity_input = "amarsw-lbl"
    opacity_output = "amarsw-ck"
    opacity_config = "amarsw-op.yaml"
    max_threads = cpu_count()

    config = load_configure(opacity_config)
    pool = Pool(max_threads)
    bnames = list(map(str, config["bands"]))
    pool.map(run_cktable_one_band, bnames)
