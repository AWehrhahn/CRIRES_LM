from os.path import dirname, join, realpath

import matplotlib.pyplot as plt
import numpy as np
from astropy import constants as const
from astropy.io import fits
import os
from molecfit_wrapper.molecfit import Molecfit
import argparse
import sys

if len(sys.argv) > 1:
    parser = argparse.ArgumentParser()
    parser.add_argument("sof", help="Set Of Files")
    parser.add_argument("--output-dir", help="he directory where the product files should be finally moved to (all products are first created in the current dir)", default=os.getcwd())
    args = parser.parse_args()
    sof = args.sof
    output_dir = args.output_dir
else:
    sof = "/scratch/ptah/anwe5599/CRIRES/2022-11-29/extr/molecfit_apply.sof"
    output_dir = "/scratch/ptah/anwe5599/CRIRES/2022-11-29/extr"

# Read the input file
with open(sof) as f:
    lines = f.readlines()

input_file = [l.split()[0] for l in lines if l.split()[1] == "UTIL_CALIB_EXTRACT_1D"][0]
model_file = [l.split()[0] for l in lines if l.split()[1] == "CAL_WAVE_MOLECFIT_MODEL"][0]
mapping_file = [l.split()[0] for l in lines if l.split()[1] == "CAL_WAVE_MOLECFIT_MAPPING"][0]


hdu = fits.open(input_file)
hdu_model = fits.open(model_file)
hdu_mapping = fits.open(mapping_file)

header = hdu[0].header
mapping = hdu_mapping[1].data
model = hdu_model[1]

chips = np.unique(mapping["CHIP"])

for chip in chips:
    ext = f"CHIP{chip}.INT1"
    data = hdu[ext].data
    orders = np.unique(mapping[mapping["CHIP"] == chip]["ORDER"])
    for order in orders:
        idx = mapping[(mapping["CHIP"] == chip) & (mapping["ORDER"] == order)]["MOLECFIT"][0]
        data[f"{order:02}_01_WL"] = model[model["CHIP"] == idx]["MLAMBDA"] * 1000

# Overwrite the results
hdu.writeto(input_file, overwrite=True)
