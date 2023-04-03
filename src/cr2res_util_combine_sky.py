import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import argparse
from os.path import join
import os
from util import make_index
import sys
from tqdm import tqdm

if len(sys.argv) > 1:
    parser = argparse.ArgumentParser()
    parser.add_argument("sof", help="Set Of Files")
    parser.add_argument("--output-dir", help="he directory where the product files should be finally moved to (all products are first created in the current dir)")
    args = parser.parse_args()
    sof = args.sof
    outfolder = args.output_dir
    if outfolder is None:
        outfolder = os.getcwd()
else:
    sof = "/scratch/ptah/anwe5599/CRIRES/2022-11-29/extr/sky.sof"
    outfolder = "/scratch/ptah/anwe5599/CRIRES/2022-11-29/extr/"

with open(sof) as f:
    lines = f.readlines()
lines = [l.split() for l in lines]

science_fname = [l[0] for l in lines if l[1] == "UTIL_CALIB"]
tw_fname = [l[0] for l in lines if l[1] == "CAL_FLAT_TW"][0]

tw = fits.open(tw_fname)
science = [fits.open(sci) for sci in science_fname]
offset = [sci[0].header["ESO SEQ CUMOFFSETY"] for sci in science] # in pixels

if len(science_fname) != 2:
    raise ValueError("Expected exactly 2 frames to combine")
if offset[0] * offset[1] >= 0:
    raise ValueError("Require two offsets that are of opposite directions")

# A is positive offset
# B is negative offset
science_A = [sci for sci, off in zip(science, offset) if off > 0][0]
science_B = [sci for sci, off in zip(science, offset) if off < 0][0]

hdus = [science[0][0]]

for chip in tqdm([1, 2, 3], desc="CHIP"):
    tw_data = tw[f"CHIP{chip}.INT1"].data

    header = science_A[f"CHIP{chip}.INT1"].header
    data_A = science_A[f"CHIP{chip}.INT1"].data
    data_B = science_B[f"CHIP{chip}.INT1"].data
    err_header = science_A[f"CHIP{chip}ERR.INT1"].header
    err_A = science_A[f"CHIP{chip}ERR.INT1"].data
    err_B = science_B[f"CHIP{chip}ERR.INT1"].data

    result = np.copy(data_A)
    result_error = np.copy(err_A)

    for order in tqdm(set(tw_data["Order"]), desc="Order"):
        idx = tw_data["Order"] == order
        x = np.arange(1, 2049)
        upper = np.polyval(tw_data[idx]["Upper"][0][::-1], x)
        lower = np.polyval(tw_data[idx]["Lower"][0][::-1], x)
        middle = np.polyval(tw_data[idx]["All"][0][::-1], x)

        height_upp = int(np.ceil(np.max(upper - middle)))
        height_low = int(np.ceil(np.max(middle - lower)))

        middle_int = middle.astype(int)
        upper_int = middle_int + height_upp
        lower_int = middle_int - height_low

        idx = make_index(middle_int, upper_int, 0, 2048)
        result[idx] = data_A[idx]
        result_error[idx] = err_A[idx]
        idx = make_index(lower_int, middle_int, 0, 2048)
        result[idx] = data_B[idx]
        result_error[idx] = err_B[idx]

    secondary = fits.ImageHDU(result, header=header)
    err_secondary = fits.ImageHDU(result_error, header=err_header)
    hdus += [secondary, err_secondary]

# save results
outfile = join(outfolder, "cr2res_util_combine_sky.fits")
hdus = fits.HDUList(hdus)
hdus.writeto(outfile, overwrite=True)

# vmin, vmax = np.nanpercentile(result, (10, 90))
# plt.imshow(result, aspect="auto", origin="lower", interpolation="none", vmin=vmin, vmax=vmax)
# plt.plot(x, upper, "r")
# plt.plot(x, lower, "r")
# plt.plot(x, middle, "r--")
# plt.savefig("test.png")
