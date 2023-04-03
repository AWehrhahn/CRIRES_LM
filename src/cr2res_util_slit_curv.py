import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import argparse

from util import make_index
import sys
import os
from scipy.ndimage import gaussian_filter1d
from pyreduce.extract import extract
from pyreduce.make_shear import Curvature

if len(sys.argv) > 1:
    parser = argparse.ArgumentParser()
    parser.add_argument("sof", help="Set Of Files")
    parser.add_argument(
        "--output-dir",
        help="he directory where the product files should be finally moved to (all products are first created in the current dir)",
    )
    args = parser.parse_args()
    sof = args.sof
    outfolder = args["output-dir"]
    if outfolder is None:
        outfolder = os.getcwd()
else:
    sof = "/scratch/ptah/anwe5599/CRIRES/2022-11-29/extr/slit_curv.sof"
    outfolder = "/scratch/ptah/anwe5599/CRIRES/2022-11-29/extr/"

with open(sof) as f:
    lines = f.readlines()
lines = [l.split() for l in lines]

tw_fname = [l[0] for l in lines if l[1] == "CAL_FLAT_TW"][0]
science_fname = [l[0] for l in lines if l[1] == "UTIL_CALIB"][0]
blaze_fname = [l[0] for l in lines if l[1] == "CAL_FLAT_EXTRACT_1D"][0]

tw = fits.open(tw_fname)
science = fits.open(science_fname)
blaze = fits.open(blaze_fname)

tilts = [0, 0, 0]
shears = [0, 0, 0]

for chip in [1, 2, 3]:
    tw_data = tw[f"CHIP{chip}.INT1"].data
    data = science[f"CHIP{chip}.INT1"].data
    err = science[f"CHIP{chip}ERR.INT1"].data
    blaze_data = blaze[f"CHIP{chip}.INT1"].data

    order_traces = []
    extraction_width = []
    column_range = []
    upper_ycen = []
    lower_ycen = []
    orders = list(set(tw_data["Order"]))

    for order in orders:
        idx = tw_data["Order"] == order
        x = np.arange(1, 2048 + 1)
        upper = np.polyval(tw_data[idx]["Upper"][0][::-1], x)
        lower = np.polyval(tw_data[idx]["Lower"][0][::-1], x)
        middle = np.polyval(tw_data[idx]["All"][0][::-1], x)

        height_upp = int(np.ceil(np.max(upper - middle)))
        height_low = int(np.ceil(np.max(middle - lower)))

        middle_int = middle.astype(int)
        upper_ycen += [middle_int + height_upp]
        lower_ycen += [middle_int - height_low]

        order_traces += [tw_data[idx]["All"][0][::-1]]
        column_range += [[10, 2048 - 10]]
        extraction_width += [[height_upp, height_low]]
    
    order_traces = np.asarray(order_traces)
    column_range = np.asarray(column_range)
    extraction_width = np.asarray(extraction_width)

    im_norm, im_ordr, spectrum, column_range = extract(
        data,
        order_traces,
        column_range=column_range,
        extraction_width=extraction_width,
        osample=10,
        swath_width=200,
        extraction_type="normalize",
        threshold_lower=-np.inf,
    )

    for i, order in enumerate(orders):
        # Reject outliers
        idx_data = make_index(lower_ycen[i], upper_ycen[i], 0, 2048)
        relative = data[idx_data] - im_ordr[idx_data]
        mask = np.isfinite(relative)
        std = 1.5 * np.median(np.abs(np.median(relative[mask]) - relative[mask]))
        mask &= np.abs(relative) < 10 * std
        # Need complex indexing to actually set the values to nan
        data[idx_data[0][~mask], idx_data[1][~mask]] = np.nan

        # Correct for the blaze
        blaze_spec = blaze_data[f"{order:02}_01_SPEC"]
        spectrum[i] /= blaze_spec

        # Smooth spectrum for easier peak detection
        spectrum[i] = gaussian_filter1d(spectrum[i], 3)

        # Limit the size of the extraction width 
        # to avoid the interorder area
        extraction_width[i, 0] -= 10
        extraction_width[i, 1] -= 10

    module = Curvature(
        order_traces,
        extraction_width,
        column_range=column_range,
        curv_degree=1,
        peak_threshold=1.5,
        window_width=21,
        plot=0,
        fit_degree=(1, 1),
        mode="2D",
        peak_function="spectrum"
    )
    data = np.ma.array(data, mask=~np.isfinite(data))
    tilt, shear = module.execute(spectrum, data)

    tilts[chip-1] = tilt
    shears[chip-1] = shear

print("Saving the results to the trace wave file")

# Create output
for chip in [1, 2, 3]:
    ext = f"CHIP{chip}.INT1"
    tw_data = tw[ext].data
    orders = np.sort(list(set(tw_data["Order"])))
    x = np.arange(1, 2049)
    for i, order in enumerate(orders):
        t = tilts[chip-1][i]
        s = shears[chip-1][i]
        ycen = np.polyval(tw_data[idx]["All"][0][::-1], x)
        # Convert to the global reference system
        t -= 2 * ycen * s

        # Fit overarching polynomial
        ct = np.polyfit(x, t, 2)
        cs = np.polyfit(x, s, 2)

        # Write the data back to the fits file
        # The indexing is chosen such that the data is actually 
        # stored in the tw object and not lost
        idx = tw_data["Order"] == order
        tw[ext].data["SlitPolyA"][idx] = [0, 1, 0]
        tw[ext].data["SlitPolyB"][idx] = ct[::-1]
        tw[ext].data["SlitPolyC"][idx] = cs[::-1]

# The recipe overwrites the existing tracewave
# With the new information
tw.writeto(tw_fname, overwrite=True)

