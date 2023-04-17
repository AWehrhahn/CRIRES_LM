from os.path import dirname, join, realpath

import matplotlib.pyplot as plt
import numpy as np
from astropy import constants as const
from astropy.io import fits
import os
from molecfit_wrapper.molecfit import Molecfit
import argparse
import sys

from cpl.ui import FrameSet

if len(sys.argv) > 1:
    parser = argparse.ArgumentParser()
    parser.add_argument("sof", help="Set Of Files")
    parser.add_argument("--output-dir", help="he directory where the product files should be finally moved to (all products are first created in the current dir)", default=os.getcwd())
    args = parser.parse_args()
    sof = args.sof
    output_dir = args.output_dir
    use_chip = None
    use_order = None
else:
    sof = "/scratch/ptah/anwe5599/CRIRES/2022-11-29/extr/molecfit_star.sof"
    output_dir = "/scratch/ptah/anwe5599/CRIRES/2022-11-29/extr"
    use_chip = 1
    use_order = None

mf = Molecfit(
    recipe_dir="/home/anwe5599/esotools/lib/esopipes-plugins",
    tmp_dir="/scratch/ptah/anwe5599/tmp",
    output_dir=output_dir,
    column_lambda="WAVE",
    column_flux="FLUX",
    column_dflux="ERR",
    wlg_to_micron=1,
    slit_width_value=0.1,
    transmission=True, # for sky emission spectrum
    tmp_path="/scratch/ptah/anwe5599/tmp",
    use_input_kernel=False,
    fit_continuum=1,
    continuum_n=0,
    continuum_const=1,
    fit_wlc=1,
    wlc_n=1,
    varkern=True,
    fit_telescope_background=True,
    telescope_background_const=0.06,
    kernmode=False,
    list_molec=["H2O", "CH4", "N2O"],
    # list_molec=["H2O", "O3", "O2", "CO2", "CH4", "N2O"],

    # list_molec=["H2O", "O3", "CH4", "C2H6", "NO2", "CO2", "OCS", "N2O"]
    # list_molec=["H2O", "O3", "O2", "CO2", "CH4", "N2O", "NH3", "HOCL", "NO", "SO2", "HCN", "C2H2"], 
    # to investigate
    # OH, HO2, "HBr", "HF", "H2", "SO", "PH3", "CS2", "H2S", "HCL", "HI"
    # available species
    #"N2,O2,CO2,O3,H2O,CH4,N2O,HNO3,CO,NO2,N2O5,CLO,HOCL,CLONO2,NO,HNO4,HCN,NH3,F11,F12,F14,F22,CCL4,COF2,H2O2,C2H2,C2H6,OCS,SO2,SF6"
    # list_molec=["H2O", "O3", "O2", "CO2", "CH4", "N2O", "C2H6", "OCS", "CO", "NO", "SO2", "NH3", "HNO3", "N2", "HCN", "H2O2", "C2H2", "COF2"]
)

def filter_frame_set(fs: FrameSet, tag:str) -> FrameSet:
    frames = [f for f in fs if f.tag == tag]
    frames = FrameSet(frames)
    return frames

# Read the input file
fs = FrameSet(sof)

input_files = filter_frame_set(fs, "UTIL_CALIB_EXTRACT_1D")
blaze_files = filter_frame_set(fs, "CAL_FLAT_EXTRACT_1D")

if len(input_files) == 0:
    raise ValueError("No input spectrum found")
elif len(input_files) > 1:
    raise ValueError("More than one input spectrum found")

if len(blaze_files) == 0:
    raise ValueError("No blaze correction found")
elif len(blaze_files) > 1:
    raise ValueError("More than one blaze correction found")


hdu = input_files[0].as_hdulist()
hdu_blaze = blaze_files[0].as_hdulist()
header = hdu[0].header

chip = 1
ext = f"CHIP{chip}.INT1"
data = hdu[ext].data
columns = data.names
orders = np.sort([int(c[:2]) for c in columns if c[-4:] == "SPEC"])
norders = len(orders)
nrows = len(data)

if use_chip is None:
    chips = [1, 2, 3]
    nchip = 3
else:
    chips = [use_chip,]
    nchip = 1

if use_order is None:
    orders = orders
    norders = norders
else:
    if np.isscalar(use_order):
        orders = [use_order,]
    else:
        orders = use_order
    norders = len(orders)

flux = np.zeros((norders * nchip, nrows))
wave = np.zeros((norders * nchip, nrows))
err = np.zeros((norders * nchip, nrows))
mapping = np.recarray(norders * nchip, formats=["i4", "i4", "i4"], names=["CHIP", "ORDER", "MOLECFIT"])

for j, chip in enumerate(chips):
    ext = f"CHIP{chip}.INT1"
    
    data = hdu[ext].data
    blaze = hdu_blaze[ext].data

    for i, order in enumerate(orders):
        flux[i + j*norders] =  data[f"{order:02}_01_SPEC"] / blaze[f"{order:02}_01_SPEC"]
        wave[i + j*norders] =  data[f"{order:02}_01_WL"]
        err[i + j*norders] = data[f"{order:02}_01_ERR"]
        mapping[i + j*norders]["CHIP"] = chip
        mapping[i + j*norders]["ORDER"] = order
        mapping[i + j*norders]["MOLECFIT"] = i + j * norders + 1

flux[flux == 0] = np.nan
# quick normalization
flux -= np.nanmin(flux, axis=1)[:, None]

x = np.arange(flux.shape[1])
for i in range(len(flux)):
    # First step, just linear
    mask = np.isfinite(flux[i])
    factor = np.max(flux[i][mask])
    flux[i] /= factor
    err[i] /= factor

x = np.arange(nrows)
for i in range(len(flux)):
    mask = np.isnan(flux[i])
    flux[i, mask] = np.interp(x[mask], x[~mask], flux[i, ~mask])
    err[i, mask] = np.interp(x[mask], x[~mask], err[i, ~mask])

# Wavelength in micron
# wave += 1
wave *= 0.001

plt.clf()
for i in range(len(flux)):
    plt.subplot(9, 2, i+1)
    plt.plot(flux[i])
    plt.xticks([])
    plt.yticks([])
plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig("test.png", dpi=600)


# Step 1:
# Since we modifed the flux and wavelength we need to write the data to a new datafile
input_file, mapping = mf.prepare_fits(header, wave, flux, err=err, mapping=mapping)
rc, sof = mf.molecfit_model_prepare(input_file)

# Save the parameters and the SOF
rc_fname = join(output_dir, "molecfit_model.rc")
sof_fname = join(output_dir, "molecfit_model.sof")
rc.write(rc_fname)
sof.write(sof_fname)

# Save the mapping
mapping_hdu = fits.BinTableHDU(data=mapping)
mapping_hdu.writeto(join(output_dir, "MAPPING.fits"), overwrite=True)
pass
