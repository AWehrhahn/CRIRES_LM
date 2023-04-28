"""Script to prepare SOF files and shell script for reduction of raw CRIRES+ 
calibration files. This is part of a series of python scripts to assist with
reducing raw CRIRES+ data, which should be run in the following order:

    1 - make_calibration_sof.py             [reducing calibration files]
    2 - make_nodding_sof_master.py          [reducing master AB nodded spectra]
    3 - make_nodding_sof_split_nAB.py       [reducing nodding time series]
    4 - blaze_correct_nodded_spectra.py     [blaze correcting all spectra]
    5 - make_diagnostic_reduction_plots.py  [creating multipage PDF diagnostic]

The reduction approach for calibration data follows 'The Simple Way' outlined
in the data reduction manual: we only use the following three esorex commands: 
cr2res_cal_dark, cr2res_cal_flat, and cr2res_cal_wave. We prepare one SOF file
for each of these commands, and a single shell script to run the  commands one
after the other. It is assumed that all raw data is located in the same
directory and was taken by the same instrument/grating settings. 

We preferentially use the BPM generated from the flats, rather than the darks.
This is because in the IR the darks aren't truly dark due to thermal emission,
and this causes issues with the global outlier detection used by default in the
YJHK bands. As such, it is more robust to use the BPM generated from the flats
while also setting a high BPM kappa threshold when calling cr2res_cal_dark.
This ensures we only ever use the flat BPM, and our master dark isn't affected
by erroneous bad pixel flags.

While static trace wave files are inappropriate to use for extracting science
data, it is acceptable to pass them to cr2res_cal_flat as cr2res_cal_wave later
updates the tracewave solution in the working directory. Static detector
linearity files are appropriate so long as the SNR of our observations do not
exceed SNR~200.

Not all calibration files will necessarily be automatically associated with the
science frames when downloading from the ESO archive. In this case, they can be
downloaded manually by querying the night in question using a specific grating.
These calibrations are taken under the program ID 60.A-9051(A). For simplicity,
the settings for querying should be:
 - DATE OBS     --> YYYY MM DD (sometimes the night, otherwise morning after)
 - DPR CATG     --> CALIB
 - INS WLEN ID  --> [wl_setting]

The CRIRES specific archive is: https://archive.eso.org/wdb/wdb/eso/crires/form

Run as
------
python make_calibration_sofs.py [wl_setting]

where [wl_setting] is the grating setting, e.g. K2148.

Output
------
The following files are created by this routine, which by default contain:

dark.sof
    file/path/dark_1.fits         DARK
    ...
    file/path/dark_n.fits         DARK

flat.sof
    file/path/flat_1.fits         FLAT
    ...
    file/path/flat_n.fits         FLAT
    static/trace_wave.fits        UTIL_WAVE_TW
    file/path/dark_master.fits    CAL_DARK_MASTER

wave.sof
    file/path/wave_une_1.fits     WAVE_UNE
    ...
    file/path/wave_une_n.fits     WAVE_UNE
    file/path/wave_fpet_1.fits    WAVE_FPET
    ...
    file/path/wave_fpet_n.fits    WAVE_FPET
    static/trace.fits             UTIL_WAVE_TW
    file/path/flat_bpm.fits       CAL_FLAT_BPM
    file/path/master_dark.fits    CAL_DARK_MASTER
    file/path/master_flat.fits    CAL_FLAT_MASTER
    static/lines.fits             EMISSION_LINES

reduce_cals.sh:
    esorex cr2res_cal_dark  dark.sof --bpm_kappa=1000
    esorex cr2res_cal_flat  flat.sof --bpm_low=0.8 --bpm_high=1.2
    esorex cr2res_cal_wave  wave.sof
"""

""" Source: https://github.com/adrains/luciferase """

import sys
import numpy as np
import os
import glob
import subprocess
import pandas as pd
# import matplotlib.pyplot as plt
from astropy.io import fits
from os.path import join, dirname, realpath

import argparse

class SofFile:
    """
    Handles the 'Set Of Files' data files that are used by esorex.
    Essentially a list of files and their descriptor that is then used by esorex.

    TODO: the class could be more sophisticated using e.g. a numpy array
    or even pandas, but that would be overkill for now
    """

    def __init__(self, data=None):
        if data is None:
            data = []
        #:list: content of the SOF file
        self.data = data

    @classmethod
    def read(cls, filename):
        """
        Reads a sof file from disk

        Parameters
        ----------
        filename : str
            name of the file to read

        Returns
        -------
        self : SofFile
            The read file
        """
        data = []
        with open(filename, "r") as f:
            for line in f.readline():
                fname, ftype = line.split(maxsplit=1)
                data += [(fname, ftype)]
        self = cls(data)
        return self

    def write(self, filename):
        """
        Writes the sof file to disk

        Parameters
        ----------
        filename : str
            The name of the file to store
        """
        content = []
        for fname, ftype in self.data:
            content += [f"{fname} {ftype}\n"]

        with open(filename, "w") as f:
            f.writelines(content)

    def append(self, filename, tag):
        self.data += [(filename, tag)]

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------

if len(sys.argv) > 1:
    parser = argparse.ArgumentParser(prog="make_sof")
    parser.add_argument("setting", help="Wavelength setting of CRIRES+")
    parser.add_argument("-r", "--raw", help="Directory with raw data")
    parser.add_argument("-o", "--output", help="Output directory with processed data")
    parser.add_argument("-e", "--exposure-time", type=int, help="Filter by Exposure Time", default=None)
    args = parser.parse_args()
    wl_setting = args.setting.upper()
    rawdir = args.raw
    outdir = args.output
    exposure_time = args.exposure_time
else:
    day, wl_setting, exposure_time = "2022-11-29", "L3262", 30
    # day, wl_setting, exposure_time = "2022-12-23", "M4318", 10
    # day, wl_setting, exposure_time = "2022-12-23", "L3262", 30
    # day, wl_setting, exposure_time = "2022-12-25", "L3340", 30
    # day, wl_setting, exposure_time = "2022-12-31", "L3426", 60
    # day, wl_setting, exposure_time = "2023-01-22", "M4318", 10
    # day, wl_setting, exposure_time = "2023-02-15", "L3340", 60
    # day, wl_setting, exposure_time = "2023-02-25", "M4318", 10 # Bad Seeing?
    # day, wl_setting, exposure_time = "2023-02-26", "L3262", 60 # High PWV

    rawdir = f"/scratch/ptah/anwe5599/CRIRES/{day}_{wl_setting}/raw"
    outdir = f"/scratch/ptah/anwe5599/CRIRES/{day}_{wl_setting}/extr"

os.makedirs(outdir, exist_ok=True)

# -----------------------------------------------------------------------------
# Read in files
# -----------------------------------------------------------------------------

# Get a list of just the calibration files
fits_fns = glob.glob(join(rawdir, "CRIRE.*.fits"))
fits_fns.sort()

# Initialise columns for our calibration data frame
cal_fns = []
objects = []
exp_times = []
wl_settings = []
ndits = []
nod = []

cal_frames_kinds = ["FLAT", "DARK", "bet Pic"]

# settings = set([fits.getheader(fn).get("HIERARCH ESO INS WLEN ID") for fn in fits_fns])
# if None in settings:
#     settings.remove(None)

# Populate our columns
for fn in fits_fns:
    # Grab the header
    header = fits.getheader(fn)

    # Skip those without required keywords (e.g."EMISSION_LINES, PHOTO_FLUX)
    if "HIERARCH ESO INS WLEN ID" not in header or "OBJECT" not in header:
        continue
    # Skip anything other than the specific calibration frames we want
    elif header["OBJECT"] not in cal_frames_kinds:
        continue
    # And skip anything not in our current wavelength setting
    elif header["HIERARCH ESO INS WLEN ID"] != wl_setting:
        continue

    # Compile info
    cal_fns.append(fn)
    objects.append(header.get("OBJECT"))
    exp_times.append(header.get("HIERARCH ESO DET SEQ1 DIT"))
    wl_settings.append(header.get("HIERARCH ESO INS WLEN ID"))
    ndits.append(header.get("HIERARCH ESO DET NDIT"))
    nod.append(header.get("HIERARCH ESO SEQ NODPOS"))

# Create dataframe
cal_frames = pd.DataFrame(
    data=np.array([cal_fns, objects, exp_times, wl_settings, ndits, nod]).T,
    index=np.arange(len(cal_fns)),
    columns=["fn", "object", "exp", "wl_setting", "ndit", "nod"],
)

# Sort this by fn
cal_frames.sort_values(by="fn", inplace=True)

# -----------------------------------------------------------------------------
# Check set of calibration files for consistency
# -----------------------------------------------------------------------------
is_flat = cal_frames["object"] == "FLAT"
is_dark = cal_frames["object"] == "DARK"
is_science = cal_frames["object"] == "bet Pic"
is_science_A = is_science & (cal_frames["nod"] == "A")
is_science_B = is_science & (cal_frames["nod"] == "B")
if exposure_time is not None:
    is_science_A &= cal_frames["exp"] == exposure_time
    is_science_B &= cal_frames["exp"] == exposure_time



# We should have a complete set of calibration frames
if np.sum(is_flat) == 0:
    raise FileNotFoundError("Missing flat fields.")
elif np.sum(is_dark) == 0:
    raise FileNotFoundError("Missing dark frames.")
elif np.sum(is_science_A) == 0:
    raise FileNotFoundError("Missing nodding position A frames")
elif np.sum(is_science_B) == 0:
    raise FileNotFoundError("Missing nodding position B frames")
elif np.sum(is_science_A) % 2 != 0:
    raise ValueError("Requiring an even number of A positions")
elif np.sum(is_science_B) % 2 != 0:
    raise ValueError("Requiring an even number of B positions")

science_exps_A = cal_frames[is_science_A]["fn"]
science_exps_B = cal_frames[is_science_B]["fn"]

# There should only be a single exposure time and NDIT for flats and wave cals
flat_exps = set(cal_frames[is_flat]["exp"])
flat_ndits = set(cal_frames[is_flat]["ndit"])
if len(flat_ndits) > 1:
    raise Exception("There should only be one flat NDIT settting")

# If we have multiples exposures for the flats, raise a warning, plot a
# diagnostic, and continue with the higher exposure. The user can then quickly
# check the plot for saturation and hopefully keep things the same.
if len(flat_exps) > 1 or len(flat_ndits) > 1:
    print("Warning, multiple sets of flats with exps: {}".format(flat_exps))
    print("Have adopted higher exp. Check flat_comparison.pdf for saturation.")

    # Grab one of each flat (since we've sorted by filename, we should be safe
    # to grab the first and last file)
    fns = cal_frames[is_flat]["fn"].values[[0, -1]]
    exps = cal_frames[is_flat]["exp"].values[[0, -1]]

    # Plot a comparison for easy reference to check for saturation
    # fig, axes = plt.subplots(2, 3)
    # for flat_i in range(2):
    #     with fits.open(fns[flat_i]) as flat:
    #         for chip_i in np.arange(1, 4):
    #             xx = axes[flat_i, chip_i - 1].imshow(flat[chip_i].data)
    #             fig.colorbar(xx, ax=axes[flat_i, chip_i - 1])

    #         axes[flat_i, 0].set_ylabel(f"Exp = {exps[flat_i]}")

    # plt.savefig(join(outdir, "flat_comparison.pdf"))
    # plt.close("all")

    # Grab the larger exposure
    flat_exp = list(flat_exps)[np.argmax(np.array(list(flat_exps)).astype(float))]
else:
    flat_exp = list(flat_exps)[0]

flat_ndit = list(flat_ndits)[0]

# Raise an exception if we don't have a dark associated with each set of cals
if not np.any(np.isin(cal_frames[is_dark]["exp"], flat_exp)):
    raise Exception("No darks for flats with exp={} sec".format(flat_exp))

# Check that we have darks with exposure times and NDIT matching our flats, and
# if so prepare the filenames for the associated master darks
matches_flat_settings_mask = np.logical_and(
    cal_frames[is_dark]["exp"] == flat_exp,
    cal_frames[is_dark]["ndit"] == flat_ndit,
)

if np.sum(matches_flat_settings_mask) == 0:
    raise Exception("No darks matching flat exposure and NDIT.")






# -----------------------------------------------------------------------------
# Write SOF file for darks
# -----------------------------------------------------------------------------
# Make SOF file for darks. The esorex recipe can do all our darks at once, so
# we may as well add them all.
dark_sof = join(outdir, "dark.sof")
dark_sof_data = SofFile()
for dark in cal_frames[is_dark]["fn"]:
    dark_sof_data.append(join(rawdir, dark), "DARK")
dark_sof_data.write(dark_sof)

master_dark = join(outdir, "cr2res_cal_dark_master.fits")
master_dark_bpm = join(outdir, "cr2res_cal_dark_bpm.fits")

# -----------------------------------------------------------------------------
# Write SOF for flats
# -----------------------------------------------------------------------------
flat_sof = join(outdir, "flat.sof")
flat_sof_data = SofFile()
for flat in cal_frames[is_flat]["fn"]:
    flat_sof_data.append(join(rawdir, flat), "FLAT")
flat_sof_data.append(master_dark, "CAL_DARK_MASTER")
flat_sof_data.append(master_dark_bpm, "CAL_DARK_BPM")
flat_sof_data.write(flat_sof)

ff_name = join(outdir, "cr2res_util_calib_calibrated_collapsed.fits")
ff_name_flat = join(outdir, "cr2res_util_calib_flat_collapsed.fits")


# -----------------------------------------------------------------------------
# Write SOF for tracewave
# -----------------------------------------------------------------------------
tw_sof = join(outdir, "tracewave.sof")
tw_sof_data = SofFile()
tw_sof_data.append(ff_name_flat, "UTIL_CALIB")
tw_sof_data.write(tw_sof)

tw_name = join(outdir, "cr2res_util_calib_flat_collapsed_tw.fits")

# -----------------------------------------------------------------------------
# Write SOF for combining the science frames
# -----------------------------------------------------------------------------
combine_A_sof = join(outdir, "combine_A.sof")
combine_A_sof_data = SofFile()
for s in science_exps_A:
    combine_A_sof_data.append(s, "OBS_NODDING_OTHER")
combine_A_sof_data.append(master_dark_bpm, "CAL_DARK_BPM")
combine_A_sof_data.append(master_dark, "CAL_DARK_MASTER")
combine_A_sof_data.write(combine_A_sof)

ff_name_science_A = join(outdir, "cr2res_util_calib_science_A_collapsed.fits")

combine_B_sof = join(outdir, "combine_B.sof")
combine_B_sof_data = SofFile()
for s in science_exps_B:
    combine_B_sof_data.append(s, "OBS_NODDING_OTHER")
combine_B_sof_data.append(master_dark_bpm, "CAL_DARK_BPM")
combine_B_sof_data.append(master_dark, "CAL_DARK_MASTER")
combine_B_sof_data.write(combine_B_sof)

ff_name_science_B = join(outdir, "cr2res_util_calib_science_B_collapsed.fits")

# -----------------------------------------------------------------------------
# Write SOF for sky only
# -----------------------------------------------------------------------------
combine_sky_sof = join(outdir, "sky.sof")
combine_sky_sof_data = SofFile()
combine_sky_sof_data.append(ff_name_science_A, "UTIL_CALIB")
combine_sky_sof_data.append(ff_name_science_B, "UTIL_CALIB")
combine_sky_sof_data.append(master_dark_bpm, "CAL_DARK_BPM")
combine_sky_sof_data.append(tw_name, "CAL_FLAT_TW")
combine_sky_sof_data.write(combine_sky_sof)

sky_name = join(outdir, "cr2res_util_combine_sky.fits")

# -----------------------------------------------------------------------------
# Write SOF for sky BPM
# -----------------------------------------------------------------------------
bpm_create_sof = join(outdir, "bpm_create.sof")
bpm_create_sof_data = SofFile()
bpm_create_sof_data.append(master_dark, "UTIL_CALIB")
bpm_create_sof_data.append(tw_name, "CAL_FLAT_TW")
bpm_create_sof_data.write(bpm_create_sof)

bpm_create2_sof = join(outdir, "bpm_create2.sof")
bpm_create2_sof_data = SofFile()
bpm_create2_sof_data.append(ff_name_flat, "UTIL_CALIB")
bpm_create2_sof_data.append(tw_name, "CAL_FLAT_TW")
bpm_create2_sof_data.write(bpm_create2_sof)

bpm_create3_sof = join(outdir, "bpm_create3.sof")
bpm_create3_sof_data = SofFile()
bpm_create3_sof_data.append(sky_name, "UTIL_CALIB")
bpm_create3_sof_data.append(tw_name, "CAL_FLAT_TW")
bpm_create3_sof_data.write(bpm_create3_sof)

bpm_create_file = join(outdir, "cr2res_util_bpm_create.fits")

bpm_merge_sof = join(outdir, "bpm_merge.sof")
bpm_merge_sof_data = SofFile()
bpm_merge_sof_data.append(bpm_create_file, "UTIL_BPM_SPLIT")
bpm_merge_sof_data.append(master_dark_bpm, "CAL_DARK_BPM")
bpm_merge_sof_data.write(bpm_merge_sof)

bpm_merge_file = join(outdir, "cr2res_util_bpm_merge.fits")



# -----------------------------------------------------------------------------
# Write SOF for flats
# -----------------------------------------------------------------------------
flat_extr_sof = join(outdir, "flat_extr.sof")
flat_extr_sof_data = SofFile()
flat_extr_sof_data.append(ff_name_flat, "UTIL_CALIB")
flat_extr_sof_data.append(tw_name, "UTIL_WAVE_TW")
flat_extr_sof_data.append(master_dark, "CAL_DARK_MASTER")
flat_extr_sof_data.append(master_dark_bpm, "CAL_DARK_BPM")
flat_extr_sof_data.write(flat_extr_sof)

flat_master_blaze = join(outdir, "cr2res_util_calib_flat_collapsed_extr1D.fits")
flat_slit_model = join(outdir, "cr2res_util_calib_flat_collapsed_extrModel.fits")

# -----------------------------------------------------------------------------
# Write SOF for normflat
# -----------------------------------------------------------------------------
norm_flat_sof = join(outdir, "norm_flat.sof")
norm_flat_sof_data = SofFile()
norm_flat_sof_data.append(ff_name_flat, "UTIL_CALIB")
norm_flat_sof_data.append(flat_slit_model, "UTIL_SLIT_MODEL")
norm_flat_sof_data.append(master_dark_bpm, "CAL_DARK_BPM")
norm_flat_sof_data.write(norm_flat_sof)

ff_norm_flat = join(outdir, "cr2res_util_normflat_Open_master_flat.fits")
ff_norm_flat_bpm = join(outdir, "cr2res_util_normflat_Open_master_bpm.fits")


# New and improved SOF for the sky extraction
combine_A_flat_sof = join(outdir, "combine_A_flat.sof")
combine_A_sof_data.append(ff_norm_flat, "CAL_FLAT_MASTER")
combine_A_sof_data.write(combine_A_flat_sof)

combine_B_flat_sof = join(outdir, "combine_B_flat.sof")
combine_B_sof_data.append(ff_norm_flat, "CAL_FLAT_MASTER")
combine_B_sof_data.write(combine_B_flat_sof)

# -----------------------------------------------------------------------------
# Write SOF for sky extr
# -----------------------------------------------------------------------------
sky_extr_sof = join(outdir, "sky_extr.sof")
sky_extr_sof_data = SofFile()
sky_extr_sof_data.append(sky_name, "UTIL_CALIB")
sky_extr_sof_data.append(tw_name, "UTIL_SLIT_CURV_TW")
sky_extr_sof_data.append(master_dark_bpm, "CAL_DARK_BPM")
sky_extr_sof_data.write(sky_extr_sof)

sky_extr_name = join(outdir, "cr2res_util_combine_sky_extr1D.fits")
sky_extr_model_name = join(outdir, "cr2res_util_combine_sky_extrModel.fits")

bpm_create_extract_sof = join(outdir, "bpm_create_extract.sof")
bpm_create_sof_extract_data = SofFile()
bpm_create_sof_extract_data.append(sky_name, "UTIL_CALIB")
bpm_create_sof_extract_data.append(tw_name, "CAL_FLAT_TW")
bpm_create_sof_extract_data.append(sky_extr_model_name, "UTIL_SLIT_MODEL")
bpm_create_sof_extract_data.write(bpm_create_extract_sof)

bpm_create_extract_file = join(outdir, "cr2res_util_bpm_create.fits")

# -----------------------------------------------------------------------------
# Write SOF for slit curvature
# -----------------------------------------------------------------------------
slit_curv_sof = join(outdir, "slit_curv.sof")
slit_curv_sof_data = SofFile()
slit_curv_sof_data.append(tw_name, "CAL_FLAT_TW")
slit_curv_sof_data.append(sky_name, "UTIL_CALIB")
slit_curv_sof_data.append(sky_extr_name, "UTIL_CALIB_EXTRACT_1D")
slit_curv_sof_data.append(master_dark_bpm, "CAL_DARK_BPM")
slit_curv_sof_data.write(slit_curv_sof)

# -----------------------------------------------------------------------------
# Write SOF for extraction of the stellar spectrum
# -----------------------------------------------------------------------------
extract_star_sof = join(outdir, "extract_stellar.sof")
extract_star_sof_data = SofFile()
extract_star_sof_data.append(ff_name_science_A, "OBS_NODDING_OTHER")
extract_star_sof_data.append(ff_name_science_B, "OBS_NODDING_OTHER")
# for s in science_exps_A:
#     extract_star_sof_data.append(s, "OBS_NODDING_OTHER")
# for s in science_exps_B:
#     extract_star_sof_data.append(s, "OBS_NODDING_OTHER")
extract_star_sof_data.append(tw_name, "CAL_FLAT_TW")
extract_star_sof_data.append(ff_norm_flat, "UTIL_MASTER_FLAT")
extract_star_sof_data.append(master_dark_bpm, "CAL_DARK_BPM")
# extract_star_sof_data.append(master_dark, "CAL_DARK_MASTER")
extract_star_sof_data.write(extract_star_sof)

extract_star = join(outdir, "cr2res_obs_nodding_extracted_combined.fits")

# -----------------------------------------------------------------------------
# Write SOF for molecfit
# -----------------------------------------------------------------------------
molecfit_prepare_sky_sof = join(outdir, "molecfit_sky.sof")
molecfit_prepare_sof_data = SofFile()
molecfit_prepare_sof_data.append(sky_extr_name, "UTIL_CALIB_EXTRACT_1D")
molecfit_prepare_sof_data.append(flat_master_blaze, "CAL_FLAT_EXTRACT_1D")
molecfit_prepare_sof_data.write(molecfit_prepare_sky_sof)

molecfit_prepare_star_sof = join(outdir, "molecfit_star.sof")
molecfit_prepare_sof_data = SofFile()
molecfit_prepare_sof_data.append(extract_star, "UTIL_CALIB_EXTRACT_1D")
molecfit_prepare_sof_data.append(flat_master_blaze, "CAL_FLAT_EXTRACT_1D")
molecfit_prepare_sof_data.write(molecfit_prepare_star_sof)

molecfit_model_rc = join(outdir, "molecfit_model.rc")
molecfit_model_sof = join(outdir, "molecfit_model.sof")
molecfit_best_model = join(outdir, "BEST_FIT_MODEL.fits")
molecfit_mapping = join(outdir, "MAPPING.fits")

molecfit_best_model_sky = join(outdir, "BEST_FIT_MODEL_SKY.fits")
molecfit_mapping_sky = join(outdir, "MAPPING_SKY.fits")

molecfit_best_model_star = join(outdir, "BEST_FIT_MODEL_STAR.fits")
molecfit_mapping_star = join(outdir, "MAPPING_STAR.fits")

molecfit_apply_sof = join(outdir, "molecfit_apply.sof")
molecfit_apply_sof_data = SofFile()
molecfit_apply_sof_data.append(extract_star, "UTIL_CALIB_EXTRACT_1D")
molecfit_apply_sof_data.append(molecfit_best_model_sky, "CAL_WAVE_MOLECFIT_MODEL_SKY")
molecfit_apply_sof_data.append(molecfit_mapping_sky, "CAL_WAVE_MOLECFIT_MAPPING_SKY")
molecfit_apply_sof_data.append(molecfit_best_model_star, "CAL_WAVE_MOLECFIT_MODEL_STAR")
molecfit_apply_sof_data.append(molecfit_mapping_star, "CAL_WAVE_MOLECFIT_MAPPING_STAR")
molecfit_apply_sof_data.write(molecfit_apply_sof)

# -----------------------------------------------------------------------------
# Write shell script
# -----------------------------------------------------------------------------


# And finally write the file containing esorex reduction commands
python_cmd_dir = realpath(join(dirname(__file__), "..", "tools"))
plot_name = join(outdir, tw_name.replace(".fits", ".png"))

# Molecfit does not work with pyesorex
# So only use pyesorex for python recipes
esorex = "esorex"
pyesorex = "pyesorex"


commands = [
    "#!/bin/bash\n"
    # Create master dark
    f"{esorex} cr2res_cal_dark --bpm_method=LOCAL {dark_sof}\n"
    # Create an initial trace wave
    f"{esorex} cr2res_util_calib --collapse=MEDIAN {flat_sof}\n"
    f"mv {ff_name} {ff_name_flat}\n"
    f"{esorex} cr2res_util_trace {tw_sof}\n"
    # Filter additional bad pixels
    f"{pyesorex} cr2res_util_bpm_create --bpm_method=NOLIGHTROWS {bpm_create_sof}\n"
    f"{esorex} cr2res_util_bpm_merge {bpm_merge_sof}\n"
    f"mv {bpm_merge_file} {master_dark_bpm}\n"
    f"{pyesorex} cr2res_util_bpm_create {bpm_create_sof}\n"
    f"{esorex} cr2res_util_bpm_merge {bpm_merge_sof}\n"
    f"mv {bpm_merge_file} {master_dark_bpm}\n"
    f"{pyesorex} cr2res_util_bpm_create {bpm_create2_sof}\n"
    f"{esorex} cr2res_util_bpm_merge {bpm_merge_sof}\n"
    f"mv {bpm_merge_file} {master_dark_bpm}\n"
    f"{pyesorex} cr2res_util_bpm_create --bpm_method=ORDER --bpm_kappa=5 {bpm_create_sof}\n"
    f"{esorex} cr2res_util_bpm_merge {bpm_merge_sof}\n"
    f"mv {bpm_merge_file} {master_dark_bpm}\n"
    f"python {join(python_cmd_dir, 'cr2res_show_trace.py')} {tw_name} {ff_name_flat} --bpm={master_dark_bpm} --out={join(outdir, 'cr2res_util_trace.png')}\n"
    f"python {join(python_cmd_dir, 'cr2res_show_trace.py')} {tw_name} {master_dark} --bpm={master_dark_bpm} --out={join(outdir, 'cr2res_cal_dark.png')}\n"
    f"python {join(python_cmd_dir, 'cr2res_show_trace.py')} {tw_name} {master_dark_bpm} --vmin=0 --vmax=1 --out={join(outdir, 'cr2res_cal_dark_bpm.png')}\n"
    # Combine the observations into a sky only observation (no curvature)
    f"{esorex} cr2res_util_calib --collapse=MEDIAN --subtract_interorder_column=FALSE {combine_A_sof}\n"
    f"mv {ff_name} {ff_name_science_A}\n"
    f"python {join(python_cmd_dir, 'cr2res_util_fits_header_update.py')} {ff_name_science_A} 'HIERARCH ESO SEQ NODPOS' A\n"
    f"python {join(python_cmd_dir, 'cr2res_show_trace.py')} {tw_name} {ff_name_science_A} --bpm={master_dark_bpm} --out={join(outdir, 'cr2res_util_combine_A.png')}\n"
    f"{esorex} cr2res_util_calib --collapse=MEDIAN --subtract_interorder_column=FALSE {combine_B_sof}\n"
    f"mv {ff_name} {ff_name_science_B}\n"
    f"python {join(python_cmd_dir, 'cr2res_util_fits_header_update.py')} {ff_name_science_B} 'HIERARCH ESO SEQ NODPOS' B\n"
    f"python {join(python_cmd_dir, 'cr2res_show_trace.py')} {tw_name} {ff_name_science_B} --bpm={master_dark_bpm} --out={join(outdir, 'cr2res_util_combine_B.png')}\n"
    f"{pyesorex} cr2res_util_combine_sky {combine_sky_sof}\n"
    # Create Bad Pixel Map of the combined sky image
    f"{pyesorex} cr2res_util_bpm_create {bpm_create3_sof}\n"
    f"{esorex} cr2res_util_bpm_merge {bpm_merge_sof}\n"
    f"mv {bpm_merge_file} {master_dark_bpm}\n"

    f"python {join(python_cmd_dir, 'cr2res_show_trace.py')} {tw_name} {sky_name} --bpm={master_dark_bpm} --out={join(outdir, 'cr2res_util_combine.png')}\n"
    # Extract sky spectrum (no curvature)
    f"{esorex} cr2res_util_extract --smooth_spec=0.0001 --swath_width=2048 {sky_extr_sof}\n"
    f"python {join(python_cmd_dir, 'cr2res_show_spectrum.py')} {tw_name} {sky_extr_name} --out={join(outdir, 'cr2res_util_combine_sky_extr1D.png')}\n"
    # Extract the mask from the optimal extraction model
    # f"{pyesorex} cr2res_util_bpm_create_from_extract --bpm_kappa=6 {bpm_create_extract_sof}\n"
    # f"{esorex} cr2res_util_bpm_merge {bpm_merge_sof}\n"
    # f"mv {bpm_merge_file} {master_dark_bpm}\n"
    ]

# For the M band we remove the 5th order as it is dominated entirely by telluric absorption
# Thus removing any useful signal
if wl_setting == "M4318":
    commands += [f"python {join(python_cmd_dir, 'cr2res_util_trace_remove.py')} {tw_name} 5\n"]

commands += [
    # Determine curvature from sky emissions
    f"{pyesorex} cr2res_util_slit_curv_sky {slit_curv_sof}\n"
    f"python {join(python_cmd_dir, 'cr2res_show_trace_curv.py')} {tw_name} {sky_name} --bpm={master_dark_bpm} --out={join(outdir, 'cr2res_util_slit_curv.png')}\n"
    # Create master flat (with curvature)
    f"{esorex} cr2res_util_extract --smooth_spec=0.0001 {flat_extr_sof}\n"
    f"{esorex} cr2res_util_normflat {norm_flat_sof}\n"
    f"python {join(python_cmd_dir, 'cr2res_show_trace.py')} {tw_name} {ff_norm_flat} --bpm={master_dark_bpm} --out={join(outdir, 'cr2res_util_normflat.png')}\n"
    # Combine the sky images (with curvature and norm flat)
    f"{esorex} cr2res_util_calib --collapse=MEDIAN --subtract_interorder_column=FALSE {combine_A_flat_sof}\n"
    f"mv {ff_name} {ff_name_science_A}\n"
    f"python {join(python_cmd_dir, 'cr2res_util_fits_header_update.py')} {ff_name_science_A} 'HIERARCH ESO SEQ NODPOS' A\n"
    f"python {join(python_cmd_dir, 'cr2res_show_trace.py')} {tw_name} {ff_name_science_A} --bpm={master_dark_bpm} --out={join(outdir, 'cr2res_util_combine_A.png')}\n"
    f"{esorex} cr2res_util_calib --collapse=MEDIAN --subtract_interorder_column=FALSE {combine_B_flat_sof}\n"
    f"mv {ff_name} {ff_name_science_B}\n"
    f"python {join(python_cmd_dir, 'cr2res_util_fits_header_update.py')} {ff_name_science_B} 'HIERARCH ESO SEQ NODPOS' B\n"
    f"python {join(python_cmd_dir, 'cr2res_show_trace.py')} {tw_name} {ff_name_science_B} --bpm={master_dark_bpm} --out={join(outdir, 'cr2res_util_combine_B.png')}\n"
    f"{pyesorex} cr2res_util_combine_sky {combine_sky_sof}\n"
    f"python {join(python_cmd_dir, 'cr2res_show_trace.py')} {tw_name} {sky_name} --bpm={master_dark_bpm} --out={join(outdir, 'cr2res_util_combine.png')}\n"
    # Extract sky spectrum (with curvature)
    f"{esorex} cr2res_util_extract --smooth_spec=0.0001 --swath_width=2048 {sky_extr_sof}\n"
    f"python {join(python_cmd_dir, 'cr2res_show_spectrum.py')} {tw_name} {sky_extr_name} --out={join(outdir, 'cr2res_util_combine_sky_extr1D.png')}\n"
    # Extract stellar spectrum
    f"{esorex} cr2res_obs_nodding --subtract_interorder_column=FALSE {extract_star_sof}\n"
    f"python {join(python_cmd_dir, 'cr2res_show_spectrum.py')} {tw_name} {extract_star} --out={join(outdir, 'cr2res_obs_nodding_extr1D.png')}\n"
    # Molecfit on sky emission spectrum
    f"{pyesorex} cr2res_wave_molecfit_prepare --transmission=False {molecfit_prepare_sky_sof}\n"
    f"{esorex} --recipe-config={molecfit_model_rc} molecfit_model {molecfit_model_sof}\n"
    f"mv {molecfit_mapping} {molecfit_mapping_sky}\n"
    f"mv {molecfit_best_model} {molecfit_best_model_sky}\n"
    # Molecfit on stellar spectrum
    f"{pyesorex} cr2res_wave_molecfit_prepare --transmission=True {molecfit_prepare_star_sof}\n"
    f"{esorex} --recipe-config={molecfit_model_rc} molecfit_model {molecfit_model_sof}\n"
    f"mv {molecfit_mapping} {molecfit_mapping_star}\n"
    f"mv {molecfit_best_model} {molecfit_best_model_star}\n"
    f"python {join(python_cmd_dir, 'cr2res_show_molecfit.py')} -e={sky_extr_name} --mfsk={molecfit_best_model_sky} --mpsk={molecfit_mapping_sky} --mfst={molecfit_best_model_star} --mpst={molecfit_mapping_sky}\n"
    # Apply the Molecfit wavelength calibration
    # using the best of the sky and stellar molecfit fits
    f"{pyesorex} cr2res_wave_molecfit_apply {molecfit_apply_sof}\n"
]

shell_script = join(outdir, "reduce_cals.sh")
print(f"Creating data reduction script : {shell_script}")

with open(shell_script, "w") as ww:
    ww.writelines(commands)

# Make the script executable
cmd = "chmod +x {}".format(shell_script)
process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
output, error = process.communicate()

pass
