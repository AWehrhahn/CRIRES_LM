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
import matplotlib.pyplot as plt
from astropy.io import fits
from os.path import join, dirname

import argparse

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------

if len(sys.argv) > 1:
    parser = argparse.ArgumentParser(prog="make_sof")
    parser.add_argument("setting", help="Wavelength setting of CRIRES+")
    parser.add_argument("-r", "--raw", help="Directory with raw data")
    parser.add_argument("-o", "--output", help="Output directory with processed data")
    args = parser.parse_args()
    wl_setting = args.setting.upper()
    rawdir = args.raw
    outdir = args.output
else:
    wl_setting = "L3262"
    rawdir = "/scratch/ptah/anwe5599/CRIRES/2022-11-29/raw"
    outdir = "/scratch/ptah/anwe5599/CRIRES/2022-11-29/extr"

os.makedirs(outdir, exist_ok=True)

# Static calibration files
CAL_FOLDER = "/home/anwe5599/esotools/cr2re-calib-1.2.3/cal"
TRACE_WAVE = "{}/{}_tw.fits".format(CAL_FOLDER, wl_setting)
# DETLIN_COEFF = "/home/tom/pCOMM/cr2res_cal_detlin_coeffs.fits"

# Reduction parameters
BPM_KAPPA = 1000  # Default: -1, controls bad pixel threshold
BPM_LOW = 0.5  # Default: 0.8, controls *relative* bad pixel threshold
BPM_HIGH = 2.0  # Default: 1.2, controls *relative* bad pixel threshold

# Detector linearity is computed from a series of flats with a range of NDIT in
# three different grating settings to illuminate all pixels. This process takes
# around ~8 hours and the resulting calibration file has SNR~200. Note that for
# science files with SNR significantly above this, the linearisation process
# will actually *degrade* the data quality. Linearity is primarily a concern
# for science cases where one is interested in the relative depths of
# absorption features rather than their locations (i.e. an abundance analysis
# would suffer more than a simply cross correlation). The use of detlin is
# *not* recommended for calibration frames.
use_detlin = False

# Whether to provide a BPM to the flat field routine. Since we by default are
# not making use of the BPM computed from the darks, there is no point in
# providing the dark BPM to the flats.
provide_bpm_to_flat = True

# -----------------------------------------------------------------------------
# Read in files
# -----------------------------------------------------------------------------
# Get current working directory
# cwd = os.getcwd()

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
is_science_A = is_science & (cal_frames["nod"] == "A") & (cal_frames["exp"] == 30)
is_science_B = is_science & (cal_frames["nod"] == "B") & (cal_frames["exp"] == 30)


# We should have a complete set of calibration frames
if np.sum(is_flat) == 0:
    raise FileNotFoundError("Missing flat fields.")
elif np.sum(is_dark) == 0:
    raise FileNotFoundError("Missing dark frames.")
elif np.sum(is_science_A) == 0:
    raise FileNotFoundError("Missing nodding position A frames")
elif np.sum(is_science_B) == 0:
    raise FileNotFoundError("Missing nodding position B frames")

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
    fig, axes = plt.subplots(2, 3)
    for flat_i in range(2):
        with fits.open(fns[flat_i]) as flat:
            for chip_i in np.arange(1, 4):
                xx = axes[flat_i, chip_i - 1].imshow(flat[chip_i].data)
                fig.colorbar(xx, ax=axes[flat_i, chip_i - 1])

            axes[flat_i, 0].set_ylabel(f"Exp = {exps[flat_i]}")

    plt.savefig(join(outdir, "flat_comparison.pdf"))
    plt.close("all")

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
else:
    # Format the flat exposure time so it writes properly. Keep up to 5 decimal
    # places, or cast to integer if they're the same.
    # if float(flat_exp) == int(float(flat_exp)):
    #     flat_exp_str = "{:0.0f}".format(int(float(flat_exp)))
    # else:
    #     flat_exp_str = "{:0.5f}".format(float(flat_exp))

    flat_master_dark = join(outdir, "cr2res_cal_dark_master.fits")
    flat_master_dark_bpm = join(outdir, "cr2res_cal_dark_bpm.fits")
    flat_master_blaze = join(outdir, "cr2res_cal_flat_Open_blaze.fits")


# -----------------------------------------------------------------------------
# Initialise new shell script
# -----------------------------------------------------------------------------
# Before we start looping, archive the old reduce.sh script if it exists
shell_script = join(outdir, "reduce_cals.sh")

if os.path.isfile(shell_script):
    bashCommand = "mv {} {}.old".format(shell_script, shell_script)
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

# And make a new splice script
with open(shell_script, "w") as rs:
    rs.write("#!/bin/bash\n")

cmd = "chmod +x {}".format(shell_script)
process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
output, error = process.communicate()

# -----------------------------------------------------------------------------
# Write SOF file for darks
# -----------------------------------------------------------------------------
# Make SOF file for darks. The esorex recipe can do all our darks at once, so
# we may as well add them all.
dark_sof = join(outdir, "dark.sof")

with open(dark_sof, "w") as sof:
    for dark in cal_frames[is_dark]["fn"]:
        sof.writelines("{}\tDARK\n".format(join(rawdir, dark)))

# -----------------------------------------------------------------------------
# Write SOF for flats
# -----------------------------------------------------------------------------
flat_sof = join(outdir, "flat.sof")

with open(flat_sof, "w") as sof:
    for flat in cal_frames[is_flat]["fn"]:
        sof.writelines("{}\tFLAT\n".format(join(rawdir, flat)))

    # sof.writelines("{}\tUTIL_WAVE_TW\n".format(TRACE_WAVE))
    # if use_detlin:
    #     sof.writelines("{}\tCAL_DETLIN_COEFFS\n".format(DETLIN_COEFF))

    sof.writelines(f"{flat_master_dark}\tCAL_DARK_MASTER\n")

    if provide_bpm_to_flat:
        sof.writelines(f"{flat_master_dark_bpm}\tCAL_DARK_BPM\n")

flat_master = join(outdir, "cr2res_cal_flat_Open_master_flat.fits")
flat_bpm = join(outdir, "cr2res_cal_flat_Open_bpm.fits")
ff_name = join(outdir, "cr2res_util_calib_calibrated_collapsed.fits")
ff_name_flat = join(outdir, "cr2res_util_calib_flat_collapsed.fits")


# -----------------------------------------------------------------------------
# Write SOF for tracewave
# -----------------------------------------------------------------------------
tw_sof = join(outdir, "tracewave.sof")

with open(tw_sof, "w") as sof:
    sof.writelines(f"{ff_name_flat}\tUTIL_CALIB\n")
    pass

tw_name = join(outdir, "cr2res_util_calib_flat_collapsed_tw.fits")

# -----------------------------------------------------------------------------
# Write SOF for combining the science frames
# -----------------------------------------------------------------------------
combine_A_sof = join(outdir, "combine_A.sof")
with open(combine_A_sof, "w") as sof:
    for s in science_exps_A:
        sof.writelines(f"{s}\tOBS_NODDING_OTHER\n")

    sof.writelines(f"{flat_master}\tCAL_FLAT_MASTER\n")
    sof.writelines(f"{flat_bpm}\tCAL_FLAT_BPM\n")
    sof.writelines(f"{flat_master_dark}\tCAL_DARK_MASTER\n")
    # sof.writelines(f"{flat_master_dark_bpm}\tCAL_DARK_BPM\n")


ff_name_science_A = join(outdir, "cr2res_util_calib_science_A_collapsed.fits")

combine_B_sof = join(outdir, "combine_B.sof")
with open(combine_B_sof, "w") as sof:
    for s in science_exps_B:
        sof.writelines(f"{s}\tOBS_NODDING_OTHER\n")

    sof.writelines(f"{flat_master}\tCAL_FLAT_MASTER\n")
    sof.writelines(f"{flat_bpm}\tCAL_FLAT_BPM\n")
    sof.writelines(f"{flat_master_dark}\tCAL_DARK_MASTER\n")
    # sof.writelines(f"{flat_master_dark_bpm}\tCAL_DARK_BPM\n")

ff_name_science_B = join(outdir, "cr2res_util_calib_science_B_collapsed.fits")

# -----------------------------------------------------------------------------
# Write SOF for sky only
# -----------------------------------------------------------------------------
combine_sky_sof = join(outdir, "sky.sof")

with open(combine_sky_sof, "w") as sof:
    sof.writelines(f"{ff_name_science_A}\tUTIL_CALIB\n")
    sof.writelines(f"{ff_name_science_B}\tUTIL_CALIB\n")
    sof.writelines(f"{tw_name}\tCAL_FLAT_TW\n")

sky_name = join(outdir, "cr2res_util_combine_sky.fits")

# -----------------------------------------------------------------------------
# Write SOF for slit curvature
# -----------------------------------------------------------------------------
slit_curv_sof = join(outdir, "slit_curv.sof")

with open(slit_curv_sof, "w") as sof:
    sof.writelines(f"{tw_name}\tCAL_FLAT_TW\n")
    sof.writelines(f"{sky_name}\tUTIL_CALIB\n")
    sof.writelines(f"{flat_master_blaze}\tCAL_FLAT_EXTRACT_1D\n")

# -----------------------------------------------------------------------------
# Write SOF for flats
# -----------------------------------------------------------------------------
flat_extr_sof = join(outdir, "flat_extr.sof")

with open(flat_extr_sof, "w") as sof:
    for flat in cal_frames[is_flat]["fn"]:
        sof.writelines("{}\tFLAT\n".format(join(rawdir, flat)))

    sof.writelines(f"{tw_name}\tUTIL_WAVE_TW\n")
    sof.writelines(f"{flat_master_dark}\tCAL_DARK_MASTER\n")
    sof.writelines(f"{flat_bpm}\tCAL_FLAT_BPM\n")
    # if use_detlin:
    #     sof.writelines("{}\tCAL_DETLIN_COEFFS\n".format(DETLIN_COEFF))

# -----------------------------------------------------------------------------
# Write SOF for sky extr
# -----------------------------------------------------------------------------
sky_extr_sof = join(outdir, "sky_extr.sof")

with open(sky_extr_sof, "w") as sof:
    sof.writelines(f"{sky_name}\tUTIL_CALIB\n")
    sof.writelines(f"{tw_name}\tUTIL_SLIT_CURV_TW\n")
    pass

sky_extr_name = join(outdir, "cr2res_util_combine_sky_extr1D.fits")

# -----------------------------------------------------------------------------
# Write shell script
# -----------------------------------------------------------------------------
# And finally write the file containing esorex reduction commands
python_cmd_dir = dirname(__file__)
plot_name = join(outdir, tw_name.replace(".fits", ".png"))

commands = []
commands += [f"esorex cr2res_cal_dark --bpm_method=GLOBAL {dark_sof}\n"]
commands += [f"esorex cr2res_util_calib --collapse=MEAN {flat_sof}\n"]
commands += [f"mv {ff_name} {ff_name_flat}\n"]
commands += [f"esorex cr2res_util_trace {tw_sof}\n"]
commands += [f"cr2res_show_trace.py {tw_name} {ff_name_flat}\n"]
commands += [f"mv {plot_name} {join(outdir, 'cr2res_util_trace.png')}"]
commands += [f"esorex cr2res_cal_flat {flat_extr_sof}\n"]
commands += [f"esorex cr2res_util_calib --collapse=MEDIAN {combine_A_sof}\n"]
commands += [f"mv {ff_name} {ff_name_science_A}\n"]
commands += [f"esorex cr2res_util_calib --collapse=MEDIAN {combine_B_sof}\n"]
commands += [f"mv {ff_name} {ff_name_science_B}\n"]
commands += [f"python {python_cmd_dir}/cr2res_util_combine_sky.py {combine_sky_sof}\n"]
commands += [f"cr2res_show_trace.py {tw_name} {sky_name}\n"]
commands += [f"mv {plot_name} {join(outdir, 'cr2res_util_combine.png')}\n"]
commands += [f"python {python_cmd_dir}/cr2res_util_slit_curv.py {slit_curv_sof}\n"]
commands += [f"cr2res_show_trace_curv.py {tw_name} {sky_name}\n"]
commands += [f"mv {plot_name} {join(outdir, 'cr2res_util_slit_curv.png')}\n"]
commands += [f"esorex cr2res_util_extract --smooth_spec=0.0001 {sky_extr_sof}\n"]
commands += [f"python {join(python_cmd_dir, 'cr2res_show_spectrum.py')} {tw_name} {sky_extr_name} -o=cr2res_util_combine_sky_extr1D.png\n"]

with open(shell_script, "a") as ww:
    ww.writelines(commands)

pass
