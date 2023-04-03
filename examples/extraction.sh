#!/bin/bash
rawdirname="/scratch/ptah/anwe5599/CRIRES/2022-11-29/raw"
extrdirname="/scratch/ptah/anwe5599/CRIRES/2022-11-29/extr"
setting="L3262"

# Step 1: create calibration files
# a) create a sof 
python make_calibration_sof.py $setting -r=$rawdirname -o=$extrdirname
# b) run esorex
esorex cr2res_cal_dark --bpm_kappa=1000 $extrdirname/dark.sof
esorex cr2res_cal_flat --bpm_low=0.5 --bpm_high=2.0 $extrdirname/flat.sof


# Step 2: create a master flat
# a) create a sof
python make_sof.py flat --setting=$setting $dirname $extrdirname/flat.sof
# b) run esorex
esorex cr2res_util_calib flat.sof

# Step 3: create a tracewave
python make_sof.py tracewave --setting=$setting $dirname $extrdirname/tw.sof
esorex cr2res_util_trace tw.sof

# Step 4: determine curvature
python make_sof.py curvature --setting=$setting $dirname $extrdirname/curv.sof
python cr2res_util_curv