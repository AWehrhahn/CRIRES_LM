from astropy.io import fits
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from os.path import join

# Locate fits files
dirname = "/scratch/ptah/anwe5599/CRIRES/2022-11-29"
fnames = glob(join(dirname, "*.fits"))

setting = "L3262"


# Load header info
ob = []
st = []
for f in fnames:
    with fits.open(f) as hdu:
        ob += [hdu[0].header.get("OBJECT", "")]
        st += [hdu[0].header.get("ESO INS WLEN ID", "")]

ob = np.asarray(ob)
st = np.asarray(st)

print(ob)
