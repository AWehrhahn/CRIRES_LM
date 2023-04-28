from astropy.io import fits
import numpy as np

fnames = ["/scratch/ptah/anwe5599/CRIRES/2022-11-29_L3262/raw/CRIRE.2022-11-29T09:03:25.725.fits",
"/scratch/ptah/anwe5599/CRIRES/2022-11-29_L3262/raw/CRIRE.2022-11-29T09:03:43.182.fits",
"/scratch/ptah/anwe5599/CRIRES/2022-11-29_L3262/raw/CRIRE.2022-11-29T09:04:00.663.fits"]

hdus = [fits.open(f) for f in fnames]
bpms = []
for chip in [1, 2, 3]:
    data = [h[f"CHIP{chip}.INT1"].data for h in hdus]
    img = np.median(data, axis=0)

    median = np.median(img)
    mad = np.median(np.abs(img - median))
    mean = np.mean(np.abs(img - median))

    lower = median - 0.5 * mean
    upper = median + 0.5 * mean

    bpm = (img > upper) | (img < lower)
    bpm = fits.ImageHDU(bpm.astype(int), header=hdus[0][f"CHIP{chip}.INT1"].header)
    bpms.append(bpm)

primary = hdus[0][0]
bpms = fits.HDUList([primary, *bpms])
bpms.writeto("cr2res_cal_dark_bpm.fits")


