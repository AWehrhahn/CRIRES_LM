import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
from tqdm import tqdm
import argparse
import sys


if len(sys.argv) > 1:
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--extracted")
    parser.add_argument("--mfsk", "--molecfit-sky")
    parser.add_argument("--mfst", "--molecfit-star")
    parser.add_argument("--mpsk", "--mapping-sky")
    parser.add_argument("--mpst", "--mapping-star")
    args = parser.parse_args()
    print(args)
    extracted_fname = args.extracted
    molecfit_fname_sky = args.mfsk
    molecfit_fname_star = args.mfst
    mapping_fname_sky = args.mpsk
    mapping_fname_star = args.mpst
else:
    molecfit_fname_sky = "/scratch/ptah/anwe5599/CRIRES/2022-11-29/extr/BEST_FIT_MODEL_SKY.fits"
    molecfit_fname_star = "/scratch/ptah/anwe5599/CRIRES/2022-11-29/extr/BEST_FIT_MODEL_STAR.fits"
    extracted_fname = "/scratch/ptah/anwe5599/CRIRES/2022-11-29/extr/cr2res_util_combine_sky_extr1D.fits"
    mapping_fname_sky = "/scratch/ptah/anwe5599/CRIRES/2022-11-29/extr/MAPPING_SKY.fits"
    mapping_fname_star = "/scratch/ptah/anwe5599/CRIRES/2022-11-29/extr/MAPPING_STAR.fits"

mf_hdu_star = fits.open(molecfit_fname_star)
mf_hdu_sky = fits.open(molecfit_fname_sky)
ex_hdu = fits.open(extracted_fname)
map_hdu_star = fits.open(mapping_fname_star)
mapping_star = map_hdu_star[1].data
map_hdu_sky = fits.open(mapping_fname_sky)
mapping_sky = map_hdu_sky[1].data


chips = np.unique(mapping_sky["CHIP"])

for chip in tqdm(chips):
    ext = f"CHIP{chip}.INT1"
    # ex_orders = np.sort([int(n[:2]) for n in ex_hdu[ext].data.names if n[-4:] == "SPEC"])
    orders = np.unique(mapping_sky[mapping_sky["CHIP"] == chip]["ORDER"])

    for order in tqdm(orders):
        mf_idx_star = (mapping_star["CHIP"] == chip) & (mapping_star["ORDER"] == order)
        mf_idx_star = mf_hdu_star[1].data["chip"] == mapping_star["MOLECFIT"][mf_idx_star][0]
        mf_data_star = mf_hdu_star[1].data[mf_idx_star]
        mf_spec_star = mf_data_star["mflux"]
        mf_wave_star = mf_data_star["mlambda"] * 1000

        mf_idx_sky = (mapping_sky["CHIP"] == chip) & (mapping_sky["ORDER"] == order)
        mf_idx_sky = mf_hdu_sky[1].data["chip"] == mapping_sky["MOLECFIT"][mf_idx_sky][0]
        mf_data_sky = mf_hdu_sky[1].data[mf_idx_sky]
        mf_spec_sky = mf_data_sky["mflux"]
        mf_wave_sky = mf_data_sky["mlambda"] * 1000

        ex_wave_star = mf_data_star["lambda"] * 1000
        ex_spec_star = mf_data_star["flux"]

        ex_wave_sky = mf_data_sky["lambda"] * 1000
        ex_spec_sky = mf_data_sky["flux"]

        mask_sky = mf_wave_sky != 0
        mask_star = mf_wave_star != 0

        plt.clf()
        plt.subplot(211)
        plt.title("STAR")
        plt.plot(mf_wave_star[mask_star], ex_spec_star[mask_star], label="Extracted")
        plt.plot(mf_wave_star[mask_star], mf_spec_star[mask_star], "--", label="Model")
        plt.subplot(212)
        plt.title("SKY")
        plt.plot(mf_wave_sky[mask_sky], ex_spec_sky[mask_sky], label="Extracted")
        plt.plot(mf_wave_sky[mask_sky], mf_spec_sky[mask_sky], "--", label="Model")
        plt.legend()
        plt.suptitle(f"CHIP: {chip} ORDER: {order:02}")
        plt.savefig(f"cr2res_wave_molecfit_c{chip}_o{order:02}.png", dpi=600)

        pass

