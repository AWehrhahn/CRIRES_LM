import pandas as pd
from os.path import dirname, join, exists, splitext, basename, getsize
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
from astropy.io import fits


def read_hitran2012_parfile(filename):
    '''
    Given a HITRAN2012-format text file, read in the parameters of the molecular absorption features.
    Parameters
    ----------
    filename : str
        The filename to read in.
    Return
    ------
    data : dict
        The dictionary of HITRAN data for the molecule.
    '''

    if not exists:
        raise ImportError('The input filename"' + filename + '" does not exist.')

    filehandle = open(filename, 'r')
    nlines = int(getsize(filename) / 162)

    data = {'M':np.empty(nlines, np.uint),               ## molecule identification number
            'I':np.empty(nlines, np.uint),               ## isotope number
            'linecenter':np.empty(nlines, np.float64),      ## line center wavenumber (in cm^{-1})
            'S':np.empty(nlines, np.float64),               ## line strength, in cm^{-1} / (molecule m^{-2})
            'Acoeff':np.empty(nlines, np.float64),          ## Einstein A coefficient (in s^{-1})
            'gamma-air':np.empty(nlines, np.float64),       ## line HWHM for air-broadening
            'gamma-self':np.empty(nlines, np.float64),      ## line HWHM for self-emission-broadening
            'Epp':np.empty(nlines, np.float64),             ## energy of lower transition level (in cm^{-1})
            'N':np.empty(nlines, np.float64),               ## temperature-dependent exponent for "gamma-air"
            'delta':np.empty(nlines, np.float64),           ## air-pressure shift, in cm^{-1} / atm
            'Vp':np.empty(nlines, "U15"),              ## upper-state "global" quanta index
            'Vpp':np.empty(nlines, "U15"),             ## lower-state "global" quanta index
            'Qp':np.empty(nlines, "U15"),              ## upper-state "local" quanta index
            'Qpp':np.empty(nlines, "U15"),             ## lower-state "local" quanta index
            'Ierr':np.empty(nlines, "U6"),            ## uncertainty indices
            'Iref':np.empty(nlines, "U12"),            ## reference indices
            'flag':np.empty(nlines, "U1"),            ## flag
            'gp':np.empty(nlines, np.float64),              ## statistical weight of the upper state
            'gpp':np.empty(nlines, np.float64)}             ## statistical weight of the lower state

    print('Reading "' + filename + '" ...')

    for i, line in tqdm(enumerate(filehandle), total=nlines):
        # if (len(line) < 160):
        #     raise ImportError('The imported file ("' + filename + '") does not appear to be a HITRAN2012-format data file.')

        data['M'][i] = np.uint(line[0:2])
        data['I'][i] = np.uint(line[2])
        data['linecenter'][i] = np.float64(line[3:15])
        data['S'][i] = np.float64(line[15:25])
        data['Acoeff'][i] = np.float64(line[25:35])
        data['gamma-air'][i] = np.float64(line[35:40])
        data['gamma-self'][i] = np.float64(line[40:45])
        data['Epp'][i] = np.float64(line[45:55])
        data['N'][i] = np.float64(line[55:59])
        data['delta'][i] = np.float64(line[59:67])
        data['Vp'][i] = line[67:82]
        data['Vpp'][i] = line[82:97]
        data['Qp'][i] = line[97:112]
        data['Qpp'][i] = line[112:127]
        data['Ierr'][i] = line[127:133]
        data['Iref'][i] = line[133:145]
        data['flag'][i] = line[145]
        data['gp'][i] = line[146:153]
        data['gpp'][i] = line[153:160]

    filehandle.close()
    return data



cwd = dirname(__file__)
fname = join(cwd, "hitran.par")
df = read_hitran2012_parfile(fname)
df = pd.DataFrame.from_dict(df)
df["wavelength"] = 1 / df["linecenter"] * 10000000

# load molecule ids
fname = join(cwd, "molecules.dat")
mol = pd.read_table(fname, header=None, names=["ID", "Formula", "Name"])


# Show Molecfit Results
molecfit_fname = "/scratch/ptah/anwe5599/CRIRES/2022-11-29/extr/BEST_FIT_MODEL.fits"
extracted_fname = "/scratch/ptah/anwe5599/CRIRES/2022-11-29/extr/cr2res_util_combine_sky_extr1D.fits"
mapping_fname = "/scratch/ptah/anwe5599/CRIRES/2022-11-29/extr/MAPPING.fits"

mf_hdu = fits.open(molecfit_fname)
ex_hdu = fits.open(extracted_fname)
map_hdu = fits.open(mapping_fname)
mapping = map_hdu[1].data

chips = np.unique(mapping["CHIP"])

for chip in chips:
    ext = f"CHIP{chip}.INT1"
    # ex_orders = np.sort([int(n[:2]) for n in ex_hdu[ext].data.names if n[-4:] == "SPEC"])
    # orders = np.unique(mapping[mapping["CHIP"] == chip]["ORDER"])
    orders = [6]

    for order in orders:
        mf_idx = (mapping["CHIP"] == chip) & (mapping["ORDER"] == order)
        mf_idx = mf_hdu[1].data["chip"] == mapping["MOLECFIT"][mf_idx][0]
        mf_data = mf_hdu[1].data[mf_idx]
        mf_spec = mf_data["mflux"]
        mf_wave = mf_data["mlambda"] * 1000

        ex_wave = mf_data["lambda"] * 1000
        ex_spec = mf_data["flux"]

        mask = mf_wave != 0
        if order == 2:
            mask[mf_wave < 3000] = False

        select = df["wavelength"] > mf_wave[mask].min()
        select &= df["wavelength"] < mf_wave[mask].max()
        sort = np.argsort(df[select]["S"])[::-1]
        # strongest = df[select][df[select]["M"] == 13]["wavelength"]
        strongest = df[select]["wavelength"].values[sort.values][:100]
        strongestID = df[select]["M"].values[sort.values]

        molid = [mol[id == mol["ID"]]["Formula"].values[0].strip() for id in tqdm(strongestID)]

        plt.clf()
        plt.plot(mf_wave[mask], ex_spec[mask], label="Extracted")
        plt.plot(mf_wave[mask], mf_spec[mask], "--", label="Model")
        plt.vlines(strongest, ex_spec[mask].min(), ex_spec[mask].max(), color="k", alpha=0.1)
        plt.legend()
        plt.savefig("test.png", dpi=600)

        pass


print(df.head())
