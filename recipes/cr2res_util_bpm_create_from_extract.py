"""
Create a BPM using an extracted model
"""
import sys
from os.path import dirname
from typing import Any, Dict

import numpy as np
from astropy.io import fits
from cpl.core import Msg
from cpl.ui import Frame, FrameSet, ParameterList, ParameterValue, PyRecipe
from tqdm import tqdm

sys.path.append(dirname(__file__))
from cr2res_recipe import CR2RES_RECIPE


class cr2res_util_bpm_create_from_extract(PyRecipe, CR2RES_RECIPE):
    _name = "cr2res_util_bpm_create_from_extract"
    _version = "1.0"
    _author = "Ansgar Wehrhahn"
    _email = "ansgar.wehrhahn@astro.su.se"
    _copyright = "GPL-3.0-or-later"
    _synopsis = "Create a BPM using an extracted model"
    _description = "Create a BPM using an extracted model"

    def __init__(self) -> None:
        super().__init__()
        self.parameters = ParameterList(
            [
                ParameterValue(
                    name="detector",
                    context="cr2res_util_bpm_create_from_extract",
                    description="Detector to run",
                    default="ALL",
                ),
                ParameterValue(
                    name="order",
                    context="cr2res_util_bpm_create_from_extract",
                    description="Order to run",
                    default="ALL",
                ),
                ParameterValue(
                    name="bpm_kappa",
                    context="cr2res_util_bpm_create_from_extract",
                    description="Sigma cutoff",
                    default=5,
                ),
            ]
        )

    def create_bpm(self, trace_wave, science, model):
        hdus = [science[0]]
        hdus[0].header["ESO PRO TYPE"] = "BPM"
        with self.redirect_stdout_tqdm() as stdout:
            for chip in tqdm([1, 2, 3], desc="CHIP", file=stdout, dynamic_ncols=True):
                ext = self.get_chip_extension(chip)
                tw_data = trace_wave[ext].data
                header = science[ext].header
                data = science[ext].data
                model_data = model[ext].data
                result = np.full(data.shape, 0, dtype=int)

                for order in tqdm(set(tw_data["Order"]), desc="Order", file=stdout, dynamic_ncols=True):
                    lower, middle, upper = self.get_order_trace(tw_data, order)

                    BPM_KAPPA = self.parameters["bpm_kappa"].value
                    idx = self.make_index(lower, upper)
                    relative = data[idx] - model_data[idx]
                    median = np.nanmedian(relative)
                    mad = 1.5 * np.nanmedian(np.abs(median - relative))
                    mask = np.abs(relative - median) > BPM_KAPPA * mad
                    Msg.info(
                        self._name,
                        f"Median: {median}, MAD: {mad}, #Rejected: {np.sum(mask)}\n",
                    )
                    # The indices are chosen so to avoid writing to a copy of the data
                    result[(idx[0][mask], idx[1][mask])] = 1

                secondary = fits.ImageHDU(result, header=header)
                hdus += [secondary]

        hdus = fits.HDUList(hdus)
        return hdus

    def run(self, frameset: FrameSet, settings: Dict[str, Any]) -> FrameSet:
        # Check the input parameters
        self.parameters = self.parse_parameters(settings)

        # Parse and validate the SOF
        frames = self.filter_frameset(frameset, science="UTIL_CALIB", model="UTIL_SLIT_MODEL", tracewave="CAL_FLAT_TW")

        # Load the data as FITS files
        science_A = frames["science"][0].as_hdulist()
        trace_wave = frames["tracewave"][0].as_hdulist()
        model = frames["model"][0].as_hdulist()

        # Run the actual work
        hdus = self.create_bpm(trace_wave, science_A, model)

        # Save the results
        outfile = "cr2res_util_bpm_create.fits"
        hdus.writeto(outfile, overwrite=True)

        # Return data back to PyEsorex
        fs = FrameSet(
            [
                Frame(outfile, tag="UTIL_BPM"),
            ]
        )
        return fs


if __name__ == "__main__":
    module = cr2res_util_bpm_create_from_extract()
    frameset = FrameSet(
        "/scratch/ptah/anwe5599/CRIRES/2022-11-29_L3262/extr/bpm_create_extract.sof"
    )
    settings = {"bpm_kappa": 5}
    module.run(frameset, settings)