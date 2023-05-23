"""
Create a BPM using the median absolute deviation within each order
"""
import sys
from os.path import dirname
from typing import Any, Dict

import numpy as np
from astropy.io import fits
from scipy.ndimage import median_filter
from cpl.core import Msg
from cpl.ui import Frame, FrameSet, ParameterList, ParameterValue, PyRecipe, ParameterEnum

sys.path.append(dirname(__file__))
from cr2res_recipe import CR2RES_RECIPE


class cr2res_util_bpm_create(PyRecipe, CR2RES_RECIPE):
    _name = "cr2res_util_bpm_create"
    _version = "1.0"
    _author = "Ansgar Wehrhahn"
    _email = "ansgar.wehrhahn@astro.su.se"
    _copyright = "GPL-3.0-or-later"
    _synopsis = "Create a BPM using the median absolute deviation within each order"
    _description = "Create a BPM using the median absolute deviation within each order"

    detectors = "ALL"
    orders = "ALL"

    def __init__(self) -> None:
        super().__init__()
        self.parameters = ParameterList(
            [
                ParameterValue(
                    name="detector",
                    context="cr2res_util_bpm_create",
                    description="Detector to run",
                    default="ALL",
                ),
                ParameterValue(
                    name="bpm_kappa",
                    context="cr2res_util_bpm_create",
                    description="Sigma cutoff",
                    default=5,
                ),
                ParameterValue(
                    name="bpm_size",
                    context="cr2res_util_bpm_create",
                    description="Size of the median filter",
                    default=10,
                ),
                ParameterEnum(
                    name="bpm_method",
                    context="cr2res_util_bpm_create",
                    description="Algorithm to use for the creation of the BPM",
                    default="LOCAL",
                    alternatives=("LOCAL", "GLOBAL", "NOLIGHTROWS"),
                ),
                ParameterValue(
                    name="bpm_value",
                    context="cr2res_util_bpm_create",
                    description="Value to set the bad pixels to in the result",
                    default=1,
                )
            ]
        )

    def create_bpm_local(self, science):
        hdus = [science[0]]
        hdus[0].header["ESO PRO TYPE"] = "BPM"

        BPM_KAPPA = self.parameters["bpm_kappa"].value
        BPM_VALUE = self.parameters["bpm_value"].value
        SIZE = self.parameters["bpm_size"].value

        for chip in self.detectors:
            Msg.info(self.name, f"Process Detector {chip}")
            ext = self.get_chip_extension(chip)
            header = science[ext].header
            data = science[ext].data
            result = np.full(data.shape, 0, dtype=int)

            tmp = np.nan_to_num(data, copy=True)
            median = median_filter(tmp, size=SIZE)
            mad = 1.5 * median_filter(np.abs(tmp - median), size=SIZE)
            mask = tmp > (median + BPM_KAPPA * mad)
            mask |= tmp < (median - BPM_KAPPA * mad)
            Msg.info(
                self.name,
                f"Median: {np.median(median):.2}, MAD: {np.median(mad):.2}, #Rejected: {np.sum(mask)}",
            )
            result[mask] = BPM_VALUE

            secondary = fits.ImageHDU(result, header=header)
            hdus += [secondary]

        hdus = fits.HDUList(hdus)
        return hdus
    
    def create_bpm_global(self, science):
        hdus = [science[0]]
        hdus[0].header["ESO PRO TYPE"] = "BPM"

        BPM_KAPPA = self.parameters["bpm_kappa"].value
        BPM_VALUE = self.parameters["bpm_value"].value

        for chip in self.detectors:
            Msg.info(self.name, f"Process Detector {chip}")
            ext = self.get_chip_extension(chip)
            header = science[ext].header
            data = science[ext].data
            result = np.full(data.shape, 0, dtype=int)

            median = np.nanmedian(data)
            mad = 1.5 * np.nanmedian(np.abs(data - median))
            mask = data > (median + BPM_KAPPA * mad)
            mask |= data < (median - BPM_KAPPA * mad)
            Msg.info(
                self.name,
                f"Median: {np.median(median):.2}, MAD: {np.median(mad):.2}, #Rejected: {np.sum(mask)}",
            )
            result[mask] = BPM_VALUE

            secondary = fits.ImageHDU(result, header=header)
            hdus += [secondary]

        hdus = fits.HDUList(hdus)
        return hdus
    
    def create_bpm_order(self, science, tracewave):
        hdus = [science[0]]
        hdus[0].header["ESO PRO TYPE"] = "BPM"

        BPM_KAPPA = self.parameters["bpm_kappa"].value
        BPM_VALUE = self.parameters["bpm_value"].value

        for chip in self.detectors:
            Msg.info(self.name, f"Process Detector {chip}")
            ext = self.get_chip_extension(chip)
            header = science[ext].header
            data = science[ext].data
            tw_data = tracewave[ext].data
            result = np.full(data.shape, 0, dtype=int)

            orders = sorted(list(set(tw_data["Order"])))
            for order in orders:
                lower, middle, upper = self.get_order_trace(tw_data, order)
                idx = self.make_index(lower, upper)
                median = np.nanmedian(data[idx])
                mad = 1.5 * np.nanmedian(np.abs(data[idx] - median))
                mask = data[idx] > (median + BPM_KAPPA * mad)
                mask |= data[idx] < (median - BPM_KAPPA * mad)
                Msg.info(
                    self.name,
                    f"Median: {np.median(median):.2}, MAD: {np.median(mad):.2}, #Rejected: {np.sum(mask)}",
                )
                result[idx[0][mask], idx[1][mask]] = BPM_VALUE

            secondary = fits.ImageHDU(result, header=header)
            hdus += [secondary]

        hdus = fits.HDUList(hdus)
        return hdus

    def create_bpm_nolight(self, science):
        hdus = [science[0]]
        hdus[0].header["ESO PRO TYPE"] = "BPM"

        BPM_VALUE = self.parameters["bpm_value"].value

        for chip in self.detectors:
            Msg.info(self.name, f"Process Detector {chip}")
            ext = self.get_chip_extension(chip)
            header = science[ext].header
            data = science[ext].data
            result = np.full(data.shape, 0, dtype=int)

            # The no light rows are the bottom 50
            # TODO: make this a parameter?
            result[:50] = BPM_VALUE

            secondary = fits.ImageHDU(result, header=header)
            hdus += [secondary]

        hdus = fits.HDUList(hdus)
        return hdus

    def run(self, frameset: FrameSet, settings: Dict[str, Any]) -> FrameSet:
        # Check the input parameters
        self.parameters = self.parse_parameters(settings)
        self.method = self.parameters["bpm_method"].value
        if (
            isinstance(self.parameters["detector"].value, str)
            and self.parameters["detector"].value.upper() == "ALL"
        ):
            self.detectors = [1, 2, 3]
        else:
            self.detectors = int(self.parameters["detector"].value)

        # Parse the SOF
        frames = self.filter_frameset(
            frameset, science="UTIL_CALIB", tracewave="CAL_FLAT_TW"
        )

        # Load the data as FITS files
        science = frames["science"][0].as_hdulist()
        trace_wave = frames["tracewave"][0].as_hdulist()

        # Run the actual work
        if self.method == "LOCAL":
            hdus = self.create_bpm_local(science)
        elif self.method == "GLOBAL":
            hdus = self.create_bpm_global(science)
        elif self.method == "NOLIGHTROWS":
            hdus = self.create_bpm_nolight(science)
        elif self.method == "ORDER":
            hdus = self.create_bpm_order(science, trace_wave)
        else:
            raise ValueError(f"Unexpected BPM method {self.method}")
        
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
    module = cr2res_util_bpm_create()
    frameset = FrameSet(
        "/scratch/ptah/anwe5599/CRIRES/2022-11-29_L3262/extr/bpm_create2.sof"
    )
    settings = {"bpm_kappa": 5}
    module.run(frameset, settings)