import sys
from os.path import dirname
"""
Combine two nodding positions to get a 'sky' observation
"""
from typing import Any, Dict

import numpy as np
from astropy.io import fits
from cpl.core import Msg
from cpl.ui import Frame, FrameSet, ParameterList, ParameterValue, PyRecipe
from tqdm import tqdm

sys.path.append(dirname(__file__))
from cr2res_recipe import CR2RES_RECIPE


class cr2res_util_combine_sky(PyRecipe, CR2RES_RECIPE):
    _name = "cr2res_util_combine_sky"
    _version = "1.0"
    _author = "Ansgar Wehrhahn"
    _email = "ansgar.wehrhahn@astro.su.se"
    _copyright = "GPL-3.0-or-later"
    _synopsis = "Combine two nodding positions to get a 'sky' observation"
    _description = "This recipe combines observations at A and B nodding positions so that the star is removed and only sky is left"


    def __init__(self) -> None:
        super().__init__()
        self.parameters = ParameterList(
            [
                ParameterValue(
                    name="detector",
                    context="cr2res_util_combine_sky",
                    description="Detector to run",
                    default="ALL",
                ),
                ParameterValue(
                    name="order",
                    context="cr2res_util_combine_sky",
                    description="Order to run",
                    default="ALL",
                ),
            ]
        )

    def combine_frames(self, trace_wave, science_A, science_B):
        hdus = [science_A[0]]

        with self.redirect_stdout_tqdm() as stdout:
            for chip in tqdm([1, 2, 3], desc="CHIP", file=stdout, dynamic_ncols=True):
                ext = self.get_chip_extension(chip)
                exterr = self.get_chip_extension(chip, error=True)
                tw_data = trace_wave[ext].data

                header = science_A[ext].header
                data_A = science_A[ext].data
                data_B = science_B[ext].data
                err_header = science_A[exterr].header
                err_A = science_A[exterr].data
                err_B = science_B[exterr].data

                result = np.copy(data_A)
                result_error = np.copy(err_A)

                for order in tqdm(set(tw_data["Order"]), desc="Order", file=stdout, dynamic_ncols=True):
                    lower, middle, upper = self.get_order_trace(tw_data, order)
                    idx = self.make_index(middle, upper)
                    result[idx] = data_A[idx]
                    result_error[idx] = err_A[idx]
                    idx = self.make_index(lower, middle)
                    result[idx] = data_B[idx]
                    result_error[idx] = err_B[idx]

                secondary = fits.ImageHDU(result, header=header)
                err_secondary = fits.ImageHDU(result_error, header=err_header)
                hdus += [secondary, err_secondary]

        hdus = fits.HDUList(hdus)
        return hdus

    def run(self, frameset: FrameSet, settings: Dict[str, Any]) -> FrameSet:
        # Check the input parameters
        self.parameters = self.parse_parameters(settings)

        # Parse the SOF
        frames = self.filter_frameset(frameset, science="UTIL_CALIB", tracewave="CAL_FLAT_TW", validate=False)
        

        # Validate the input
        if len(frames["science"]) != 2:
            raise ValueError("Expected exactly 2 frames to combine")
        if len(frames["tracewave"]) != 1:
            raise ValueError("Expected one trace wave frame")

        science = [f.as_hdulist() for f in frames["science"]]
        offset = [sci[0].header["ESO SEQ CUMOFFSETY"] for sci in science]  # in pixels

        if offset[0] * offset[1] >= 0:
            raise ValueError("Require two offsets that are of opposite directions")

        # A is positive offset
        # B is negative offset
        science_A = [sci for sci, off in zip(science, offset) if off > 0][0]
        science_B = [sci for sci, off in zip(science, offset) if off < 0][0]
        trace_wave = frames["tracewave"][0].as_hdulist()

        # Run the actual work
        hdus = self.combine_frames(trace_wave, science_A, science_B)

        # Save the results
        outfile = "cr2res_util_combine_sky.fits"
        hdus.writeto(outfile, overwrite=True)

        # Return data back to PyEsorex
        fs = FrameSet([Frame(outfile, tag="UTIL_CALIB"),])
        return fs

if __name__ == "__main__":
    module = cr2res_util_combine_sky()
    frameset = FrameSet("/scratch/ptah/anwe5599/CRIRES/2022-12-23_M4318/extr/sky.sof")
    settings = {}
    module.run(frameset, settings)