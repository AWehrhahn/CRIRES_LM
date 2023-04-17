from typing import Any, Dict

import numpy as np
from astropy.io import fits
from cpl.core import Msg
from cpl.ui import Frame, FrameSet, ParameterList, ParameterValue, PyRecipe
from tqdm import tqdm


def get_chip_extension(chip: int, error: bool = False) -> str:
    if error:
        return f"CHIP{chip}ERR.INT1"
    else:
        return f"CHIP{chip}.INT1"

def make_index(ymin, ymax, xmin, xmax, zero=0):
    """Create an index (numpy style) that will select part of an image with changing position but fixed height
    The user is responsible for making sure the height is constant, otherwise it will still work, but the subsection will not have the desired format
    Parameters
    ----------
    ymin : array[ncol](int)
        lower y border
    ymax : array[ncol](int)
        upper y border
    xmin : int
        leftmost column
    xmax : int
        rightmost colum
    zero : bool, optional
        if True count y array from 0 instead of xmin (default: False)
    Returns
    -------
    index : tuple(array[height, width], array[height, width])
        numpy index for the selection of a subsection of an image
    """

    # TODO
    # Define the indices for the pixels between two y arrays, e.g. pixels in an order
    # in x: the rows between ymin and ymax
    # in y: the column, but n times to match the x index
    ymin = np.asarray(ymin, dtype=int)
    ymax = np.asarray(ymax, dtype=int)
    xmin = int(xmin)
    xmax = int(xmax)

    if zero:
        zero = xmin

    index_x = np.array(
        [np.arange(ymin[col], ymax[col] + 1) for col in range(xmin - zero, xmax - zero)]
    )
    index_y = np.array(
        [
            np.full(ymax[col] - ymin[col] + 1, col)
            for col in range(xmin - zero, xmax - zero)
        ]
    )
    index = index_x.T, index_y.T + zero

    return index

class cr2res_util_combine_sky(PyRecipe):
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

        for chip in tqdm([1, 2, 3], desc="CHIP"):
            ext = get_chip_extension(chip)
            exterr = get_chip_extension(chip, error=True)
            tw_data = trace_wave[ext].data

            header = science_A[ext].header
            data_A = science_A[ext].data
            data_B = science_B[ext].data
            err_header = science_A[exterr].header
            err_A = science_A[exterr].data
            err_B = science_B[exterr].data

            result = np.copy(data_A)
            result_error = np.copy(err_A)

            for order in tqdm(set(tw_data["Order"]), desc="Order"):
                idx = tw_data["Order"] == order
                x = np.arange(1, 2049)
                upper = np.polyval(tw_data[idx]["Upper"][0][::-1], x)
                lower = np.polyval(tw_data[idx]["Lower"][0][::-1], x)
                middle = np.polyval(tw_data[idx]["All"][0][::-1], x)

                height_upp = int(np.ceil(np.max(upper - middle)))
                height_low = int(np.ceil(np.max(middle - lower)))

                middle_int = middle.astype(int)
                upper_int = middle_int + height_upp
                lower_int = middle_int - height_low

                idx = make_index(middle_int, upper_int, 0, 2048)
                result[idx] = data_A[idx]
                result_error[idx] = err_A[idx]
                idx = make_index(lower_int, middle_int, 0, 2048)
                result[idx] = data_B[idx]
                result_error[idx] = err_B[idx]

            secondary = fits.ImageHDU(result, header=header)
            err_secondary = fits.ImageHDU(result_error, header=err_header)
            hdus += [secondary, err_secondary]

        hdus = fits.HDUList(hdus)
        return hdus

    def run(self, frameset: FrameSet, settings: Dict[str, Any]) -> FrameSet:
        # Check the input parameters
        for key, value in settings.items():
            try:
                self.parameters[key].value = value
            except KeyError:
                Msg.warning(
                    self.name,
                    f"Settings includes {key}:{value} but {self} has no parameter named {key}.",
                )

        # Parse the SOF
        science_frames = FrameSet()
        trace_wave_frames = FrameSet()
        # Go through the list of input frames, check the tag and act accordingly
        for frame in frameset:
            if frame.tag == "UTIL_CALIB":
                frame.group = Frame.FrameGroup.RAW
                science_frames.append(frame)
                Msg.debug(self.name, f"Got UTIL_CALIB frame: {frame.file}.")
            elif frame.tag == "CAL_FLAT_TW":
                frame.group = Frame.FrameGroup.CALIB
                trace_wave_frames.append(frame)
                Msg.debug(self.name, f"Got TRACE_WAVE frame: {frame.file}.")
            else:
                Msg.warning(
                    self.name,
                    f"Got frame {frame.file!r} with unexpected tag {frame.tag!r}, ignoring.",
                )

        # Validate the input
        if len(science_frames) != 2:
            raise ValueError("Expected exactly 2 frames to combine")
        if len(trace_wave_frames) == 0:
            raise ValueError("Expected one trace wave frame")
        elif len(trace_wave_frames) >= 2:
            raise ValueError("Expected only one trace wave frame")

        science = [f.as_hdulist() for f in science_frames]
        offset = [sci[0].header["ESO SEQ CUMOFFSETY"] for sci in science]  # in pixels

        if offset[0] * offset[1] >= 0:
            raise ValueError("Require two offsets that are of opposite directions")

        # A is positive offset
        # B is negative offset
        science_A = [sci for sci, off in zip(science, offset) if off > 0][0]
        science_B = [sci for sci, off in zip(science, offset) if off < 0][0]
        trace_wave = trace_wave_frames[0].as_hdulist()

        # Run the actual work
        hdus = self.combine_frames(trace_wave, science_A, science_B)

        # Save the results
        outfile = "cr2res_util_combine_sky.fits"
        hdus.writeto(outfile, overwrite=True)

        # Return data back to PyEsorex
        fs = FrameSet([Frame(outfile, tag="UTIL_CALIB"),])
        return fs
