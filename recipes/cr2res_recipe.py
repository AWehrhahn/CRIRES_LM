import contextlib
import sys
from os.path import exists
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from cpl.core import Msg
from cpl.ui import FrameSet, ParameterValue
from tqdm.contrib import DummyTqdmFile

INDEX1D = List[int]
INDEX2D = List[List[int]]

# This can't be a subclass of PyRecipe or it gets recognized as a recipe itself
class CR2RES_RECIPE:
    name = "CR2RES_RECIPE"

    @staticmethod
    def get_chip_extension(chip: int, error: bool = False) -> str:
        """ Get the fits extension name for this chip """
        if error:
            return f"CHIP{chip}ERR.INT1"
        else:
            return f"CHIP{chip}.INT1"

    @staticmethod
    def get_table_column(order: int, trace: int, column: str) -> str:
        """ Get the table column name in the fits extracted spectrum bin table """
        return f"{order:02}_{trace:02}_{column}"

    @staticmethod
    def get_orders_trace_wave(trace_wave: Dict[str, Any]) -> List[int]:
        """ Get the orders contained in a trace wave """
        return sorted(list(set(trace_wave["Order"])))
    
    @staticmethod
    def get_orders_spectrum(spectrum: Dict[str, Any]) -> List[int]:
        """ Get the order numbers contained in a spectrum """
        return sorted([int(c[:2]) for c in spectrum.names if c[-4:] == "SPEC"])

    @staticmethod
    def get_order_trace(
        trace_wave: Dict[str, Any], order: int, clip:bool=True
    ) -> Tuple[INDEX1D, INDEX1D, INDEX1D]:
        """ Load the order trace indices from a trace wave table """
        idx = trace_wave["Order"] == order
        x = np.arange(1, 2049)
        upper = np.polyval(trace_wave[idx]["Upper"][0][::-1], x)
        lower = np.polyval(trace_wave[idx]["Lower"][0][::-1], x)
        middle = np.polyval(trace_wave[idx]["All"][0][::-1], x)

        if clip:
            height_upp = int(np.ceil(np.min(upper - middle)))
            height_low = int(np.ceil(np.min(middle - lower)))

            middle_int = middle.astype(int)
            upper_int = middle_int + height_upp
            lower_int = middle_int - height_low
            return lower_int, middle_int, upper_int
        else:
            return lower, middle, upper

    @staticmethod
    def make_index(
        ymin: INDEX1D, ymax: INDEX1D, xmin: int = 0, xmax: int = 2048, zero: int = 0
    ) -> Tuple[INDEX2D, INDEX2D]:
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
            [
                np.arange(ymin[col], ymax[col] + 1)
                for col in range(xmin - zero, xmax - zero)
            ]
        )
        index_y = np.array(
            [
                np.full(ymax[col] - ymin[col] + 1, col)
                for col in range(xmin - zero, xmax - zero)
            ]
        )
        index = index_x.T, index_y.T + zero

        return index

    @staticmethod
    @contextlib.contextmanager
    def redirect_stdout_tqdm():
        """ Redirect the standard output when using tqdm progress bars"""
        orig_out_err = sys.stdout, sys.stderr
        try:
            sys.stdout, sys.stderr = map(DummyTqdmFile, orig_out_err)
            yield orig_out_err[0]
        # Relay exceptions
        except Exception as exc:
            raise exc
        # Always restore sys.stdout/err if necessary
        finally:
            sys.stdout, sys.stderr = orig_out_err

    def parse_parameters(self, settings: Dict[str, Any]) -> Dict[str, ParameterValue]:
        for key, value in settings.items():
            try:
                self.parameters[key].value = value
            except KeyError:
                Msg.warning(
                    self.name,
                    f"Settings includes {key}:{value} but {self.name} has no parameter named {key}.",
                )
        return self.parameters

    def filter_frameset(
        self, frameset: FrameSet, validate=True, **tags: Dict[str, Union[str, List[str]]]
    ) -> Dict[str, FrameSet]:
        """
        Sort the frameset into groups based on the tags
        """
        result = {}

        for frame in frameset:
            found = False
            for key, value in tags.items():
                # If value is just a string convert it to a list
                if type(value) is str:
                    value = [value,]
                if frame.tag in value:
                    # Make a new frameset if its the first time we encounter it
                    if key not in result.keys():
                        result[key] = FrameSet()
                    result[key].append(frame)
                    Msg.debug(self.name, f"Got {frame.tag!r} frame: {frame.file!r}.")
                    found = True
            if not found:
                Msg.warning(
                    self.name,
                    f"Got frame {frame.file!r} with unexpected tag {frame.tag!r}, ignoring.",
                )
        if validate:
            for key, value in tags.items():
                if key not in result.keys() or len(result[key]) == 0:
                    raise FileNotFoundError(f"No {key} frames found in SOF. Expected tag {value}")
                elif len(result[key]) > 1:
                    raise ValueError(f"Got more than 1 {key} frame, but expected only one.")
                elif not exists(result[key][0].file):
                    raise FileNotFoundError(f"Specified {key} frame with filename {result[key][0]} not found.")
        return result