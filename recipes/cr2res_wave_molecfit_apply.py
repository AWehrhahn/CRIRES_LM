from typing import Any, Dict
from os.path import basename

import numpy as np
from cpl.core import Msg
from cpl.ui import Frame, FrameSet, ParameterList, ParameterValue, PyRecipe


def get_chip_extension(chip: int, error: bool = False) -> str:
    if error:
        return f"CHIP{chip}ERR.INT1"
    else:
        return f"CHIP{chip}.INT1"


def get_spectrum_table_header(order: int, trace: int, column: str) -> str:
    return f"{order:02}_{trace:02}_{column}"


class cr2res_wave_molecfit_apply(PyRecipe):
    _name = "cr2res_wave_molecfit_apply"
    _version = "1.0"
    _author = "Ansgar Wehrhahn"
    _email = "ansgar.wehrhahn@astro.su.se"
    _copyright = "GPL-3.0-or-later"
    _synopsis = "Prepare data for molecfit"
    _description = "This recipe formats the CRIRES+ data into a format that is suitable for Molecfit"

    def __init__(self) -> None:
        super().__init__()
        self.parameters = ParameterList(
            [
                ParameterValue(
                    name="detector",
                    context="cr2res_wave_molecfit_apply",
                    description="Detector to run",
                    default="ALL",
                ),
                ParameterValue(
                    name="order",
                    context="cr2res_wave_molecfit_apply",
                    description="Order to run",
                    default="ALL",
                ),
            ]
        )

    def parse_parameters(self, settings):
        for key, value in settings.items():
            try:
                self.parameters[key].value = value
            except KeyError:
                Msg.warning(
                    self.name,
                    f"Settings includes {key}:{value} but {self} has no parameter named {key}.",
                )

    def filter_frameset(self, frameset: FrameSet, **tags) -> Dict[str, FrameSet]:
        result = {}

        for frame in frameset:
            found = False
            for key, value in tags.items():
                if frame.tag == value:
                    if key not in result.keys():
                        result[key] = FrameSet()
                    result[key].append(frame)
                    found = True
                    Msg.debug(self.name, f"Got {value} frame: {frame.file}.")
                    break
            if not found:
                Msg.warning(
                    self.name,
                    f"Got frame {frame.file!r} with unexpected tag {frame.tag!r}, ignoring.",
                )

        return result

    def run(self, frameset: FrameSet, settings: Dict[str, Any]) -> FrameSet:

        # Check the input parameters
        self.parse_parameters(settings)

        # sort the frames by type
        frames = self.filter_frameset(
            frameset,
            spectrum="UTIL_CALIB_EXTRACT_1D",
            model="CAL_WAVE_MOLECFIT_MODEL",
            mapping="CAL_WAVE_MOLECFIT_MAPPING",
        )

        if len(frames["spectrum"]) != 1:
            raise ValueError("Expected exactly 1 spectrum frame")
        if len(frames["model"]) != 1:
            raise ValueError("Expected exactly 1 model frame")
        if len(frames["mapping"]) != 1:
            raise ValueError("Expected exactly 1 mapping frame")

        spectrum_fname = frames["spectrum"][0].file
        spectrum = frames["spectrum"][0].as_hdulist()
        model = frames["model"][0].as_hdulist()
        mapping = frames["mapping"][0].as_hdulist()

        header = spectrum[0].header
        mapping = mapping[1].data
        model = model[1].data

        chips = np.unique(mapping["CHIP"])

        for chip in chips:
            ext = get_chip_extension(chip)
            data = spectrum[ext].data
            orders = np.unique(mapping[mapping["CHIP"] == chip]["ORDER"])
            for order in orders:
                idx = mapping[(mapping["CHIP"] == chip) & (mapping["ORDER"] == order)][
                    "MOLECFIT"
                ][0]
                column = get_spectrum_table_header(order, 1, "WL")
                data[column] = model[model["CHIP"] == idx]["MLAMBDA"] * 1000

        # Overwrite the results
        spectrum.writeto(basename(spectrum_fname), overwrite=True)

        result = FrameSet(
            [
                Frame(spectrum_fname, tag="UTIL_CALIB_EXTRACT_1D"),
            ]
        )
        return result
