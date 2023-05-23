"""
Apply wavelength calibration from molecfit to extracted spectra
"""
import sys
from os.path import basename, dirname
from typing import Any, Dict

import numpy as np
from cpl.core import Msg
from cpl.ui import Frame, FrameSet, ParameterList, ParameterValue, PyRecipe
from tqdm import tqdm

sys.path.append(dirname(__file__))
from cr2res_recipe import CR2RES_RECIPE


class cr2res_wave_molecfit_apply(PyRecipe, CR2RES_RECIPE):
    _name = "cr2res_wave_molecfit_apply"
    _version = "1.0"
    _author = "Ansgar Wehrhahn"
    _email = "ansgar.wehrhahn@astro.su.se"
    _copyright = "GPL-3.0-or-later"
    _synopsis = "Apply wavelength calibration from molecfit to extracted spectra"
    _description = "This recipe formats the Molecfit wavelength calibration back into a CRIRES+ spectrum"

    SKY = "SKY"
    STAR = "STAR"

    ASSIGNMENT = {
        "L3262": {
                (1, 2): STAR,
                (1, 3): SKY,
                (1, 4): STAR,
                (1, 5): STAR,
                (1, 6): STAR,
                (1, 7): STAR,
                (2, 2): SKY,
                (2, 3): SKY,
                (2, 4): STAR,
                (2, 5): STAR,
                (2, 6): STAR,
                (2, 7): STAR,
                (3, 2): SKY,
                (3, 3): SKY,
                (3, 4): STAR,
                (3, 5): STAR,
                (3, 6): STAR,
                (3, 7): STAR,
            },
        "L3340": {
                (1, 2): SKY,
                (1, 3): SKY,
                (1, 4): STAR,
                (1, 5): STAR,
                (1, 6): STAR,
                (1, 7): STAR,
                (2, 2): SKY,
                (2, 3): SKY,
                (2, 4): STAR,
                (2, 5): STAR,
                (2, 6): STAR,
                (2, 7): STAR,
                (3, 2): SKY,
                (3, 3): SKY,
                (3, 4): STAR,
                (3, 5): STAR,
                (3, 6): STAR,
                (3, 7): STAR,
            },
        "L3426": {
                (1, 2): SKY,
                (1, 3): STAR,
                (1, 4): STAR,
                (1, 5): STAR,
                (1, 6): STAR,
                (1, 7): STAR,
                (1, 8): STAR,
                (2, 2): SKY,
                (2, 3): STAR,
                (2, 4): STAR,
                (2, 5): STAR,
                (2, 6): STAR,
                (2, 7): STAR,
                (2, 8): STAR,
                (3, 2): SKY,
                (3, 3): STAR,
                (3, 4): STAR,
                (3, 5): STAR,
                (3, 6): STAR,
                (3, 7): STAR,
                (3, 8): STAR,
        },
        "M4318": {
                (1, 3): STAR,
                (1, 4): SKY,
                (1, 6): STAR,
                (1, 7): SKY,
                (1, 8): STAR,
                (2, 3): STAR,
                (2, 4): SKY,
                (2, 6): STAR,
                (2, 7): SKY,
                (2, 8): STAR,
                (3, 3): STAR,
                (3, 4): SKY,
                (3, 6): STAR,
                (3, 7): SKY,
                (3, 8): SKY
            }
    }

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

    def run(self, frameset: FrameSet, settings: Dict[str, Any]) -> FrameSet:

        # Check the input parameters
        self.parse_parameters(settings)

        # sort the frames by type
        frames = self.filter_frameset(
            frameset,
            spectrum="UTIL_CALIB_EXTRACT_1D",
            model_sky="CAL_WAVE_MOLECFIT_MODEL_SKY",
            mapping_sky="CAL_WAVE_MOLECFIT_MAPPING_SKY",
            model_star="CAL_WAVE_MOLECFIT_MODEL_STAR",
            mapping_star="CAL_WAVE_MOLECFIT_MAPPING_STAR",
        )

        spectrum_fname = frames["spectrum"][0].file
        spectrum = frames["spectrum"][0].as_hdulist()
        model_sky = frames["model_sky"][0].as_hdulist()
        mapping_sky = frames["mapping_sky"][0].as_hdulist()
        model_star = frames["model_star"][0].as_hdulist()
        mapping_star = frames["mapping_star"][0].as_hdulist()

        header = spectrum[0].header
        mapping_sky = mapping_sky[1].data
        model_sky = model_sky[1].data
        mapping_star = mapping_star[1].data
        model_star = model_star[1].data

        chips = np.unique(mapping_sky["CHIP"])
        wl_set = header["ESO INS WLEN ID"]
        try:
            assignment = self.ASSIGNMENT[wl_set]
        except KeyError:
            raise ValueError(f"Unexpected wavelength setting {wl_set}")

        with self.redirect_stdout_tqdm() as orig_stdout:
            for chip in tqdm(chips, desc="Detector", file=orig_stdout, dynamic_ncols=True):
                ext = self.get_chip_extension(chip)
                data = spectrum[ext].data
                orders = np.unique(mapping_sky[mapping_sky["CHIP"] == chip]["ORDER"])
                xs, ys, zs, os = [], [], [], []
                for order in tqdm(orders, leave=False, desc="Order", file=orig_stdout, dynamic_ncols=True):
                    if assignment[(chip, order)] == self.STAR:
                        mapping = mapping_star
                        model = model_star
                    elif assignment[(chip, order)] == self.SKY:
                        mapping = mapping_sky
                        model = model_sky
                    else:
                        raise ValueError("Unexpected Chip/Order combination")

                    idx = (mapping["CHIP"] == chip) & (mapping["ORDER"] == order)
                    idx = mapping[idx]["MOLECFIT"][0]
                    column = self.get_table_column(order, 1, "WL")
                    data[column] = model[model["CHIP"] == idx]["MLAMBDA"] * 1000

                    zs += [data[column]]
                    ys += [np.full_like(zs[-1], order)]
                    xs += [np.arange(len(zs[-1]))]
                    os += [order]

        # Overwrite the results
        out_fname = basename(spectrum_fname)[:-5] + "_molecfit_wave.fits"
        spectrum.writeto(basename(out_fname), overwrite=True)

        result = FrameSet(
            [
                Frame(out_fname, tag="UTIL_CALIB_EXTRACT_1D"),
            ]
        )
        return result


if __name__ == "__main__":
    recipe = cr2res_wave_molecfit_apply()
    sof = "/scratch/ptah/anwe5599/CRIRES/2022-11-29_OLD/extr/molecfit_apply.sof"
    sof = FrameSet(sof)
    res = recipe.run(sof, {})
    pass