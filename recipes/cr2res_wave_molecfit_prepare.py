import sys
from os.path import dirname
from typing import Any, Dict

import numpy as np
from astropy.io import fits
from cpl.core import Msg
from cpl.ui import Frame, FrameSet, ParameterList, ParameterValue, PyRecipe
from pyesorex.pyesorex import Pyesorex

sys.path.append(dirname(__file__))
from cr2res_recipe import CR2RES_RECIPE


def save_frameset(frameset, filename):
    content = []
    for frame in frameset:
        content += [f"{frame.file} {frame.tag}\n"]

    with open(filename, "w") as f:
        f.writelines(content)

class RecipeConfig(dict):
    """
    This class handles the configuration parameters settings file for esorex recipes.
    The files can be created using ´´´esorex --create-config=<filename> <recipe-name>```
    """

    def __init__(self, parameters=None, replacement=None):
        super().__init__()
        if parameters is not None:
            parameters = {p.name:p.value for p in parameters}
            self.update(parameters)
        if replacement is None:
            replacement = {"FALSE": False, "TRUE": True, "NULL": None}
        #:dict: The strings replacements that are used to convert from esorex format to python
        self.replacement = replacement

    @property
    def replacement_inverse(self):
        return {v: k for k, v in self.replacement.items()}

    def parse(self, default):
        """
        Parse the values in the input dictonary default from strings
        to more useful and pythonic data types.
        Replacements for specific strings are defined in self.replacement
        This method is shallow, i.e. it only works on the top level of default
        Note also that it operates in place, i.e. the input is modified

        Parameters
        ----------
        default : dict
            Dictionary with parameters to convert.

        Returns
        -------
        default : dict
            The same object as was input, but with the items converted
        """
        for key, value in default.items():
            if value in self.replacement.keys():
                value = self.replacement[value]
            else:
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
            default[key] = value
        return default

    def unparse_value(self, value, inverse):
        if isinstance(value, (list, tuple)):
            list_inverse = {True: "1", False: "0"}
            value = [self.unparse_value(v, list_inverse) for v in value]
            value = ",".join(value)
        elif any([value is key for key in inverse]):
            # Turns out that True == 1 and False == 0, as far as the dictionary keys are concerned
            # but we want to treat them as different objects, so we have to use is
            value = inverse[value]
        else:
            value = str(value)
        return value

    def unparse(self, parameters):
        """
        Similar to parse, but in reverse it replaces the values in parameters
        with strings that are understood by esorex.

        Parameters
        ----------
        parameters : dict
            Input dictionary with python datatypes

        Returns
        -------
        parameters : dict
            The same dictionary but with all strings
        """
        default = {}
        inverse = self.replacement_inverse
        for key, value in parameters.items():
            key = key.upper()
            value = self.unparse_value(value, inverse)
            default[key] = value
        return default

    @classmethod
    def read(cls, config:Dict, replacement=None):
        """
        Read a configuration file from disk

        TODO: right now we are using config parser for the file parsing
        TODO: also this removes the comments

        Parameters
        ----------
        filename : str
            Name of the configuration file to read
        replacement : dict, optional
            replacement for speciific strings, see parse, by default None

        Returns
        -------
        self : RecipeConfig
            The read and parsed configuration
        """
        self = cls(replacement=replacement)
        self.update(self.parse(config))
        return self

    def write(self, filename):
        """
        Write the configuration file to disk

        Parameters
        ----------
        filename : str
            filename of the destination
        """
        # We copy self.parameters, since unparse works in place
        params = self.unparse(self)
        content = []
        for key, value in params.items():
            content += [f"{key}={value}\n"]

        with open(filename, "w") as f:
            f.writelines(content)

    def get_recipe_options(self):
        """
        Generate the recipe options parameters that can be passed to esorex
        instead of the recipe configuration file

        Returns
        -------
        options : list
            recipe options for exorex
        """
        params = self.unparse(self.parameters)
        options = []
        for key, value in params.items():
            options += [f"--{key.upper()}={value}"]
        return options

class cr2res_wave_molecfit_prepare(PyRecipe, CR2RES_RECIPE):
    _name = "cr2res_wave_molecfit_prepare"
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
                    context="cr2res_wave_molecfit_prepare",
                    description="Detector to run",
                    default="ALL",
                ),
                ParameterValue(
                    name="order",
                    context="cr2res_wave_molecfit_prepare",
                    description="Order to run",
                    default="ALL",
                ),
                ParameterValue(
                    name="transmission",
                    context="cr2res_wave_molecfit_prepare",
                    description="Transmission or Emission spectrum",
                    default=True,
                )
            ]
        )

    def load_data(self, spectrum, blaze):
        # Read the data from all chips into one big array
        flux = []
        wave = []
        err = []
        # This stores the chip and order of the data as it relates to the
        # Molecfit internal order
        mapping = []
        k = 0
        for chip in [1, 2, 3]:
            ext = self.get_chip_extension(chip)
            data = spectrum[ext].data
            dblaze = blaze[ext].data
            columns = data.names

            orders = np.sort([int(c[:2]) for c in columns if c[-4:] == "SPEC"])

            for order in orders:
                # Division but avoiding zero blaze elements
                # So we dont get a warning message
                col = self.get_table_column(order, 1, "SPEC")
                col_wl = self.get_table_column(order, 1, "WL")
                col_err = self.get_table_column(order, 1, "ERR")

                ftmp = np.full_like(data[col], np.nan)
                np.divide(data[col], dblaze[col], where=dblaze[col] != 0, out=ftmp)
                flux += [ftmp]
                wave += [data[col_wl]]
                err += [data[col_err]]
                k += 1
                mapping += [
                    [chip, order, k],
                ]

        flux = np.array(flux)
        wave = np.array(wave)
        err = np.array(err)
        mapping = np.rec.fromrecords(mapping, names="CHIP,ORDER,MOLECFIT")
        return wave, flux, err, mapping

    def normalize_spectrum_star(self, wave, flux, err):
        # quick normalization
        flux -= np.nanpercentile(flux, 1, axis=1)[:, None]

        x = np.arange(flux.shape[1])
        for i in range(len(flux)):
            # First step, just linear
            mask = np.isfinite(flux[i])
            factor = np.nanpercentile(flux[i][mask], 99)
            flux[i] /= factor
            err[i] /= factor
            flux[i, flux[i] < 0] = np.nan
            err[i, np.isnan(flux[i])] = 0

        x = np.arange(flux.shape[1])
        for i in range(len(flux)):
            mask = np.isnan(flux[i])
            flux[i, mask] = np.interp(x[mask], x[~mask], flux[i, ~mask], left=1, right=1)
            err[i, mask] = np.interp(x[mask], x[~mask], err[i, ~mask], left=0, right=0)

        return flux, err

    def normalize_spectrum_sky(self, wave, flux, err):
        # Quick normalization
        flux -= np.nanpercentile(flux, 1, axis=1)[:, None]

        # further normalization with a linear fit
        x = np.arange(flux.shape[1])
        for i in range(len(flux)):
            # First step, just linear
            mask = np.isfinite(flux[i])
            coef = np.polyfit(x[mask], flux[i, mask], 1)
            cont = np.polyval(coef, x)
            flux[i] -= cont

            # Second step, only use the bottom half of the points
            mask &= flux[i] < np.nanmedian(flux[i])
            coef = np.polyfit(x[mask], flux[i, mask], 1)
            cont = np.polyval(coef, x)
            flux[i] -= cont

            # flux[i] -= np.nanpercentile(flux[i], 1)
            flux[i, flux[i] < 0] = np.nan
            err[i, np.isnan(flux[i])] = 0

        # Replace NaNs with interpolated/extrapolated values
        x = np.arange(flux.shape[1])
        for i in range(len(flux)):
            mask = np.isnan(flux[i])
            flux[i, mask] = np.interp(x[mask], x[~mask], flux[i, ~mask], left=0, right=0)
            err[i, mask] = np.interp(x[mask], x[~mask], err[i, ~mask], left=0, right=0)

        return flux, err

    def prepare_fits(
        self, header, wave, flux, parameters, err=None, sort=True, mapping=None
    ):
        """Create a new fits file that can be read by Molecfit
        The new file is created in the self.output_dir directory

        Parameters
        ----------
        header : fits.Header
            fits header of the original file, contains all the keywords
            that are used by Molecfit
        wave : array
            wavelength array
        flux : array
            flux (spectrum) array
        err : array, optional
            flux (spectrum) uncertainties, if not set, we use the
            square root of flux, by default None
        sort : bool, optional
            whether to sort the data by increasing wavelength as is required
            by molecfit. If False it is up to the user to ensure the order

        Returns
        -------
        filename : str
            the name of the new fits file
        """
        if err is None:
            err = [np.sqrt(f) for f in flux]
        nseg = len(wave)

        # The wavelenhgth needs to be sorted in ascending order
        # Both in each segment, as well as overall
        if sort:
            mean = [np.mean(w) for w in wave]
            sort = np.argsort(mean)
            wave = [wave[s] for s in sort]
            flux = [flux[s] for s in sort]
            err = [err[s] for s in sort]

            if mapping is not None:
                mapping = mapping[sort]
                mapping["MOLECFIT"] = np.arange(1, len(mapping) + 1)

            for i in range(nseg):
                sort = np.argsort(wave[i])
                wave[i] = wave[i][sort]
                flux[i] = flux[i][sort]
                err[i] = err[i][sort]

        prihdu = fits.PrimaryHDU(header=header)
        thdulist = [prihdu]

        for i in range(nseg):

            col1 = fits.Column(
                name=parameters["COLUMN_LAMBDA"].value,
                format="1D",
                array=wave[i],
            )
            col2 = fits.Column(
                name=parameters["COLUMN_FLUX"].value,
                format="1D",
                array=flux[i],
            )
            col3 = fits.Column(
                name=parameters["COLUMN_DFLUX"].value,
                format="1D",
                array=err[i],
            )
            cols = fits.ColDefs([col1, col2, col3])
            tbhdu = fits.BinTableHDU.from_columns(cols)
            thdulist += [tbhdu]

        thdulist = fits.HDUList(thdulist)
        return thdulist, mapping

    def run(self, frameset: FrameSet, settings: Dict[str, Any]) -> FrameSet:
        # Check the input parameters
        self.parameters = self.parse_parameters(settings)
        transmission = bool(self.parameters["transmission"].value)

        # sort the frames by type
        frames = self.filter_frameset(
            frameset, spectrum="UTIL_CALIB_EXTRACT_1D", blaze="CAL_FLAT_EXTRACT_1D"
        )

        # Load the data
        spectrum = frames["spectrum"][0].as_hdulist()
        blaze = frames["blaze"][0].as_hdulist()
        header = spectrum[0].header
        # Normalize the input spectrum
        wave, flux, err, mapping = self.load_data(spectrum, blaze)

        # Fix the wavelength scale using the expected wavelength range
        x = np.arange(wave.shape[1])
        for i in range(len(wave)):
            order = mapping["ORDER"][i]
            lower = header[f"ESO INS WLEN MIN{order}"]
            upper = header[f"ESO INS WLEN MAX{order}"]
            mask = (~np.isfinite(wave[i])) | (wave[i] <= lower) | (wave[i] >= upper)
            wave[i, mask] = np.polyval(np.polyfit(x[~mask], wave[i, ~mask], 1), x[mask])

        flux[(~np.isfinite(flux))] = np.nan
        flux[(~np.isfinite(wave)) | (wave <= 0)] = np.nan
        flux[(~np.isfinite(err)) | (err <= 0)] = np.nan

        # Remove the edges of the orders
        # As they are badly reduced
        # and often contain outliers
        left, right = 20, 20
        for i in range(len(flux)):
            flux[i][:left] = np.nan
            err[i][:left] = np.nan
            flux[i][-right:] = np.nan
            err[i][-right:] = np.nan

        if transmission:
            flux, err = self.normalize_spectrum_star(wave, flux, err)
        else:
            flux, err = self.normalize_spectrum_sky(wave, flux, err)

        # Ensure errors are 0 when necessary
        err[flux <= 0] = 0
        err[~np.isfinite(err)] = 0
        for i in range(len(flux)):
            err[i][:left] = 0
            err[i][-right:] = 0

        # Remove a strong line in the spectrum of this order
        # This of course depends on the wavelength setting used
        # TODO: customize this for all settings used here (if required)
        wl_set = header["ESO INS WLEN ID"]
        if wl_set == "L3262":
            if not transmission:
                idx = mapping["MOLECFIT"][(mapping["CHIP"] == 2) & (mapping["ORDER"] == 3)][0] - 1
                flux[idx][1024-500:1024+500] = 0

        
            idx = mapping["MOLECFIT"][(mapping["CHIP"] == 2) & (mapping["ORDER"] == 3)][0] - 1
            flux[idx][-100:] = 0
            err[idx][-100:] = 0

            idx = mapping["MOLECFIT"][(mapping["CHIP"] == 3) & (mapping["ORDER"] == 2)][0] - 1
            flux[idx][-100:] = 0
            err[idx][-100:] = 0
        if wl_set == "L3340":
            if not transmission:
                idx = mapping["MOLECFIT"][(mapping["CHIP"] == 2) & (mapping["ORDER"] == 2)][0] - 1
                flux[idx][1500:] = 0

        # Convert the wavelength in micron for Molecfit
        wave *= 0.001

        # Initialize the recipe parameters
        molecules = ["H2O", "CH4", "N2O"]
        esorex = Pyesorex()
        esorex.recipe = "molecfit_model"
        esorex.recipe_parameters["TMP_PATH"] = "/scratch/ptah/anwe5599/tmp"
        esorex.recipe_parameters["COLUMN_LAMBDA"] = "WAVE"
        esorex.recipe_parameters["COLUMN_FLUX"] = "FLUX"
        esorex.recipe_parameters["COLUMN_DFLUX"] = "ERR"
        esorex.recipe_parameters["SILENT_EXTERNAL_BINS"] = False

        esorex.recipe_parameters["WLG_TO_MICRON"] = 1
        esorex.recipe_parameters["PIX_SCALE_VALUE"] = 0.056
        esorex.recipe_parameters["TRANSMISSION"] = transmission
        esorex.recipe_parameters["USE_INPUT_KERNEL"] = False
        esorex.recipe_parameters["VARKERN"] = True
        esorex.recipe_parameters["FIT_CONTINUUM"] = "1"
        esorex.recipe_parameters["CONTINUUM_N"] = "0"
        esorex.recipe_parameters["FIT_WLC"] = "1"
        esorex.recipe_parameters["WLC_N"] = 1
        esorex.recipe_parameters["WLC_CONST"] = 0
        esorex.recipe_parameters["LIST_MOLEC"] = ",".join(molecules)
        esorex.recipe_parameters["REL_COL"] = ",".join(["1"] * len(molecules))
        esorex.recipe_parameters["FIT_MOLEC"] = ",".join(["1"] * len(molecules))
        # Fix the broadening
        esorex.recipe_parameters["FIT_RES_BOX"] = False
        esorex.recipe_parameters["RES_BOX"] = 0
        esorex.recipe_parameters["FIT_RES_GAUSS"] = False
        esorex.recipe_parameters["RES_GAUSS"] = 10
        esorex.recipe_parameters["FIT_RES_LORENTZ"] = False
        esorex.recipe_parameters["RES_LORENTZ"] = 0

        # Since we modifed the flux and wavelength we need to write the data to a new datafile
        header = spectrum[0].header

        input_file, mapping = self.prepare_fits(
            header, wave, flux, esorex.recipe_parameters, err=err, mapping=mapping
        )

        # Read the included wavelength, now in the correct order
        wave_include = []
        for i in range(1, len(input_file)):

            wave = input_file[i].data[esorex.recipe_parameters["COLUMN_LAMBDA"].value]
            wmin, wmax = np.nanmin(wave), np.nanmax(wave)
            wmin, wmax = (
                wmin * esorex.recipe_parameters["WLG_TO_MICRON"].value,
                wmax * esorex.recipe_parameters["WLG_TO_MICRON"].value,
            )
            wave_include += [wmin, wmax]

        nseg = len(wave_include) // 2
        wave_include = ",".join([f"{w:.5}" for w in wave_include])
        esorex.recipe_parameters["WAVE_INCLUDE"] = wave_include
        esorex.recipe_parameters["MAP_REGIONS_TO_CHIP"] = ",".join(
            [str(i) for i in range(1, nseg + 1)]
        )
        esorex.recipe_parameters["CHIP_EXTENSIONS"] = True

        # Save the prepared input spectrum
        input_file_name = "INPUT_SPECTRUM.fits"
        input_file.writeto(input_file_name, overwrite=True)

        # Save the recipe parameters
        # The pyesorex method does not work since 
        # MOLECFIT parameters are formatted differently
        # Thus we use our own implementation
        # Specifically molecfit expects bool values in uppercase
        # esorex.write_recipe_config(rc_fname)
        rc_fname = "molecfit_model.rc"
        rc = RecipeConfig(esorex.recipe_parameters)
        rc.write(rc_fname)

        # Save the SOF
        science_frame = Frame(input_file_name, tag="SCIENCE")
        sof = FrameSet(
            [science_frame],
        )
        sof_fname = "molecfit_model.sof"
        save_frameset(sof, sof_fname)

        # Save the mapping
        mapping_hdu = fits.BinTableHDU(data=mapping)
        mapping_fname = "MAPPING.fits"
        mapping_hdu.writeto(mapping_fname, overwrite=True)

        # Return the results
        result = FrameSet(
            [
                Frame(input_file_name, tag="CAL_WAVE_MOLECFIT_SCIENCE"),
                Frame(mapping_fname, tag="CAL_WAVE_MOLECFIT_MAPPING"),
                Frame(rc_fname, tag="CAL_WAVE_MOLECFIT_RC"),
                Frame(sof_fname, tag="CAL_WAVE_MOLECFIT_SOF"),
            ]
        )
        return result
