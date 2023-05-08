import sys
from os.path import basename, dirname
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
from cpl.core import Msg
from cpl.ui import (
    Frame,
    FrameSet,
    ParameterList,
    ParameterValue,
    PyRecipe,
    ParameterEnum,
)
from numpy.polynomial.polynomial import polyval2d
from scipy import signal
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.optimize import least_squares
from scipy.linalg import svd
from skimage import exposure


sys.path.append(dirname(__file__))
from cr2res_recipe import CR2RES_RECIPE
Msg.increase_indent = lambda : Msg.set_config(indent=Msg.get_config()["indent"] + 1)
Msg.decrease_indent = lambda : Msg.set_config(indent=Msg.get_config()["indent"] - 1)

def extend_orders(orders, nrow):
    """Extrapolate extra orders above and below the existing ones

    Parameters
    ----------
    orders : array[nord, degree]
        order tracing coefficients
    nrow : int
        number of rows in the image

    Returns
    -------
    orders : array[nord + 2, degree]
        extended orders
    """

    nord, ncoef = orders.shape

    if nord > 1:
        order_low = 2 * orders[0] - orders[1]
        order_high = 2 * orders[-1] - orders[-2]
    else:
        order_low = [0 for _ in range(ncoef)]
        order_high = [0 for _ in range(ncoef - 1)] + [nrow]

    return np.array([order_low, *orders, order_high])


def fix_extraction_width(xwd, orders, cr, ncol):
    """Convert fractional extraction width to pixel range

    Parameters
    ----------
    extraction_width : array[nord, 2]
        current extraction width, in pixels or fractions (for values below 1.5)
    orders : array[nord, degree]
        order tracing coefficients
    column_range : array[nord, 2]
        column range to use
    ncol : int
        number of columns in image

    Returns
    -------
    extraction_width : array[nord, 2]
        updated extraction width in pixels
    """

    if not np.all(xwd > 1.5):
        # if extraction width is in relative scale transform to pixel scale
        x = np.arange(ncol)
        for i in range(1, len(xwd) - 1):
            for j in [0, 1]:
                if xwd[i, j] < 1.5:
                    k = i - 1 if j == 0 else i + 1
                    left = max(cr[[i, k], 0])
                    right = min(cr[[i, k], 1])

                    if right < left:
                        raise ValueError(
                            f"Check your column ranges. Orders {i} and {k} are weird"
                        )

                    current = np.polyval(orders[i], x[left:right])
                    below = np.polyval(orders[k], x[left:right])
                    xwd[i, j] *= np.min(np.abs(current - below))

        xwd[0] = xwd[1]
        xwd[-1] = xwd[-2]

    xwd = np.ceil(xwd).astype(int)

    return xwd


def fix_column_range(column_range, orders, extraction_width, nrow, ncol):
    """Fix the column range, so that no pixels outside the image will be accessed (Thus avoiding errors)

    Parameters
    ----------
    img : array[nrow, ncol]
        image
    orders : array[nord, degree]
        order tracing coefficients
    extraction_width : array[nord, 2]
        extraction width in pixels, (below, above)
    column_range : array[nord, 2]
        current column range
    no_clip : bool, optional
        if False, new column range will be smaller or equal to current column range, otherwise it can also be larger (default: False)

    Returns
    -------
    column_range : array[nord, 2]
        updated column range
    """

    ix = np.arange(ncol)
    # Loop over non extension orders
    for i, order in zip(range(1, len(orders) - 1), orders[1:-1]):
        # Shift order trace up/down by extraction_width
        coeff_bot, coeff_top = np.copy(order), np.copy(order)
        coeff_bot[-1] -= extraction_width[i, 0]
        coeff_top[-1] += extraction_width[i, 1]

        y_bot = np.polyval(coeff_bot, ix)  # low edge of arc
        y_top = np.polyval(coeff_top, ix)  # high edge of arc

        # find regions of pixels inside the image
        # then use the region that most closely resembles the existing column range (from order tracing)
        # but clip it to the existing column range (order tracing polynomials are not well defined outside the original range)
        points_in_image = np.where((y_bot >= 0) & (y_top < nrow))[0]

        if len(points_in_image) == 0:
            raise ValueError(
                f"No pixels are completely within the extraction width for order {i}"
            )

        regions = np.where(np.diff(points_in_image) != 1)[0]
        regions = [(r, r + 1) for r in regions]
        regions = [
            points_in_image[0],
            *points_in_image[(regions,)].ravel(),
            points_in_image[-1],
        ]
        regions = [[regions[i], regions[i + 1] + 1] for i in range(0, len(regions), 2)]
        overlap = [
            min(reg[1], column_range[i, 1]) - max(reg[0], column_range[i, 0])
            for reg in regions
        ]
        iregion = np.argmax(overlap)
        column_range[i] = np.clip(
            regions[iregion], column_range[i, 0], column_range[i, 1]
        )

    column_range[0] = column_range[1]
    column_range[-1] = column_range[-2]

    return column_range


def fix_parameters(xwd, cr, orders, nrow, ncol, nord, ignore_column_range=False):
    """Fix extraction width and column range, so that all pixels used are within the image.
    I.e. the column range is cut so that the everything is within the image

    Parameters
    ----------
    xwd : float, array
        Extraction width, either one value for all orders, or the whole array
    cr : 2-tuple(int), array
        Column range, either one value for all orders, or the whole array
    orders : array
        polynomial coefficients that describe each order
    nrow : int
        Number of rows in the image
    ncol : int
        Number of columns in the image
    nord : int
        Number of orders in the image
    ignore_column_range : bool, optional
        if true does not change the column range, however this may lead to problems with the extraction, by default False

    Returns
    -------
    xwd : array
        fixed extraction width
    cr : array
        fixed column range
    orders : array
        the same orders as before
    """

    if xwd is None:
        xwd = 0.5
    if np.isscalar(xwd):
        xwd = np.tile([xwd, xwd], (nord, 1))
    else:
        xwd = np.asarray(xwd)
        if xwd.ndim == 1:
            xwd = np.tile(xwd, (nord, 1))

    if cr is None:
        cr = np.tile([0, ncol], (nord, 1))
    else:
        cr = np.asarray(cr)
        if cr.ndim == 1:
            cr = np.tile(cr, (nord, 1))

    orders = np.asarray(orders)

    xwd = np.array([xwd[0], *xwd, xwd[-1]])
    cr = np.array([cr[0], *cr, cr[-1]])
    orders = extend_orders(orders, nrow)

    xwd = fix_extraction_width(xwd, orders, cr, ncol)
    if not ignore_column_range:
        cr = fix_column_range(cr, orders, xwd, nrow, ncol)

    orders = orders[1:-1]
    xwd = xwd[1:-1]
    cr = cr[1:-1]

    return xwd, cr, orders


def polyfit2d(x, y, z, w=None, degree=1, x0=None, loss="arctan", method="trf"):

    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    z = np.asarray(z).ravel()
    if w is not None:
        w = np.asarray(w).ravel()
    else:
        w = np.ones_like(x)

    if np.isscalar(degree):
        degree_x = degree_y = degree + 1
    else:
        degree_x = degree[0] + 1
        degree_y = degree[1] + 1

    polyval2d = np.polynomial.polynomial.polyval2d

    def func(c):
        c = c.reshape(degree_x, degree_y)
        value = polyval2d(x, y, c)
        return (value - z) / w

    if x0 is None:
        x0 = np.zeros(degree_x * degree_y)
    else:
        x0 = x0.ravel()

    res = least_squares(func, x0, loss=loss, method=method)
    coef = res.x
    coef = coef.reshape(degree_x, degree_y)

    return coef


class cr2res_util_slit_curv_sky(PyRecipe, CR2RES_RECIPE):
    _name = "cr2res_util_slit_curv_sky"
    _version = "1.0"
    _author = "Ansgar Wehrhahn"
    _email = "ansgar.wehrhahn@astro.su.se"
    _copyright = "GPL-3.0-or-later"
    _synopsis = "Determine the slit curvature from a sky observation"
    _description = "This recipe determines the slit curvature from a sky emission spectrum instead of a wavelength calibration spectrum"

    peak_width = 1
    sigma_cutoff = 3
    curv_degree = 1
    threshold = 1.5
    window_width = 41
    fit_degree = (1, 1)
    mode = "2D"
    peak_function = "spectrum"
    plot = True
    chip = 0
    order = 0
    wl_set = ""

    def __init__(self) -> None:
        super().__init__()
        self.parameters = ParameterList(
            [
                ParameterValue(
                    name="detector",
                    context="cr2res_util_slit_curv_sky",
                    description="Detector to run",
                    default=-1,
                ),
                ParameterValue(
                    name="order",
                    context="cr2res_util_slit_curv_sky",
                    description="Order to run",
                    default=-1,
                ),
                ParameterEnum(
                    name="fit_mode",
                    context="cr2res_util_slit_curv_sky",
                    description="Whether to perform a 1d or 2d polynomial fit",
                    default="2D",
                    alternatives=("1D", "2D"),
                ),
                ParameterValue(
                    name="degree_x",
                    context="cr2res_util_slit_curv_sky",
                    description="Polynomial degree along x",
                    default=1,
                ),
                ParameterValue(
                    name="degree_y",
                    context="cr2res_util_slit_curv_sky",
                    description="Polynomial degree along y",
                    default=1,
                ),
                ParameterValue(
                    name="plot",
                    context="cr2res_util_slit_curv_sky",
                    description="Activates plots",
                    default=True,
                ),
            ]
        )

    def determine_slit_curvature(self, trace_wave, science, extracted, bpm=None):
        tilts = {}
        shears = {}

        header = science[0].header
        self.wl_set = header["ESO INS WLEN ID"]

        for chip in self.detectors:
            Msg.info(self.name, f"Processing Detector {chip}")
            Msg.increase_indent()
            self.chip = chip

            ext = self.get_chip_extension(chip)
            exterr = self.get_chip_extension(chip, error=True)

            tw_data = trace_wave[ext].data
            data = science[ext].data
            err = science[exterr].data
            # blaze_data = blaze[ext].data
            extract_data = extracted[ext].data
            if bpm is not None:
                bpm_data = bpm[ext].data != 0
                data[bpm_data] = np.nan

            order_traces = []
            extraction_width = []
            column_range = []
            upper_ycen = []
            lower_ycen = []
            orders = self.get_orders_trace_wave(tw_data)
            xsize = len(extract_data)

            for order in orders:
                lower, middle, upper = self.get_order_trace(tw_data, order)
                upper_ycen += [upper]
                lower_ycen += [lower]
                height_upp = int(np.ceil(np.min(upper - middle)))
                height_low = int(np.ceil(np.min(middle - lower)))
                idx = tw_data["Order"] == order
                order_traces += [tw_data[idx]["All"][0][::-1]]
                column_range += [[10, xsize - 10]]
                extraction_width += [[height_upp, height_low]]

            order_traces = np.asarray(order_traces)
            column_range = np.asarray(column_range)
            extraction_width = np.asarray(extraction_width)

            spectrum = np.array(
                [
                    extract_data[self.get_table_column(order, 1, "SPEC")]
                    for order in orders
                ]
            )

            for i, order in enumerate(orders):
                # Smooth spectrum for easier peak detection
                spectrum[i][np.isnan(spectrum[i])] = 0
                spectrum[i] = gaussian_filter1d(spectrum[i], 3)
                # Limit the size of the extraction width
                # to avoid the interorder area
                extraction_width[i, 0] -= 10
                extraction_width[i, 1] -= 10

            order_range = (0, order_traces.shape[0])
            data = np.ma.array(data, mask=~np.isfinite(data))
            tilt, shear = self.execute(
                spectrum,
                data,
                order_traces,
                extraction_width,
                column_range,
                order_range,
                chip=chip,
                order_nrs=orders,
            )

            tilts[chip] = tilt
            shears[chip] = shear
            Msg.decrease_indent()

        # Create output
        for chip in self.detectors:
            ext = self.get_chip_extension(chip)
            tw_data = trace_wave[ext].data
            orders = self.get_orders_trace_wave(tw_data)
            x = np.arange(1, 2048 + 1)
            for i, order in enumerate(orders):
                t = tilts[chip][i]
                s = shears[chip][i]
                _, ycen, _ = self.get_order_trace(tw_data, order, clip=False)
                # Convert to the global reference system
                t -= 2 * ycen * s

                # Fit overarching polynomial
                ct = np.polyfit(x, t, 2)
                cs = np.polyfit(x, s, 2)

                Msg.debug(
                    self.name,
                    f"CHIP: {chip} ORDER: {order} TILT: {np.array2string(ct, precision=2)}",
                )

                # Write the data back to the fits file
                # The indexing is chosen such that the data is actually
                # stored in the tw object and not lost
                idx = tw_data["Order"] == order
                trace_wave[ext].data["SlitPolyA"][idx] = [0, 1, 0]
                trace_wave[ext].data["SlitPolyB"][idx] = ct[::-1]
                trace_wave[ext].data["SlitPolyC"][idx] = cs[::-1]

        return trace_wave

    def run(self, frameset: FrameSet, settings: Dict[str, Any]) -> FrameSet:
        # Check the input parameters
        self.parameters = self.parse_parameters(settings)
        self.detectors = self.parameters["detector"].value
        if self.detectors <= 0:
            self.detectors = [1, 2, 3]
        else:
            self.detectors = [
                self.detectors,
            ]

        self.orders = self.parameters["order"].value

        self.mode = self.parameters["fit_mode"].value
        if self.mode == "1D":
            self.fit_degree = self.parameters["degree_x"].value
        elif self.mode == "2D":
            self.fit_degree = (
                self.parameters["degree_x"].value,
                self.parameters["degree_y"].value,
            )
        else:
            raise ValueError
        
        self.plot = self.parameters["plot"].value

        Msg.debug(self.name, f"Fit degree {self.fit_degree}")

        frames = self.filter_frameset(
            frameset,
            science="UTIL_CALIB",
            trace_wave="CAL_FLAT_TW",
            spectrum="UTIL_CALIB_EXTRACT_1D",
            bpm="CAL_DARK_BPM",
            validate=False,
        )
        # Validate the input
        if len(frames["science"]) != 1:
            raise ValueError("Expected exactly 1 science frame")
        if len(frames["trace_wave"]) != 1:
            raise ValueError("Expected exactly 1 trace wave frame")
        if len(frames["spectrum"]) != 1:
            raise ValueError("Expected exactly 1 spectrum frame")

        # Run the actual work
        trace_wave = frames["trace_wave"][0].as_hdulist()
        science = frames["science"][0].as_hdulist()
        spectrum = frames["spectrum"][0].as_hdulist()
        if len(frames["bpm"]) >= 1:
            bpm = frames["bpm"][0].as_hdulist()
        else:
            bpm = None

        trace_wave = self.determine_slit_curvature(trace_wave, science, spectrum, bpm)

        # Save the results
        outfile = basename(frames["trace_wave"][0].file)
        trace_wave.writeto(outfile, overwrite=True)

        # Return data back to PyEsorex
        fs = FrameSet(
            [
                Frame(outfile, tag="CAL_SLIT_CURV_TW"),
            ]
        )
        return fs

    @staticmethod
    def _fix_inputs(shape, orders, extraction_width, column_range, order_range):
        nrow, ncol = shape
        nord = len(orders)

        extraction_width, column_range, orders = fix_parameters(
            extraction_width, column_range, orders, nrow, ncol, nord
        )

        column_range = column_range[order_range[0] : order_range[1]]
        extraction_width = extraction_width[order_range[0] : order_range[1]]
        orders = orders[order_range[0] : order_range[1]]

        n = order_range[1] - order_range[0]
        order_range = (0, n)

        return orders, extraction_width, column_range, order_range

    def _find_peaks(self, vec, cr):
        # TODO: Test this heuristic on other spectra?
        # First find all peaks regardless of height
        # Then only use the peaks above the mean
        # We specifically want the mean and not the median
        # as we need the inluence of the large peaks to be stronger
        # This should work well for "natural" spectra that have random small peaks
        # But how well does it work on well performing wavelength calibration images?
        vec -= np.nanpercentile(vec, 10)
        vec[~np.isfinite(vec)] = 0
        peaks, stats = signal.find_peaks(
            vec, width=self.peak_width, distance=self.window_width
        )
        prominence = stats["prominences"]
        # height = np.mean(prominence)
        height = np.median(prominence)
        # height = np.nanpercentile(vec, 20)
        Msg.info(self.name, f"Peak detection threshold height: {height}")
        peaks, _ = signal.find_peaks(
            vec, prominence=height, width=self.peak_width, distance=self.window_width
        )

        # Remove peaks at the edge
        peaks = peaks[
            (peaks >= self.window_width + 1)
            & (peaks < len(vec) - self.window_width - 1)
        ]

        # Manually remove bad areas of peaks
        if self.wl_set == "L3262":
            if self.chip == 3 and self.order == 2:
                peaks = peaks[peaks < 1000]
            elif self.chip == 3 and self.order == 3:
                peaks = peaks[peaks > 150]
                peaks = peaks[(peaks < 700) | (peaks > 1100)]
                peaks = peaks[(peaks < 1600) | (peaks > 1880)]
        elif self.wl_set == "L3340":
            if self.chip == 1 and self.order == 3:
                peaks = peaks[peaks > 200]
                peaks = peaks[(peaks < 300) | (peaks > 600)]
            elif self.chip == 3 and self.order == 2:
                peaks = peaks[peaks < 1200]
            elif self.chip == 3 and self.order == 3:
                peaks = peaks[peaks < 1500]
        elif self.wl_set == "L3426":
            if self.chip == 2 and self.order == 2:
                peaks = []
        elif self.wl_set == "M4318":
            if self.chip == 1 and self.order == 3:
                peaks = peaks[peaks > 400]

        if self.plot:
            plt.clf()
            plt.plot(vec)
            plt.plot(peaks, vec[peaks], "d")
            plt.savefig(f"cr2res_util_slit_curv_sky_peaks_c{self.chip}_o{self.order}.png")

        # Remove the offset, due to vec being a subset of extracted
        peaks += cr[0]
        return vec, peaks

    def _determine_curvature_single_line(
        self, original, extracted, peak, ycen, ycen_int, xwd, chip=0, order=0, npeak=0
    ):
        """
        Fit the curvature of a single peak in the spectrum

        This is achieved by fitting a model, that consists of gaussians
        in spectrum direction, that are shifted by the curvature in each row.

        Parameters
        ----------
        original : array of shape (nrows, ncols)
            whole input image
        peak : int
            column position of the peak
        ycen : array of shape (ncols,)
            row center of the order of the peak
        xwd : 2 tuple
            extraction width above and below the order center to use

        Returns
        -------
        tilt : float
            first order curvature
        shear : float
            second order curvature
        """
        _, ncol = original.shape

        # look at +- width pixels around the line
        # Extract short horizontal strip for each row in extraction width
        # Then fit a gaussian to each row, to find the center of the line
        x = peak + np.arange(-self.window_width, self.window_width + 1)
        x = x[(x >= 0) & (x < ncol)]
        xmin, xmax = x[0], x[-1] + 1

        # Look above and below the line center
        y = np.arange(-xwd[0], xwd[1] + 1)[:, None] - ycen[xmin:xmax][None, :]

        x = x[None, :]
        idx = self.make_index(ycen_int - xwd[0], ycen_int + xwd[1], xmin, xmax)
        img = original[idx]

        # Remove the background
        sl = np.ma.median(img, axis=1)
        sl = gaussian_filter1d(sl.data.ravel(), 10)
        img = img - sl[:, None]

        # Normalize the image
        img_compressed = np.ma.compressed(img)
        img_min = np.percentile(img_compressed, 5)
        img_max = np.percentile(img_compressed, 95)
        img -= img_min
        img /= img_max - img_min

        img[:] = exposure.equalize_hist(img.data, mask=~img.mask)
        img[np.isnan(img)] = np.ma.masked

        # Normalize the spectrum
        sp = extracted
        sp -= sp[xmin:xmax].min()
        sp /= sp[xmin:xmax].max()
        sp_x = np.arange(0, len(sp))

        peak_func = {
            "gaussian": self.gaussian,
            "lorentzian": self.lorentzian,
            "spectrum": self.collapsed,
        }
        peak_func = peak_func[self.peak_function]

        def model(coef):
            A, middle, sig, *curv = coef
            mu = middle + shift(curv)
            if self.peak_function in ["gaussian", "lorentzian"]:
                mod = peak_func(x, A, mu, sig)
            elif self.peak_function in ["spectrum"]:
                mod = peak_func(x, A, sp, sp_x - peak, mu)
            # mod = mod * sl[:, None]
            return mod

        def model_compressed(coef):
            return np.ma.compressed((model(coef) - img).ravel())

        A = 1  # np.nanpercentile(img_compressed, 95)
        sig = (xmax - xmin) / 4  # TODO
        if self.curv_degree == 1:
            shift = lambda curv: curv[0] * y
        elif self.curv_degree == 2:
            shift = lambda curv: (curv[0] + curv[1] * y) * y
        else:
            raise ValueError("Only curvature degrees 1 and 2 are supported")
        # res = least_squares(model, x0=[A, middle, sig, 0], loss="soft_l1", bounds=([0, xmin, 1, -10],[np.inf, xmax, xmax, 10]))
        x0 = [A, peak, sig] + [0] * self.curv_degree
        bounds = [(-np.inf, np.inf), (xmin, xmax), (0, 10 * sig), (-0.1, 0.1)]
        if self.curv_degree == 2:
            bounds += [(-1, 1)]
        bounds = np.array(bounds).T
        res = least_squares(
            model_compressed,
            x0=x0,
            method="trf",
            loss="soft_l1",
            f_scale=0.1,
            bounds=bounds,
        )

        # To get the ucnertainties of the fit
        # Do Moore-Penrose inverse discarding zero singular values.
        # See scipy curve_fit
        _, s, VT = svd(res.jac, full_matrices=False)
        threshold = np.finfo(float).eps * max(res.jac.shape) * s[0]
        s = s[s > threshold]
        VT = VT[:s.size]
        pcov = np.dot(VT.T / s**2, VT)
        unc = np.sqrt(np.diag(pcov))

        if self.curv_degree == 1:
            tilt, shear = res.x[3], 0
            # utilt, ushear = unc[3], 1
            model_data = model(res.x)
            utilt = unc[3]
            ushear = 1
        elif self.curv_degree == 2:
            tilt, shear = res.x[3], res.x[4]
            utilt, ushear = unc[3], unc[4]
        else:
            tilt, shear = 0, 0
            utilt, ushear = 1, 1

        if self.plot:
            nrows, _ = img.shape
            model_data = model(res.x)
            middle = nrows // 2
            y = np.arange(-middle, -middle + nrows)
            x = res.x[1] - xmin + (tilt + shear * y) * y
            y += middle
            vmin, vmax = 0, 1
            plt.clf()
            plt.subplot(121)
            plt.imshow(img, interpolation="none", origin="lower", vmin=vmin, vmax=vmax)
            plt.plot(x, y, "r")
            plt.subplot(122)
            plt.imshow(model_data, interpolation="none", origin="lower", vmin=vmin, vmax=vmax)
            plt.plot(x, y, "r")
            plt.suptitle(f"UNC: {utilt}")
            plt.savefig(f"cr2res_util_slit_curv_sky_c{chip}_o{order}_p{npeak}.png")

        return tilt, shear, utilt, ushear

    def _fit_curvature_single_order(self, peaks, tilt, shear):
        try:
            middle = np.median(tilt)
            sigma = np.percentile(tilt, (32, 68))
            sigma = middle - sigma[0], sigma[1] - middle
            mask = (tilt >= middle - 5 * sigma[0]) & (tilt <= middle + 5 * sigma[1])
            peaks, tilt, shear = peaks[mask], tilt[mask], shear[mask]

            coef_tilt = np.zeros(self.fit_degree + 1)
            res = least_squares(
                lambda coef: np.polyval(coef, peaks) - tilt,
                x0=coef_tilt,
                loss="arctan",
            )
            coef_tilt = res.x

            coef_shear = np.zeros(self.fit_degree + 1)
            res = least_squares(
                lambda coef: np.polyval(coef, peaks) - shear,
                x0=coef_shear,
                loss="arctan",
            )
            coef_shear = res.x

        except:
            coef_tilt = np.zeros(self.fit_degree + 1)
            coef_shear = np.zeros(self.fit_degree + 1)

        return coef_tilt, coef_shear, peaks

    def _determine_curvature_all_lines(
        self,
        original,
        extracted,
        orders,
        extraction_width,
        column_range,
        order_range,
        chip=0,
        order_nrs=None,
    ):
        ncol = original.shape[1]
        # Store data from all orders
        all_peaks = []
        all_tilt = []
        all_shear = []
        all_utilt = []
        all_ushear = []
        plot_vec = []

        n = order_range[1] - order_range[0]
        if order_nrs is None:
            order_nrs = np.arange(n)

        for j in range(n):
            Msg.debug(self.name, f"Processing Order {order_nrs[j]}")
            self.order = order_nrs[j]

            cr = column_range[j]
            xwd = extraction_width[j]
            ycen = np.polyval(orders[j], np.arange(ncol))
            ycen_int = ycen.astype(int)
            ycen -= ycen_int

            # Find peaks
            vec = extracted[j, cr[0] : cr[1]]
            vec, peaks = self._find_peaks(vec, cr)

            npeaks = len(peaks)
            Msg.info(self._name, f"{npeaks} peaks found in Order {order_nrs[j]}")

            # Determine curvature for each line seperately
            tilt = np.zeros(npeaks)
            shear = np.zeros(npeaks)
            utilt = np.zeros(npeaks)
            ushear = np.zeros(npeaks)

            mask = np.full(npeaks, True)
            for ipeak, peak in enumerate(peaks):
                Msg.debug(self.name, f"Processing Peak {ipeak}")
                try:
                    tilt[ipeak], shear[ipeak], utilt[ipeak], ushear[ipeak] = self._determine_curvature_single_line(
                        original,
                        extracted[j],
                        peak,
                        ycen,
                        ycen_int,
                        xwd,
                        chip=chip,
                        order=order_nrs[j],
                        npeak=ipeak,
                    )
                except RuntimeError:  # pragma: no cover
                    mask[ipeak] = False

            # Store results
            all_peaks += [peaks[mask]]
            all_tilt += [tilt[mask]]
            all_shear += [shear[mask]]
            all_utilt += [utilt[mask]]
            all_ushear += [ushear[mask]]
            plot_vec += [vec]
        return all_peaks, all_tilt, all_shear, all_utilt, all_ushear, plot_vec

    def fit(self, peaks, tilt, shear, utilt, ushear, order_range):
        n = order_range[1] - order_range[0]
        if self.mode == "1D":
            coef_tilt = np.zeros((n, self.fit_degree + 1))
            coef_shear = np.zeros((n, self.fit_degree + 1))
            for j in range(n):
                coef_tilt[j], coef_shear[j], _ = self._fit_curvature_single_order(
                    peaks[j], tilt[j], shear[j]
                )
        elif self.mode == "2D":
            x = np.concatenate(peaks)
            y = [np.full(len(p), i) for i, p in enumerate(peaks)]
            y = np.concatenate(y)
            z = np.concatenate(tilt)
            w = np.concatenate(utilt)
            coef_tilt = polyfit2d(x, y, z, w=w, degree=self.fit_degree, loss="soft_l1")

            zs = np.concatenate(shear)
            ws = np.concatenate(ushear)
            coef_shear = polyfit2d(x, y, zs, w=ws, degree=self.fit_degree, loss="arctan")

            if self.plot:
                plt.clf()
                orders = np.unique(y)
                zm = [np.median(z[y==o]) for o in orders]
                zstep = np.median(np.diff(zm)) * 10
                X, Y = np.meshgrid(
                    np.linspace(0, 2048, 100), orders
                )
                Z = np.polynomial.polynomial.polyval2d(X, Y, coef_tilt)
                for i, o in enumerate(orders):
                    l1 = plt.plot(x[y==o], z[y==o] + i * zstep, "+")
                    plt.plot(X[Y==o], Z[Y==o] + i * zstep, "--", color=l1[0]._color)
                plt.savefig(f"cr2res_util_slit_curv_sky_tilt_c{self.chip}.png")


        Msg.debug(self.name, f"Tilt Coef:\n{np.array2string(coef_tilt, precision=2)}")
        Msg.debug(self.name, f"Shear Coef:\n{coef_shear}")

        return coef_tilt, coef_shear

    def eval(self, peaks, order, coef_tilt, coef_shear):
        if self.mode == "1D":
            tilt = np.zeros(peaks.shape)
            shear = np.zeros(peaks.shape)
            for i in np.unique(order):
                idx = order == i
                tilt[idx] = np.polyval(coef_tilt[i], peaks[idx])
                shear[idx] = np.polyval(coef_shear[i], peaks[idx])
        elif self.mode == "2D":
            tilt = polyval2d(peaks, order, coef_tilt)
            shear = polyval2d(peaks, order, coef_shear)
        return tilt, shear

    def execute(
        self,
        extracted,
        original,
        orders,
        extraction_width,
        column_range,
        order_range,
        chip=0,
        order_nrs=None,
    ):

        (orders, extraction_width, column_range, order_range,) = self._fix_inputs(
            original.shape,
            orders,
            extraction_width,
            column_range,
            order_range,
        )

        peaks, tilt, shear, utilt, ushear, vec = self._determine_curvature_all_lines(
            original,
            extracted,
            orders,
            extraction_width,
            column_range,
            order_range,
            chip=chip,
            order_nrs=order_nrs,
        )

        coef_tilt, coef_shear = self.fit(peaks, tilt, shear, utilt, ushear, order_range)

        iorder, ipeaks = np.indices(extracted.shape)
        tilt, shear = self.eval(ipeaks, iorder, coef_tilt, coef_shear)

        return tilt, shear

    @staticmethod
    def gaussian(x, A, mu, sig):
        """
        A: height
        mu: offset from central line
        sig: standard deviation
        """
        return A * np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0)))

    @staticmethod
    def lorentzian(x, A, x0, mu):
        """
        A: height
        x0: offset from central line
        mu: width of lorentzian
        """
        return A * mu / ((x - x0) ** 2 + 0.25 * mu ** 2)

    @staticmethod
    def collapsed(x, A, sp, sp_x, mu):
        interp = interp1d(sp_x, sp, bounds_error=False, fill_value="extrapolate")
        res = A * interp(x - mu)
        return res


if __name__ == "__main__":
    fs = FrameSet("/scratch/ptah/anwe5599/CRIRES/2022-12-23_M4318/extr/slit_curv.sof")
    settings = {"degree_x": 2, "degree_y": 1, "detector": 1}
    module = cr2res_util_slit_curv_sky()
    module.run(fs, settings)