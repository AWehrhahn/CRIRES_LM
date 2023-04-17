import argparse
import os
import sys

import numpy as np
from astropy.io import fits
from numpy.polynomial.polynomial import polyval2d
from scipy import signal
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from tqdm import tqdm

try:
    from .util import fix_parameters, make_index, polyfit2d
except ImportError:
    from util import fix_parameters, make_index, polyfit2d


class Curvature:
    def __init__(
        self,
        orders,
        extraction_width=0.5,
        column_range=None,
        order_range=None,
        window_width=9,
        peak_threshold=10,
        peak_width=1,
        fit_degree=2,
        sigma_cutoff=3,
        mode="1D",
        peak_function="gaussian",
        curv_degree=2,
    ):
        self.orders = orders
        self.extraction_width = extraction_width
        self.column_range = column_range
        if order_range is None:
            order_range = (0, self.nord)
        self.order_range = order_range
        self.window_width = window_width
        self.threshold = peak_threshold
        self.peak_width = peak_width
        self.fit_degree = fit_degree
        self.sigma_cutoff = sigma_cutoff
        self.mode = mode
        self.curv_degree = curv_degree
        self.peak_function = peak_function

        if self.mode == "1D":
            # fit degree is an integer
            if not np.isscalar(self.fit_degree):
                self.fit_degree = self.fit_degree[0]
        elif self.mode == "2D":
            # fit degree is a 2 tuple
            if np.isscalar(self.fit_degree):
                self.fit_degree = (self.fit_degree, self.fit_degree)

    @property
    def nord(self):
        return self.orders.shape[0]

    @property
    def n(self):
        return self.order_range[1] - self.order_range[0]

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        if value not in ["1D", "2D"]:
            raise ValueError(
                f"Value for 'mode' not understood. Expected one of ['1D', '2D'] but got {value}"
            )
        self._mode = value

    def _fix_inputs(self, original):
        orders = self.orders
        extraction_width = self.extraction_width
        column_range = self.column_range

        nrow, ncol = original.shape
        nord = len(orders)

        extraction_width, column_range, orders = fix_parameters(
            extraction_width, column_range, orders, nrow, ncol, nord
        )

        self.column_range = column_range[self.order_range[0] : self.order_range[1]]
        self.extraction_width = extraction_width[
            self.order_range[0] : self.order_range[1]
        ]
        self.orders = orders[self.order_range[0] : self.order_range[1]]
        self.order_range = (0, self.n)

    def _find_peaks(self, vec, cr):
        # This should probably be the same as in the wavelength calibration
        vec -= np.nanmedian(vec)
        height = np.nanpercentile(vec, 68) * self.threshold
        vec[~np.isfinite(vec)] = 0
        peaks, _ = signal.find_peaks(
            vec, prominence=height, width=self.peak_width, distance=self.window_width
        )

        # Remove peaks at the edge
        peaks = peaks[
            (peaks >= self.window_width + 1)
            & (peaks < len(vec) - self.window_width - 1)
        ]
        # Remove the offset, due to vec being a subset of extracted
        peaks += cr[0]
        return vec, peaks

    def _determine_curvature_single_line(
        self, original, extracted, peak, ycen, ycen_int, xwd
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
        idx = make_index(ycen_int - xwd[0], ycen_int + xwd[1], xmin, xmax)
        img = original[idx]
        img_compressed = np.ma.compressed(img)

        img_min = np.percentile(img_compressed, 5)
        img_max = np.percentile(img_compressed, 95)
        img -= img_min
        img /= img_max - img_min
        # img = np.ma.clip(img, 0, 1)

        sl = np.ma.mean(img, axis=1)
        sl = sl[:, None]
        sp = extracted
        sp -= sp[xmin:xmax].min()
        sp /= sp[xmin:xmax].max()
        sp_x = np.arange(0, len(sp))

        peak_func = {
            "gaussian": gaussian,
            "lorentzian": lorentzian,
            "spectrum": collapsed,
        }
        peak_func = peak_func[self.peak_function]

        def model(coef):
            A, middle, sig, *curv = coef
            mu = middle + shift(curv)
            if self.peak_function in ["gaussian", "lorentzian"]:
                mod = peak_func(x, A, mu, sig)
            elif self.peak_function in ["spectrum"]:
                mod = peak_func(x, A, sp, sp_x - peak, mu)
            # mod *= sl
            return (mod - img).ravel()

        def model_compressed(coef):
            return np.ma.compressed(model(coef))

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
        bounds = [(-np.inf, np.inf), (xmin, xmax), (0, 10 * sig), (-2, 2)]
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

        if self.curv_degree == 1:
            tilt, shear = res.x[3], 0
        elif self.curv_degree == 2:
            tilt, shear = res.x[3], res.x[4]
        else:
            tilt, shear = 0, 0

        return tilt, shear

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

    def _determine_curvature_all_lines(self, original, extracted):
        ncol = original.shape[1]
        # Store data from all orders
        all_peaks = []
        all_tilt = []
        all_shear = []
        plot_vec = []

        for j in tqdm(range(self.n), desc="Order"):
            cr = self.column_range[j]
            xwd = self.extraction_width[j]
            ycen = np.polyval(self.orders[j], np.arange(ncol))
            ycen_int = ycen.astype(int)
            ycen -= ycen_int

            # Find peaks
            vec = extracted[j, cr[0] : cr[1]]
            vec, peaks = self._find_peaks(vec, cr)

            npeaks = len(peaks)
            print(f"{npeaks} found in Order {j}")

            # Determine curvature for each line seperately
            tilt = np.zeros(npeaks)
            shear = np.zeros(npeaks)
            mask = np.full(npeaks, True)
            for ipeak, peak in tqdm(
                enumerate(peaks), total=len(peaks), desc="Peak", leave=False
            ):
                try:
                    tilt[ipeak], shear[ipeak] = self._determine_curvature_single_line(
                        original, extracted[j], peak, ycen, ycen_int, xwd
                    )
                except RuntimeError:  # pragma: no cover
                    mask[ipeak] = False

            # Store results
            all_peaks += [peaks[mask]]
            all_tilt += [tilt[mask]]
            all_shear += [shear[mask]]
            plot_vec += [vec]
        return all_peaks, all_tilt, all_shear, plot_vec

    def fit(self, peaks, tilt, shear):
        if self.mode == "1D":
            coef_tilt = np.zeros((self.n, self.fit_degree + 1))
            coef_shear = np.zeros((self.n, self.fit_degree + 1))
            for j in range(self.n):
                coef_tilt[j], coef_shear[j], _ = self._fit_curvature_single_order(
                    peaks[j], tilt[j], shear[j]
                )
        elif self.mode == "2D":
            x = np.concatenate(peaks)
            y = [np.full(len(p), i) for i, p in enumerate(peaks)]
            y = np.concatenate(y)
            z = np.concatenate(tilt)
            coef_tilt = polyfit2d(x, y, z, degree=self.fit_degree, loss="arctan")

            z = np.concatenate(shear)
            coef_shear = polyfit2d(x, y, z, degree=self.fit_degree, loss="arctan")

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

    def execute(self, extracted, original):

        _, ncol = original.shape

        self._fix_inputs(original)

        peaks, tilt, shear, vec = self._determine_curvature_all_lines(
            original, extracted
        )

        coef_tilt, coef_shear = self.fit(peaks, tilt, shear)

        iorder, ipeaks = np.indices(extracted.shape)
        tilt, shear = self.eval(ipeaks, iorder, coef_tilt, coef_shear)

        return tilt, shear


# TODO allow other line shapes
def gaussian(x, A, mu, sig):
    """
    A: height
    mu: offset from central line
    sig: standard deviation
    """
    return A * np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0)))


def lorentzian(x, A, x0, mu):
    """
    A: height
    x0: offset from central line
    mu: width of lorentzian
    """
    return A * mu / ((x - x0) ** 2 + 0.25 * mu ** 2)


def collapsed(x, A, sp, sp_x, mu):
    interp = interp1d(sp_x, sp, bounds_error=False, fill_value="extrapolate")
    res = A * interp(x - mu)
    return res


if len(sys.argv) > 1:
    parser = argparse.ArgumentParser()
    parser.add_argument("sof", help="Set Of Files")
    parser.add_argument(
        "--output-dir",
        help="he directory where the product files should be finally moved to (all products are first created in the current dir)",
    )
    args = parser.parse_args()
    sof = args.sof
    outfolder = args["output-dir"]
    if outfolder is None:
        outfolder = os.getcwd()
else:
    sof = "/scratch/ptah/anwe5599/CRIRES/2022-11-29/extr/slit_curv.sof"
    outfolder = "/scratch/ptah/anwe5599/CRIRES/2022-11-29/extr/"

with open(sof) as f:
    lines = f.readlines()
lines = [l.split() for l in lines]

tw_fname = [l[0] for l in lines if l[1] == "CAL_FLAT_TW"][0]
science_fname = [l[0] for l in lines if l[1] == "UTIL_CALIB"][0]
blaze_fname = [l[0] for l in lines if l[1] == "CAL_FLAT_EXTRACT_1D"][0]
extract_fname = [l[0] for l in lines if l[1] == "UTIL_CALIB_EXTRACT_1D"][0]
model_fname = [l[0] for l in lines if l[1] == "UTIL_CALIB_EXTRACT_MODEL"][0]

tw = fits.open(tw_fname)
science = fits.open(science_fname)
blaze = fits.open(blaze_fname)
extract = fits.open(extract_fname)
model = fits.open(model_fname)

tilts = [0, 0, 0]
shears = [0, 0, 0]

for chip in [1, 2, 3]:
    ext = f"CHIP{chip}.INT1"
    exterr = f"CHIP{chip}ERR.INT1"
    tw_data = tw[ext].data
    data = science[ext].data
    err = science[exterr].data
    blaze_data = blaze[ext].data
    extract_data = extract[ext].data
    model_data = model[ext].data

    order_traces = []
    extraction_width = []
    column_range = []
    upper_ycen = []
    lower_ycen = []
    orders = np.sort(list(set(tw_data["Order"])))

    for order in orders:
        idx = tw_data["Order"] == order
        x = np.arange(1, 2048 + 1)
        upper = np.polyval(tw_data[idx]["Upper"][0][::-1], x)
        lower = np.polyval(tw_data[idx]["Lower"][0][::-1], x)
        middle = np.polyval(tw_data[idx]["All"][0][::-1], x)

        height_upp = int(np.ceil(np.max(upper - middle)))
        height_low = int(np.ceil(np.max(middle - lower)))

        middle_int = middle.astype(int)
        upper_ycen += [middle_int + height_upp]
        lower_ycen += [middle_int - height_low]

        order_traces += [tw_data[idx]["All"][0][::-1]]
        column_range += [[10, 2048 - 10]]
        extraction_width += [[height_upp, height_low]]

    order_traces = np.asarray(order_traces)
    column_range = np.asarray(column_range)
    extraction_width = np.asarray(extraction_width)

    spectrum = np.array([extract_data[f"{order:02}_01_SPEC"] for order in orders])

    for i, order in enumerate(orders):
        # Reject outliers
        idx_data = make_index(lower_ycen[i], upper_ycen[i], 0, 2048)
        relative = data[idx_data] - model_data[idx_data]
        mask = np.isfinite(relative)
        std = 1.5 * np.median(np.abs(np.median(relative[mask]) - relative[mask]))
        mask &= np.abs(relative) < 10 * std
        # Need complex indexing to actually set the values to nan
        data[idx_data[0][~mask], idx_data[1][~mask]] = np.nan

        # Correct for the blaze
        blaze_spec = blaze_data[f"{order:02}_01_SPEC"]
        spectrum[i] /= blaze_spec
        spectrum[i][np.isnan(spectrum[i])] = 0
        
        # Smooth spectrum for easier peak detection
        spectrum[i] = gaussian_filter1d(spectrum[i], 3)

        # Limit the size of the extraction width
        # to avoid the interorder area
        extraction_width[i, 0] -= 10
        extraction_width[i, 1] -= 10

    module = Curvature(
        order_traces,
        extraction_width,
        column_range=column_range,
        curv_degree=1,
        peak_threshold=1.5,
        window_width=21,
        fit_degree=(1, 1),
        mode="2D",
        peak_function="spectrum",
    )
    data = np.ma.array(data, mask=~np.isfinite(data))
    tilt, shear = module.execute(spectrum, data)

    tilts[chip - 1] = tilt
    shears[chip - 1] = shear

print("Saving the results to the trace wave file")

# Create output
for chip in [1, 2, 3]:
    ext = f"CHIP{chip}.INT1"
    tw_data = tw[ext].data
    orders = np.sort(list(set(tw_data["Order"])))
    x = np.arange(1, 2049)
    for i, order in enumerate(orders):
        t = tilts[chip - 1][i]
        s = shears[chip - 1][i]
        ycen = np.polyval(tw_data[idx]["All"][0][::-1], x)
        # Convert to the global reference system
        t -= 2 * ycen * s

        # Fit overarching polynomial
        ct = np.polyfit(x, t, 2)
        cs = np.polyfit(x, s, 2)

        # Write the data back to the fits file
        # The indexing is chosen such that the data is actually
        # stored in the tw object and not lost
        idx = tw_data["Order"] == order
        tw[ext].data["SlitPolyA"][idx] = [0, 1, 0]
        tw[ext].data["SlitPolyB"][idx] = ct[::-1]
        tw[ext].data["SlitPolyC"][idx] = cs[::-1]

# The recipe overwrites the existing tracewave
# With the new information
tw.writeto(tw_fname, overwrite=True)
