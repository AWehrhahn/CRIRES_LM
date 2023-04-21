from typing import Any, Dict
from os.path import basename

from scipy.special import binom
from scipy.linalg import lstsq

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

import numpy as np
from tqdm import tqdm
from cpl.core import Msg
from cpl.ui import Frame, FrameSet, ParameterList, ParameterValue, PyRecipe


def _get_coeff_idx(coeff):
    idx = np.indices(coeff.shape)
    idx = idx.T.swapaxes(0, 1).reshape((-1, 2))
    # degree = coeff.shape
    # idx = [[i, j] for i, j in product(range(degree[0]), range(degree[1]))]
    # idx = np.asarray(idx)
    return idx


def _scale(x, y):
    # Normalize x and y to avoid huge numbers
    # Mean 0, Variation 1
    offset_x, offset_y = np.mean(x), np.mean(y)
    norm_x, norm_y = np.std(x), np.std(y)
    if norm_x == 0:
        norm_x = 1
    if norm_y == 0:
        norm_y = 1
    x = (x - offset_x) / norm_x
    y = (y - offset_y) / norm_y
    return x, y, (norm_x, norm_y), (offset_x, offset_y)


def _unscale(x, y, norm, offset):
    x = x * norm[0] + offset[0]
    y = y * norm[1] + offset[1]
    return x, y


def polyvander2d(x, y, degree):
    # A = np.array([x ** i * y ** j for i, j in idx], dtype=float).T
    A = np.polynomial.polynomial.polyvander2d(x, y, degree)
    return A


def polyscale2d(coeff, scale_x, scale_y, copy=True):
    if copy:
        coeff = np.copy(coeff)
    idx = _get_coeff_idx(coeff)
    for k, (i, j) in enumerate(idx):
        coeff[i, j] /= scale_x ** i * scale_y ** j
    return coeff


def polyshift2d(coeff, offset_x, offset_y, copy=True):
    if copy:
        coeff = np.copy(coeff)
    idx = _get_coeff_idx(coeff)
    # Copy coeff because it changes during the loop
    coeff2 = np.copy(coeff)
    for k, m in idx:
        not_the_same = ~((idx[:, 0] == k) & (idx[:, 1] == m))
        above = (idx[:, 0] >= k) & (idx[:, 1] >= m) & not_the_same
        for i, j in idx[above]:
            b = binom(i, k) * binom(j, m)
            sign = (-1) ** ((i - k) + (j - m))
            offset = offset_x ** (i - k) * offset_y ** (j - m)
            coeff[k, m] += sign * b * coeff2[i, j] * offset
    return coeff


def plot2d(x, y, z, coeff, title=None, fname=None):
    # regular grid covering the domain of the data
    if x.size > 500:
        choice = np.random.choice(x.size, size=500, replace=False)
    else:
        choice = slice(None, None, None)
    x, y, z = x[choice], y[choice], z[choice]
    XS, YS = np.linspace(np.min(x), np.max(x), 20), np.linspace(np.min(y), np.max(y), 20)
    X, Y = np.meshgrid(XS, YS)
    Z = np.polynomial.polynomial.polyval2d(X, Y, coeff)
    # Plotly
    fig = go.Figure(data=[
            go.Surface(x=XS, y=YS, z=Z), 
            go.Scatter3d(x=x, y=y, z=z, mode="markers", marker=dict(color="green"))
        ])
    if fname is None or fname is True:
        fig.show()
    else:
        fig.write_html(fname, include_plotlyjs=True)

    # # Matplotlib.pyplot
    # fig = plt.figure()
    # ax = fig.gca(projection="3d")
    # ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
    # ax.scatter(x, y, z, c="r", s=50)
    # plt.xlabel("X")
    # plt.ylabel("Y")
    # ax.set_zlabel("Z")
    # if title is not None:
    #     plt.title(title)
    # # ax.axis("equal")
    # # ax.axis("tight")
    # if fname is None or fname is True:
    #     plt.show()
    # else:
    #     plt.savefig(fname)
            

polyval2d = np.polynomial.polynomial.polyval2d

def polyfit2d(
    x, y, z, degree=1, max_degree=None, scale=True, plot=False, plot_title=None
):
    """A simple 2D plynomial fit to data x, y, z
    The polynomial can be evaluated with numpy.polynomial.polynomial.polyval2d

    Parameters
    ----------
    x : array[n]
        x coordinates
    y : array[n]
        y coordinates
    z : array[n]
        data values
    degree : int, optional
        degree of the polynomial fit (default: 1)
    max_degree : {int, None}, optional
        if given the maximum combined degree of the coefficients is limited to this value
    scale : bool, optional
        Wether to scale the input arrays x and y to mean 0 and variance 1, to avoid numerical overflows.
        Especially useful at higher degrees. (default: True)
    plot : bool, optional
        wether to plot the fitted surface and data (slow) (default: False)

    Returns
    -------
    coeff : array[degree+1, degree+1]
        the polynomial coefficients in numpy 2d format, i.e. coeff[i, j] for x**i * y**j
    """
    # Flatten input
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    z = np.asarray(z).ravel()

    # Removed masked values
    mask = ~(np.ma.getmask(z) | np.ma.getmask(x) | np.ma.getmask(y))
    x, y, z = x[mask].ravel(), y[mask].ravel(), z[mask].ravel()

    if scale:
        x, y, norm, offset = _scale(x, y)

    # Create combinations of degree of x and y
    # usually: [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), ....]
    if np.isscalar(degree):
        degree = (int(degree), int(degree))
    assert len(degree) == 2, "Only 2D polynomials can be fitted"
    degree = [int(degree[0]), int(degree[1])]
    # idx = [[i, j] for i, j in product(range(degree[0] + 1), range(degree[1] + 1))]
    coeff = np.zeros((degree[0] + 1, degree[1] + 1))
    idx = _get_coeff_idx(coeff)

    # Calculate elements 1, x, y, x*y, x**2, y**2, ...
    A = polyvander2d(x, y, degree)

    # We only want the combinations with maximum order COMBINED power
    if max_degree is not None:
        mask = idx[:, 0] + idx[:, 1] <= int(max_degree)
        idx = idx[mask]
        A = A[:, mask]

    # Do least squares fit
    C, *_ = lstsq(A, z)

    # Reorder coefficients into numpy compatible 2d array
    for k, (i, j) in enumerate(idx):
        coeff[i, j] = C[k]

    # # Backup copy of coeff
    if scale:
        coeff = polyscale2d(coeff, *norm, copy=False)
        coeff = polyshift2d(coeff, *offset, copy=False)

    if plot:  # pragma: no cover
        if scale:
            x, y = _unscale(x, y, norm, offset)
        plot2d(x, y, z, coeff, title=plot_title, fname=plot)

    return coeff

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
            model_sky="CAL_WAVE_MOLECFIT_MODEL_SKY",
            mapping_sky="CAL_WAVE_MOLECFIT_MAPPING_SKY",
            model_star="CAL_WAVE_MOLECFIT_MODEL_STAR",
            mapping_star="CAL_WAVE_MOLECFIT_MAPPING_STAR",
        )

        if len(frames["spectrum"]) != 1:
            raise ValueError("Expected exactly 1 spectrum frame")
        if len(frames["model_sky"]) != 1:
            raise ValueError("Expected exactly 1 model frame")
        if len(frames["mapping_sky"]) != 1:
            raise ValueError("Expected exactly 1 mapping frame")
        if len(frames["model_star"]) != 1:
            raise ValueError("Expected exactly 1 model frame")
        if len(frames["mapping_star"]) != 1:
            raise ValueError("Expected exactly 1 mapping frame")

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
        if wl_set == "L3262":
            assignment = {
                (1, 2): "STAR",
                (1, 3): "SKY",
                (1, 4): "SKY",
                (1, 5): "STAR",
                (1, 6): "STAR",
                (1, 7): "STAR",
                (2, 2): "SKY",
                (2, 3): "SKY",
                (2, 4): "STAR",
                (2, 5): "STAR",
                (2, 6): "STAR",
                (2, 7): "STAR",
                (3, 2): "SKY",
                (3, 3): "SKY",
                (3, 4): "STAR",
                (3, 5): "STAR",
                (3, 6): "STAR",
                (3, 7): "STAR",
            }

        for chip in tqdm(chips, desc="Detector"):
            ext = get_chip_extension(chip)
            data = spectrum[ext].data
            orders = np.unique(mapping_sky[mapping_sky["CHIP"] == chip]["ORDER"])
            xs, ys, zs, os = [], [], [], []
            for order in tqdm(orders, leave=False, desc="Order"):
                if assignment[(chip, order)] == "STAR":
                    mapping = mapping_star
                    model = model_star
                elif assignment[(chip, order)] == "SKY":
                    mapping = mapping_sky
                    model = model_sky
                else:
                    raise ValueError("Unexpected Chip/Order combination")

                idx = (mapping["CHIP"] == chip) & (mapping["ORDER"] == order)
                idx = mapping[idx]["MOLECFIT"][0]
                column = get_spectrum_table_header(order, 1, "WL")
                data[column] = model[model["CHIP"] == idx]["MLAMBDA"] * 1000

                zs += [data[column]]
                ys += [np.full_like(zs[-1], order)]
                xs += [np.arange(len(zs[-1]))]
                os += [order]

            xsr = np.asarray(xs).ravel()
            ysr = np.asarray(ys).ravel()
            zsr = np.asarray(zs).ravel()
            mask = zs != 0
            xsr, ysr, zsr = xsr[mask], ysr[mask], zsr[mask]
            coef = polyfit2d(xsr, ysr, zsr, degree=(2, 3), plot=f"cr2res_wave_molecfit_plot_chip{chip}_2D.html", plot_title=f"CHIP {chip}")
            ws = polyval2d(xs, ys, coef)

            for i, order in enumerate(orders):
                idx = (mapping["CHIP"] == chip) & (mapping["ORDER"] == order)
                idx = mapping[idx]["MOLECFIT"][0]

                wave_star = model_star[model_star["CHIP"] == idx]["MLAMBDA"] * 1000
                mask_star = wave_star != 0
                spec_star = model_star[model_star["CHIP"] == idx]["MFLUX"]
                extr_star = model_star[model_star["CHIP"] == idx]["FLUX"]

                wave_sky = model_sky[model_sky["CHIP"] == idx]["MLAMBDA"] * 1000
                mask_sky = wave_sky != 0
                spec_sky = model_sky[model_sky["CHIP"] == idx]["MFLUX"]
                extr_sky = model_sky[model_sky["CHIP"] == idx]["FLUX"]
                plt.clf()
                plt.subplot(211)
                plt.title("STAR")
                plt.plot(wave_star[mask_star], extr_star[mask_star], label="Extracted")
                plt.plot(wave_star[mask_star], spec_star[mask_star], "--", label="Model")
                plt.plot(ws[i], extr_star, label="Corrected")

                plt.subplot(212)
                plt.title("SKY")
                plt.plot(wave_sky[mask_sky], extr_sky[mask_sky], label="Extracted")
                plt.plot(wave_sky[mask_sky], spec_sky[mask_sky], "--", label="Model")
                plt.plot(ws[i], extr_sky, label="Corrected")

                plt.legend()
                plt.suptitle(f"CHIP: {chip} ORDER: {order:02}")
                plt.savefig(f"cr2res_wave_molecfit_apply_c{chip}_o{order:02}.png", dpi=600)

        # Overwrite the results
        spectrum.writeto(basename(spectrum_fname), overwrite=True)

        result = FrameSet(
            [
                Frame(spectrum_fname, tag="UTIL_CALIB_EXTRACT_1D"),
            ]
        )
        return result


if __name__ == "__main__":
    recipe = cr2res_wave_molecfit_apply()
    sof = "/scratch/ptah/anwe5599/CRIRES/2022-11-29_OLD/extr/molecfit_apply.sof"
    sof = FrameSet(sof)
    res = recipe.run(sof, {})
    pass