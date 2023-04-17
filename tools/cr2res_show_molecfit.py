import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
from scipy.optimize import least_squares
from scipy.special import binom
from scipy.linalg import lstsq, solve, solve_banded
from tqdm import tqdm

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


def plot2d(x, y, z, coeff, title=None):
    # regular grid covering the domain of the data
    if x.size > 500:
        choice = np.random.choice(x.size, size=500, replace=False)
    else:
        choice = slice(None, None, None)
    x, y, z = x[choice], y[choice], z[choice]
    X, Y = np.meshgrid(
        np.linspace(np.min(x), np.max(x), 20), np.linspace(np.min(y), np.max(y), 20)
    )
    Z = np.polynomial.polynomial.polyval2d(X, Y, coeff)
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
    ax.scatter(x, y, z, c="r", s=50)
    plt.xlabel("X")
    plt.ylabel("Y")
    ax.set_zlabel("Z")
    if title is not None:
        plt.title(title)
    # ax.axis("equal")
    # ax.axis("tight")
    # plt.show()
    plt.savefig("test.png")


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
        plot2d(x, y, z, coeff, title=plot_title)

    return coeff

molecfit_fname_sky = "/scratch/ptah/anwe5599/CRIRES/2022-11-29/extr/BEST_FIT_MODEL_SKY.fits"
molecfit_fname_star = "/scratch/ptah/anwe5599/CRIRES/2022-11-29/extr/BEST_FIT_MODEL_STAR.fits"
extracted_fname = "/scratch/ptah/anwe5599/CRIRES/2022-11-29/extr/cr2res_util_combine_sky_extr1D.fits"
mapping_fname_sky = "/scratch/ptah/anwe5599/CRIRES/2022-11-29/extr/MAPPING_SKY.fits"
mapping_fname_star = "/scratch/ptah/anwe5599/CRIRES/2022-11-29/extr/MAPPING_STAR.fits"

mf_hdu_star = fits.open(molecfit_fname_star)
mf_hdu_sky = fits.open(molecfit_fname_sky)
ex_hdu = fits.open(extracted_fname)
map_hdu_star = fits.open(mapping_fname_star)
mapping_star = map_hdu_star[1].data
map_hdu_sky = fits.open(mapping_fname_sky)
mapping_sky = map_hdu_sky[1].data


chips = np.unique(mapping_sky["CHIP"])

# x = np.tile(np.arange(2048), 6)
# y = mf_hdu[1].data["chip"]
# z = mf_hdu[1].data["mlambda"] * 1000

# coef = polyfit2d(x, y, z, degree=1, plot=True)
# Z = np.polynomial.polynomial.polyval2d(x, y, coef)
# mf_hdu[1].data["mlambda"] = Z / 1000

for chip in tqdm(chips):
    ext = f"CHIP{chip}.INT1"
    # ex_orders = np.sort([int(n[:2]) for n in ex_hdu[ext].data.names if n[-4:] == "SPEC"])
    orders = np.unique(mapping_sky[mapping_sky["CHIP"] == chip]["ORDER"])

    for order in tqdm(orders):
        mf_idx_star = (mapping_star["CHIP"] == chip) & (mapping_star["ORDER"] == order)
        mf_idx_star = mf_hdu_star[1].data["chip"] == mapping_star["MOLECFIT"][mf_idx_star][0]
        mf_data_star = mf_hdu_star[1].data[mf_idx_star]
        mf_spec_star = mf_data_star["mflux"]
        mf_wave_star = mf_data_star["mlambda"] * 1000

        mf_idx_sky = (mapping_sky["CHIP"] == chip) & (mapping_sky["ORDER"] == order)
        mf_idx_sky = mf_hdu_sky[1].data["chip"] == mapping_sky["MOLECFIT"][mf_idx_sky][0]
        mf_data_sky = mf_hdu_sky[1].data[mf_idx_sky]
        mf_spec_sky = mf_data_sky["mflux"]
        mf_wave_sky = mf_data_sky["mlambda"] * 1000

        ex_wave_star = mf_data_star["lambda"] * 1000
        ex_spec_star = mf_data_star["flux"]

        ex_wave_sky = mf_data_sky["lambda"] * 1000
        ex_spec_sky = mf_data_sky["flux"]

        mask_sky = mf_wave_sky != 0
        mask_star = mf_wave_star != 0
        # if chip == 1 and order == 2:
        #     mask[mf_wave < 3900] = False
        

        plt.clf()
        plt.subplot(211)
        plt.title("STAR")
        plt.plot(mf_wave_star[mask_star], ex_spec_star[mask_star], label="Extracted")
        plt.plot(mf_wave_star[mask_star], mf_spec_star[mask_star], "--", label="Model")
        plt.subplot(212)
        plt.title("SKY")
        plt.plot(mf_wave_sky[mask_sky], ex_spec_sky[mask_sky], label="Extracted")
        plt.plot(mf_wave_sky[mask_sky], mf_spec_sky[mask_sky], "--", label="Model")
        plt.legend()
        plt.suptitle(f"CHIP: {chip} ORDER: {order:02}")
        plt.savefig(f"test_c{chip}_o{order:02}.png", dpi=600)

        pass

