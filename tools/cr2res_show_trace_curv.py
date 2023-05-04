#!/usr/bin/env python3
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

from skimage import exposure

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

def compare(fname_trace, fname_img=None, outfile=None, dpi=180, normalize=True, bpm=None):
    """ compare img and trace """
    trace = fits.open(fname_trace)
    if fname_img:
        img = fits.open(fname_img)
        linecol = "w"
    else:
        linecol = "k"
    if bpm is not None:
        bpm = fits.open(bpm)

    X = np.arange(2048)
    FIG = plt.figure(figsize=(15, 5))

    for det in [1, 2, 3]:
        ax = FIG.add_subplot(1, 3, det)
        ax.set_xticks([])
        ax.set_yticks([])

        try:
            tdata = trace['CHIP%d.INT1'%det].data
        except:
            print("extension %s is missing, skipping." % i)
            continue
        if tdata is None:
            print("Data for CHIP%s is empty, skipping." % i)
            continue

        if fname_img:
            imgdata = img['CHIP%d.INT1'%det].data
            imgdata = np.nan_to_num(imgdata)
            if bpm is not None:
                bpmdata = bpm["CHIP%d.INT1" % det].data
                if bpmdata is not None:
                    imgdata[bpmdata != 0] = 0

            if normalize:
                for t in tdata:
                    upper = t["Upper"]
                    lower = t["Lower"]
                    alla = t["All"]
                    middle = np.polyval(alla[::-1], X)
                    upper = np.polyval(upper[::-1], X)
                    lower = np.polyval(lower[::-1], X)
                    up = int(np.ceil(np.min(upper - middle)))
                    low = int(np.ceil(np.min(middle - lower)))
                    middle = np.asarray(middle, dtype=int)
                    idx = make_index(middle - low, middle + up, 0, 2048)
                    vmin, vmax = np.nanpercentile(imgdata[idx], (5, 95))
                    mask = (imgdata[idx] > vmin) & (imgdata[idx] < vmax)
                    imgdata[idx] = exposure.equalize_hist(imgdata[idx], mask=mask)
                vmin, vmax = 0, 1
            else:
                vmin, vmax = np.percentile(imgdata[imgdata != 0], (5, 95))
                vmax += (vmax-vmin)*0.4

            ax.imshow(imgdata, origin="lower", vmin=vmin, vmax=vmax,
                cmap='plasma')

        for t in tdata:
            upper = t["Upper"]
            lower = t["Lower"]
            alla = t["All"]
            order = t["Order"]
            ca = t["SlitPolyA"]
            cb = t["SlitPolyB"]
            cc = t["SlitPolyC"]

            upper = np.polyval(upper[::-1], X)
            ax.plot(X, upper, ":" + linecol)

            lower = np.polyval(lower[::-1], X)
            ax.plot(X, lower, ":" + linecol)

            middle = np.polyval(alla[::-1], X)
            ax.plot(X, middle, "--" + linecol)

            i1 = tdata[tdata["order"] == order]["Slitfraction"][:, 1]
            i2 = tdata[tdata["order"] == order]["All"]
            coeff = [np.interp(0.5, i1, i2[:, k]) for k in range(i2.shape[1])]


            for i in range(30, 2048, 100):
                ew = [int(middle[i] - lower[i]), int(upper[i] - middle[i])]
                x = np.zeros(ew[0] + ew[1] + 1)
                y = np.arange(-ew[0], ew[1] + 1).astype(float)

                # Evaluate the curvature polynomials to get coefficients
                a = np.polyval(ca[::-1], i)
                b = np.polyval(cb[::-1], i)
                c = np.polyval(cc[::-1], i)
                yc = np.polyval(coeff[::-1], i)

                # Shift polynomials to the local frame of reference
                a = a - i + yc * b + yc * yc * c
                b += 2 * yc * c

                for j, yt in enumerate(y):
                    x[j] = i + yt * b + yt ** 2 * c
                y += middle[i]
                plt.plot(x, y, "-"+linecol)

            if np.isnan(middle[1024]):
                continue
            ax.text(
                500,
                middle[1024],
                "order: %s\ntrace: %s" % (t["order"], t["TraceNb"]),
                color=linecol,
                horizontalalignment="center",
                verticalalignment="center",
                size=9,
            )
        ax.axis((1,2048,1,2048))

    FIG.tight_layout(pad=0.02)
    if outfile is None:
        outfile = fname_trace.replace(".fits", ".png")
    print(f"Saving plot: {outfile}")
    plt.savefig(outfile, dpi=dpi)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("tracewave")
    parser.add_argument("image")
    parser.add_argument("--out")
    parser.add_argument("--bpm")
    parser.add_argument("--dpi", default=600)
    parser.add_argument("--normalize", default=True)
    args = parser.parse_args()

    compare(args.tracewave, args.image, outfile=args.out, dpi=args.dpi, normalize=args.normalize, bpm=args.bpm)
