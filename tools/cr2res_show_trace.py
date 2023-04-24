#!/usr/bin/env python3
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt


def compare(fname_trace, fname_img=None, fname_spec=None, figname=None, dpi=240, fname_bpm=None):
    """ compare img and trace """
    trace = fits.open(fname_trace)
    if fname_img:
        img = fits.open(fname_img)
        linecol = "w"
    else:
        linecol = "k"
    if fname_spec:
        spec = fits.open(fname_spec)
    else:
        spec = None
    if fname_bpm:
        bpm = fits.open(fname_bpm)
    else:
        bpm = None

    X = np.arange(2048)
    FIG = plt.figure(figsize=(10, 3.5))

    for i in [1, 2, 3]:
        ax = FIG.add_subplot(1, 3, i)
        ax.set_xticks([])
        ax.set_yticks([])

        try:
            tdata = trace[i].data
        except:
            print("extension %s is missing, skipping." % i)
            continue
        if tdata is None:
            print("Data for CHIP%s is empty, skipping." % i)
            continue

        if fname_img:
            imgdata = img["CHIP%d.INT1" % i].data
            if imgdata is not None:
                # imgdata = np.ma.masked_where(np.isnan(imgdata),imgdata)
                imgdata = np.nan_to_num(imgdata)
                if bpm is not None:
                    bpmdata = bpm["CHIP%d.INT1" % i].data
                    if bpmdata is not None:
                        imgdata[bpmdata != 0] = 0
                vmin, vmax = np.nanpercentile(imgdata, (5, 98))
                ax.imshow(imgdata, origin="lower", cmap="plasma", vmin=vmin, vmax=vmax)

        for t in tdata:
            pol = np.polyval(t["Upper"][::-1], X)
            ax.plot(X, pol, ":" + linecol)
            pol = np.polyval(t["Lower"][::-1], X)
            ax.plot(X, pol, ":" + linecol)
            pol = np.polyval(t["All"][::-1], X)
            ax.plot(X, pol, "--" + linecol)
            if np.isnan(pol[1024]):
                continue
            ax.text(
                1024,
                pol[1024],
                "order: %s     trace: %s" % (t["order"], t["TraceNb"]),
                color=linecol,
                horizontalalignment="center",
                verticalalignment="center",
                size=8,
            )
        ax.axis((1, 2048, 1, 2048))

    FIG.tight_layout(pad=0.02)
    if figname is None:
        figname = fname_trace.replace(".fits", ".png")
    plt.savefig(figname, dpi=dpi)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("img")
    parser.add_argument("--spec")
    parser.add_argument("--bpm")
    parser.add_argument("--out")
    parser.add_argument("--dpi", default=240)
    args = parser.parse_args()

    fname_trace = args.trace
    fname_img = args.img
    fname_spec = args.spec
    fname_out = args.out
    fname_bpm = args.bpm
    dpi = args.dpi

    compare(fname_trace, fname_img, fname_spec, fname_out, fname_bpm=fname_bpm, dpi=dpi)
