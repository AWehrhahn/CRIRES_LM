import argparse
import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tracewave")
    parser.add_argument("extract")
    parser.add_argument("-o", "--out")
    args = parser.parse_args()

    tw_fname = args.tracewave
    extr_fname = args.extract
    output = args.out
    if output is None:
        output = tw_fname.replace(".fits", ".png")

    tw = fits.open(tw_fname)
    extr = fits.open(extr_fname)

    fig = plt.figure(figsize=(14,4))
    axs = [fig.add_subplot(131)]
    axs += [fig.add_subplot(132, sharey=axs[0]), fig.add_subplot(133, sharey=axs[0])]

    for chip in [1, 2, 3]:
        ext = f"CHIP{chip}.INT1"
        tw_data = tw[ext].data
        extr_data = extr[ext].data
        ax = axs[chip-1]

        orders = np.sort(tw_data["Order"])
        for order in orders:
            spec = extr_data[f"{order:02}_01_SPEC"]
            spec -= np.nanmin(spec[20:-20])
            spec /= np.nanmax(spec[20:-20])
            spec[:20] = np.nan
            spec[-20:] = np.nan
            ax.plot(spec + order - orders[0], lw=0.1)

    fig.subplots_adjust(wspace=0)
    print(f"Saving plot: {output}")
    fig.savefig(output, dpi=1200)