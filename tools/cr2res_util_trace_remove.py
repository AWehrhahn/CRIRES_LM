"""
Remove an order from a trace wave table
"""
from astropy.io import fits
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_wave")
    parser.add_argument("order", type=int)
    args = parser.parse_args()
    tw_fname = args.trace_wave
    order = args.order

    hdu = fits.open(tw_fname)

    for chip in [1, 2, 3]:
        table = hdu[f"CHIP{chip}.INT1"].data
        idx = table["Order"] == order
        hdu[f"CHIP{chip}.INT1"].data = table[~idx]

    hdu.writeto(tw_fname, overwrite=True)
