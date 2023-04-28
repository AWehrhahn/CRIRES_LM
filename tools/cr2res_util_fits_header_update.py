from astropy.io import fits
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    parser.add_argument("keyword")
    parser.add_argument("value")

    args = parser.parse_args()
    fname = args.filename
    kw = args.keyword
    val = args.value

    hdu = fits.open(fname, mode="update")
    hdu[0].header[kw] = val
    hdu.flush()
    hdu.close()
