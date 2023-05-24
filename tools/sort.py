import argparse
import os
import re
import shutil
import sys
from glob import glob
from os.path import basename, dirname, join

from astropy.io import fits
from tqdm import tqdm

if len(sys.argv) > 1:
    parser = argparse.ArgumentParser()
    parser.add_argument("src")
    parser.add_argument("dst")
    args = parser.parse_args()
    src = args.src
    dst = args.dst
else:
    cwd = dirname(__file__)
    src = join(cwd, "raw")
    dst = join(cwd, "{date}_{setting}", "raw")

xml_files = glob(join(src, "*.xml"))

for file in tqdm(xml_files, desc="XML"):
    with open(file) as f:
        content = f.read()
    names = re.findall(r'name="(.*)"', content)
    names = [join(src, f"{n}.fits") for n in names]

    if len(names) < 2:
        continue

    wl_set = None
    for name in names:
        header = fits.getheader(name)
        wl_set = header.get("HIERARCH ESO INS WLEN ID")
        if wl_set is not None:
            break

    dir_name = dst.format(date=names[0][-28:-18], setting=wl_set)
    os.makedirs(dir_name, exist_ok=True)
    for name in tqdm(names, leave=False, desc="Files"):
        shutil.copyfile(name, join(dir_name, basename(name)))
    shutil.copyfile(file, join(dir_name, basename(file)))

pass