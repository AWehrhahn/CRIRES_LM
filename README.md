# CRIRES_LM
CRIRES+ data reduction for the L and M band.
The slit curvature is obtained from the sky emission spectrum which is created
using the two nodding position observations. Then the wavelength calibration is 
performed using the sky emission as well as the telluric transmission spectra
with MOLECFIT. Special care is taken to create extensive bad pixel maps as 
the sky emission spectrum is quite sensitive to those.

# Details
This package consists of three parts, Pyesorex recipes to do the actual work, 
plotting scripts to show of the results, and finally a script to create the
data reduction script. This script then calls the new and existing recipes as 
well as plotting scripts.

# How To
First create the data reduction scipt and SOF file for the esorex recipes using:
```python make_sof.py {setting} -r={input_dir} -o={output_dir} -e={exposure_time}```
where setting is the CRIRES+ wavelength setting (e.g. L3262), input_dir is the 
path to the directory containing the raw observation and calibration fits files,
output_dir is the path to the directory where all products will be stored, 
and finally exposure_time is necessary to filter the available files in case
more than one exposure time has been observed (e.g. since the exposure time was 
adjusted to avoid over exposure on the detector).

Then you can run the newly created reduce_cals.sh script created in output_dir from within
output_dir. This will run the entire data reduction pipeline including plotting scripts.

# Parallel
To help run this package for a number of observations at once, 
the two steps can be replaced with make_sof.sh and parallel.sh in src.
make_sof.sh will call make_sof.py for each of the existing observation dates and
parallel.sh will then call the reduce_cals.sh script in each output_dir. Make sure
that the terminal session will stay alive for the whole data reduction as
it will take some time.

# FAQ
1. Some of the plotting scripts are drop in replacements for the existing plotting
scripts of the cr2res pipeline with the same name. They work the same, but have additional 
options that are useful here.
2. cr2res_recipe.py does not actually contain a recipe, but instead contains shared 
functionality between the recipes used here. Due to the way that Pyesorex imports 
recipes it is not itself a recipe, and the other recipes need to modify the python 
search path to import it.
