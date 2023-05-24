# CRIRES LM
This package provides additional recipes to reduce raw data of the CRIRES+ instrument in the L and M bands using sky emission and telluric absorption features. In this introduction we will walk through one example dataset and showcase how the reduction is performed.

# The Dataset
Here we are going to use the observations of Beta Pic in the L3262 setting of CRIRES+ from the night of 2022-11-28 (available here: http://archive.eso.org/eso/eso_archive_main.html). Make sure to include the calibration files in the download as they are necessary for the data reduction. Additionally note that we need to unzip the datafiles from the .Z format that they are provided in by the archive. This can be done for example with the `gunzip` utility.

# Requirements
For this package to work we need to install some additional software, notably we require the CR2RES pipeline as well as MOLECFIT from the ESO instrument pipelines. Both are available here: https://www.eso.org/sci/software/pipelines/
We also need to install pycpl and pyesorex to run the recipes, the installation instructions are found here: https://www.eso.org/sci/software/pycpl/
Make sure that the pyesorex plugin dir environment variable `PYESOREX_PLUGIN_DIR` points at the recipes of this package in crires_lm/recipes.
Finally we need Python version 3.9 (others might work, but are untested) as well as the following packages as specified in requirements.txt
astropy>=5.2.2
matplotlib>=3.3.4
numpy>=1.24.3
pandas>=1.1.5
scikit_image>=0.18.1
scipy>=1.6.0
tqdm>=4.65.0

# Workflow
## Sorting the datafiles
The first step is to sort the datafiles into the different observing nights and wavelength settings. This is less important if we only have one dataset, but is crucial if we get all the data. For this purpose we can use the `sort_files.py` script. For example you can call it like so:
```
python sort_files.py src dst
```
where src is the path to the directory with all the downloaded files, and dst is the path where the files should be located after sorting. This will create new directories called `{date}_{setting}/raw` in the destination, where {date} and {setting} will be replaced by the observing date and wavelength setting of the datasets. This script uses the .xml files provided by the ESO archive to determine which calibration files belong to which dataset.
The command might then be:
```
python sort_files.py /scratch/ptah/anwe5599/CRIRES/raw /scratch/ptah/anwe5599/CRIRES
```

## Creating the data reduction pipeline
Before we can start the data reduction we first need to create the necessary set-of-files (sof) for the esorex recipes, as well as create a bash script that calls the recipes in the correct order and with the right parameters. This is done by the `prepare_pipeline.py` script. It is called like so:
```
python prepare_pipeline.py setting raw out exptime
```
where setting is the name of the wavelength setting (e.g. L3262), raw is the path to the path to the raw files as created by `cr2res_util_sort_files.py`, out is the path to a new working directory that will have all the data products as well as sof files, and finally exptime is the exposure time of the observations. This is necessary in case there is more than one exposure time per dataset, which can happen if the exposure time was changed after one exposure to avoid overexposure. For this dataset the exposure time is 30s. This will then create a script called `reduce_cals.sh` in the output directory.
The final command could then be:
```
python prepare_pipeline.py L3262 /scratch/ptah/anwe5599/CRIRES/2022-11-29_L3262/raw /scratch/ptah/anwe5599/CRIRES/2022-11-29_L3262/extr 30
```

## Running the pipeline
With all parts in place we can then run the data reduction pipeline. For this we simply call the `reduce_cals.sh` script that was generated in the previous step. It will be located in the specified output directory (i.e. here: /scratch/ptah/anwe5599/CRIRES/2022-11-29_L3262/extr). This will then call all the necessary recipes and scripts to make the data reduction work. For the sake of explanation we will go through the steps here.

## Calibrations
The first steps are to create the master dark and master flat files based on the calibrations. These are then used to create bad pixel maps for the observations. The bad pixels maps are especially important for us, since the sky emission spectra are more easily influenced by strong outliers. This uses the recipes: `cr2re_cal_dark`, `cr2res_util_calib`, `cr2res_util_bpm_create`, and `cr2res_util_bpm_merge`.

![cr2res_util_trace](https://github.com/AWehrhahn/CRIRES_LM/assets/31626864/6ca8bdea-136e-44f9-911e-8fcd32705197)


## Master Observations
The next step is to combine all observations of each of the two nodding positions (A and B) into a master observation using `cr2res_util_calib`. We also need to update the header information of these master files as they drop the information about the nodding position from their headers, but we need it for the further steps. So we add it back in with the `cr2res_util_fits_header_update.py` tool.

![cr2res_util_combine_A](https://github.com/AWehrhahn/CRIRES_LM/assets/31626864/8a2392d1-fe2a-49c7-852f-2357b28956ad)


## Creating a sky emission spectrum
Using the two nodding positions, we can remove the stellar spectrum and obtain a pure sky observation. This then uses half of the pixels from the A position and half the pixels from the B position.

![cr2res_util_combine](https://github.com/AWehrhahn/CRIRES_LM/assets/31626864/df9cf880-b434-4ee6-93a4-3953a6a91924)

## Slit Curvature
With this sky observation we can then extract the sky emission spectrum and determine the slit curvature, since the sky spectrum fills the entire slit. This is done using the new `cr2res_util_slit_curv_sky` recipe, which works similarly to the existing slit curvature recipe from the CRIRES+ pipeline. It first finds all peaks in the spectrum and then fits a curvature to each of those lines. It then fits a 2D polynomial to all the lines to make sure there is a smooth transition in slit curvature. Here we unfortunately have to remove some parts of the spectrum as they are not fit well. The exact ranges can be found in the recipe.

![cr2res_util_slit_curv](https://github.com/AWehrhahn/CRIRES_LM/assets/31626864/b586dee7-7ab8-47c3-b702-7c64639d72ac)

## Wavelength calibration
Using this slit curvature we can then extract the sky emission spectrum again as well as the stellar spectrum. Here the stellar spectrum is essentially the telluric spectrum, since beta Pic is a fast rotator. For the wavelength calibration we can therefore use the MOLECFIT code on both the emission and transmission spectrum. We are using both as one might work better in one wavelength segment and the other is better in another. Which solution to use in which order is hardcoded into `cr2res_wave_molecfit_apply`. To prepare the data into a format that MOLECFIT can understand we use the `cr2res_wave_molecfit_prepare` recipe, which also includes the recipe parameters for MOLECFIT. Here we only use three molecules for the fit H2O, CH4, and N2O, as this gives good results. The solution is then applied to the extracted spectrum using te `cr2res_wave_molecfit_apply` recipe

![cr2res_wave_molecfit_c1_o05](https://github.com/AWehrhahn/CRIRES_LM/assets/31626864/174f07b9-b7b8-4611-8321-c7f753434ade)
