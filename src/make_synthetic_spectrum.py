from astropy.io import fits
from pysme.sme import SME_Structure
from pysme.linelist.vald import ValdFile
from pysme.synthesize import synthesize_spectrum
from os.path import join, dirname
import numpy as np
from pysme.gui.plot_plotly import FinalPlot

from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u

fname = "/scratch/ptah/anwe5599/CRIRES/2022-11-29/extr/cr2res_obs_nodding_extracted_combined.fits"
linelist_fname = join(dirname(__file__), "beta_pic.lin")
hdu = fits.open(fname)

sme = SME_Structure()
# Only use chip 1 for now
sme.wave = [hdu[f"CHIP1.INT1"].data[f"{o:02}_01_WL"] for o in [2, 3, 4, 5, 6, 7]]
sme.spec = [hdu[f"CHIP1.INT1"].data[f"{o:02}_01_SPEC"] for o in [2, 3, 4, 5, 6, 7]]
sme.mask = 1

# ignore the edges of the orders
sme.spec[:, :5] = 0
sme.spec[:, -5:] = 0
sme.mask[:, :5] = 0
sme.mask[:, -5:] = 0

for i in range(sme.nseg):
    sme.spec[i] /= np.max(sme.spec[i])

# Estimates from Nasa Exoplanet Archive
# Rounded to nearest even numbers
sme.teff = 8000
sme.monh = -0.2
sme.logg = 4.
sme.vsini = 105


vlt = EarthLocation.of_site('Paranal')  # the easiest way... but requires internet
ra = 5 * u.deg +  47 * u.arcmin +  17 * u.arcsec
dec = -(51 * u.deg + 3 * u.arcmin +  59 * u.arcsec)
sc = SkyCoord(ra=ra, dec=dec)
time = Time('2022-11-29')
barycorr = sc.radial_velocity_correction(obstime=time, location=vlt)  

sme.vrad = 20 + barycorr.to_value("km/s")

sme.linelist = ValdFile(linelist_fname)

sme.vrad_flag = "fix"
sme.cscale_flag = "none"

sme = synthesize_spectrum(sme)

fig = FinalPlot(sme)
fig.save(filename="test.html")

pass