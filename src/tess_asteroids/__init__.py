import logging
from os import path
from urllib.error import HTTPError

import numpy as np
import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

# Read in straps table
loc = path.abspath(path.dirname(__file__))
straps = pd.read_csv(f"{loc}/data/straps.csv", comment="#")

# Retrieve file containing TESS data downlink times
try:
    file_path = "https://tess.mit.edu/public/files/TESS_orbit_times.csv"
    downlinks = pd.read_csv(file_path, comment="#")

    # Sector 3 was used for testing ACS and downlink times are not accurate.
    # Manual override using values from https://tess.mit.edu/public/files/TESS_FFI_observation_times.csv
    downlinks.loc[
        np.logical_and(downlinks["Sector"] == 3, downlinks["Orbit"] == 13),
        ["Start of Orbit", "End of Orbit"],
    ] = ["2018-09-20 12:56:15", "2018-10-05 01:29:40"]
    downlinks.loc[
        np.logical_and(downlinks["Sector"] == 3, downlinks["Orbit"] == 14),
        ["Start of Orbit", "End of Orbit"],
    ] = ["2018-10-05 03:29:40", "2018-10-17 21:17:58"]

except HTTPError:
    downlinks = None
    logger.warning("The file {0} was not found.".format(file_path))

# TESS zero-point magnitude and error
TESSmag_zero_point = 20.44
TESSmag_zero_point_err = 0.05

# Default bad bits for SPOC quality masking, as recommended in TESS archive manual
# (https://outerspace.stsci.edu/display/TESS/2.0+-+Data+Product+Overview)
default_bad_spoc_bits = [1, 2, 3, 4, 5, 6, 8, 10, 13, 15]
# Default bad bits for LC quality masking: at least one pixel inside mask is non-science, saturated
# or negative raw flux, or cadence is in region of non-science data during sector 3.
default_bad_lc_bits = [2, 4, 10, 14]

__version__ = "1.5.0"
__all__ = ["MovingTPF"]

from .movingtpf import MovingTPF  # noqa: E402
