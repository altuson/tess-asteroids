import logging
from os import path

import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

# Read in straps table
loc = path.abspath(path.dirname(__file__))
straps = pd.read_csv(f"{loc}/data/straps.csv", comment="#")

from .movingtpf import MovingTPF  # noqa: E402

__version__ = "0.6.0"
__all__ = ["MovingTPF"]
