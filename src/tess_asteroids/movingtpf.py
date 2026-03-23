import time
import warnings
from copy import deepcopy
from datetime import datetime
from typing import Optional, Tuple, Union

import lkprf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tesswcs
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.time import Time
from astropy.utils.exceptions import AstropyUserWarning
from fbpca import pca
from lkspacecraft import TESSSpacecraft
from lkspacecraft.spacecraft import BadEphemeris
from numpy.polynomial import Polynomial
from patsy import dmatrix
from scipy import ndimage, stats
from scipy.interpolate import CubicSpline
from tess_ephem import ephem
from tesscube import TESSCube
from tesscube.fits import get_wcs_header_by_extension
from tesscube.query import async_get_primary_hdu
from tesscube.utils import _sync_call, convert_coordinates_to_runs
from tqdm import tqdm

from . import (
    TESSmag_zero_point,
    __version__,
    default_bad_lc_bits,
    default_bad_spoc_bits,
    downlinks,
    logger,
    straps,
)
from .utils import (
    add_column_header_comments,
    animate_cube,
    calculate_TESSmag,
    compute_moments,
    create_bad_bitmask,
    inside_ellipse,
    make_wcs_header,
)


class MovingTPF:
    """
    Create a TPF for a moving target (e.g. asteroid) from a TESS FFI. Includes methods to efficiently retrieve the data,
    correct the background, define an aperture mask and save a TPF in the SPOC format.

    Extract a lightcurve from the TPF, using `aperture` or `psf` photometry. Includes methods to create quality flags and
    save the lightcurve as a FITS file.

    Parameters
    ----------
    target : str
        Target ID. This is only used when saving the TPF.
    ephem : DataFrame
        Target ephemeris with columns ['time', 'sector', 'camera', 'ccd', 'column', 'row'].
        Optional columns: ['vmag', 'hmag', 'sun_distance', 'obs_distance', 'sto_angle'].

        - 'time' : float in format (JD - 2457000) in TDB. See also `barycentric` below.
        - 'sector', 'camera', 'ccd' : int
        - 'column', 'row' : float. These must be one-indexed, where the lower left pixel of the FFI is (1,1).
        - 'vmag' : float, optional. Visual magnitude.
        - 'hmag' : float, optional. Absolute magnitude.
        - 'sun_distance' : float, optional. Distance from Sun, in AU.
        - 'obs_distance' : float, optional. Distance from observer (TESS), in AU.
        - 'sto_angle' : float, optional. Sun-Target-Observer angle, in degrees. Closely approximates true phase angle. See Horizons documentation for definition: https://ssd.jpl.nasa.gov/horizons/manual.html#obsquan
    barycentric : bool, default=True

        - If True, the input `ephem['time']` must be in TDB measured at the solar system barycenter. This is the case for the
            'TSTART'/'TSTOP' keywords in SPOC FFI headers and the 'TIME' column in SPOC TPFs and LCFs.
        - If False, the input `ephem['time']` must be in TDB measured at the spacecraft. This can be recovered from the SPOC data
            products: for FFIs subtract header keyword 'BARYCORR' from 'TSTART'/'TSTOP' and for TPFs/LCFs subtract the
            'TIMECORR' column from the 'TIME' column.
    metadata : dict
        A dictionary with optional keys {'eccentricity': float, 'inclination': float, 'perihelion': float, 'ephem_date': str}.

        - 'eccentricity' : Target's orbital eccentricity. This is saved in the TPF/LCF headers.
        - 'inclination' : Target's orbital inclination, in degrees. This is saved in the TPF/LCF headers.
        - 'perihelion' : Target's perihelion distance, in AU. This is saved in the TPF/LCF headers.
        - 'ephem_date' : Date the ephemeris was generated. Format must be YYYY-MM-DD. This is saved in the TPF/LCF headers.
    """

    def __init__(
        self,
        target: str,
        ephem: pd.DataFrame,
        barycentric: bool = True,
        metadata: dict = {},
    ):
        self.target = target
        self.ephem = ephem
        self.barycentric = barycentric

        # Check self.ephem has more than one row
        if len(self.ephem) < 2:
            raise ValueError("ephem must have at least two rows.")

        # Check self.ephem['time'] has correct units
        if min(self.ephem["time"]) >= 2457000:
            raise ValueError("ephem['time'] must have units (JD - 2457000).")

        # Check self.ephem['sector'] has one unique value
        if self.ephem["sector"].nunique() == 1:
            self.sector = int(self.ephem["sector"][0])
        else:
            raise ValueError("ephem['sector'] must have one unique value.")

        # Check if target is only observed on one camera/ccd during sector
        if self.ephem["camera"].nunique() == 1 and self.ephem["ccd"].nunique() == 1:
            self.camera = int(self.ephem["camera"][0])
            self.ccd = int(self.ephem["ccd"][0])
        else:
            # >>>>> INCLUDE A WAY TO GET MULTIPLE CUBES. <<<<<
            raise NotImplementedError(
                "Target crosses multiple camera/ccd. Not yet implemented."
            )

        # Check the values of the observing geometry parameters are physical.
        if "sun_distance" in self.ephem:
            # Distance from Sun cannot be negative:
            if (self.ephem["sun_distance"] < 0).any():
                raise ValueError(
                    "`sun_distance` is the distance from the Sun in AU and every value must be >= 0."
                )
        if "obs_distance" in self.ephem:
            # Distance from observer cannot be negative:
            if (self.ephem["obs_distance"] < 0).any():
                raise ValueError(
                    "`obs_distance` is the distance from the observer in AU and every value must be >= 0"
                )
        if "sto_angle" in self.ephem:
            # Sun-Target-Observer angle runs from 0 to 180 degrees:
            if (self.ephem["sto_angle"] < 0).any() or (
                self.ephem["sto_angle"] > 180
            ).any():
                raise ValueError(
                    "`sto_angle` is the Sun-Target-Observer angle in degrees and every value must satisfy: 0 <= sto_angle <= 180."
                )

        # Save orbital elements and check the values are physical.
        if "eccentricity" in metadata:
            self.ecc = float(metadata["eccentricity"])
            # Eccentricity cannot be negative:
            if self.ecc < 0:
                raise ValueError(
                    "`ecc` is the orbital eccentricity and it must satisfy: ecc >= 0. Not `{0}`".format(
                        self.ecc
                    )
                )
        if "inclination" in metadata:
            self.inc = float(metadata["inclination"])
            # Orbital inclination runs from 0 to 180 degrees:
            if self.inc < 0 or self.inc > 180:
                raise ValueError(
                    "`inc` is the orbital inclination in degrees and it must satisfy: 0 <= inc <= 180. Not `{0}`".format(
                        self.inc
                    )
                )
        if "perihelion" in metadata:
            self.peri = float(metadata["perihelion"])
            # Perihelion distance cannot be negative:
            if self.peri < 0:
                raise ValueError(
                    "`peri` is the perihelion distance in AU and it must satisfy: peri >= 0. Not `{0}`".format(
                        self.peri
                    )
                )

        # Save ephemeris date
        if "ephem_date" in metadata:
            self.ephem_date = str(metadata["ephem_date"])
        else:
            self.ephem_date = None  # type: ignore

        # Initialise tesscube
        self.cube = TESSCube(sector=self.sector, camera=self.camera, ccd=self.ccd)

        # Retrieve original primary header of FFI cube
        self.primary_hdu = _sync_call(
            async_get_primary_hdu, object_key=self.cube.object_key
        ).header

        logger.info(
            "Initialised MovingTPF for target {0} in sector {1}, camera {2}, ccd {3}.".format(
                self.target, self.sector, self.camera, self.ccd
            )
        )

    def make_tpf(
        self,
        shape: Tuple[int, int] = (11, 11),
        bg_method: str = "linear_model",
        ap_method: str = "prf",
        save: bool = False,
        outdir: str = "",
        file_name: Optional[str] = None,
        **kwargs,
    ):
        """
        Performs all steps to create and save a SPOC-like TPF for a moving target.

        Parameters
        ----------
        shape : Tuple(int,int)
            Defined as (nrows,ncols), in pixels.
            Defines the pixels that will be retrieved, centred on the target, at each timestamp.
        bg_method : str
            Method used for background correction.
            One of [`rolling`, `linear_model`].
        ap_method : str
            Method used to create aperture.
            One of [`threshold`, `prf`, `ellipse`].
        save : bool, default=False
            If True, save the TPF HDUList to a FITS file.
        outdir : str
            If `save`, this is the directory into which the file will be saved.
        file_name : str
            If `save`, this is the filename that will be used. Format must be '.fits'.
            If no filename is given, a default one will be generated.
        **kwargs
            Keyword arguments passed to `create_pixel_quality()`, `background_correction()`,
            `create_aperture()` and `to_fits()`.

        Returns
        -------
        """
        # >>>>> ADD REFINE_COORDINATES() WHEN IMPLEMENTED <<<<<
        self.get_data(shape=shape)
        self.reshape_data()
        self.background_correction(method=bg_method, **kwargs)
        self.create_pixel_quality(**kwargs)
        self.create_aperture(method=ap_method, **kwargs)
        self.to_fits(
            file_type="tpf", save=save, outdir=outdir, file_name=file_name, **kwargs
        )

    def make_lc(
        self,
        method: str = "all",
        save: bool = False,
        file_name: Optional[str] = None,
        outdir: str = "",
        **kwargs,
    ):
        """
        Performs all steps to create a lightcurve from the moving TPF, with the option to save.

        Parameters
        ----------
        method : str
            Method to extract lightcurve. One of [`all`, `aperture`, `psf`].
        save : bool
            If True, save the lightcurve HDUList to a FITS file.
        outdir : str
            If `save`, this is the directory into which the file will be saved.
        file_name : str
            If `save`, this is the filename that will be used. Format must be '.fits'.
            If no filename is given, a default one will be generated.
        **kwargs
            Keyword arguments passed to `to_lightcurve()` and `to_fits()`.

        Returns
        -------
        """
        if method == "all":
            self.to_lightcurve(method="aperture", **kwargs)
            self.to_lightcurve(method="psf", **kwargs)
        else:
            self.to_lightcurve(method=method, **kwargs)
        self.to_fits(
            file_type="lc", save=save, outdir=outdir, file_name=file_name, **kwargs
        )

    def refine_coordinates(self):
        """
        Apply correction to target ephemeris.

        Parameters
        ----------

        Returns
        -------
        """
        raise NotImplementedError("refine_coordinates() is not yet implemented.")

    def get_data(self, shape: Tuple[int, int] = (11, 11)):
        """
        Retrieve pixel data for a moving target from a TESS FFI.

        Parameters
        ----------
        shape : Tuple(int,int)
            Defined as (nrows,ncols), in pixels.
            Defines the pixels that will be retrieved, centred on the target, at each timestamp.

        Returns
        -------
        """
        # Shape needs to be >=3 in both dimensions, otherwise `make_wcs_header()` errors.
        self.shape = shape

        # Use interpolation to get target (row,column) at cube time.
        # If input ephem["time"] is in TDB at spacecraft, convert cube time for interpolation.
        column_interp = CubicSpline(
            self.ephem["time"].astype(float),
            self.ephem["column"].astype(float),
            extrapolate=False,
        )(
            self.cube.time
            if self.barycentric
            else self.cube.time - self.cube.last_hdu.data["BARYCORR"]
        )
        row_interp = CubicSpline(
            self.ephem["time"].astype(float),
            self.ephem["row"].astype(float),
            extrapolate=False,
        )(
            self.cube.time
            if self.barycentric
            else self.cube.time - self.cube.last_hdu.data["BARYCORR"]
        )

        # Remove nans from interpolated position
        nan_mask = np.logical_and(~np.isnan(column_interp), ~np.isnan(row_interp))
        row_interp, column_interp = row_interp[nan_mask], column_interp[nan_mask]

        # Coordinates (row,column) of lower left corner of region to retrieve around target
        self.corner = np.ceil(
            np.asarray(
                [row_interp - self.shape[0] / 2, column_interp - self.shape[1] / 2]
            ).T
        ).astype(int)

        # Pixel positions to retrieve around target
        row, column = (
            np.mgrid[: self.shape[0], : self.shape[1]][:, None, :, :]
            + self.corner.T[:, :, None, None]
        )

        # Remove frames that include pixels outside bounds of FFI.
        # >>>>> COULD KEEP FRAMES WITH PIXELS OUTSIDE OF BOUNDS AND FILL WITH NANS INSTEAD? <<<<<
        bound_mask = np.logical_and(
            [r.all() for r in np.logical_and(row[:, :, 0] >= 1, row[:, :, 0] <= 2078)],
            [
                c.all()
                for c in np.logical_and(column[:, 0, :] >= 1, column[:, 0, :] <= 2136)
            ],
        )
        self.time_original = self.cube.time[
            nan_mask
        ][
            bound_mask
        ]  # Original FFI timestamps of each frame in the data cube, in TDB at SS barycenter.
        self.timecorr_original = self.cube.last_hdu.data["BARYCORR"][nan_mask][
            bound_mask
        ]  # Original time correction of each frame in the data cube.
        self.tstart = self.cube.tstart[nan_mask][
            bound_mask
        ]  # Time at start of the exposure.
        self.tstop = self.cube.tstop[nan_mask][
            bound_mask
        ]  # Time at end of the exposure.
        self.quality = self.cube.quality[nan_mask][
            bound_mask
        ]  # SPOC quality flag of each frame in the data cube.
        self.cadence_number = self.cube.cadence_number[
            nan_mask
        ][
            bound_mask
        ]  # Unique cadence number of each frame in the data cube, as defined by tesscube.
        self.ephemeris = np.asarray(
            [row_interp[bound_mask], column_interp[bound_mask]]
        ).T  # Predicted (row,column) of target.
        self.corner = self.corner[bound_mask]
        row, column = row[bound_mask], column[bound_mask]
        pixel_coordinates = np.asarray([row.ravel(), column.ravel()]).T

        # Check there are pixels inside FFI bounds.
        if len(pixel_coordinates) == 0:
            raise RuntimeError(
                "All pixels are outside of FFI bounds (1<=row<=2078, 1<=col<=2136)."
            )
        # Warn user if some of the pixels are outside of FFI bounds.
        elif sum(~bound_mask) > 0:
            logger.warning(
                "Some of the requested pixels are outside of the FFI bounds (1<=row<=2078, 1<=col<=2136) and will not be returned."
            )

        # Convert pixels to byte runs
        runs = convert_coordinates_to_runs(pixel_coordinates)

        # Retrieve the data
        logger.info("Started data retrieval.")
        start_time = time.time()
        result = _sync_call(self.cube.async_get_data_per_rows, runs)
        logger.info(
            "Finished data retrieval in {0:.2f} sec.".format(time.time() - start_time)
        )

        # Split result into flux, flux_err and reshape into (ntimes, npixels)
        self.all_flux, self.all_flux_err = np.vstack(result).transpose([2, 1, 0])
        # Apply masks to remove rejected frames.
        self.all_flux, self.all_flux_err = (
            self.all_flux[nan_mask][bound_mask],
            self.all_flux_err[nan_mask][bound_mask],
        )

        # Transform unique pixel indices back into (row,column)
        self.pixels = np.asarray(
            [
                j
                for i in [
                    np.asarray(
                        [
                            np.full(run["ncolumns"], run["row"]),
                            np.arange(
                                run["start_column"],
                                run["start_column"] + run["ncolumns"],
                            ),
                        ]
                    ).T
                    for run in runs
                ]
                for j in i
            ]
        )

        # Set non-science pixels to nan values.
        non_science_pixel_mask = ~np.logical_and(
            self.pixels[:, 0] <= 2048,
            np.logical_and(self.pixels[:, 1] >= 45, self.pixels[:, 1] <= 2092),
        )
        self.all_flux[:, non_science_pixel_mask] = np.nan
        self.all_flux_err[:, non_science_pixel_mask] = np.nan
        # Check there are pixels inside FFI science array.
        if np.all(non_science_pixel_mask):
            raise RuntimeError(
                "All pixels are outside of FFI science array (1<=row<=2048, 45<=col<=2092)."
            )
        # Warn user if there are pixels outside of FFI science array.
        if np.sum(non_science_pixel_mask) > 0:
            logger.warning(
                "Some of the requested pixels are outside of the FFI science array (1<=row<=2048, 45<=col<=2092), but they will be set to NaN in your TPF."
            )

        # Pixel mask that tracks moving target
        target_mask = []
        for t in range(len(self.time_original)):
            target_mask.append(
                np.logical_and(
                    np.isin(self.pixels.T[0], row[t].ravel()),
                    np.isin(self.pixels.T[1], column[t].ravel()),
                ),
            )
        self.target_mask = np.asarray(target_mask)

        # Convert (row,column) ephemeris to (ra,dec) using WCS from tesswcs.
        # Note: if MovingTPF was initialised from_name, then tess-ephem
        # internally converted (ra,dec) to (row,column) using tesswcs.
        # `self.coords` does not recover these original values because the
        # ephemeris has since been interpolated.
        # Note: `pixel_to_world()` assumes zero-indexing so subtract one from (row,col).
        self.wcs = tesswcs.WCS.from_sector(
            sector=self.sector, camera=self.camera, ccd=self.ccd
        )
        self.coords = np.asarray(
            [
                self.wcs.pixel_to_world(
                    self.ephemeris[t, 1] - 1, self.ephemeris[t, 0] - 1
                )
                for t in range(len(self.time_original))
            ]
        )

        # Calculate barycentric correction using position of object.
        # If SPICE kernels are missing, no correction will be computed.
        try:
            time_sc = self.time_original - self.timecorr_original
            # Catch warnings.
            with warnings.catch_warnings(record=True) as recorded_warnings:
                tess = TESSSpacecraft()
                # Get unique warnings and save to logger.
                for w in list(
                    {
                        "{0}-{1}".format(w.filename, w.lineno): w
                        for w in recorded_warnings
                    }.values()
                ):
                    logger.warning("Warning from TESSSpacecraft(): {0}".format(w))

            timecorr = []
            for t in range(len(time_sc)):
                timecorr.append(
                    tess.get_barycentric_time_correction(
                        time=Time(time_sc[t] + 2457000, scale="tdb", format="jd"),
                        ra=self.coords[t].ra.value,
                        dec=self.coords[t].dec.value,
                    )[0]
                    / 60
                    / 60
                    / 24
                )
            self.timecorr = np.asarray(timecorr)
            self.time = time_sc + self.timecorr
        except BadEphemeris:
            logger.warning(
                "Barycentric correction was not calculated due to missing SPICE kernels."
            )
            self.timecorr = self.timecorr_original
            self.time = self.time_original

        # Interpolate observing geometry parameters onto observation times
        # Note: interpolation will fail if any value is not finite.
        self.obs_params = {}
        for param in ["sun_distance", "obs_distance", "sto_angle"]:
            if param in self.ephem and np.isfinite(self.ephem[param].values).all():
                val = CubicSpline(
                    self.ephem["time"].astype(float),
                    self.ephem[param].astype(float),
                    extrapolate=False,
                )(self.time if self.barycentric else self.time - self.timecorr)
            else:
                val = np.full(len(self.time), np.nan)

            self.obs_params[param] = val

    def reshape_data(self):
        """
        Reshape flux data into cube with shape (len(self.time), self.shape).
        """
        if not hasattr(self, "all_flux"):
            raise AttributeError("Must run `get_data()` before reshaping data.")

        self.flux = []
        self.flux_err = []
        # Reshape flux data.
        for t in range(len(self.time)):
            self.flux.append(self.all_flux[t][self.target_mask[t]].reshape(self.shape))
            self.flux_err.append(
                self.all_flux_err[t][self.target_mask[t]].reshape(self.shape)
            )

        self.flux = np.asarray(self.flux)
        self.flux_err = np.asarray(self.flux_err)

    def background_correction(self, method: str = "linear_model", **kwargs):
        """
        Apply background correction to reshaped flux data.

        Parameters
        ----------
        method : str
            Method used for background correction. One of [`rolling`, `linear_model`].
        **kwargs
            Keyword arguments passed to `_bg_rolling_median()` and `_bg_linear_model()`.

        Returns
        -------
        """
        if not hasattr(self, "all_flux") or not hasattr(self, "flux"):
            raise AttributeError(
                "Must run `get_data()` and `reshape_data()` before computing background."
            )

        # Initialise masks that will flag NaNs in scattered light or star model, only used
        # if `linear_model` is the most recent method run.
        self.sl_nan_mask = np.zeros_like(self.time, dtype=bool)
        self.star_nan_mask = np.zeros_like(self.all_flux, dtype=bool)

        # Initialise mask that will flag pixels with poor star model fit, only used
        # if `linear_model` is the most recent method run.
        self.star_fit_mask = np.zeros(len(self.pixels), dtype=bool)

        # Initialise bad SPOC bits, only used if `linear_model` is the most recent method run.
        self.bad_spoc_bits = "n/a"

        # Define SL correction method
        self.sl_method = "n/a"

        # Get background via chosen method
        if method == "rolling":
            self.bg, self.bg_err = self._bg_rolling_median(**kwargs)

        elif method == "linear_model":
            self.bg, self.bg_err, _, _, _, _ = self._bg_linear_model(
                reshape=True, **kwargs
            )

        else:
            raise ValueError(
                "`method` must be one of: [`rolling`,`linear_model`]. Not `{0}`".format(
                    method
                )
            )

        # Apply background correction
        self.corr_flux = self.flux - self.bg
        self.corr_flux_err = np.sqrt(self.flux_err**2 + self.bg_err**2)

        self.bg_method = method

        logger.info("Corrected background using method {0}.".format(self.bg_method))

    def _bg_rolling_median(self, nframes: int = 25, **kwargs):
        """
        Calculate the background using a rolling median of nearby frames.

        Parameters
        ----------
        nframes : int
            Number of frames either side of current frame to use in estimate of background.

        Returns
        -------
        bg : ndarray
            Background flux estimate.
            Array with same shape as `self.flux`.

        bg_err : ndarray
            Error on background flux estimate.
            Array with same shape as `self.flux`.
        """

        if not hasattr(self, "all_flux"):
            raise AttributeError("Must run `get_data()` before computing background.")

        bg = []
        bg_err = []
        for i in range(len(self.all_flux)):
            # Get flux window.
            flux_window = self.all_flux[
                i - nframes if i >= nframes else 0 : i + nframes + 1
                if i <= len(self.all_flux) - nframes
                else len(self.all_flux)
            ][:, self.target_mask[i]]
            # Catch warnings that arise if pixel is nan throughout window (e.g. non-science pixels).
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="All-NaN slice encountered",
                    category=RuntimeWarning,
                )
                # Compute background flux.
                bg.append(np.nanmedian(flux_window, axis=0).reshape(self.shape))
                # Use the Median Absolute Deviation (MAD) for error on the background flux.
                bg_err.append(
                    np.nanmedian(
                        np.abs(flux_window - np.nanmedian(flux_window, axis=0)), axis=0
                    ).reshape(self.shape)
                )

        return np.asarray(bg), np.asarray(bg_err)

    def _create_source_mask(
        self,
        include_target: bool = True,
        include_stars: bool = True,
        target_threshold: float = 0.01,
        star_flux_threshold: float = 1.05,
        star_gradient_threshold: float = 4,
        all_flux: bool = True,
        **kwargs,
    ):
        """
        Creates a boolean mask that masks stationary sources (e.g. stars) and the moving target.

        The moving target mask is created using the PRF model and selecting all pixels that contain more than the
        `target_threshold` fraction of the object's flux. This mask is defined per time. This is only included in the mask
        if `include_target` is `True`.

        The star mask is created using two condiditons: i) a high flux value and ii) a high flux gradient. This mask is
        constant across all times. This is only included in the mask if `include_stars` is `True`.

        The source mask can be returned in all_flux or TPF space, controlled by the `all_flux` parameter. In all_flux
        space the source mask has shape (ntimes, npixels) and in the TPF space the shape is (ntimes, nrows, ncols).

        Parameters
        ----------
        include_target : bool
            If `True`, includes moving target in the source mask.
        include_stars : bool
            If `True`, includes stars in the source mask.
        target_threshold : float
            Pixels where the PRF model is greater than this threshold are included in the target mask. Must be between 0 and 1
            because the PRF model is normalised so that all values sum to one.
        star_flux_threshold : float
            Used to define the threshold above which a pixel has a high flux.
        star_gradient_threshold : float
            Used to define the threshold above which a pixel has a high flux gradient.
        all_flux : boolean
            If True, the mask is returned with shape (ntimes, npixels).
            If False, the mask is returned with shape (ntimes, nrows, ncols).
        kwargs : dict
            Keywords arguments passed to `self._create_target_prf_model`, e.g `time_step`.

        Returns
        -------
        source_mask : ndarray
            Boolean mask. The mask is `True` when the moving target or a star is present.
        """

        # Check thresholds are physical.
        if target_threshold < 0 or target_threshold >= 1:
            raise ValueError(
                f"`target_threshold` must be between 0 and 1. Not '{target_threshold}'"
            )
        if star_flux_threshold <= 0:
            raise ValueError(
                f"`star_flux_threshold` must be greater than 0. Not '{star_flux_threshold}'"
            )
        if star_gradient_threshold <= 0:
            raise ValueError(
                f"`star_gradient_threshold` must be greater than 0. Not '{star_gradient_threshold}'"
            )

        # Create mask for moving target.
        if include_target:
            target_mask, _ = self._create_target_prf_model(all_flux=True, **kwargs)
            target_mask = target_mask >= target_threshold
        else:
            target_mask = np.zeros_like(self.all_flux, dtype=bool)

        # Create mask for stationary sources e.g. stars (high flux values AND high flux gradients).
        if include_stars:
            # Compute median of each pixel across all times and median of all pixels across all times.
            # Catch warnings that arise if value is nan at all times (e.g. non-science pixels).
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="All-NaN slice encountered",
                    category=RuntimeWarning,
                )
                med = np.nanmedian(self.all_flux, axis=0)
                med_all = np.nanmedian(self.all_flux)

            # Mask pixels whose average flux value over all time is some fraction greater than average flux value over all
            # pixels and all times.
            star_flux_mask = med >= star_flux_threshold * med_all

            # Reshape median to match 2D all_flux region. This is necessary to be able to compute mask in terms of gradient.
            # Origin is minimum row/column (not a pair) and shape is entire all_flux region.
            origin = tuple(self.pixels.min(axis=0).astype(int))
            shape = tuple(
                (self.pixels.max(axis=0) - self.pixels.min(axis=0) + 1).astype(int)
            )
            med_reshaped = np.full(shape, np.nan)
            med_reshaped[
                self.pixels[:, 0] - np.asarray(origin[0]),
                self.pixels[:, 1] - np.asarray(origin[1]),
            ] = med

            # Mask pixels whose average flux gradient over all time is some fraction greater than average flux gradient
            # over all pixels and all times. Have to use binary_fill_holes to fill holes in binary mask (otherwise bright
            # pixels in center of star are sometimes excluded from gradient mask).
            # Catch warnings that arise if all pixels are nan at all times (e.g. non-science pixels).
            med_gradient = np.gradient(med_reshaped)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="All-NaN slice encountered",
                    category=RuntimeWarning,
                )
                star_gradient_mask = np.hypot(
                    *med_gradient
                ) >= star_gradient_threshold * np.nanmedian(np.hypot(*med_gradient))
            star_gradient_mask = ndimage.binary_fill_holes(star_gradient_mask)

            # Combine flux and gradient mask - gradient mask is reshaped back to match star_flux_mask.
            star_mask = (
                star_flux_mask
                & star_gradient_mask[
                    self.pixels[:, 0] - np.asarray(origin[0]),
                    self.pixels[:, 1] - np.asarray(origin[1]),
                ]
            )
        else:
            star_mask = np.zeros(self.all_flux.shape[1], dtype=bool)

        source_mask = np.asarray(
            [np.logical_or(targ_mask, star_mask) for targ_mask in target_mask]
        )
        if not all_flux:
            source_mask_reshaped = []
            for t in range(len(self.time)):
                source_mask_reshaped.append(
                    source_mask[t][self.target_mask[t]].reshape(self.shape)
                )
            source_mask = np.asarray(source_mask_reshaped)

        return source_mask

    def _data_chunks(self, data_chunks: Optional[Union[list, np.ndarray]] = None):
        """
        This function creates and/or checks the format of data chunks.
        If `data_chunks=None`, this function will define `data_chunks` using TESS data downlink times.
        Otherwise, this function will check the format of user-defined `data_chunks`.
        These chunks are used by `_create_scattered_light_model()` and `_bg_linear_model()`.

        Parameters
        ----------
        data_chunks : ndarray
            A boolean mask that defines chunks of data. The array must have shape (nchunks, ntimes).
            If None, chunks will be defined using TESS data downlink times.

        Returns
        -------
        data_chunks : ndarray
            A boolean mask that defines chunks of data. The array has shape (nchunks, ntimes).

        """

        # If user has not provided data chunks, split data based upon data downlink times.
        if data_chunks is None:
            # If downlinks file was not found, data is not split into chunks.
            if downlinks is None:
                data_chunks = np.asarray([np.full(len(self.time), True)])
                logger.warning(
                    "The downlink file was not found. Data was not split into chunks and all cadences will be modelled simultaneously."
                )

            else:
                sector_downlinks = downlinks[downlinks["Sector"] == self.sector]

                # Check for expected number of chunks.
                if self.sector in [97, 98] and len(sector_downlinks) != 8:
                    logger.warning(
                        "For sector {0} there should be 8 data chunks, but there are actually {1} data chunks. Investigate or define your own `data_chunks`.".format(
                            self.sector, len(sector_downlinks)
                        )
                    )
                elif self.sector <= 55 and len(sector_downlinks) != 2:
                    logger.warning(
                        "For sector {0} there should be 2 data chunks, but there are actually {1} data chunks. Investigate or define your own `data_chunks`.".format(
                            self.sector, len(sector_downlinks)
                        )
                    )
                elif self.sector > 55 and len(sector_downlinks) != 4:
                    logger.warning(
                        "For sector {0} there should be 4 data chunks, but there are actually {1} data chunks. Investigate or define your own `data_chunks`.".format(
                            self.sector, len(sector_downlinks)
                        )
                    )

                # Define data chunk mask
                data_chunks = []
                for _, downlink in sector_downlinks.iterrows():
                    start = (
                        Time(downlink["Start of Orbit"], format="iso", scale="utc").jd
                        - 2457000
                    )
                    end = (
                        Time(downlink["End of Orbit"], format="iso", scale="utc").jd
                        - 2457000
                    )
                    # The buffer of 0.05 days accounts for inaccuracies in the downlink times.
                    chunk = np.logical_and(
                        self.time - self.timecorr >= start - 0.05,
                        self.time - self.timecorr < end + 0.05,
                    )
                    # Only append chunks which have at least one True value i.e. target was observed during that chunk.
                    if chunk.any():
                        data_chunks.append(chunk)

            logger.info("Defined {0} data chunks.".format(len(data_chunks)))

        # Enforce that `data_chunks` is an array of arrays.
        data_chunks = np.atleast_2d(data_chunks)
        # Check data_chunks has correct length and number of dimensions.
        if data_chunks.ndim != 2 or data_chunks.shape[1] != len(self.time):  # type: ignore
            raise ValueError(
                "`data_chunks` must be a two-dimensional boolean array, where each sub-array has length equal to `self.time`."
            )

        # Check that all data is included in exactly one chunk.
        if not np.all(np.sum(data_chunks, axis=0) == 1):  # type: ignore
            raise ValueError(
                "All data must be included in exactly one data chunk, but this is not the case. Try re-defining `data_chunks`."
            )

        return data_chunks

    def _create_scattered_light_model(
        self,
        knot_width: float = 20,
        data_chunks: Optional[Union[list, np.ndarray]] = None,
        spoc_quality_mask: Optional[np.ndarray] = None,
        ncomponents: int = 8,
        niter: int = 5,
        sigma: float = 5,
        niter_clip: int = 3,
        diagnostic_plot: bool = False,
        outdir: str = "",
        **kwargs,
    ):
        """
        Uses PCA and linear modelling to create a scattered light model with the same shape as `self.all_flux`.

        - The design matrix is defined with a 3rd degree B-spline in both row and column. The knots of each spline can be
        controlled with the parameter `knot_width`.
        - The PCA components are computed per data chunk (defined by `data_chunks`) and fit to the data with optional
        outlier clipping.
        - The frames flagged as bad quality by `spoc_quality_mask` are not used to compute the scattered light model
        and those cadences do not have a SL model (it will be NaNs).

        Parameters
        ----------
        knot_width : float
            Approximate pixel spacing between spline knots. A default of 20 pixels is used because this represents
            the typical scale over which scattered light varies.
        data_chunks : ndarray
            A boolean mask that defines chunks of data to be independently fit. The array must have shape
            (nchunks, ntimes). If you want to fit all data simultaneously, define as
            `np.asarray([np.full(len(self.time), True)])`. If None, chunks will be defined using TESS data downlink times.
        spoc_quality_mask : ndarray
            A boolean mask that defines data with good SPOC quality. The array must have the same length as `self.time`.
            It can be defined using the function `_create_spoc_quality_mask()`. If None, the default quality mask will be used.
        ncomponents : int
            Number of PCA components.
        niter : int
            Number of iterations that will be run to compute the PCA components.
        sigma : float
            Sigma threshold for outlier detection. A larger value of `sigma` will mask less data.
        niter_clip : int
            Number of iterations of outlier clipping.

            - To turn off outlier clipping, set `niter_clip` to zero.
            - To clip until no outliers remain, set `niter_clip` to `np.inf`.
        diagnostic_plot : bool
            If True, shows two diagnostic plots to check the scattered light model.
        outdir : str
            If `diagnostic_plot`, this is the directory into which the plot will be saved.
        kwargs : dict
            Keywords arguments passed to `self._create_source_mask`, e.g `target_threshold`, `star_flux_threshold`.

        Returns
        -------
        sl_model : ndarray
            Scattered light model, with same shape as `self.all_flux`.
        sl_model_err : ndarray
            Error on scattered light model, with same shape as `self.all_flux`.
        """

        if not hasattr(self, "all_flux"):
            raise AttributeError(
                "Must run `get_data()` before computing scattered light model."
            )

        # Parameter logic checks
        if sigma <= 0:
            raise ValueError(f"`sigma` must be greater than zero. Not '{sigma}'")
        if niter_clip < 0:
            raise ValueError(
                f"`niter_clip` must be greater than or equal to zero. Not '{niter_clip}'"
            )

        # If no SPOC quality mask is provided, the default mask is used.
        if spoc_quality_mask is None:
            spoc_quality_mask, self.bad_spoc_bits = self._create_spoc_quality_mask(
                bad_spoc_bits="default"
            )

        # Create mask for moving target and stars.
        source_mask = self._create_source_mask(
            include_stars=True, include_target=True, all_flux=True, **kwargs
        )

        # Add nan flux values to the mask (PCA cannot have nan in flux array).
        source_mask = np.logical_or(source_mask, np.isnan(self.all_flux))

        # Mask all moving target pixels at all times, even when it is not present.
        source_mask = source_mask.any(axis=0)

        # Define equally spaced knots between bounds, with spacing approximately equal to `knot_width`.
        # It is important to include an extra knot at the lower bound - this gives proper behaviour of spline.
        row, col = self.pixels.T
        knots_col = np.linspace(
            col.min(),
            col.max(),
            int(np.round((col.max() - col.min()) / knot_width)) + 1,
        )[:-1]
        knots_row = np.linspace(
            row.min(),
            row.max(),
            int(np.round((row.max() - row.min()) / knot_width)) + 1,
        )[:-1]

        # If `knot_width` is too large for the dataset, `knots_col` or `knots_row` can be returned as empty arrays.
        # The function will run, but we need to ensure that there is always one knot at the lower bound.
        if len(knots_col) == 0:
            knots_col = np.asarray([col.min()])
        if len(knots_row) == 0:
            knots_row = np.asarray([row.min()])

        # Warn the user if there are no interior knots. This can lead to a poorly constrained spline.
        if len(knots_row) == 1 or len(knots_col) == 1:
            logger.warning(
                "There are no interior knots in column and/or row with a `knot_width` of {0}. Try re-defining `knot_width` to a smaller value.".format(
                    knot_width
                )
            )

        # Create design matrix - pairwise combination of a third degree b-spline in row and column, with a global intercept.
        X = np.asarray(
            dmatrix(
                "bs(row, knots=knots_row, degree=3, include_intercept=False) : bs(col, knots=knots_col, degree=3, include_intercept=False)",
                {
                    "row": row,
                    "knots_row": knots_row,
                    "col": col,
                    "knots_col": knots_col,
                },
            )
        )
        # Exclude all zeroes from design matrix.
        X = X[:, X.sum(axis=0) != 0]
        # Account for strap columns in the design matrix.
        X = np.hstack([np.isin(col, straps["Column"] + 44)[:, None], X])

        # Create and/or check data chunks
        data_chunks = self._data_chunks(data_chunks)

        # Initialise priors on LM components
        prior_mu = np.zeros(X.shape[1])
        prior_sigma = np.ones(X.shape[1]) * 0.5

        # Initialise scattered light model and error.
        sl_model = np.full(self.all_flux.shape, np.nan)
        sl_model_err = np.full(self.all_flux.shape, np.nan)

        logger.info("Started computation of scattered light model.")
        start_time = time.time()

        for i, chunk in enumerate(data_chunks):  # type: ignore
            # Define good quality cadences in the data chunk.
            cadence_mask = np.logical_and(spoc_quality_mask, chunk)
            # Initialise outlier mask
            outlier_mask = np.zeros_like(source_mask, dtype=bool)

            # If there is no data to fit, SL model and error will be nan for entire chunk.
            if (~cadence_mask).all():
                # Update SL NaN mask
                if hasattr(self, "sl_nan_mask"):
                    self.sl_nan_mask[chunk] = True

                logger.warning(
                    "When computing the scattered light model for data chunk {0}, there was no good quality data. The scattered light model for the entire data chunk was set to nan.".format(
                        i
                    )
                )

                continue

            # Get the PCA components for data chunk, excluding pixels with sources.
            # This will return an AssertionError if `ncomponents` is greater than the smallest dimension of the
            # input matrix (i.e. too few times or pixels). In this case, SL model and error will be nan for entire chunk.
            try:
                U, s, V = pca(
                    self.all_flux[cadence_mask][:, ~source_mask],
                    k=ncomponents,
                    raw=True,
                    n_iter=niter,
                )
            except AssertionError:
                # Update SL NaN mask
                if hasattr(self, "sl_nan_mask"):
                    self.sl_nan_mask[chunk] = True

                logger.warning(
                    "When computing the scattered light model for data chunk {0}, the PCA failed with an AssertionError. This means either niter < 0, ncomponents <= 0 or ncomponents is greater than the smallest dimension of the input matrix (i.e. masking of `self.all_flux` has removed too much data). The scattered light model for the entire data chunk was set to nan.".format(
                        i
                    )
                )

                continue

            # Prime while loop used for outlier clipping.
            n_clip = np.inf
            i_clip = 0
            exit_loop = False
            while n_clip != 0 and i_clip <= niter_clip:
                # Define pixels to fit.
                pixel_mask = np.logical_and(~source_mask, ~outlier_mask)

                # If there are no pixels to fit, SL model and error will be nan for entire chunk.
                if (~pixel_mask).all():
                    # Update SL NaN mask
                    if hasattr(self, "sl_nan_mask"):
                        self.sl_nan_mask[chunk] = True

                    logger.warning(
                        "When computing the scattered light model for data chunk {0}, there were no pixels to fit. The scattered light model for the entire data chunk was set to nan.".format(
                            i
                        )
                    )

                    exit_loop = True

                    break

                # Compute best-fitting weights with Bayesian least squares
                try:
                    w = np.linalg.solve(
                        X[pixel_mask].T.dot(X[pixel_mask])
                        + np.diag(1 / prior_sigma**2),
                        X[pixel_mask].T.dot(V[:, ~outlier_mask[~source_mask]].T)
                        + prior_mu[:, None] / prior_sigma[:, None] ** 2,
                    )
                # If no solution is found, SL model and error will be nan for entire chunk.
                except np.linalg.LinAlgError:
                    # Update SL NaN mask
                    if hasattr(self, "sl_nan_mask"):
                        self.sl_nan_mask[chunk] = True

                    logger.warning(
                        "When computing the scattered light model for data chunk {0}, no solution was found. The scattered light model for the entire data chunk was set to nan.".format(
                            i
                        )
                    )

                    exit_loop = True
                    break

                # Clip outliers if there are remaining iterations.
                if i_clip < niter_clip:
                    n_clip_prev = np.sum(outlier_mask)
                    outlier_mask[~source_mask] = np.logical_or(
                        outlier_mask[~source_mask],
                        sigma_clip(
                            ((V.T - X.dot(w)[~source_mask]) ** 2).sum(axis=1) ** 0.5,
                            sigma=sigma,
                        ).mask,
                    )
                    n_clip = np.sum(outlier_mask) - n_clip_prev
                    logger.info(
                        "When computing the scattered light model, clipped {1} pixels in iteration {2} for data chunk {0}.".format(
                            i, n_clip, i_clip
                        )
                    )
                i_clip += 1

            if not exit_loop:
                # Create model that can predict V for excluded pixels (sources and outliers).
                V_model = X.dot(w).T

                # Compute scattered light model and error.
                sl_model[cadence_mask] = U.dot(np.diag(s)).dot(V_model)
                # >>>>> Add error on SL model <<<<<

        logger.info(
            "Finished computation of scattered light model in {0:.2f} sec.".format(
                time.time() - start_time
            )
        )

        # Add bad SPOC quality data to SL nan mask - these cadences do not have a SL model.
        if hasattr(self, "sl_nan_mask"):
            self.sl_nan_mask[~spoc_quality_mask] = True

        if diagnostic_plot:
            logger.info("Making diagnostic plots for scattered light model...")

            # Check format of outdir
            if len(outdir) > 0 and not outdir.endswith("/"):
                outdir += "/"

            # Create file name prefix
            file_name = "tess-{0}-s{1:04}-{2}-{3}-shape{4}x{5}".format(
                str(self.target).replace(" ", "").replace("/", ""),
                self.sector,
                self.camera,
                self.ccd,
                *self.shape,
            )

            # Plot one: SL model on 2D pixel grid in all_flux region for selection of frames.
            # Origin is minimum row/column (not a pair) and shape is entire all_flux region.
            origin = tuple(self.pixels.min(axis=0).astype(int))
            shape = tuple(
                (self.pixels.max(axis=0) - self.pixels.min(axis=0) + 1).astype(int)
            )
            # Re-shape SL model to all_flux region.
            sl_reshaped = np.full((len(self.time), *shape), np.nan)
            for t in range(len(self.time)):
                sl_reshaped[t][
                    self.pixels[:, 0] - np.asarray(origin[0]),
                    self.pixels[:, 1] - np.asarray(origin[1]),
                ] = sl_model[t]
            # Plot SL model from first, middle and last frame.
            # Note: setting aspect="auto" means pixels are not square in these visualisations.
            frames = [0, len(self.time) // 2, -1]
            extent = (
                origin[1] - 0.5,
                origin[1] + shape[1] - 0.5,
                origin[0] - 0.5,
                origin[0] + shape[0] - 0.5,
            )
            fig, ax = plt.subplots(
                1, len(frames), sharex=True, sharey=True, figsize=(len(frames) * 4, 4)
            )
            for i, frame in enumerate(frames):
                im = ax[i].imshow(
                    sl_reshaped[frame],
                    origin="lower",
                    extent=extent,
                    aspect="auto",
                    interpolation="none",
                )
                ax[i].set(
                    xlabel="Column Pixel",
                    title="CAD {0} | BTJD {1:.4f}".format(
                        self.cadence_number[frame], self.time[frame]
                    ),
                )
                if i == 0:
                    ax[i].set(ylabel="Row Pixel")
                cbar = fig.colorbar(im, ax=ax[i], location="right")
                cbar.set_label("Flux [$e^-/s$]")
            plt.savefig(outdir + file_name + "-sl_2d.png")
            plt.close(fig)
            logger.info("Created file: {0}".format(outdir + file_name + "-sl_2d.png"))

            # Plot two: time-series of SL model for each pixel.
            # Note: setting aspect="auto" means pixels are not square in these visualisations.
            fig, ax = plt.subplots()
            im = ax.imshow(
                sl_model.T, origin="lower", aspect="auto", interpolation="none"
            )
            ax.set(
                xlabel="Time Index", ylabel="Pixel Index", title="Scattered Light Model"
            )
            # Add colorbar
            cbar = fig.colorbar(im, ax=ax, location="right")
            cbar.set_label("Flux [$e^-/s$]")
            plt.savefig(outdir + file_name + "-sl_time.png")
            plt.close(fig)
            logger.info("Created file: {0}".format(outdir + file_name + "-sl_time.png"))

        return np.asarray(sl_model), np.asarray(sl_model_err)

    def _create_spoc_quality_mask(
        self,
        spoc_quality: Optional[np.ndarray] = None,
        bad_spoc_bits: Union[list[int], str] = "default",
    ):
        """
        Creates a boolean mask using SPOC quality flags.

        In all cases except `bad_spoc_bits="none"`, non-science data in sector 3 is also masked.

        Parameters
        ----------
        spoc_quality : ndarray
            Array of SPOC quality flags to turn into a quality mask.
            If None, `self.quality` will be used.
        bad_spoc_bits : list or str
            Defines SPOC bits corresponding to bad quality data. Can be one of:

            - "default" - mask bits defined by `default_bad_spoc_bits`.
            - "all" - mask all data with a SPOC quality flag.
            - "none" - mask no data.
            - list - mask custom bits provided in list.
            More information about the SPOC quality flags can be found in Section 9 of the TESS Science
            Data Products Description Document.

        Returns
        -------
        spoc_quality_mask : ndarray
            A boolean mask, with the same length as `spoc_quality`.
            Good quality data is `True` and bad quality data is `False`.
        bad_spoc_bits : list or str
            The SPOC bits used to define bad quality data.
        """

        # Assign default value to `spoc_quality` if none is provided by user:
        if spoc_quality is None:
            if not hasattr(self, "quality"):
                raise AttributeError(
                    "Must run `get_data()` before computing SPOC quality mask, if you have not provided `spoc_quality`."
                )
            spoc_quality = self.quality

        # Define bitmask for user-defined bad SPOC quality bits.
        bad_spoc_bit_value = create_bad_bitmask(
            bad_bits=bad_spoc_bits, default_bad_bits=default_bad_spoc_bits
        )

        # Define SPOC quality mask.
        if bad_spoc_bit_value == "all":
            spoc_quality_mask = spoc_quality == 0
        else:
            spoc_quality_mask = spoc_quality & bad_spoc_bit_value == 0

        # Add non-science data in sector 3 to quality mask, as defined in data release notes.
        if self.sector == 3:
            t = self.time_original - self.timecorr_original
            spoc_quality_mask = np.logical_and(
                spoc_quality_mask, np.logical_and(t >= 1385.89663, t <= 1406.29247)
            )

        return spoc_quality_mask, bad_spoc_bits

    def _bg_linear_model(
        self,
        bad_spoc_bits: Union[list, str] = "default",
        data_chunks: Optional[Union[list, np.ndarray]] = None,
        sl_method: str = "pca",
        sl_knot_width: float = 20,
        sl_sigma: float = 5,
        sl_niter_clip: int = 3,
        knot_width: float = 0.5,
        window_length: Optional[float] = 2,
        sigma: float = 3,
        niter_clip: int = 3,
        red_chi2_tol: float = 2,
        reshape: bool = True,
        progress_bar: bool = True,
        diagnostic_plot: bool = False,
        outdir: str = "",
        **kwargs,
    ):
        """
        Calculate the background flux using linear modelling. There are two components:
            1. Scattered light model: use PCA and linear modelling to compute a scattered light model at each cadence.
            2. Star model: use linear modelling to compute a model for the rest of the background (e.g. stars) at
               each cadence.
        These two components get summed to create a global background model.

        Step one is done using the `_create_scattered_light_model` function. Step two loops through each pixel to model
        the scattered light corrected flux:

        - There is the option to only model cadences in a time window around when the pixel is in the TPF region. This is
        controlled with the parameter `window_length`.
        - The design matrix is defined with a 3rd degree B-spline in time. The knots of the spline can be controlled with
        the parameter `knot_width`.
        - The cadences flagged as bad quality by `bad_spoc_bits` do not have a BG model (it will be nan).

        Parameters
        ----------
        bad_spoc_bits : list or str
            Defines SPOC bits corresponding to bad quality data. Can be one of:

            - "default" - mask bits defined by `default_bad_spoc_bits`.
            - "all" - mask all data with a SPOC quality flag.
            - "none" - mask no data.
            - list - mask custom bits provided in list.
            Data that is masked will not be used when creating the background model and its resulting background model will be NaN.
            More information about the SPOC quality flags can be found in Section 9 of the TESS Science Data Products
            Description Document.
        data_chunks : ndarray
            A boolean mask that defines chunks of data. The star model will account for discontinuities in the flux
            time-series between chunks. `data_chunks` is also used by `_create_scattered_light_model()`. The array must have
            shape (nchunks, ntimes). If you don't want to split data into chunks, define as `np.asarray([np.full(len(self.time), True)])`.
            If None, chunks will be defined using TESS data downlink times.
        sl_method : str
            Method used to compute scattered light model. One of [`pca`].
        sl_knot_width : float
            Approximate pixel spacing between spline knots used for scattered light model.
        sl_sigma : float
            Sigma threshold used for outlier detection when creating scattered light model.
        sl_niter_clip : int
            Number of iterations of outlier clipping when creating scattered light model. To turn off outlier clipping, set `sl_niter_clip=0`.
        knot_width : float
            Approximate spacing between spline knots used for star model, in days.
        window_length : float
            This defines the width of the window, in days, used for fitting the star model to each pixel. For each pixel, the fitting window
            is centred on the time it is present in the TPF region. If None, the full time-series is modelled. A longer `window_length` will
            increase the runtime of this function.
        sigma : float
            Sigma threshold for outlier detection when creating star model. A larger value of `sigma` will mask less data.
        niter_clip : int
            Number of iterations of outlier clipping when creating star model.

            - To turn off outlier clipping, set `niter_clip` to zero.
            - To clip until no outliers remain, set `niter_clip` to `np.inf`.
        red_chi2_tol: float
            Pixels with reduced chi-squared >= `red_chi2_tol` will be flagged to indicate the star model fit was poor. This flag
            is used by `create_pixel_quality()` and `_create_lc_quality()`.
        reshape : boolean
            If True, the background model is returned with a shape (ntimes, nrows, ncols) i.e. same shape as `self.flux`.
            If False, the background model is returned with a shape (ntimes, npixels) i.e. same shape as `self.all_flux`.
        progress_bar : bool
            If `True`, a progress bar will be displayed for the computation of the star model.
        diagnostic_plot : bool
            If True, shows diagnostic plots to check the scattered light model and star model.
        outdir : str
            If `diagnostic_plot`, this is the directory into which the plot will be saved.
        kwargs : dict
            Keywords arguments passed to `_create_scattered_light_model` (e.g. `niter`, `ncomponents`)
            and `_create_source_mask` (e.g `target_threshold`, `star_flux_threshold`).

        Returns
        -------
        bg : ndarray
            Background flux estimate. This is the sum of the scattered light model and star model.
        bg_err : ndarray
            Error on background flux estimate.
        sl_model : ndarray
            Scattered light model.
        sl_model_err : ndarray
            Error on scattered light model.
        star_model : ndarray
            Star model.
        star_model_err : ndarray
            Error on star model.
        """

        if not hasattr(self, "all_flux"):
            raise AttributeError("Must run `get_data()` before computing background.")

        # Parameter logic checks
        if window_length is not None and window_length <= 0:
            raise ValueError(
                f"`window_length` must be greater than zero. Not '{window_length}'"
            )
        if sigma <= 0:
            raise ValueError(f"`sigma` must be greater than zero. Not '{sigma}'")
        if niter_clip < 0:
            raise ValueError(
                f"`niter_clip` must be greater than or equal to zero. Not '{niter_clip}'"
            )

        # Define good quality data using user-defined SPOC bits.
        spoc_quality_mask, self.bad_spoc_bits = self._create_spoc_quality_mask(
            bad_spoc_bits=bad_spoc_bits
        )

        # Create and/or check data chunks.
        data_chunks = self._data_chunks(data_chunks)

        # Compute scattered light model
        if sl_method == "pca":
            sl_model, sl_model_err = self._create_scattered_light_model(
                knot_width=sl_knot_width,
                sigma=sl_sigma,
                niter_clip=sl_niter_clip,
                data_chunks=data_chunks,
                spoc_quality_mask=spoc_quality_mask,
                diagnostic_plot=diagnostic_plot,
                outdir=outdir,
                **kwargs,
            )
        else:
            raise ValueError(
                "`sl_method` must be one of: [`pca`]. Not `{0}`".format(sl_method)
            )

        # Remove scattered light from flux
        sl_corr_flux = self.all_flux - sl_model
        sl_corr_flux_err = np.sqrt(
            np.nansum([self.all_flux_err**2, sl_model_err**2], axis=0)
        )
        # If both errors have nan value, propagate nan:
        sl_corr_flux_err = np.where(
            np.logical_and(np.isnan(self.all_flux_err), np.isnan(sl_model_err)),
            np.nan,
            sl_corr_flux_err,
        )
        self.sl_method = sl_method

        # Identify nans in SL corrected flux (e.g. SL model failed).
        nan_mask = np.isnan(sl_corr_flux)

        # Create mask for moving target.
        source_mask = self._create_source_mask(
            include_stars=False, include_target=True, all_flux=True, **kwargs
        )

        # Define equally spaced knots between bounds, with spacing approximately equal to `knot_width`.
        # It is important to include an extra knot at the lower bound - this gives proper behaviour of spline.
        t = self.time
        knots = np.linspace(
            t.min(),
            t.max(),
            int(np.round((t.max() - t.min()) / knot_width)) + 1,
        )[:-1]

        # If `knot_width` is too large for the dataset, `knots` can be returned as an empty array.
        # The function will run, but we need to ensure that there is always one knot at the lower bound.
        if len(knots) == 0:
            knots = np.asarray([t.min()])

        # Warn the user if there are no interior knots. This can lead to a poorly constrained spline.
        if len(knots) == 1:
            logger.warning(
                "There are no interior knots with a `knot_width` of {0} days. Try re-defining `knot_width` to a smaller value.".format(
                    knot_width
                )
            )

        # Create design matrix - third degree b-spline in time, with a global intercept
        X = np.asarray(
            dmatrix(
                "bs(t, knots=knots, degree=3, include_intercept=False)",
                {"t": t, "knots": knots},
            )
        )
        # Break design matrix into data chunks.
        X = np.hstack([X * chunk[:, None] for chunk in data_chunks])  # type: ignore
        # Remove components that don't contribute
        X = X[:, X.sum(axis=0) != 0]

        # Compute half-width of fitting window, in cadences
        if window_length is not None:
            ncadences = np.round(
                window_length / 2 / np.nanmedian(np.diff(self.time))
            ).astype(int)

        # Initialise priors on LM components (broad normal distributions). Prior on sigma uses saturation level.
        prior_mu = np.zeros(X.shape[1])
        prior_sigma = np.ones(X.shape[1]) * 1e5

        # Initialise star model.
        star_model, star_model_err = (
            np.full_like(self.all_flux, np.nan),
            np.full_like(self.all_flux, np.nan),
        )

        # Initialise reduced chi-squared
        red_chi2 = np.full(len(self.pixels), np.nan)

        logger.info("Started computation of star model.")
        start_time = time.time()
        # Loop through each pixel and fit time-series
        for pdx in tqdm(range(len(self.pixels)), disable=not progress_bar):
            # If all flux values are nan (e.g. non-science pixels), star model will also be nan.
            if np.isnan(sl_corr_flux[:, pdx]).all():
                continue

            # Good quality cadences with no nans.
            mask = np.logical_and(spoc_quality_mask, ~nan_mask[:, pdx])

            # If using `window_length`, create a mask that defines a window around when the pixel is in TPF region.
            if window_length is not None:
                # Create a copy of target mask for pdx
                window = self.target_mask[:, pdx].copy()

                # Define window around time when pixel is in TPF.
                true_indices = np.nonzero(window)[0]
                indices_to_set = true_indices[:, None] + np.arange(
                    -ncadences, ncadences + 1
                )

                # Clip the indices to ensure they are within the array bounds
                clipped_indices = np.clip(indices_to_set.flatten(), 0, window.size - 1)

                # Set the elements at the calculated indices to True
                window[clipped_indices] = True

                # Add to mask
                mask = np.logical_and(mask, window)

                # Create mask for model predicition - only when the pixel is in TPF region.
                mask_predict = np.logical_and(mask, self.target_mask[:, pdx])
            else:
                # Create mask for model predicition - all times.
                mask_predict = mask.copy()

            # Remove pixels containing target for fitting
            mask = np.logical_and(mask, ~source_mask[:, pdx])

            # Prime while loop used for outlier clipping.
            n_clip = np.inf
            i_clip = 0
            exit_loop = False
            while n_clip != 0 and i_clip <= niter_clip:
                # If there are no times to fit, star model and error will be nan for pixel at all times.
                if (~mask).all():
                    logger.warning(
                        "When computing the star model for pixel {} (row {}, column {}) in iteration {}, there were no times to fit. The model was set to nan at all times.".format(
                            pdx, *self.pixels[pdx], i_clip
                        )
                    )

                    exit_loop = True
                    break

                # Get flux time-series for pixel, excluding masked data
                y, yerr = sl_corr_flux[mask, pdx], sl_corr_flux_err[mask, pdx]

                # Use weighted Bayesian LS.
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", message="divide by zero encountered in divide"
                    )
                    warnings.filterwarnings(
                        "ignore", message="invalid value encountered in divide"
                    )
                    sigma_w_inv = X[mask].T.dot(X[mask] / yerr[:, None] ** 2) + np.diag(
                        1 / prior_sigma**2
                    )
                    B = X[mask].T.dot(y / yerr**2) + prior_mu / prior_sigma**2

                # Find the best-fitting weights
                w = np.linalg.solve(sigma_w_inv, B)

                # Compute residuals, scaled by error
                # If yerr is zero, catch warning. We have seen this happen for S1, Cam1, CCD4
                # where all_flux and all_flux_err are zero for several cadences.
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="divide by zero encountered in divide",
                        category=RuntimeWarning,
                    )
                    resid = (y - X[mask].dot(w)) / yerr

                # Clip outliers, if there are remaining iterations.
                if i_clip < niter_clip:
                    n_clip_prev = np.sum(~mask)
                    clipped = np.zeros_like(mask, dtype=bool)
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=AstropyUserWarning)
                        clipped[mask] = ~sigma_clip(resid, sigma=sigma).mask
                        mask = np.logical_and(mask, clipped)
                        # Don't predict model for outliers, but ensure TPF pixels are always included.
                        mask_predict = np.logical_and(
                            mask_predict,
                            np.logical_or(clipped, self.target_mask[:, pdx]),
                        )

                    n_clip = np.sum(~mask) - n_clip_prev
                i_clip += 1

            if not exit_loop:
                # Turn best-fitting weights and errors into a distribution
                wcov = np.linalg.inv(sigma_w_inv)
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            message="covariance is not symmetric positive-semidefinite",
                            category=RuntimeWarning,
                        )
                        wdist = np.random.multivariate_normal(w, wcov, size=100)
                except np.linalg.LinAlgError:
                    logger.warning(
                        "When computing the star model for pixel {} (row {}, column {}), the computation of `wdist` failed with a `LinAlgError`. The model was set to nan at all times.".format(
                            pdx, *self.pixels[pdx]
                        )
                    )
                    break

                # Calculate model distribution
                star_model_dist = X[mask_predict].dot(wdist.T)

                # Compute star model and error from distribution.
                # Catch warnings that arise because of nan pixels.
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", message="Mean of empty slice", category=RuntimeWarning
                    )
                    warnings.filterwarnings(
                        "ignore",
                        message="Degrees of freedom <= 0 for slice.",
                        category=RuntimeWarning,
                    )
                    star_model[mask_predict, pdx], star_model_err[mask_predict, pdx] = (
                        np.nanmean(star_model_dist, axis=1),
                        np.nanstd(star_model_dist, ddof=1, axis=1)
                        / np.sqrt(star_model_dist.shape[1]),
                    )

                # Compute reduced chi-squared using fit data.
                # Catch warnings that arise if denominator is zero (reduced chi squared will be returned as inf).
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="divide by zero encountered in scalar divide",
                    )
                    red_chi2[pdx] = np.sum(resid**2) / (
                        np.sum(mask) - np.linalg.matrix_rank(X[mask])
                    )

        logger.info(
            "Finished computation of star model in {0:.2f} sec.".format(
                time.time() - start_time
            )
        )

        # Flag cadences where star model is nan
        if hasattr(self, "star_nan_mask"):
            if window_length is not None:
                self.star_nan_mask[
                    np.where(np.logical_and(np.isnan(star_model), self.target_mask))
                ] = True
            else:
                self.star_nan_mask[np.where(np.isnan(star_model))] = True

        # Flag pixels where model fit was poor, using reduced chi-squared
        if hasattr(self, "star_fit_mask"):
            self.star_fit_mask = red_chi2 > red_chi2_tol
            logger.info(
                "{0} pixels have poor fitting star model.".format(
                    sum(self.star_fit_mask)
                )
            )

        if diagnostic_plot:
            logger.info("Making diagnostic plots for background model...")

            # Check format of outdir
            if len(outdir) > 0 and not outdir.endswith("/"):
                outdir += "/"

            # Create file name
            file_name = "tess-{0}-s{1:04}-{2}-{3}-shape{4}x{5}-bg_model.png".format(
                str(self.target).replace(" ", "").replace("/", ""),
                self.sector,
                self.camera,
                self.ccd,
                *self.shape,
            )

            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))

            vmin = np.nanpercentile(self.all_flux, 1)
            vmax = np.nanpercentile(self.all_flux, 99)

            ax1.set(ylabel="Pixel Index", title="Raw Flux")
            im = ax1.imshow(
                self.all_flux.T,
                origin="lower",
                aspect="auto",
                interpolation="none",
                vmin=vmin,
                vmax=vmax,
            )
            # Add colorbar
            cbar = fig.colorbar(im, ax=ax1, location="right")
            cbar.set_label("Flux [$e^-/s$]")

            ax2.set(
                ylabel="Pixel Index", title="Background Model (Scattered Light + Star)"
            )
            im = ax2.imshow(
                (star_model + sl_model).T,
                origin="lower",
                aspect="auto",
                interpolation="none",
                vmin=vmin,
                vmax=vmax,
            )
            # Add colorbar
            cbar = fig.colorbar(im, ax=ax2, location="right")
            cbar.set_label("Flux [$e^-/s$]")

            ax3.set(
                xlabel="Time Index",
                ylabel="Pixel Index",
                title="Residuals (excluding target, vmin=-5, vmax=5)",
            )
            im = ax3.imshow(
                np.where(
                    (~source_mask).T, (self.all_flux - star_model - sl_model).T, np.nan
                ),
                origin="lower",
                aspect="auto",
                interpolation="none",
                cmap="RdBu",
                vmin=-5,
                vmax=5,
            )
            # Add colorbar
            cbar = fig.colorbar(im, ax=ax3, location="right")
            cbar.set_label("Flux [$e^-/s$]")

            fig.tight_layout()
            plt.savefig(outdir + file_name)
            plt.close(fig)
            logger.info("Created file: {0}".format(outdir + file_name))

        # Reshape arrays to match `self.flux`
        if reshape:
            sl_model_reshaped = []
            sl_model_err_reshaped = []
            star_model_reshaped = []
            star_model_err_reshaped = []
            for t in range(len(self.time)):
                sl_model_reshaped.append(
                    sl_model[t][self.target_mask[t]].reshape(self.shape)
                )
                sl_model_err_reshaped.append(
                    sl_model_err[t][self.target_mask[t]].reshape(self.shape)
                )
                star_model_reshaped.append(
                    star_model[t][self.target_mask[t]].reshape(self.shape)
                )
                star_model_err_reshaped.append(
                    star_model_err[t][self.target_mask[t]].reshape(self.shape)
                )
            sl_model = np.asarray(sl_model_reshaped)
            sl_model_err = np.asarray(sl_model_err_reshaped)
            star_model = np.asarray(star_model_reshaped)
            star_model_err = np.asarray(star_model_err_reshaped)

        # Combine star model and scattered light model to create global BG model
        bg = sl_model + star_model
        bg_err = np.sqrt(
            np.nansum(
                [
                    star_model_err**2,
                    sl_model_err**2,
                ],
                axis=0,
            )
        )
        # If both errors have nan value, propagate nan:
        bg_err = np.where(
            np.logical_and(
                np.isnan(star_model_err),
                np.isnan(sl_model_err),
            ),
            np.nan,
            bg_err,
        )

        return (
            bg,
            bg_err,
            sl_model,
            sl_model_err,
            star_model,
            star_model_err,
        )

    def create_aperture(self, method: str = "prf", **kwargs):
        """
        Creates an aperture mask using method `threshold`, `prf` or `ellipse`.
        It creates the `self.aperture_mask` attribute with the 3D mask.

        Parameters
        ----------
        method : str
            Method used for aperture estimation. One of [`threshold`, `prf`, `ellipse`].
        kwargs : dict
            Keywords arguments passed to aperture mask method, e.g
            `self._create_threshold_mask` takes `threshold` and `reference_pixel`.

        Returns
        -------
        """
        # Get mask via chosen method
        if method == "threshold":
            self.aperture_mask, self.ap_param = self._create_threshold_aperture(
                **kwargs
            )
        elif method == "prf":
            self.aperture_mask, self.ap_param = self._create_prf_aperture(**kwargs)
        elif method == "ellipse":
            self.aperture_mask, self.ap_param = self._create_ellipse_aperture(
                return_params=False, **kwargs
            )
        else:
            raise ValueError(
                f"Method must be one of: ['threshold', 'prf', 'ellipse']. Not '{method}'"
            )

        self.ap_method = method

        logger.info(
            "Created aperture using method {0} with {1} = {2}.".format(
                self.ap_method,
                "threshold" if self.ap_method in ["prf", "threshold"] else "R value",
                self.ap_param,
            )
        )

    def _create_threshold_aperture(
        self,
        threshold: float = 3.0,
        reference_pixel: Union[str, Tuple[float, float]] = "center",
        **kwargs,
    ):
        """
        Creates an threshold aperture mask of shape [ntimes, nrows, ncols].
        Pixels with flux values above the median flux value times threshold * MAD * 1.4826
        are set to True, rest are out of the mask.
        If the thresholding method yields multiple contiguous regions, then
        only the region closest to the (col, row) coordinate specified by
        `reference_pixel` is returned.

        For more details see `lightkurve.TargetPixelFile.create_threshold_mask`.

        Parameters
        ----------
        threshold : float
            A value for the number of sigma by which a pixel needs to be
            brighter than the median flux to be included in the aperture mask.
        reference_pixel: (int, int) tuple, 'center', or None
            (row, column) pixel coordinate closest to the desired region.
            For example, use `reference_pixel=(0,0)` to select the region
            closest to the bottom left corner of the target pixel file.
            If 'center' (default) then the region closest to the center pixel
            will be selected. If `None` then all regions will be selected.

        Returns
        -------
        aperture_mask : ndarray
            Boolean numpy array containing `True` for pixels above the
            threshold. Shape is (ntimes, nrows, ncols)
        threshold : float
            Threshold value used to create aperture.
        """
        if (
            not hasattr(self, "all_flux")
            or not hasattr(self, "flux")
            or not hasattr(self, "corr_flux")
        ):
            raise AttributeError(
                "Must run `get_data()`, `reshape_data()` and `background_correction()` before creating aperture."
            )

        if reference_pixel == "center":
            reference_pixel = ((self.shape[0] - 1) / 2, (self.shape[1] - 1) / 2)

        aperture_mask = np.zeros_like(self.flux).astype(bool)

        # mask with value threshold
        median = np.nanmedian(self.corr_flux)
        mad = stats.median_abs_deviation(self.corr_flux.ravel(), nan_policy="omit")

        # iterate over frames
        for nt in range(len(self.time)):
            # Calculate the threshold value and mask
            # std is estimated in a robust way by multiplying the MAD with 1.4826
            aperture_mask[nt] = (
                self.corr_flux[nt] >= (1.4826 * mad * threshold) + median
            )
            # skip frames with zero mask
            if aperture_mask[nt].sum() == 0:
                continue
            # keep all pixels above threshold if asked
            if reference_pixel is None:
                continue

            # find mask patch closest to reference_pixel
            # `label` assigns number labels to each contiguous `True`` values in the threshold
            # mask, this is useful to find unique mask patches and isolate the one closer to
            # `reference pixel`
            labels = ndimage.label(aperture_mask[nt])[0]
            # For all pixels above threshold, compute distance to reference pixel:
            label_args = np.argwhere(labels > 0)
            distances = [
                np.hypot(crd[0], crd[1])
                for crd in label_args
                - np.array([reference_pixel[0], reference_pixel[1]])
            ]
            # Which label corresponds to the closest pixel
            closest_arg = label_args[np.argmin(distances)]
            closest_label = labels[closest_arg[0], closest_arg[1]]
            # update mask with closest patch
            aperture_mask[nt] = labels == closest_label

        return aperture_mask, threshold

    def _create_prf_aperture(self, threshold: float = 0.01, **kwargs):
        """
        Creates an aperture mask from the PRF model.

        Parameters
        ----------
        threshold : float
            Must be in the range [0,1). Only pixels where the prf model >= `threshold`
            will be included in the aperture.
        **kwargs
            Keyword arguments to be passed to `_create_target_prf_model()`.

        Returns
        -------
        aperture_mask : ndarray
            Boolean numpy array, where pixels inside the aperture are 'True'.
            Shape is (ntimes, nrows, ncols).
        threshold : float
            Threshold used to create aperture.
        """

        # Create PRF model
        self.prf_model, self.ap_prf_nan_mask = self._create_target_prf_model(
            all_flux=False, **kwargs
        )

        # Use PRF model to define aperture
        if isinstance(threshold, float) and threshold < 1 and threshold >= 0:
            aperture_mask = self.prf_model >= threshold  # type: ignore
            # No pixels above threshold
            if (~aperture_mask).all():
                logger.warning(
                    f"When computing the PRF aperture, none of the pixels in the PRF model were greater than the threshold ({threshold})."
                )
        else:
            raise ValueError(
                f"Threshold must be a float between 0 and 1. Not '{threshold}'"
            )

        return aperture_mask, threshold

    def _create_target_prf_model(
        self, time_step: Optional[float] = None, all_flux: bool = False, **kwargs
    ):
        """
        Creates a PRF model of the target as a function of time, using the `lkprf` package.
        Since the target is moving, the PRF model per time is made by summing models on a high
        resolution time grid during the exposure. The PRF model represents the fraction of the
        total flux in that pixel at that time and at each time all values sum to one.

        The PRF model can be returned in the all_flux or TPF space, controlled by the
        `all_flux` parameter. In all_flux space the PRF model has shape (ntimes, npixels) and in
        the TPF space the shape is (ntimes, nrows, ncols).

        Parameters
        ----------
        time_step : float
            Resolution of time grid used to build PRF model, in minutes. A smaller time_step
            will increase the runtime, but the PRF model will better match the extended shape
            of the moving target.
            If `None`, a value will be computed based upon the average speed of the target
            during the observation.
        all_flux : boolean
            If True, the PRF model is returned with a shape (ntimes, npixels).
            If False, the PRF model is returned with a shape (ntimes, nrows, ncols).

        Returns
        -------
        prf_model : ndarray
            numpy array containing the PRF model of the moving target.
        prf_nan_mask : ndarray
            Mask to identify cadences where PRF model contained NaNs and was replaced with model from
            preceding/following frame.
        """

        if not hasattr(self, "all_flux"):
            raise AttributeError("Must run `get_data()` before creating PRF model.")

        # If input ephem["time"] is in TDB at spacecraft, convert tstart and tstop to match.
        if self.barycentric:
            tstart = self.tstart
            tstop = self.tstop
        else:
            tstart = self.tstart - self.timecorr_original
            tstop = self.tstop - self.timecorr_original

        # Initialise PRF - don't specify sector => uses post-sector4 models in all cases.
        prf = lkprf.TESSPRF(camera=self.camera, ccd=self.ccd)  # , sector=self.sector)

        # Initialise mask that will flag NaNs in PRF model.
        prf_nan_mask = np.zeros_like(self.time, dtype=bool)

        # If no time_step is given, compute a value based upon the average target speed.
        # Note: some asteroids significantly change speed during the sector. Our tests
        # have shown that defining time_step for the average speed does not significantly
        # affect their PRF models.
        if time_step is None:
            # Average cadence, in minutes
            cadence = np.nanmean(tstop - tstart) * 24 * 60
            # Average target track length, in pixels
            track_length = np.nanmedian(
                np.sqrt(np.sum(np.diff(self.ephemeris, axis=0) ** 2, axis=1))
            )
            # Pixel resolution at which to evaluate PRF model
            resolution = 0.1
            # Time resolution at which to evaluate PRF model
            time_step = (cadence / track_length) * resolution
            # If time_step is greater than observing cadence, set time_step a small fraction less
            # than observing cadence. time_step must be less than cadence for inteprolation.
            if time_step >= cadence:
                time_step = 0.99 * cadence
            logger.info(
                "`_create_target_prf_model()` calculated a time_step of {0} minutes.".format(
                    time_step
                )
            )

        # Use interpolation to get target row,column for high-resolution time grid
        high_res_time = np.linspace(
            tstart[0],
            tstop[-1],
            int(np.ceil((tstop[-1] - tstart[0]) * 24 * 60 / time_step)),
        )
        column_interp = CubicSpline(
            self.ephem["time"].astype(float),
            self.ephem["column"].astype(float),
            extrapolate=False,
        )(high_res_time)
        row_interp = CubicSpline(
            self.ephem["time"].astype(float),
            self.ephem["row"].astype(float),
            extrapolate=False,
        )(high_res_time)

        # Define origin and shape for all_flux
        if all_flux:
            # Origin is minimum row/column (not a pair) and shape is entire all_flux region
            origin = tuple(self.pixels.min(axis=0).astype(int))
            shape = tuple(
                (self.pixels.max(axis=0) - self.pixels.min(axis=0) + 1).astype(int)
            )

        # Build PRF model at each timestamp
        prf_model = []
        recorded_warnings = []
        for t in range(len(self.time)):
            # Define origin and shape per time
            if not all_flux:
                origin = (self.corner[t][0], self.corner[t][1])
                shape = self.shape

            # Find indices in `high_res_time` between corresponding tstart/tstop.
            inds = np.where(
                np.logical_and(high_res_time >= tstart[t], high_res_time <= tstop[t])
            )[0]

            # Get PRF model throughout exposure, sum and normalise.
            # If `row_interp` or `col_interp` contain nans (i.e. outside range of interpolation),
            # then prf.evaluate breaks. In that case, manually define model with nans.
            try:
                # Catch warnings.
                with warnings.catch_warnings(record=True) as recorded_warning:
                    model = prf.evaluate(
                        targets=[
                            (r, c)
                            for r, c in zip(row_interp[inds], column_interp[inds])
                        ],
                        origin=origin,
                        shape=shape,
                    )
                    recorded_warnings.extend(recorded_warning)
                model = sum(model) / np.sum(model)
            except ValueError:
                model = np.full(shape, np.nan)

            # Save PRF models to array, only saving pixels that have been retrieved.
            if all_flux:
                prf_model.append(
                    model[
                        self.pixels[:, 0] - np.asarray(origin[0]),
                        self.pixels[:, 1] - np.asarray(origin[1]),
                    ]
                )
            else:
                prf_model.append(model)

        # Get unique warnings and save to logger.
        for w in list(
            {
                "{0}-{1}".format(w.filename, w.lineno): w for w in recorded_warnings
            }.values()
        ):
            logger.warning("Warning from `prf.evaluate()`: {0}".format(w))

        # If first/last frame contains nans, replace PRF model with following/preceding frame.
        if np.isnan(prf_model).any():
            nan_ind = np.unique(np.where(np.isnan(prf_model))[0])

            # Update mask of NaNs in PRF model.
            prf_nan_mask[nan_ind] = True

            for i in nan_ind:
                # First frame, use following PRF model.
                if i == 0:
                    prf_model[i] = prf_model[i + 1]
                    logger.warning(
                        "The PRF model contained nans in the first frame (cadence number {0}). The model was replaced with that from the following frame (cadence number {1}).".format(
                            self.cadence_number[i], self.cadence_number[i + 1]
                        )
                    )
                # Last frame, use preceding PRF model.
                elif i == len(prf_model) - 1:
                    prf_model[i] = prf_model[i - 1]
                    logger.warning(
                        "The PRF model contained nans in the last frame (cadence number {0}). The model was replaced with that from the preceding frame (cadence number {1}).".format(
                            self.cadence_number[i], self.cadence_number[i - 1]
                        )
                    )
                # Warn user if other nans exist because this is unexpected.
                else:
                    logger.warning(
                        "The PRF model contains unexpected nans in cadence number {0}. This should be investigated.".format(
                            self.cadence_number[i]
                        )
                    )

        return np.asarray(prf_model), prf_nan_mask

    def _create_ellipse_aperture(
        self,
        R: float = 3.0,
        smooth: bool = True,
        return_params: bool = False,
        plot: bool = False,
        **kwargs,
    ):
        """
        Uses second-order moments of 2d flux distribution to compute ellipse parameters
        (cxx, cyy and cxy) and get an aperture mask with pixels inside the ellipse.
        The function can also optionally return the x/y centroids and the ellipse
        parameters (semi-major axis, A, semi-minor axis, B, and position angle, theta).
        Pixels with distance <= R^2 from the pixel center to the target position are
        considered inside the aperture.
        Ref: https://astromatic.github.io/sextractor/Position.html#ellipse-iso-def

        Parameters
        ----------
        R: float
            Value to scale the ellipse, the default is 3.0 which typically represents
            well the isophotal limits of the object.
        smooth: boolean
            Whether to smooth the second-order moments by fitting a 3rd-order polynomial.
            This helps to remove outliers and keep ellipse parameters more stable.
        return_params: boolean
            Return a ndarray with x/y centroids and ellipse parameters computed from
            first- and second-order moments [X_cent, Y_cent, A, B, theta_deg].
        plot: boolean
            Generate a diagnostic plot with first- and second-order moments.

        Returns
        -------
        aperture_mask: ndarray
            Boolean 3D mask array with pixels within the ellipse.
        R : float
            R value used to create aperture.
        ellipse_parameters: ndarray
            If `return_params`, will return centroid and ellipse parameters
            [X_cent, Y_cent, A, B, theta_deg] with shape (5, n_times).
        """
        # create a threshold mask to select pixels to use for moments
        threshold_mask, _ = self._create_threshold_aperture(
            threshold=3.0, reference_pixel="center"
        )

        X, Y, X2, Y2, XY = compute_moments(self.corr_flux, threshold_mask)

        if plot:
            fig, ax = plt.subplots(2, 2, figsize=(9, 7))
            fig.suptitle("Moments", y=0.94)
            ax[0, 0].plot(
                X + self.corner[:, 1], Y + self.corner[:, 0], label="Centroid"
            )
            ax[0, 0].plot(self.ephemeris[:, 1], self.ephemeris[:, 0], label="Ephem")
            ax[0, 0].legend()
            ax[0, 0].set_title("")
            ax[0, 0].set_ylabel("Y")
            ax[0, 0].set_xlabel("X")

            ax[0, 1].plot(self.time, XY, c="tab:blue", lw=1, label="Moments")
            ax[0, 1].set_ylabel("XY")
            ax[0, 1].set_xlabel("Time")

            ax[1, 0].plot(self.time, X2, c="tab:blue", lw=1)
            ax[1, 0].set_ylabel("X2")
            ax[1, 0].set_xlabel("Time")
            ax[1, 1].plot(self.time, Y2, c="tab:blue", lw=1)
            ax[1, 1].set_ylabel("Y2")
            ax[1, 1].set_xlabel("Time")
            if not smooth:
                plt.show()

        # fit a 3rd deg polynomial to smooth X2, Y2 and XY
        # due to orbit projections, some tracks can show change in directions,
        # a 3rd order polynomial can capture this.
        if smooth:
            # mask zeros and outliers
            mask = ~np.logical_or(Y2 == 0, np.logical_or(X2 == 0, XY == 0))
            mask &= ~sigma_clip(Y2, sigma=5).mask
            mask &= ~sigma_clip(X2, sigma=5).mask
            mask &= ~sigma_clip(XY, sigma=5).mask
            if plot:
                # we plot outliers before they are replaced by interp
                ax[0, 1].scatter(
                    self.time[~mask],
                    XY[~mask],
                    c="tab:red",
                    label="Outliers",
                    marker=".",
                    lw=1,
                )
                ax[1, 0].scatter(
                    self.time[~mask], X2[~mask], c="tab:red", marker=".", lw=1
                )
                ax[1, 1].scatter(
                    self.time[~mask],
                    Y2[~mask],
                    c="tab:red",
                    marker=".",
                    lw=1,
                )
            # fit and eval polynomials
            Y2 = Polynomial.fit(self.time[mask], Y2[mask], deg=3)(self.time)
            X2 = Polynomial.fit(self.time[mask], X2[mask], deg=3)(self.time)
            XY = Polynomial.fit(self.time[mask], XY[mask], deg=3)(self.time)
            if plot:
                ax[0, 1].plot(
                    self.time, XY, c="tab:orange", label="Smooth (3rd-deg poly)", lw=1.5
                )
                ax[1, 0].plot(self.time, X2, c="tab:orange", lw=1.5)
                ax[1, 1].plot(self.time, Y2, c="tab:orange", lw=1.5)
                ax[0, 1].legend()
                plt.show()

        if return_params:
            # compute A, B, and theta
            semi_sum = (X2 + Y2) / 2
            semi_sub = (X2 - Y2) / 2
            A = np.sqrt(semi_sum + np.sqrt(semi_sub**2 + XY**2))
            B = np.sqrt(semi_sum - np.sqrt(semi_sub**2 + XY**2))
            theta_rad = np.arctan(2 * XY / (X2 - Y2)) / 2

            # convert theta to degrees and fix angle change when A and B swap
            # due to change in track direction
            theta_deg = np.rad2deg(theta_rad)
            gradA = np.gradient(A)
            idx = np.where(gradA[:-1] * gradA[1:] < 0)[0] + 1
            theta_deg[idx[0] :] += 90

        # compute CXX, CYY and CXY which is a better param for an ellipse
        den = X2 * Y2 - XY**2
        CXX = Y2 / den
        CYY = X2 / den
        CXY = (-2) * XY / den

        # use CXX, CYY, CXY to create an elliptical mask of size R
        aperture_mask = np.zeros_like(self.flux).astype(bool)
        for nt in range(len(self.time)):
            rr, cc = np.mgrid[
                self.corner[nt, 0] : self.corner[nt, 0] + self.shape[0],
                self.corner[nt, 1] : self.corner[nt, 1] + self.shape[1],
            ]

            # for the moment we center the ellipse on the ephemeris to avoid
            # poorly constrained centroid in bad frames and when the background
            # subtraction has remaining artifacts.
            # TODO:
            # iterate in the centroiding on bad frames to remove contaminated pixels
            # and refine solution. That will result in better centroid estimation
            # usable for the ellipse mask center.
            if np.isnan(self.corr_flux[nt]).all():
                # If all pixels at that time are nan (e.g. non-science region), mask should be False.
                aperture_mask[nt] = np.full(self.shape, False)
            else:
                aperture_mask[nt] = inside_ellipse(
                    cc,
                    rr,
                    CXX[nt],
                    CYY[nt],
                    CXY[nt],
                    x0=self.ephemeris[nt, 1],
                    # x0=self.corner[nt, 1] + X[nt],
                    y0=self.ephemeris[nt, 0],
                    # y0=self.corner[nt, 0] + Y[nt],
                    R=R,
                )
        if return_params:
            return (
                aperture_mask,
                R,
                np.array(
                    [self.corner[:, 1] + X, self.corner[:, 0] + Y, A, B, theta_deg]
                ).T,
            )
        else:
            return aperture_mask, R

    def create_pixel_quality(
        self, sat_level: float = 1e5, sat_buffer_rad: int = 1, **kwargs
    ):
        """
        Create 3D pixel quality flags. This is stored in the `self.pixel_quality` attribute.

        Each flag is a bit-wise combination of the following bits:

        Bit - Description

        - 1 - pixel is outside of science array
        - 2 - pixel is in a strap column
        - 3 - pixel is saturated
        - 4 - pixel is within `sat_buffer_rad` pixels of a saturated pixel
        - 5 - pixel has no scattered light correction. Only relevant if `linear_model` background correction was used.
        - 6 - pixel had no background star model, value is nan. Only relevant if `linear_model` background correction was used.
        - 7 - pixel had negative flux value BEFORE background correction was applied. This can happen near bleed columns from saturated stars (e.g. see Sector 6, Camera 1, CCD 4).
        - 8 - pixel has a poor fitting background star model. Only relevant if `linear_model` background correction was used.

        Parameters
        ----------
        sat_level : float
            Flux (e-/s) above which to consider a pixel saturated.
        sat_buffer_rad : int
            Approximate radius of saturation buffer (in pixels) around each saturated pixel.

        Returns
        -------
        """
        if (
            not hasattr(self, "pixels")
            or not hasattr(self, "flux")
            or not hasattr(self, "corr_flux")
        ):
            raise AttributeError(
                "Must run `get_data()`, `reshape_data()` and `background_correction()` before creating pixel quality flags."
            )

        # Pixel mask that identifies non-science pixels
        science_mask = ~np.logical_and(
            np.logical_and(self.pixels.T[0] >= 1, self.pixels.T[0] <= 2048),
            np.logical_and(self.pixels.T[1] >= 45, self.pixels.T[1] <= 2092),
        )

        # Pixel mask that identifies strap columns
        # Must add 44 to straps['Column'] because this is one-indexed from first science pixel.
        strap_mask = np.isin(self.pixels.T[1], straps["Column"] + 44)

        # Pixel mask that identifies saturated pixels
        sat_mask = self.flux > sat_level

        # Pixel mask that identifies negative flux values
        negative_mask = self.flux < 0

        # >>>>> ADD A MASK FOR OTHER SATURATION FEATURES? <<<<<

        # Define bits
        pixel_quality = []
        for t in range(len(self.time)):
            # Define dictionary containing each mask and corresponding binary digit.
            # Masks are reshaped to (len(self.time), self.shape), if necessary
            masks = {
                "science_mask": {
                    "bit": 1,
                    "value": science_mask[self.target_mask[t]].reshape(self.shape),
                },
                "strap_mask": {
                    "bit": 2,
                    "value": strap_mask[self.target_mask[t]].reshape(self.shape),
                },
                "sat_mask": {"bit": 3, "value": sat_mask[t]},
                # Saturation buffer:
                # Computes pixels that are 4-adjacent to a saturated pixel and repeats
                # `sat_buffer_rad` times such that the radius of the saturated buffer
                # mask is approximately `sat_buffer_rad` around each saturated pixel.
                # Excludes saturated pixels themselves.
                "sat_buffer_mask": {
                    "bit": 4,
                    "value": ndimage.binary_dilation(
                        sat_mask[t], iterations=sat_buffer_rad
                    )
                    & ~sat_mask[t],
                },
                # Pixel was not corrected for scattered light. This either applies to no pixels or all pixels
                # at a given time. It is only meaningful if the `linear_model` background correction was used.
                "sl_nan_mask": {
                    "bit": 5,
                    "value": np.full(self.shape, self.sl_nan_mask[t]),
                },
                # Pixel does not have a background star model (value is nan).
                # It is only meaningful if the `linear_model` background correction was used.
                "star_nan_mask": {
                    "bit": 6,
                    "value": self.star_nan_mask[t][self.target_mask[t]].reshape(
                        self.shape
                    ),
                },
                "negative_mask": {"bit": 7, "value": negative_mask[t]},
                # Pixel has a poor fitting background star model.
                # It is only meaningful if the `linear_model` background correction was used.
                "star_fit_mask": {
                    "bit": 8,
                    "value": self.star_fit_mask[self.target_mask[t]].reshape(
                        self.shape
                    ),
                },
            }
            # Compute bit-wise flags
            pixel_quality.append(
                np.sum(
                    [
                        (2 ** (masks[mask]["bit"] - 1)) * masks[mask]["value"]
                        for mask in masks
                    ],
                    axis=0,
                ).astype("int16")
            )
        self.pixel_quality = np.asarray(pixel_quality)

    def to_lightcurve(self, method: str = "aperture", **kwargs):
        """
        Extract lightcurve from the moving TPF, using either `aperture` or `psf` photometry.
        This function creates the `self.lc` attribute, which stores the time series data.

        Parameters
        ----------
        method : str
            Method to extract lightcurve. One of `aperture` or `psf`.
        kwargs : dict
            Keyword arguments, e.g `self._aperture_photometry` takes `bad_bits`,
            `self._psf_photometry` takes `time_bin_size`

        Returns
        -------
        """

        # Initialise lightcurve dictionary
        if not hasattr(self, "lc"):
            self.lc = {}

        # Get lightcurve via aperture photometry
        if method == "aperture":
            self.lc["aperture"] = self._aperture_photometry(**kwargs)

        # Get lightcurve via PSF photometry
        elif method == "psf":
            self.lc["psf"] = self._psf_photometry(**kwargs)

        else:
            raise ValueError(
                f"Method must be one of: ['aperture', 'psf']. Not '{method}'"
            )

    def _psf_photometry(
        self,
        time_bin_size: Optional[float] = None,
        bad_spoc_bits: Union[list, str] = "default",
        progress_bar: bool = True,
        **kwargs,
    ):
        """
        Computes PSF photometry by fitting the amplitude of the target's PRF model to the data.
        The model is fitted using a linear model as in the LFD paper (Hedges et al. 2021).

        This function can optionally fit all of the data in a time window simultaneously,
        effectively binning the data to improve SNR.

        Parameters
        ----------
        time_bin_size : float
            Width of time window, in days. All cadences in the time window will be used to fit the PRF model simultaneously,
            effectively doing data binning. Default is None, which means each cadence is fit independently.
        bad_spoc_bits : list or str
            Defines SPOC bits corresponding to bad quality data. Can be one of:

            - "default" - mask bits defined by `default_bad_spoc_bits`.
            - "all" - mask all data with a SPOC quality flag.
            - "none" - mask no data.
            - list - mask custom bits provided in list.
            Data that is masked will not be used when fitting the PRF model. If all cadences in a binning window are defined
            as bad quality, there will be no lightcurve data for that window.
            More information about the SPOC quality flags can be found in Section 9 of the TESS Science Data Products
            Description Document.
        progress_bar : bool
            If `True`, a progress bar will be displayed for the computation of the PSF photometry.

        Returns:
        --------

        psf_phot : dict
            A dictionary containing the PSF photometry, with the following keys:

            - `time` is the average time in the binning window.
            - `time_uerr`/`time_lerr` are the upper/lower error on time (corresponds to limits of binning window).
                These will be nan if `time_bin_size = None`.
            - `time_corr` is the average barycentric time correction in the binning window.
            - `cadenceno` is the average cadence number, as defined by `tesscube`, in the binning window.
            - `flux` and `flux_err` from the PRF fitting.
            - `TESSmag`, `TESSmag_err`, `TESSmag_uerr`, `TESSmag_lerr`: magnitude, symmetric and asymmetric errors in the TESS bandpass.
            - `red_chi2` is the reduced chi-squared of the fitted PRF model.
            - `spoc_quality` is the combined SPOC quality flag in the binning window.
            - `quality`: cadence quality flag from `_create_lc_quality()`.
            - `n_cadences` is the number of cadences that were used to simultaneously fit the PRF model in the binning window.
            - `qual_frac` is the average fraction of flux in the PRF model that falls on good quality pixels (`self.pixel_quality` = 0).
            - `bg_std` is the standad deviation of the background pixels (where PRF model < 0.1%) in each binning window.
            - `bg_mad` is the median absolute deviation of background pixels (where PRF model < 0.1%) in each binning window.
            - `sun_distance` is the average distance from the Sun (AU) in the binning window.
            - `obs_distance` is the average distance from TESS (AU) in the binning window.
            - `sto_angle` is the average Sun-Target-Observer angle (degrees) in the binning window.
            - `time_bin_size` is the width of window used for binning.
            - `bad_spoc_bits` is the SPOC bits used to define bad quality data.

            Note: if `time_bin_size = None` then `time`, `time_corr`, `cadenceno` and `spoc_quality` are equal to
            `self.time`, `self.timecorr`, `self.cadence_number` and `self.quality`, respectively.
            Note: all values are ndarrays with the same length, except `time_bin_size` which is a float and
            `bad_spoc_bits` which is a list or str.
        """
        if (
            not hasattr(self, "all_flux")
            or not hasattr(self, "flux")
            or not hasattr(self, "corr_flux")
            or not hasattr(self, "pixel_quality")
        ):
            raise AttributeError(
                "Must run `get_data()`, `reshape_data()`, `background_correction()` and `create_pixel_quality()` before doing PSF photometry."
            )

        # Define good quality cadences using user-defined SPOC quality flags.
        spoc_quality_mask, bad_spoc_bits = self._create_spoc_quality_mask(
            bad_spoc_bits=bad_spoc_bits
        )

        # Apply quality mask to data
        time = self.time[spoc_quality_mask]
        timecorr = self.timecorr[spoc_quality_mask]
        prf_model, prf_nan_mask = self._create_target_prf_model()
        prf_model, prf_nan_mask = (
            prf_model[spoc_quality_mask],
            prf_nan_mask[spoc_quality_mask],
        )
        cube = self.corr_flux[spoc_quality_mask]
        cube_err = self.corr_flux_err[spoc_quality_mask]
        spoc_quality = self.quality[spoc_quality_mask]
        cadno = self.cadence_number[spoc_quality_mask]
        pixel_quality = self.pixel_quality[spoc_quality_mask]
        sun_distance = self.obs_params["sun_distance"][spoc_quality_mask]
        obs_distance = self.obs_params["obs_distance"][spoc_quality_mask]
        sto_angle = self.obs_params["sto_angle"][spoc_quality_mask]

        # If all data has been masked, raise warning and return empty arrays.
        if len(time) == 0:
            logger.warning(
                "During PSF photometry, all times were masked and no PSF light curve was derived."
            )
            return {
                "time": np.array([]),
                "time_uerr": np.array([]),
                "time_lerr": np.array([]),
                "time_corr": np.array([]),
                "cadenceno": np.array([]),
                "flux": np.array([]),
                "flux_err": np.array([]),
                "TESSmag": np.array([]),
                "TESSmag_err": np.array([]),
                "TESSmag_uerr": np.array([]),
                "TESSmag_lerr": np.array([]),
                "red_chi2": np.array([]),
                "spoc_quality": np.array([]),
                "quality": np.array([]),
                "n_cadences": np.array([]),
                "quality_fraction": np.array([]),
                "bg_std": np.array([]),
                "bg_mad": np.array([]),
                "sun_distance": np.array([]),
                "obs_distance": np.array([]),
                "sto_angle": np.array([]),
                "time_bin_size": time_bin_size,
                "bad_spoc_bits": bad_spoc_bits,
            }

        # Derive flux prior from Vmag (if available).
        # Use Vmag to Tmag conversion from Woods et al 2021 (T = V - 0.671).
        # Note: interpolation will fail if Vmag is not finite.
        if "vmag" in self.ephem and np.isfinite(self.ephem["vmag"].values).all():
            tmag_prior = (
                CubicSpline(
                    self.ephem["time"].astype(float),
                    self.ephem["vmag"].astype(float),
                    extrapolate=False,
                )(self.time if self.barycentric else self.time - self.timecorr)
                - 0.671
            )
            flux_prior = 10 ** ((TESSmag_zero_point - tmag_prior) / 2.5)
            flux_prior = flux_prior[spoc_quality_mask]
        else:
            flux_prior = np.zeros_like(time)

        # Define time bins
        if time_bin_size is not None:
            dt = time[-1] - time[0]
            cadence = np.nanmedian(np.diff(time))

            if time_bin_size < cadence or time_bin_size > dt:
                raise ValueError(
                    f"`time_bin_size` must be larger than the observing cadence ({cadence:.5f} d) and less than the data span ({dt:.5f} d)."
                )

            else:
                n_bins = int(dt / time_bin_size)
                # Compute the bin edges and indices that fall inside each bin
                bin_edges = np.histogram_bin_edges(time, bins=n_bins)
                bin_index = [
                    np.where((time >= le) & (time <= ra))[0]
                    for le, ra in zip(bin_edges[:-1], bin_edges[1:])
                ]
                # Remove empty bins
                bin_index = [x for x in bin_index if len(x) > 0]

        # No time binning
        else:
            bin_index = np.arange(len(time)).tolist()

        psf_phot = []
        pixel_masks = []
        pixel_qualities = []
        nfails = 0

        # Iterate over each binning window to compute PSF photometry
        for bdx in tqdm(bin_index, total=len(bin_index), disable=not progress_bar):
            # Assign variables inside the loop
            bdx = np.atleast_1d(bdx)
            t = time[bdx]
            f = cube[bdx]
            fe = cube_err[bdx]
            p = prf_model[bdx]
            pn = prf_nan_mask[bdx]
            qu = spoc_quality[bdx]
            cn = cadno[bdx]
            tc = timecorr[bdx]
            pmu = flux_prior[bdx]
            pixel_qualities.append(pixel_quality[bdx])

            # Compute average observing geometry parameters inside binning window
            # Catch warnings that arise if self.obs_params contains empty arrays
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Mean of empty slice",
                    category=RuntimeWarning,
                )
                sun = np.nanmean(sun_distance[bdx])
                obs = np.nanmean(obs_distance[bdx])
                sto = np.nanmean(sto_angle[bdx])

            # Use pixels with PRF value > 0.001% for fitting.
            # This value is small to include all pixels where the PRF has contribution.
            pixel_masks.append(p > 0.00001)
            # Exclude nan values from mask:
            j = np.logical_and(
                pixel_masks[-1], np.logical_and(np.isfinite(f), np.isfinite(fe))
            ).ravel()

            # Compute standard deviation and MAD for BG pixels (PRF model < 0.1%).
            bg_mask = np.logical_and(
                p < 0.001, np.logical_and(np.isfinite(f), np.isfinite(fe))
            )
            # Catch warnings that arise because of nan pixels.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Degrees of freedom <= 0 for slice.",
                    category=RuntimeWarning,
                )
                bg_std = np.nanstd(f[bg_mask], ddof=1)
            bg_mad = stats.median_abs_deviation(f[bg_mask], nan_policy="omit")

            # Derive SPOC quality value in window
            qual = 0
            for q in qu:
                qual |= q

            # Compute average fraction of flux in PRF model that falls on good quality pixels
            qual_frac = p.ravel()[
                np.logical_and(j, (pixel_qualities[-1] == 0).ravel())
            ].sum() / len(bdx)

            # Compute errors on time window
            t_mean = t.mean()
            if time_bin_size is None:
                te_u, te_l = np.nan, np.nan
            else:
                te_u, te_l = np.nanmax(t) - t_mean, t_mean - np.nanmin(t)

            # Create DM from PRF model
            X = p.ravel()[:, None]

            # Initialise prior on flux
            pmu = np.atleast_1d(np.nanmean(pmu))
            psigma = np.ones_like(pmu) * 1e5

            # Use weighted Bayesian LS
            sigma_w_inv = X[j].T.dot(X[j] / fe.ravel()[j, None] ** 2) + np.diag(
                1 / psigma**2
            )

            # Solve linear model
            try:
                # If there are no pixels to fit, raise a LinAlgError.
                if (~j).all():
                    raise np.linalg.LinAlgError()

                # Compute flux
                amp = np.linalg.solve(
                    sigma_w_inv,
                    X[j].T.dot(f.ravel()[j] / fe.ravel()[j] ** 2) + pmu / psigma**2,
                )[0]
                # Compute flux error
                amp_err = (np.linalg.inv(sigma_w_inv).diagonal() ** 0.5)[0]
                # Catch warnings that arise because of divide by zero.
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="divide by zero encountered in scalar divide",
                        category=RuntimeWarning,
                    )
                    # Compute reduced chi-squared (X has 1 component)
                    red_chi2 = np.sum(
                        ((f.ravel() - X.dot(amp).ravel()) ** 2 / (fe.ravel() ** 2))[j],
                        axis=0,
                    ) / (j.sum() - 1)

                # Save results
                psf_phot.append(
                    [
                        t_mean,
                        te_u,
                        te_l,
                        tc.mean(),
                        np.median(cn),
                        amp,
                        amp_err,
                        red_chi2,
                        qual,
                        len(bdx),
                        qual_frac,
                        pn.any(),
                        bg_std,
                        bg_mad,
                        sun,
                        obs,
                        sto,
                    ]
                )

            # Catch linalg errors and fill with nan values for this time
            except np.linalg.LinAlgError:
                psf_phot.append(
                    [
                        t_mean,
                        te_u,
                        te_l,
                        tc.mean(),
                        np.median(cn),
                        np.nan,
                        np.nan,
                        np.nan,
                        qual,
                        len(bdx),
                        qual_frac,
                        pn.any(),
                        bg_std,
                        bg_mad,
                        sun,
                        obs,
                        sto,
                    ]
                )
                nfails += 1
                continue

        if nfails > 0:
            logger.warning(
                f"During PSF photometry, {nfails} cadences did not solve. These cadences will be replaced by `NaN`."
            )

        (
            time,
            time_uerr,
            time_lerr,
            time_corr,
            cadenceno,
            flux,
            flux_err,
            red_chi2,
            spoc_quality,
            n_cadences,
            qual_frac,
            prf_nan_mask,
            bg_std,
            bg_mad,
            sun,
            obs,
            sto,
        ) = np.asarray(psf_phot).T

        # Convert measured flux to TESS magnitude.
        # Flux fraction is one - the fitted amplitude is 100% of the target's flux.
        mag, mag_err, mag_uerr, mag_lerr = calculate_TESSmag(
            flux, flux_err, np.ones_like(flux)
        )

        return {
            "time": time,
            "time_uerr": time_uerr,
            "time_lerr": time_lerr,
            "time_corr": time_corr,
            "cadenceno": cadenceno,
            "flux": flux,
            "flux_err": flux_err,
            "TESSmag": mag,
            "TESSmag_err": mag_err,
            "TESSmag_uerr": mag_uerr,
            "TESSmag_lerr": mag_lerr,
            "red_chi2": red_chi2,
            "spoc_quality": spoc_quality.astype(int),
            "quality": self._create_lc_quality(
                pixel_mask=pixel_masks,
                pixel_quality=pixel_qualities,
                prf_nan_mask=prf_nan_mask,
                psf_flux=flux,
            ),
            "n_cadences": n_cadences.astype(int),
            "quality_fraction": qual_frac,
            "bg_std": bg_std,
            "bg_mad": bg_mad,
            "sun_distance": sun,
            "obs_distance": obs,
            "sto_angle": sto,
            "time_bin_size": time_bin_size,
            "bad_spoc_bits": bad_spoc_bits,
        }

    def _aperture_photometry(self, bad_bits: list = [1, 3, 7], **kwargs):
        """
        Gets flux and BG flux inside aperture and computes flux-weighted centroid.

        Parameters
        ----------
        bad_bits : list
            Bits to mask during computation of aperture flux, BG flux and centroid. These bits correspond
            to the `self.pixel_quality` flags. By default, bits 1 (non-science pixel), 3 (saturated pixel)
            and 7 (negative flux before BG correction) are masked.

        Returns
        -------

        ap_phot : dict
            A dictionary containing the aperture photometry, with the following keys:

            - `time`: equal to `self.time`.
            - `flux`, `flux_err`: sum of flux inside aperture and error.
            - `TESSmag`, `TESSmag_err`, `TESSmag_uerr`, `TESSmag_lerr`: magnitude, symmetric and asymmetric errors in the TESS bandpass.
            - `bg`, `bg_err`: sum of background flux inside aperture and error.
            - `col_cen`, `col_cen_err`, `row_cen`, `row_cen_err`: flux-weighted centroids inside aperture and errors.
            - `quality`: cadence quality flag from `_create_lc_quality()`.
            - `flux_fraction`: fraction of PRF model flux inside aperture.
            - `n_pix`: number of pixels inside aperture, excluding pixels flagged as `bad_bits`.
            - `n_pix_star`: number of pixels inside aperture, excluding pixels flagged as `bad_bits`, that overlap with star mask.
            - `bg_std`: standad deviation of background pixels (where PRF model < 0.1%).
            - `bg_mad`: median absolute deviation of background pixels (where PRF model < 0.1%).
            - `ra`, `dec`: flux-weighted centroids converted to world coordinates using WCS.
            - `ap_mask`: aperture mask used to create the lightcurve.
            - `ap_method`: method used to create the aperture. One of [`threshold`, `prf`, `ellipse`].
            - `ap_param`: threshold (`threshold` or `prf` method) or R value (`ellipse` method) used to create aperture.
            - `bad_bits`: bits masked during computation of photometry, as defined by user.

            Note: The row and column centroids are one-indexed and correspond to the position in the full FFI, where the
            lower left pixel has the value (1,1).
            Note: `flux` is corrected using `flux_fraction` to represent 100% of the target's flux.
        """

        if (
            not hasattr(self, "all_flux")
            or not hasattr(self, "flux")
            or not hasattr(self, "corr_flux")
            or not hasattr(self, "pixel_quality")
            or not hasattr(self, "aperture_mask")
        ):
            raise AttributeError(
                "Must run `get_data()`, `reshape_data()`, `background_correction()`, `create_pixel_quality()` and `create_aperture()` before doing aperture photometry."
            )

        # Get target's PRF model.
        if not hasattr(self, "prf_model"):
            self.prf_model, self.ap_prf_nan_mask = self._create_target_prf_model()

        # Compute `value` to mask bad bits.
        bad_bit_value = create_bad_bitmask(bad_bits)

        mask = []
        ap_flux = []
        ap_flux_err = []
        ap_bg = []
        ap_bg_err = []
        flux_fraction = []
        bg_std = []
        bg_mad = []

        for t in range(len(self.time)):
            # Combine aperture mask with masking of bad bits.
            mask.append(
                np.logical_and(
                    self.aperture_mask[t],
                    self.pixel_quality[t] & bad_bit_value == 0,
                )
            )

            # Compute flux and bg flux inside aperture (sum values).
            # (If no pixels in mask, these values will be nan.)
            ap_flux.append(np.nansum(self.corr_flux[t][mask[-1]]))
            ap_flux_err.append(np.sqrt(np.nansum(self.corr_flux_err[t][mask[-1]] ** 2)))
            ap_bg.append(np.nansum(self.bg[t][mask[-1]]))
            ap_bg_err.append(np.sqrt(np.nansum(self.bg_err[t][mask[-1]] ** 2)))

            # If all pixels in aperture have nan value, propagate nan:
            if np.isnan(self.corr_flux[t][mask[-1]]).all():
                ap_flux[-1] = np.nan
            if np.isnan(self.corr_flux_err[t][mask[-1]]).all():
                ap_flux_err[-1] = np.nan
            if np.isnan(self.bg[t][mask[-1]]).all():
                ap_bg[-1] = np.nan
            if np.isnan(self.bg_err[t][mask[-1]]).all():
                ap_bg_err[-1] = np.nan

            # Compute fraction of PRF model flux inside aperture, excluding bad_bits.
            flux_fraction.append(np.nansum(self.prf_model[t][mask[-1]]))

            # PRF model should not contain NaNs, but have this catch here just in case.
            if np.isnan(self.prf_model[t][mask[-1]]).all():
                flux_fraction[-1] = np.nan

            # Correct flux to represent 100% of target's flux.
            ap_flux[-1] /= flux_fraction[-1]
            ap_flux_err[-1] /= flux_fraction[-1]

            # Compute standard deviation and MAD for BG pixels (PRF model < 0.1%).
            bg_mask = np.logical_and(
                self.prf_model[t] < 0.001,
                self.pixel_quality[t] & bad_bit_value == 0,
            )
            # Catch warnings that arise because of nan pixels.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Degrees of freedom <= 0 for slice.",
                    category=RuntimeWarning,
                )
                bg_std.append(np.nanstd(self.corr_flux[t][bg_mask], ddof=1))
            bg_mad.append(
                stats.median_abs_deviation(
                    self.corr_flux[t][bg_mask], nan_policy="omit"
                )
            )

        # Compute flux-weighted centroid inside aperture
        # These values are zero-indexed i.e. lower left pixel in TPF is (0,0).
        col_cen, row_cen, col_cen_err, row_cen_err = compute_moments(
            self.corr_flux, np.asarray(mask), second_order=False, return_err=True
        )
        # Replace zero with nan (i.e. no pixels in mask => no centroid measured)
        col_cen[col_cen == 0] = np.nan
        row_cen[row_cen == 0] = np.nan
        col_cen_err[col_cen_err == 0] = np.nan
        row_cen_err[row_cen_err == 0] = np.nan

        # Sum centroid with `self.corner` to get pixel position in full FFI.
        col_cen += self.corner[:, 1]
        row_cen += self.corner[:, 0]

        # Convert measured centroid from (row,col) to (ra,dec) using WCS from tesswcs.
        # Note: `pixel_to_world()` assumes zero-indexing so subtract one from (row,col).
        measured_coords = np.asarray(
            [
                self.wcs.pixel_to_world(col_cen[t] - 1, row_cen[t] - 1)
                for t in range(len(self.time_original))
            ]
        )

        # Convert measured flux to TESS magnitude.
        # Flux fraction is always one - the flux has already been corrected with `flux_fraction`.
        mag, mag_err, mag_uerr, mag_lerr = calculate_TESSmag(
            np.asarray(ap_flux), np.asarray(ap_flux_err), np.ones_like(ap_flux)
        )

        return {
            "time": self.time,
            "flux": np.asarray(ap_flux),
            "flux_err": np.asarray(ap_flux_err),
            "TESSmag": mag,
            "TESSmag_err": mag_err,
            "TESSmag_uerr": mag_uerr,
            "TESSmag_lerr": mag_lerr,
            "bg": np.asarray(ap_bg),
            "bg_err": np.asarray(ap_bg_err),
            "col_cen": col_cen,
            "col_cen_err": col_cen_err,
            "row_cen": row_cen,
            "row_cen_err": row_cen_err,
            "quality": self._create_lc_quality(
                pixel_mask=self.aperture_mask,
                pixel_quality=self.pixel_quality,
                prf_nan_mask=self.ap_prf_nan_mask,
                bad_bit_value=bad_bit_value,
            ),
            "flux_fraction": np.asarray(flux_fraction),
            "n_pix": np.asarray(mask).sum(axis=(1, 2)),
            "n_pix_star": np.logical_and(
                mask,
                self._create_source_mask(
                    include_target=False, include_stars=True, all_flux=False, **kwargs
                ),
            ).sum(axis=(1, 2)),
            "bg_std": np.asarray(bg_std),
            "bg_mad": np.asarray(bg_mad),
            "ra": [coord.ra.value for coord in measured_coords],
            "dec": [coord.dec.value for coord in measured_coords],
            "ap_mask": self.aperture_mask,
            "ap_method": self.ap_method,
            "ap_param": self.ap_param,
            "bad_bits": ",".join([str(bit) for bit in bad_bits]),
        }

    def _create_lc_quality(
        self,
        pixel_mask: Union[list, np.ndarray],
        pixel_quality: Union[list, np.ndarray],
        prf_nan_mask: np.ndarray,
        bad_bit_value: Optional[int] = None,
        psf_flux: Optional[np.ndarray] = None,
    ):
        """
        Creates quality flags for lightcurve. This is defined independently of SPOC quality flags.
        This function is called internally by `_aperture_photometry()` and `_psf_photometry()`.

        If you are creating LC quality for aperture photometry:

        - `pixel_mask` should correspond to the pixels inside the aperture mask.
        - `bad_bit_value` should correspond to `bad_bits` defined by user in `_aperture_photometry()`.
        - `psf_flux` should be None.

        If you are creating LC quality for PSF photometry:

        - `pixel_mask` should correspond to the pixels that were used to fit the PRF model in `_psf_photometry()`.
        - `bad_bit_value` should be None.
        - `psf_flux` should be the PSF flux time-series from `_psf_photometry()`.

        The flag is a bit-wise combination of the following bits:

        Bit - Description

        - 1 - no pixels inside mask.
        - 2 - at least one non-science pixel inside mask.
        - 3 - at least one pixel inside mask is in a strap column.
        - 4 - at least one saturated pixel inside mask.
        - 5 - at least one pixel inside mask is 4-adjacent to a saturated pixel.
        - 6 - all pixels inside aperture are `bad_bits`. Only defined if `bad_bit_value` is not None (relevant for aperture photometry).
        - 7 - PRF model contained nans.
        - 8 - at least one pixel inside mask does not have scattered light correction. Only relevant if `linear_model` background correction was used.
        - 9 - at least one pixel inside mask had no star model (value is nan). Only relevant if `linear_model` background correction was used.
        - 10 - at least one pixel inside mask had negative value BEFORE background correction was applied.
        - 11 - PSF fit failed due to singular matrix (see np.linalg.LinAlgError) or because all pixels used to fit the PRF model had NaN flux values. Only relevant if `method=psf`.
        - 12 - at least one pixel inside mask had a poor fitting background star model. Only relevant if `linear_model` background correction was used.
        - 13 - mask has two or more discrete regions.

        Parameters
        ----------
        pixel_mask : ndarray or list
            Pixels used to define LC quality. For aperture photometry, this should be the aperture mask. For PSF photometry, this
            should be the pixels used to fit the PRF model.
        pixel_quality : ndarray or list
            Pixel quality mask, defined by `create_pixel_quality()`. This must have the same shape as `pixel_mask`.
        prf_nan_mask : ndarray
            Cadences where the PRF model is NaN, as returned by `_create_target_prf_model()`. This must have length equal to the first
            dimension of `pixel_mask`.
        bad_bit_value : int
            Value corresponding to `bad_bits`, as defined by user in `_aperture_photometry()`.
            If creating quality mask for PSF photmetry, this should be None.
        psf_flux : ndarray
            PSF flux time-series from `_psf_photometry()`.
            If creating quality mask for aperture photmetry, this should be None.

        Returns
        -------
        lc_quality : ndarray
            Array of lightcurve quality flags with length [ntimes].
        """

        if (bad_bit_value is not None and psf_flux is not None) or (
            bad_bit_value is None and psf_flux is None
        ):
            raise ValueError(
                "Must define either `bad_bit_value` or `psf_flux`, but not both."
            )

        # Define bits
        masks = {
            # No pixels in mask
            "no_pixel_mask": {
                "bit": 1,
                "value": np.array([pixel_mask[t].sum() for t in range(len(pixel_mask))])
                == 0,
            },
            # Non-science pixel in mask
            "science_mask": {
                "bit": 2,
                "value": [
                    (pixel_quality[t][pixel_mask[t]] & 1 != 0).any()
                    for t in range(len(pixel_mask))
                ],
            },
            # Strap in mask
            "strap_mask": {
                "bit": 3,
                "value": [
                    (pixel_quality[t][pixel_mask[t]] & 2 != 0).any()
                    for t in range(len(pixel_mask))
                ],
            },
            # Saturated pixel in mask
            "sat_mask": {
                "bit": 4,
                "value": [
                    (pixel_quality[t][pixel_mask[t]] & 4 != 0).any()
                    for t in range(len(pixel_mask))
                ],
            },
            # Pixel in mask is 4-adjacent to saturated pixel
            "sat_buffer_mask": {
                "bit": 5,
                "value": [
                    (pixel_quality[t][pixel_mask[t]] & 8 != 0).any()
                    for t in range(len(pixel_mask))
                ],
            },
            # All pixels in aperture are `bad_bits`, as defined by user in `_aperture_photometry()`
            # Only relevant for aperture photometry.
            "bad_bit_mask": {
                "bit": 6,
                "value": [
                    (pixel_quality[t][pixel_mask[t]] & bad_bit_value != 0).all()
                    if pixel_mask[t].sum() != 0
                    else False
                    for t in range(len(pixel_mask))
                ]
                if bad_bit_value is not None
                else np.zeros(len(pixel_mask), dtype=bool),
            },
            # PRF model contained nans and was replaced with preceding/following frame.
            "prf_nan_mask": {"bit": 7, "value": prf_nan_mask},
            # Pixel in mask with no scattered light correction
            # Only relevant if `linear_model` background correction was used.
            "sl_nan_mask": {
                "bit": 8,
                "value": [
                    (pixel_quality[t][pixel_mask[t]] & 16 != 0).any()
                    for t in range(len(pixel_mask))
                ],
            },
            # Pixel in aperture with no background star model (value is nan).
            # Only relevant if `linear_model` background correction was used.
            "star_nan_mask": {
                "bit": 9,
                "value": [
                    (pixel_quality[t][pixel_mask[t]] & 32 != 0).any()
                    for t in range(len(pixel_mask))
                ],
            },
            # Negative pixel (before BG correction) in mask
            "negative_mask": {
                "bit": 10,
                "value": [
                    (pixel_quality[t][pixel_mask[t]] & 64 != 0).any()
                    for t in range(len(pixel_mask))
                ],
            },
            # PSF fit failed.
            # Only relevant for PSF photometry.
            "psf_fit_fail": {
                "bit": 11,
                "value": np.isnan(psf_flux)
                if psf_flux is not None
                else np.zeros(len(pixel_mask), dtype=bool),
            },
            # Pixel in mask with a poor fitting background star model.
            # Only relevant if the `linear_model` background correction was used.
            "star_fit_mask": {
                "bit": 12,
                "value": [
                    (pixel_quality[t][pixel_mask[t]] & 128 != 0).any()
                    for t in range(len(pixel_mask))
                ],
            },
            # Discontinuous mask
            "discon_mask": {
                "bit": 13,
                "value": [
                    any(
                        [
                            ndimage.label(
                                p, structure=ndimage.generate_binary_structure(2, 2)
                            )[1]
                            > 1
                            for p in pixel_mask[t]
                        ]
                    )
                    if len(pixel_mask[t].shape) == 3
                    else ndimage.label(
                        pixel_mask[t],
                        structure=ndimage.generate_binary_structure(2, 2),
                    )[1]
                    > 1
                    for t in range(len(pixel_mask))
                ],
            },
            # Add flag for negative pixels (after BG correction) in aperture?
        }

        # Compute bit-wise flags
        lc_quality = np.sum(
            [
                (2 ** (masks[mask]["bit"] - 1)) * np.asarray(masks[mask]["value"])
                for mask in masks
            ],
            axis=0,
        ).astype("int16")

        return np.asarray(lc_quality)

    def to_fits(
        self,
        file_type: str,
        save: bool = False,
        overwrite: bool = True,
        outdir: str = "",
        file_name: Optional[str] = None,
        **kwargs,
    ):
        """
        Convert the moving TPF or lightcurve data to FITS format. This function creates the
        `self.tpf_hdulist` or `self.lc_hdulist` attribute, which can be optionally saved
        to a file.

        Parameters
        ----------
        file_type : str
            Type of file to be converted to FITS. One of [`tpf`, `lc`].
        save : bool
            If True, write the HDUList to a file.
        overwrite : bool
            If `save`, this determines whether to overwrite an existing file with the
            same name.
        outdir : str
            If `save`, this is the directory into which the file will be saved.
        file_name : str
            If `save`, this is the filename that will be used. Format must be '.fits'.
            If no filename is given, a default one will be generated.

        Returns
        -------
        """

        # Make HDUList
        if file_type == "tpf":
            self.tpf_hdulist = self._make_tpf_hdulist()
        elif file_type == "lc":
            self.lc_hdulist = self._make_lc_hdulist()
        else:
            raise ValueError(
                f"`file_type` must be one of: ['tpf', 'lc']. Not '{file_type}'"
            )

        # Write HDUList to file
        if save:
            self._save_hdulist(
                file_type=file_type,
                overwrite=overwrite,
                outdir=outdir,
                file_name=file_name,
            )

    def _make_tpf_hdulist(
        self, ap_masks: list = [], ap_methods: list = [], ap_params: list = []
    ):
        """
        Make HDUList for moving TPF, using similar format to SPOC.

        Parameters
        ----------
        ap_masks : list
            A list of 3D aperture masks.
            If empty, `self.aperture_mask` will be used.
        ap_methods : list
            A list of the methods (`threshold`, `prf` or `ellipse`) used to create the aperture masks.
            If empty, `self.ap_method` will be used.
        ap_params : list
            A list of the parameters used to create the aperture masks.
            If empty, `self.ap_param` will be used.

        Returns
        -------
        hdulist : astropy.io.fits.HDUList
            HDUList for moving TPF.
        """

        # Attribute checks
        if (
            not hasattr(self, "all_flux")
            or not hasattr(self, "flux")
            or not hasattr(self, "corr_flux")
            or not hasattr(self, "pixel_quality")
        ):
            raise AttributeError(
                "Must run `get_data()`, `reshape_data()`, `background_correction()` and `create_pixel_quality()` before creating TPF HDUList."
            )

        # Assign default values to `ap_masks`, `ap_methods` and `ap_params` if none are provided by user:
        if len(ap_masks) == 0 and len(ap_methods) == 0 and len(ap_params) == 0:
            if not hasattr(self, "aperture_mask"):
                raise AttributeError(
                    "Must run `create_aperture()` before creating TPF HDUList if you have not provided `ap_masks`, `ap_methods` and `ap_params`."
                )
            ap_masks = [self.aperture_mask]
            ap_methods = [self.ap_method]
            ap_params = [self.ap_param]

        # Logic check
        if (
            len(ap_masks) != len(ap_methods)
            or len(ap_masks) != len(ap_params)
            or len(ap_params) != len(ap_methods)
        ):
            raise ValueError(
                "`ap_masks`, `ap_methods` and `ap_params` must have the same length."
            )

        # Initialize HDUList
        tpf_hdulist = [
            self._make_primary_hdu(
                file_type="tpf",
                ap_masks=ap_masks,
                ap_methods=ap_methods,
                ap_params=ap_params,
            )
        ]

        # Compute WCS header
        wcs_header = make_wcs_header(self.shape)

        # Compute form and dimensions
        tform = str(self.corr_flux[0].size) + "E"
        dims = str(self.corr_flux[0].shape[::-1])

        # Offset between expected target position and center of TPF
        pos_corr1 = self.ephemeris[:, 1] - self.corner[:, 1] - 0.5 * (self.shape[1] - 1)
        pos_corr2 = self.ephemeris[:, 0] - self.corner[:, 0] - 0.5 * (self.shape[0] - 1)

        # Define SPOC-like FITS columns
        cols = [
            # Times in TDB at SS barycenter.
            fits.Column(
                name="TIME",
                format="D",
                unit="BJD - 2457000, days",
                disp="D14.7",
                array=self.time,
            ),
            # Barycentric time correction.
            fits.Column(
                name="TIMECORR",
                format="E",
                unit="d",
                disp="E14.7",
                array=self.timecorr,
            ),
            # Cadence number, as defined by tesscube.
            fits.Column(name="CADENCENO", format="I", array=self.cadence_number),
            fits.Column(
                name="FLUX",
                format=tform,
                dim=dims,
                unit="e-/s",
                disp="E14.7",
                array=self.corr_flux,
            ),
            fits.Column(
                name="FLUX_ERR",
                format=tform,
                dim=dims,
                unit="e-/s",
                disp="E14.7",
                array=self.corr_flux_err,
            ),
            fits.Column(
                name="FLUX_BKG",
                format=tform,
                dim=dims,
                unit="e-/s",
                disp="E14.7",
                array=self.bg,
            ),
            fits.Column(
                name="FLUX_BKG_ERR",
                format=tform,
                dim=dims,
                unit="e-/s",
                disp="E14.7",
                array=self.bg_err,
            ),
            fits.Column(
                name="QUALITY",
                format="J",
                disp="B16.16",
                array=self.quality,
            ),
            fits.Column(
                name="POS_CORR1",
                format="E",
                unit="pixel",
                disp="E14.7",
                array=pos_corr1,
            ),
            fits.Column(
                name="POS_CORR2",
                format="E",
                unit="pixel",
                disp="E14.7",
                array=pos_corr2,
            ),
        ]

        # Create SPOC-like table HDU
        table_hdu_spoc = fits.BinTableHDU.from_columns(
            cols, header=fits.Header([*self.cube.output_first_header.cards])
        )

        # Add WCS cards for relevant columns and insert after TDIMn keyword
        for ext in [4, 5, 6, 7]:
            for card in reversed(
                list(get_wcs_header_by_extension(wcs_header, ext=ext).cards)
            ):
                table_hdu_spoc.header.insert(f"TDIM{ext}", card, after=True)

        # Add comments to column keywords
        add_column_header_comments(table_hdu_spoc)

        table_hdu_spoc.header["EXTNAME"] = ("PIXELS", "name of extension")

        # Remove irrelevent keywords
        for keyword in ["TICID", "RA_OBJ", "DEC_OBJ"]:
            table_hdu_spoc.header.remove(keyword)

        # Update existing keywords
        for keyword in ["TSTART", "TSTOP", "DATE-OBS", "DATE-END", "OBJECT"]:
            table_hdu_spoc.header.set(
                keyword,
                tpf_hdulist[0].header[keyword],
                comment=tpf_hdulist[0].header.comments[keyword],
            )
        table_hdu_spoc.header.set(
            "TELAPSE", table_hdu_spoc.header["TSTOP"] - table_hdu_spoc.header["TSTART"]
        )
        table_hdu_spoc.header.set(
            "LIVETIME",
            table_hdu_spoc.header["TELAPSE"] * table_hdu_spoc.header["DEADC"],
            comment="[d] TELAPSE multiplied by DEADC",
        )
        table_hdu_spoc.header.set("BACKAPP", 1.0)

        # Save to TPF HDUList
        tpf_hdulist.append(table_hdu_spoc)

        # Create image HDU containing flattened aperture(s).
        # Aperture has values 0 and 2, where 0/2 indicates the pixel is outside/inside the aperture.
        # This format is used to be consistent with the aperture HDU from SPOC.
        for i, ap_mask in enumerate(ap_masks):
            aperture_hdu = fits.ImageHDU(
                data=(np.nansum(ap_mask, axis=0) > 0).astype("int32") * 2,
                header=fits.Header(
                    [*self.cube.output_secondary_header.cards, *wcs_header.cards]
                ),
            )
            aperture_hdu.header["EXTNAME"] = (
                "APERTURE{0}".format(i if len(ap_masks) > 1 else ""),
                "name of extension",
            )
            aperture_hdu.header.set("OBJECT", self.target, comment="object name")
            aperture_hdu.header.comments["NAXIS1"] = "length of dimension 1"
            aperture_hdu.header.comments["NAXIS2"] = "length of dimension 2"

            # Remove irrelevent keywords
            for keyword in ["TICID", "RA_OBJ", "DEC_OBJ"]:
                aperture_hdu.header.remove(keyword)

            # Save to TPF HDUList
            tpf_hdulist.append(aperture_hdu)

        # Define extra FITS columns
        cols = [
            # Times in TDB at SS barycenter.
            fits.Column(
                name="TIME",
                format="D",
                unit="BJD - 2457000, days",
                disp="D14.7",
                array=self.time,
            ),
            # Cadence number, as defined by tesscube.
            fits.Column(name="CADENCENO", format="I", array=self.cadence_number),
            # Original TESS FFI timestamps, in TDB at SS barycenter.
            # This was calculated for the center of the FFI, not the target position.
            fits.Column(
                name="ORIGINAL_TIME",
                format="D",
                unit="BJD - 2457000, days",
                disp="D14.7",
                array=self.time_original,
            ),
            # Original barycentric time correction.
            # This was calculated for the center of the FFI, not the target position.
            fits.Column(
                name="ORIGINAL_TIMECORR",
                format="E",
                unit="d",
                disp="E14.7",
                array=self.timecorr_original,
            ),
            # Predicted position of target, in world coordinates.
            fits.Column(
                name="RA_PRED",
                format="E",
                unit="deg",
                disp="E14.7",
                array=[coord.ra.value for coord in self.coords],
            ),
            fits.Column(
                name="DEC_PRED",
                format="E",
                unit="deg",
                disp="E14.7",
                array=[coord.dec.value for coord in self.coords],
            ),
            # Original FFI column of lower-left pixel in TPF.
            fits.Column(
                name="CORNER1",
                format="I",
                unit="pixel",
                array=self.corner[:, 1],
            ),
            # Original FFI row of lower-left pixel in TPF.
            fits.Column(
                name="CORNER2",
                format="I",
                unit="pixel",
                array=self.corner[:, 0],
            ),
            # 3D pixel quality flags
            fits.Column(
                name="PIXEL_QUALITY",
                format=tform.replace("E", "I"),
                dim=dims,
                disp="B16.16",
                array=self.pixel_quality,
            ),
        ]

        # Add aperture(s) as a function of time.
        for i, ap_mask in enumerate(ap_masks):
            # Aperture has values 0 and 2, where 0/2 indicates the pixel is outside/inside the aperture.
            # This format is used to be consistent with the aperture HDU from SPOC.
            cols.append(
                fits.Column(
                    name="APERTURE{0}".format(i if len(ap_masks) > 1 else ""),
                    format=tform.replace("E", "J"),
                    dim=dims,
                    array=ap_mask.astype("int32") * 2,
                ),
            )

        # Create table HDU for extra columns
        table_hdu_extra = fits.BinTableHDU.from_columns(cols)
        # Add comments to column keywords
        add_column_header_comments(table_hdu_extra)
        table_hdu_extra.header["EXTNAME"] = ("EXTRAS", "name of extension")
        tpf_hdulist.append(table_hdu_extra)

        # Return hdulist
        hdulist = fits.HDUList(tpf_hdulist)
        hdulist[0].header.comments["EXTEND"] = "file contains extensions"
        return hdulist

    def _make_lc_hdulist(self, lc: Optional[dict] = None):
        """
        Make HDUList for lightcurve file. This includes separate HDUs for aperture and PSF photometry.

        Parameters
        ----------
        lc : dict
            A dictionary containing aperture and PSF photometry. This must have the same structure as that
            produced by `self.to_lightcurve`. If None, `self.lc` will be used by default.

        Returns
        -------
        hdulist : astropy.io.fits.HDUList
            HDUList for lightcurve file.
        """

        # Attribute checks
        if not hasattr(self, "time"):
            raise AttributeError("Must run `get_data()` before creating LCF HDUList.")

        # Assign default value to `lc` if none is provided by user:
        if lc is None:
            if not hasattr(self, "lc"):
                raise AttributeError(
                    "Must run `to_lightcurve()` before creating LCF HDUList if you have not provided `lc`."
                )
            lc = self.lc

        # Make a copy of dictionary
        lc = deepcopy(lc)

        # Logic checks
        if len(lc) == 0:
            raise ValueError("Must have at least one lightcurve in `lc`.")

        # Initialize HDUList
        lc_hdulist = [self._make_primary_hdu(file_type="lc")]

        # Ensure lc["aperture"] is at least 1D
        if "aperture" in lc:
            lc["aperture"] = np.atleast_1d(lc["aperture"])
        else:
            lc["aperture"] = []

        # Create HDU for each aperture
        for i, ap_lc in enumerate(lc["aperture"]):
            cols_ap = [
                # Times in TDB at SS barycenter.
                fits.Column(
                    name="TIME",
                    format="D",
                    unit="BJD - 2457000, days",
                    disp="D14.7",
                    array=self.time,
                ),
                # Cadence number, as defined by tesscube.
                fits.Column(name="CADENCENO", format="I", array=self.cadence_number),
                # SPOC quality flag
                fits.Column(
                    name="QUALITY",
                    format="J",
                    disp="B16.16",
                    array=self.quality,
                ),
                # --------------
                # Aperture photometry
                # Sum of flux inside aperture and err
                fits.Column(
                    name="FLUX",
                    format="E",
                    unit="e-/s",
                    disp="E14.7",
                    array=ap_lc["flux"],
                ),
                fits.Column(
                    name="FLUX_ERR",
                    format="E",
                    unit="e-/s",
                    disp="E14.7",
                    array=ap_lc["flux_err"],
                ),
                # TESS magnitude and errors
                fits.Column(
                    name="TESSMAG",
                    format="E",
                    unit="mag",
                    disp="E14.7",
                    array=ap_lc["TESSmag"],
                ),
                fits.Column(
                    name="TESSMAG_ERR",
                    format="E",
                    unit="mag",
                    disp="E14.7",
                    array=ap_lc["TESSmag_err"],
                ),
                fits.Column(
                    name="TESSMAG_UERR",
                    format="E",
                    unit="mag",
                    disp="E14.7",
                    array=ap_lc["TESSmag_uerr"],
                ),
                fits.Column(
                    name="TESSMAG_LERR",
                    format="E",
                    unit="mag",
                    disp="E14.7",
                    array=ap_lc["TESSmag_lerr"],
                ),
                # Sum of BG flux inside aperture and err
                fits.Column(
                    name="FLUX_BKG",
                    format="E",
                    unit="e-/s",
                    disp="E14.7",
                    array=ap_lc["bg"],
                ),
                fits.Column(
                    name="FLUX_BKG_ERR",
                    format="E",
                    unit="e-/s",
                    disp="E14.7",
                    array=ap_lc["bg_err"],
                ),
                # Measured column centroid and err
                fits.Column(
                    name="MOM_CENTR1",
                    format="E",
                    unit="pixel",
                    disp="E14.7",
                    array=ap_lc["col_cen"],
                ),
                fits.Column(
                    name="MOM_CENTR1_ERR",
                    format="E",
                    unit="pixel",
                    disp="E14.7",
                    array=ap_lc["col_cen_err"],
                ),
                # Measured row centroid and err
                fits.Column(
                    name="MOM_CENTR2",
                    format="E",
                    unit="pixel",
                    disp="E14.7",
                    array=ap_lc["row_cen"],
                ),
                fits.Column(
                    name="MOM_CENTR2_ERR",
                    format="E",
                    unit="pixel",
                    disp="E14.7",
                    array=ap_lc["row_cen_err"],
                ),
                # Measured position of target, in world coordinates.
                fits.Column(
                    name="RA",
                    format="E",
                    unit="deg",
                    disp="E14.7",
                    array=ap_lc["ra"],
                ),
                fits.Column(
                    name="DEC",
                    format="E",
                    unit="deg",
                    disp="E14.7",
                    array=ap_lc["dec"],
                ),
                # Quality from `_create_lc_quality()`
                fits.Column(
                    name="AP_QUALITY",
                    format="I",
                    disp="B16.16",
                    array=ap_lc["quality"],
                ),
                # Fraction of PRF model flux inside aperture.
                fits.Column(
                    name="FLUX_FRACTION",
                    format="E",
                    disp="E14.7",
                    array=ap_lc["flux_fraction"],
                ),
                # Number of pixels inside aperture, excluding pixels flagged as
                # `bad_bits` by the user.
                fits.Column(
                    name="NPIX",
                    format="I",
                    array=ap_lc["n_pix"],
                ),
                # Number of pixels inside aperture, excluding pixels flagged as
                # `bad_bits` by the user, that overlap with star mask.
                fits.Column(
                    name="NPIX_STAR",
                    format="I",
                    array=ap_lc["n_pix_star"],
                ),
                # Standard deviation of background pixels.
                fits.Column(
                    name="BKG_STD",
                    format="E",
                    unit="e-/s",
                    disp="E14.7",
                    array=ap_lc["bg_std"],
                ),
                # Median absolute deviation of background pixels.
                fits.Column(
                    name="BKG_MAD",
                    format="E",
                    unit="e-/s",
                    disp="E14.7",
                    array=ap_lc["bg_mad"],
                ),
            ]

            table_hdu_ap = fits.BinTableHDU.from_columns(cols_ap)
            # Add comments to column keywords
            add_column_header_comments(table_hdu_ap)
            table_hdu_ap.header["EXTNAME"] = (
                "LIGHTCURVE_AP{0}".format(i if len(lc["aperture"]) > 1 else ""),
                "name of extension",
            )

            # Add extra keywords to header
            table_hdu_ap.header.set(
                "AP_TYPE",
                ap_lc["ap_method"],
                comment="method used to create aperture",
            )
            table_hdu_ap.header.set(
                "AP_PAR",
                ap_lc["ap_param"],
                comment="{0} used to create aperture".format(
                    "threshold"
                    if ap_lc["ap_method"] in ["prf", "threshold"]
                    else "R value"
                ),
            )
            table_hdu_ap.header.set(
                "AP_NPIX",
                np.nanmedian([np.nansum(mask) for mask in ap_lc["ap_mask"]]),
                comment="average number of pixels in aperture",
            )
            table_hdu_ap.header.set(
                "BAD_BITS",
                ap_lc["bad_bits"],
                comment="bits excluded during aperture photometry",
            )
            # Average measured TESS magnitude of target
            # Catch warnings that arise if LC is nan at all times.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="All-NaN slice encountered",
                    category=RuntimeWarning,
                )
                mag_header = round(np.nanmedian(ap_lc["TESSmag"]), 3)
            table_hdu_ap.header.set(
                "TESSMAG",
                mag_header if ~np.isnan(mag_header) else None,
                comment="[mag] avg measured TESS magnitude",
            )

            # Add aperture to overall HDUList
            lc_hdulist.append(table_hdu_ap)

        # Create HDU for PSF photometry.
        if "psf" in lc:
            cols_psf = [
                # Average time in binning window, in TDB at SS barycenter
                fits.Column(
                    name="TIME",
                    format="D",
                    unit="BJD - 2457000, days",
                    disp="D14.7",
                    array=lc["psf"]["time"],
                ),
                # Upper and lower error on time (corresponds to limits of binning window)
                fits.Column(
                    name="TIME_UERR",
                    format="E",
                    unit="d",
                    disp="E14.7",
                    array=lc["psf"]["time_uerr"],
                ),
                fits.Column(
                    name="TIME_LERR",
                    format="E",
                    unit="d",
                    disp="E14.7",
                    array=lc["psf"]["time_lerr"],
                ),
                # Average barycentric time correction in binning window
                fits.Column(
                    name="TIMECORR",
                    format="E",
                    unit="d",
                    disp="E14.7",
                    array=lc["psf"]["time_corr"],
                ),
                # Average cadence number, as defined by tesscube, in binning window
                fits.Column(
                    name="CADENCENO",
                    format="I",
                    array=lc["psf"]["cadenceno"],
                ),
                # Combined SPOC quality flag in the cadence binning window
                fits.Column(
                    name="QUALITY",
                    format="J",
                    disp="B16.16",
                    array=lc["psf"]["spoc_quality"],
                ),
                # PSF flux and error
                fits.Column(
                    name="FLUX",
                    format="E",
                    unit="e-/s",
                    disp="E14.7",
                    array=lc["psf"]["flux"],
                ),
                fits.Column(
                    name="FLUX_ERR",
                    format="E",
                    unit="e-/s",
                    disp="E14.7",
                    array=lc["psf"]["flux_err"],
                ),
                # TESS magnitude and error
                fits.Column(
                    name="TESSMAG",
                    format="E",
                    unit="mag",
                    disp="E14.7",
                    array=lc["psf"]["TESSmag"],
                ),
                fits.Column(
                    name="TESSMAG_ERR",
                    format="E",
                    unit="mag",
                    disp="E14.7",
                    array=lc["psf"]["TESSmag_err"],
                ),
                fits.Column(
                    name="TESSMAG_UERR",
                    format="E",
                    unit="mag",
                    disp="E14.7",
                    array=lc["psf"]["TESSmag_uerr"],
                ),
                fits.Column(
                    name="TESSMAG_LERR",
                    format="E",
                    unit="mag",
                    disp="E14.7",
                    array=lc["psf"]["TESSmag_lerr"],
                ),
                # Model fit reduced chi-squared
                fits.Column(
                    name="RED_CHI2",
                    format="E",
                    disp="E14.7",
                    array=lc["psf"]["red_chi2"],
                ),
                # Quality from `_create_lc_quality()`
                fits.Column(
                    name="PSF_QUALITY",
                    format="I",
                    disp="B16.16",
                    array=lc["psf"]["quality"],
                ),
                # Average fraction of flux in PRF model that falls on good quality pixels (`self.pixel_quality` = 0).
                fits.Column(
                    name="QUALITY_FRACTION",
                    format="E",
                    disp="E14.7",
                    array=lc["psf"]["quality_fraction"],
                ),
                # Number of cadences used for simultaneous PSF fit
                fits.Column(
                    name="N_CADENCES",
                    format="I",
                    array=lc["psf"]["n_cadences"],
                ),
                # Standard deviation of background pixels.
                fits.Column(
                    name="BKG_STD",
                    format="E",
                    unit="e-/s",
                    disp="E14.7",
                    array=lc["psf"]["bg_std"],
                ),
                # Median absolute deviation of background pixels.
                fits.Column(
                    name="BKG_MAD",
                    format="E",
                    unit="e-/s",
                    disp="E14.7",
                    array=lc["psf"]["bg_mad"],
                ),
            ]

            # Add observing geometry parameters, if they exist:
            if (
                "sun_distance" in self.ephem
                and "obs_distance" in self.ephem
                and "sto_angle" in self.ephem
            ):
                cols_psf.extend(
                    [
                        # Distance from Sun, in AU.
                        fits.Column(
                            name="SUN_DISTANCE",
                            format="E",
                            unit="AU",
                            disp="E14.7",
                            array=lc["psf"]["sun_distance"],
                        ),
                        # Distance from observer (TESS), in AU.
                        fits.Column(
                            name="OBS_DISTANCE",
                            format="E",
                            unit="AU",
                            disp="E14.7",
                            array=lc["psf"]["obs_distance"],
                        ),
                        # Sun-Target-Observer angle, in degrees.
                        fits.Column(
                            name="STO_ANGLE",
                            format="E",
                            unit="deg",
                            disp="E14.7",
                            array=lc["psf"]["sto_angle"],
                        ),
                    ]
                )

            # Create table HDU
            table_hdu_psf = fits.BinTableHDU.from_columns(cols_psf)
            # Add comments to column keywords
            add_column_header_comments(table_hdu_psf)
            table_hdu_psf.header["EXTNAME"] = ("LIGHTCURVE_PSF", "name of extension")

            # Add extra keywords to header
            table_hdu_psf.header.set(
                "TBINSIZE",
                lc["psf"]["time_bin_size"],
                comment="[d] width of window for PSF fitting",
            )
            table_hdu_psf.header.set(
                "BAD_SPOC",
                ",".join([str(bit) for bit in lc["psf"]["bad_spoc_bits"]])
                if isinstance(lc["psf"]["bad_spoc_bits"], list)
                else lc["psf"]["bad_spoc_bits"],
                comment="bad quality SPOC bits for PRF fitting",
            )
            # Average measured TESS magnitude of target
            # Catch warnings that arise if LC is nan at all times.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="All-NaN slice encountered",
                    category=RuntimeWarning,
                )
                warnings.filterwarnings(
                    "ignore",
                    message="Mean of empty slice",
                    category=RuntimeWarning,
                )

                mag_header = round(np.nanmedian(lc["psf"]["TESSmag"]), 3)
            table_hdu_psf.header.set(
                "TESSMAG",
                mag_header if ~np.isnan(mag_header) else None,
                comment="[mag] avg measured TESS magnitude",
            )

            # Add PSF to overall HDUList
            lc_hdulist.append(table_hdu_psf)

        # Define table HDU for extra columns.
        cols_extra = [
            # Times in TDB at SS barycenter.
            fits.Column(
                name="TIME",
                format="D",
                unit="BJD - 2457000, days",
                disp="D14.7",
                array=self.time,
            ),
            # Barycentric time correction.
            fits.Column(
                name="TIMECORR",
                format="E",
                unit="d",
                disp="E14.7",
                array=self.timecorr,
            ),
            # Original TESS FFI timestamps, in TDB at SS barycenter.
            # This was calculated for the center of the FFI, not the target position.
            fits.Column(
                name="ORIGINAL_TIME",
                format="D",
                unit="BJD - 2457000, days",
                disp="D14.7",
                array=self.time_original,
            ),
            # Original barycentric time correction.
            # This was calculated for the center of the FFI, not the target position.
            fits.Column(
                name="ORIGINAL_TIMECORR",
                format="E",
                unit="d",
                disp="E14.7",
                array=self.timecorr_original,
            ),
            # Cadence number, as defined by tesscube.
            fits.Column(name="CADENCENO", format="I", array=self.cadence_number),
            # Original FFI column of lower-left pixel in TPF.
            fits.Column(
                name="CORNER1",
                format="I",
                unit="pixel",
                array=self.corner[:, 1],
            ),
            # Original FFI row of lower-left pixel in TPF.
            fits.Column(
                name="CORNER2",
                format="I",
                unit="pixel",
                array=self.corner[:, 0],
            ),
            # Predicted position of target in pixel coordinates. Corresponds
            # to position in full FFI.
            fits.Column(
                name="EPHEM1",
                format="E",
                unit="pixel",
                disp="E14.7",
                array=self.ephemeris[:, 1],
            ),
            fits.Column(
                name="EPHEM2",
                format="E",
                unit="pixel",
                disp="E14.7",
                array=self.ephemeris[:, 0],
            ),
            # Predicted position of target, in world coordinates.
            fits.Column(
                name="RA_PRED",
                format="E",
                unit="deg",
                disp="E14.7",
                array=[coord.ra.value for coord in self.coords],
            ),
            fits.Column(
                name="DEC_PRED",
                format="E",
                unit="deg",
                disp="E14.7",
                array=[coord.dec.value for coord in self.coords],
            ),
        ]

        # Add observing geometry parameters, if they exist:
        if (
            "sun_distance" in self.ephem
            and "obs_distance" in self.ephem
            and "sto_angle" in self.ephem
        ):
            cols_extra.extend(
                [
                    # Distance from Sun, in AU.
                    fits.Column(
                        name="SUN_DISTANCE",
                        format="E",
                        unit="AU",
                        disp="E14.7",
                        array=self.obs_params["sun_distance"],
                    ),
                    # Distance from observer (TESS), in AU.
                    fits.Column(
                        name="OBS_DISTANCE",
                        format="E",
                        unit="AU",
                        disp="E14.7",
                        array=self.obs_params["obs_distance"],
                    ),
                    # Sun-Target-Observer angle, in degrees.
                    fits.Column(
                        name="STO_ANGLE",
                        format="E",
                        unit="deg",
                        disp="E14.7",
                        array=self.obs_params["sto_angle"],
                    ),
                ]
            )

        # Create table HDU
        table_hdu_extra = fits.BinTableHDU.from_columns(cols_extra)
        # Add comments to column keywords
        add_column_header_comments(table_hdu_extra)
        table_hdu_extra.header["EXTNAME"] = ("EXTRAS", "name of extension")

        # Add extras to overall HDUList
        lc_hdulist.append(table_hdu_extra)

        # Return hdulist
        hdulist = fits.HDUList(lc_hdulist)
        hdulist[0].header.comments["EXTEND"] = "file contains extensions"
        return hdulist

    def _make_primary_hdu(
        self,
        file_type: str,
        ap_masks: list = [],
        ap_methods: list = [],
        ap_params: list = [],
    ):
        """
        Make primary header for moving TPF and LCF. Initialises using tesscube
        and updates/adds keywords, e.g. properties of target, settings used to create files.

        Parameters
        ----------
        file_type : str
            Type of file to create a primary HDU for. One of ['tpf', 'lc'].
        ap_masks : list
            A list of 3D aperture masks. This is only necessary if `file_type=tpf`.
            If empty, `self.aperture_mask` will be used.
        ap_methods : list
            A list of the methods (`threshold`, `prf` or `ellipse`) used to create the aperture masks.
            This is only necessary if `file_type=tpf`. If empty, `self.ap_method` will be used.
        ap_params : list
            A list of the parameters used to create the aperture masks. This is only necessary if `file_type=tpf`.
            If empty, `self.ap_param` will be used.

        Returns
        -------
        hdu : astropy.io.fits.PrimaryHDU
            Primary HDU to use in moving TPF or lightcurve file.
        """

        # Attribute checks
        if file_type in ["tpf", "lc"]:
            if not hasattr(self, "time") or not hasattr(self, "bg_method"):
                raise AttributeError(
                    "Must run `get_data()` and `background_correction()` before creating primary header for {0}".format(
                        file_type.upper()
                    )
                )
        else:
            raise ValueError(
                f"`file_type` must be one of: ['tpf', 'lc']. Not '{file_type}'"
            )

        # For TPF, assign default values to `ap_masks`, `ap_methods` and `ap_params` if none are provided by user:
        if (
            file_type == "tpf"
            and len(ap_masks) == 0
            and len(ap_methods) == 0
            and len(ap_params) == 0
        ):
            if not hasattr(self, "aperture_mask"):
                raise AttributeError(
                    "Must run `create_aperture()` before creating primary HDU for TPF if you have not provided `ap_masks`, `ap_methods` and `ap_params`."
                )
            ap_masks = [self.aperture_mask]
            ap_methods = [self.ap_method]
            ap_params = [self.ap_param]

        # Logic check
        if (
            len(ap_masks) != len(ap_methods)
            or len(ap_masks) != len(ap_params)
            or len(ap_params) != len(ap_methods)
        ):
            raise ValueError(
                "`ap_masks`, `ap_methods` and `ap_params` must have the same length."
            )

        # Get primary hdu from tesscube
        hdu = self.cube.output_primary_ext.copy()
        hdu.header["EXTNAME"] = ("PRIMARY", "name of extension")

        # Remove irrelevent keywords
        for keyword in [
            "TESSMAG",
            "TICID",
            "PXTABLE",
            "RA_OBJ",
            "DEC_OBJ",
            "PMRA",
            "PMDEC",
            "PMTOTAL",
            "TEFF",
            "LOGG",
            "MH",
            "RADIUS",
            "TICVER",
        ]:
            hdu.header.remove(keyword)

        # Update existing keywords
        hdu.header.set(
            "DATE", datetime.now().strftime("%Y-%m-%d"), comment="file creation date"
        )
        hdu.header.set(
            "TSTART",
            self.time[0],
            comment="observation start time in BTJD of first frame",
        )
        hdu.header.set(
            "TSTOP",
            self.time[-1],
            comment="observation start time in BTJD of last frame",
        )
        hdu.header.set(
            "DATE-OBS", Time(self.time[0] + 2457000, scale="tdb", format="jd").utc.isot
        )
        hdu.header.set(
            "DATE-END", Time(self.time[-1] + 2457000, scale="tdb", format="jd").utc.isot
        )
        hdu.header.set("CREATOR", "tess-asteroids")
        hdu.header.set("PROCVER", __version__)
        hdu.header.set("DATA_REL", comment="SPOC data release version number")
        hdu.header.set("OBJECT", self.target, comment="object name")

        # Add keywords from original FFI header
        hdu.header.set(
            "SPOCDATE",
            self.primary_hdu["DATE"],
            comment="original SPOC FFI creation date",
            after="ORIGIN",
        )
        hdu.header.set(
            "SPOCVER",
            self.primary_hdu["PROCVER"],
            comment="SPOC version that processed FFI data",
            after="SPOCDATE",
        )

        # Add ephemeris date
        hdu.header.set(
            "EPHDATE",
            self.ephem_date,
            comment="ephemeris date",
            after="DATE",
        )

        # Add keywords to describe how file was created
        hdu.header.set(
            "SHAPE",
            "({0},{1})".format(*self.shape),
            comment="shape of TPF (row, column)",
            after="CCD",
        )
        hdu.header.set(
            "BAD_SPOC",
            f"{','.join([str(bit) for bit in self.bad_spoc_bits])}"
            if isinstance(self.bad_spoc_bits, list)
            else self.bad_spoc_bits,
            comment="bad quality SPOC bits for BG correction",
            after="SHAPE",
        )
        hdu.header.set(
            "BG_CORR",
            self.bg_method,
            comment="method used for BG correction",
            after="BAD_SPOC",
        )
        hdu.header.set(
            "SL_CORR",
            self.sl_method,
            comment="method used for scattered light correction",
            after="BG_CORR",
        )

        # Add extra information to LCF
        if file_type == "lc":
            # TESS magnitude zero-point
            hdu.header.set(
                "TESSMAG0",
                round(TESSmag_zero_point, 3),
                comment="[mag] TESS zero-point magnitude",
                after="SL_CORR",
            )
            # Copy over header keywords from TESS
            for keyword in reversed(
                [
                    "TIMEREF",
                    "TASSIGN",
                    "TIMESYS",
                    "BJDREFI",
                    "BJDREFF",
                    "TIMEUNIT",
                    "EXPOSURE",
                    "RADESYS",
                ]
            ):
                hdu.header.set(
                    keyword,
                    self.cube.output_first_header[keyword],
                    comment=self.cube.output_first_header.comments[keyword],
                    after="DATE-END",
                )
            hdu.header.set(
                "TELAPSE",
                hdu.header["TSTOP"] - hdu.header["TSTART"],
                before="TSTART",
                comment="[d] TSTOP - TSTART",
            )

        # Add aperture information to TPF
        if file_type == "tpf":
            for i in range(len(ap_masks)):
                hdu.header.set(
                    "AP{0}_TYPE".format(i if len(ap_masks) > 1 else ""),
                    ap_methods[i],
                    comment="method used to create aperture",
                    after="SL_CORR" if i == 0 else "AP{0}_NPIX".format(i - 1),
                )
                hdu.header.set(
                    "AP{0}_PAR".format(i if len(ap_masks) > 1 else ""),
                    ap_params[i],
                    comment="{0} used to create aperture".format(
                        "threshold"
                        if ap_methods[i] in ["prf", "threshold"]
                        else "R value"
                    ),
                    after="AP{0}_TYPE".format(i if len(ap_masks) > 1 else ""),
                )
                hdu.header.set(
                    "AP{0}_NPIX".format(i if len(ap_masks) > 1 else ""),
                    np.nanmedian([np.nansum(mask) for mask in ap_masks[i]]),
                    comment="average number of pixels in aperture",
                    after="AP{0}_PAR".format(i if len(ap_masks) > 1 else ""),
                )

        # Compute average predicted V and H magnitude of target
        # Catch warnings that arise if arrays are empty:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Mean of empty slice",
                category=RuntimeWarning,
            )
            vmag = (
                round(
                    np.nanmean(
                        self.ephem.loc[
                            np.logical_and(
                                self.ephem["time"]
                                >= (
                                    self.time_original[0]
                                    if self.barycentric
                                    else self.time_original[0]
                                    - self.timecorr_original[0]
                                ),
                                self.ephem["time"]
                                <= (
                                    self.time_original[-1]
                                    if self.barycentric
                                    else self.time_original[-1]
                                    - self.timecorr_original[-1]
                                ),
                            ),
                            "vmag",
                        ]
                    ),
                    3,
                )
                if "vmag" in self.ephem
                else None
            )
            hmag = (
                round(
                    np.nanmean(
                        self.ephem.loc[
                            np.logical_and(
                                self.ephem["time"]
                                >= (
                                    self.time_original[0]
                                    if self.barycentric
                                    else self.time_original[0]
                                    - self.timecorr_original[0]
                                ),
                                self.ephem["time"]
                                <= (
                                    self.time_original[-1]
                                    if self.barycentric
                                    else self.time_original[-1]
                                    - self.timecorr_original[-1]
                                ),
                            ),
                            "hmag",
                        ]
                    ),
                    3,
                )
                if "hmag" in self.ephem and ~np.isnan(self.ephem["hmag"]).all()
                else None
            )

        # Add keywords for object properties
        hdu.header.set(
            "VMAG",
            vmag if ~np.isnan(vmag) else None,
            comment="[mag] predicted V magnitude",
            after="EQUINOX",
        )
        hdu.header.set(
            "HMAG",
            hmag if ~np.isnan(hmag) else None,
            comment="[mag] predicted H absolute magnitude",
            after="VMAG",
        )
        hdu.header.set(
            "PERIHEL",
            round(self.peri, 3) if hasattr(self, "peri") else None,
            comment="[AU] perihelion distance",
            after="HMAG",
        )
        hdu.header.set(
            "ORBECC",
            round(self.ecc, 3) if hasattr(self, "ecc") else None,
            comment="orbit eccentricity",
            after="PERIHEL",
        )
        hdu.header.set(
            "ORBINC",
            round(self.inc, 3) if hasattr(self, "inc") else None,
            comment="[deg] orbit inclination",
            after="ORBECC",
        )

        # RA and Dec rates are computed from predicted coordinates so TPF and LCF can use
        # consistent values.
        # Use np.unwrap() for RA to ensure angles correctly wrap at 0/360.
        hdu.header.set(
            "RARATE",
            round(
                np.nanmean(
                    np.gradient(
                        np.unwrap(
                            [coord.ra.value for coord in self.coords], period=360
                        ),
                        self.time,
                    )
                    * 3600
                    / 24
                ),
                3,
            ),
            comment="[arcsec/h] average RA rate",
            after="ORBINC",
        )
        hdu.header.set(
            "DECRATE",
            round(
                np.nanmean(
                    np.gradient([coord.dec.value for coord in self.coords], self.time)
                    * 3600
                    / 24
                ),
                3,
            ),
            comment="[arcsec/h] average Dec rate",
            after="RARATE",
        )
        # Pixel speed is computed from input ephemeris so TPF and LCF can use consistent values.
        hdu.header.set(
            "PIXVEL",
            round(
                np.nanmean(
                    np.hypot(
                        np.gradient(self.ephemeris[:, 0], self.time),
                        np.gradient(self.ephemeris[:, 1], self.time),
                    )
                    / 24
                ),
                3,
            ),
            comment="[pix/h] average speed",
            after="DECRATE",
        )

        hdu.header.set(
            "SUN_DIST",
            round(np.nanmean(self.obs_params["sun_distance"]), 3)
            if "sun_distance" in self.ephem
            and ~np.isnan(self.ephem["sun_distance"]).all()
            else None,
            comment="[AU] average distance from Sun",
            after="PIXVEL",
        )
        hdu.header.set(
            "OBS_DIST",
            round(np.nanmean(self.obs_params["obs_distance"]), 3)
            if "obs_distance" in self.ephem
            and ~np.isnan(self.ephem["obs_distance"]).all()
            else None,
            comment="[AU] average distance from TESS",
            after="SUN_DIST",
        )
        hdu.header.set(
            "STO_ANG",
            round(np.nanmean(self.obs_params["sto_angle"]), 3)
            if "sto_angle" in self.ephem and ~np.isnan(self.ephem["sto_angle"]).all()
            else None,
            comment="[deg] average Sun-Target-Observer angle",
            after="OBS_DIST",
        )

        return hdu

    def _save_hdulist(
        self,
        file_type: str,
        overwrite: bool = True,
        outdir: str = "",
        file_name: Optional[str] = None,
    ):
        """
        Write HDUList to a FITS file.

        Parameters
        ----------
        file_type : str
            Type of file to be saved. One of ['tpf', 'lc'].
        overwrite : bool
            Whether to overwrite an existing file of the same name.
        outdir : str
            Directory into which the file will be saved.
        file_name : str
            Filename that will be used. Format must be `.fits`.
            If no filename is given, a default one will be generated.

        Returns
        -------
        """

        # Attribute checks
        if file_type == "tpf" and not hasattr(self, "tpf_hdulist"):
            raise AttributeError("Must run `_make_tpf_hdulist()` before saving file.")
        elif file_type == "lc" and not hasattr(self, "lc_hdulist"):
            raise AttributeError("Must run `_make_lc_hdulist()` before saving file.")

        # Create default file name
        if file_name is None:
            file_name = "tess-{0}-s{1:04}-{2}-{3}-shape{4}x{5}".format(
                str(self.target).replace(" ", "").replace("/", ""),
                self.sector,
                self.camera,
                self.ccd,
                *self.shape,
            )
            if file_type == "tpf":
                file_name += "-moving_tp.fits"
            elif file_type == "lc":
                file_name += "_lc.fits"
            else:
                raise ValueError(
                    f"`file_type` must be one of: ['tpf', 'lc']. Not '{file_type}'"
                )

        # Check format of file_name and outdir
        if not file_name.endswith(".fits"):
            raise ValueError(
                "`file_name` must be a .fits file. Not `{0}`".format(file_name)
            )
        if len(outdir) > 0 and not outdir.endswith("/"):
            outdir += "/"

        # Write hdulist to file
        if file_type == "tpf":
            hdulist = self.tpf_hdulist
        elif file_type == "lc":
            hdulist = self.lc_hdulist
        else:
            raise ValueError(
                f"`file_type` must be one of: ['tpf', 'lc']. Not '{file_type}'"
            )
        hdulist.writeto(outdir + file_name, overwrite=overwrite, checksum=True)
        logger.info("Created file: {0}".format(outdir + file_name))

    def animate_tpf(
        self,
        show_aperture: bool = True,
        show_ephemeris: bool = True,
        step: Optional[int] = None,
        save: bool = False,
        outdir: str = "",
        file_name: Optional[str] = None,
        **kwargs,
    ):
        """
        Plot animation of TPF data with optional information overlay.

        Parameters
        ----------
        show_aperture : bool
            If True, the aperture used for photometry is displayed in the animation.
        show_ephemeris : bool
            If True, the predicted position of the target is included in the animation.
        step : int or None
            Spacing between frames, i.e. plot every nth frame.  If `None`, the spacing will be determined such
            that about 50 frames are shown. Showing more frames will increase the runtime and, if `save`, the
            file size.
        save : bool
            If True, save the animation.
        outdir : str
            If `save`, this is the directory into which the file will be saved.
        file_name : str or None
            If `save`, this is the filename that will be used. Format must be '.gif'.
            If no filename is given, a default one will be generated.
        kwargs:
            Keyword arguments passed to `utils.animate_cube` such as `interval`, `repeat_delay`, `cnorm`,
            `vmin`, `vmax`.

        Returns
        --------
        animation : html
            If in a notebook environment, the animation is returned in HTML format for display purposes.
        """

        # Attribute checks
        if (
            not hasattr(self, "all_flux")
            or not hasattr(self, "flux")
            or not hasattr(self, "corr_flux")
            or not hasattr(self, "aperture_mask")
        ):
            raise AttributeError(
                "Must run `get_data()`, `reshape_data()`, `background_correction()` and `create_aperture()` before animating."
            )

        # Compute default step
        if step is None:
            step = len(self.time) // 50 if len(self.time) >= 50 else 1

        # Create animation
        ani = animate_cube(
            self.corr_flux,
            aperture_mask=self.aperture_mask if show_aperture else None,
            corner=self.corner,
            ephemeris=self.ephemeris if show_ephemeris else None,
            cadenceno=self.cadence_number,
            time=self.time,
            step=step,
            suptitle=f"Target {self.target} in Sector {self.sector} Camera {self.camera} CCD {self.ccd}",
            **kwargs,
        )

        # Save animation
        if save:
            # Create default file name
            if file_name is None:
                file_name = (
                    "tess-{0}-s{1:04}-{2}-{3}-shape{4}x{5}-moving_tp.gif".format(
                        str(self.target).replace(" ", "").replace("/", ""),
                        self.sector,
                        self.camera,
                        self.ccd,
                        *self.shape,
                    )
                )
            # Check format of file_name and outdir
            if not file_name.endswith(".gif"):
                raise ValueError(
                    "`file_name` must be a .gif file. Not `{0}`".format(file_name)
                )
            if len(outdir) > 0 and not outdir.endswith("/"):
                outdir += "/"

            ani.save(outdir + file_name, writer="pillow")
            logger.info("Created file: {0}".format(outdir + file_name))

        # Return animation in HTML format.
        # If in notebook environment, this allows animation to be displayed.
        try:
            from IPython.display import HTML

            return HTML(ani.to_jshtml())
        except ModuleNotFoundError:
            # To make installing `tess-asteroids` easier, ipython is not a dependency
            # because we can assume it is installed when notebook-specific features are called.
            logger.error(
                "ipython needs to be installed for `animate()` to work (e.g., `pip install ipython`)"
            )

    def create_lc_quality_mask(
        self,
        spoc_quality: np.ndarray,
        lc_quality: np.ndarray,
        bad_spoc_bits: Union[list[int], str] = "default",
        bad_lc_bits: Union[list[int], str] = "default",
    ):
        """
        Combines SPOC quality flags and tess_asteroids lightcurve quality flags to
        create a boolean quality mask that can be applied to the aperture or PSF
        lightcurve.

        Parameters
        ----------
        spoc_quality : ndarray
            Array of SPOC quality flags.
        lc_quality : ndarray
            Array of tess_asteroids lightcurve quality flags, as defined by `_create_lc_quality()`.
            This must have the same length as `spoc_quality`.
        bad_spoc_bits : list or str
            Defines SPOC bits corresponding to bad quality data. Can be one of:

            - "default" - mask bits defined by `default_bad_spoc_bits`.
            - "all" - mask all data with a SPOC quality flag.
            - "none" - mask no data.
            - list - mask custom bits provided in list.

            More information about the SPOC quality flags can be found in Section 9 of the TESS Science
            Data Products Description Document.
        bad_lc_bits : list or str
            Defines bits corresponding to bad quality data from `_create_lc_quality()`.
            Can be one of:

            - "default" - mask bits defined by `default_bad_lc_bits`.
            - "all" - mask all data with a quality flag.
            - "none" - mask no data.
            - list - mask custom bits provided in list.

        Returns
        -------
        quality_mask : ndarray
            A boolean mask with the same length as `spoc_quality` and `lc_quality`.
            `True` indicates a good quality cadence and `False` indicates a bad quality cadence.
        """

        # Create SPOC quality mask using user-defined bad bits.
        spoc_quality_mask, _ = self._create_spoc_quality_mask(
            spoc_quality=spoc_quality, bad_spoc_bits=bad_spoc_bits
        )

        # Define bitmask for user-defined tess_asteroids lightcurve quality bits.
        bad_lc_value = create_bad_bitmask(
            bad_bits=bad_lc_bits, default_bad_bits=default_bad_lc_bits
        )

        # Define lightcurve quality mask.
        if bad_lc_value == "all":
            quality_mask = lc_quality == 0
        else:
            quality_mask = lc_quality & bad_lc_value == 0

        return np.logical_and(quality_mask, spoc_quality_mask)

    def plot_lc(
        self,
        lc: Optional[dict] = None,
        method: str = "all",
        bad_spoc_bits: Union[list[int], str] = "default",
        bad_ap_bits: Union[list[int], str] = "default",
        bad_psf_bits: Union[list[int], str] = "default",
        plot_err: bool = False,
        plot_bad_quality: bool = True,
        xlim: Optional[tuple] = None,
        ylim: Optional[tuple] = None,
        save: bool = False,
        outdir: str = "",
        file_name: Optional[str] = None,
    ):
        """
        Plot aperture and/or PSF lightcurves, with options to customise and save.

        Parameters
        ----------
        lc : dict
            A dictionary containing aperture and PSF photometry. This must have the same structure as that
            produced by `self.to_lightcurve()`. If None, `self.lc` will be used by default.
        method : str
            Lightcurve to plot. One of [`all`, `aperture`, `psf`].
        bad_spoc_bits : list or str
            Defines SPOC bits corresponding to bad quality data. Can be one of:

            - "default" - mask bits defined by `default_bad_spoc_bits`.
            - "all" - mask all data with a SPOC quality flag.
            - "none" - mask no data.
            - list - mask custom bits provided in list.

            More information about the SPOC quality flags can be found in Section 9 of the TESS Science
            Data Products Description Document.
        bad_ap_bits/bad_psf_bits : list or str
            Defines bits corresponding to bad quality data for `aperture`/`psf` photometry, as defined by `_create_lc_quality()`.
            Can be one of:

            - "default" - mask bits defined by `default_bad_lc_bits`.
            - "all" - mask all data with a quality flag.
            - "none" - mask no data.
            - list - mask custom bits provided in list.
        plot_err : bool
            If True, plot errorbars on lightcurve.
        plot_bad_quality : bool
            If True, include data flagged as bad quality in the plot.
        xlim/ylim : tuple
            (min, max) axes limits.
        save : bool
            If True, save the figure.
        outdir : str
            If `save`, this is the directory into which the file will be saved.
        file_name : str or None
            If `save`, this is the filename that will be used.
            If no filename is given, a default one will be generated.

        Returns
        --------
        fig : matplotlib.figure.Figure
            Matplotlib figure object.

        """

        # Attribute checks
        if not hasattr(self, "quality"):
            raise AttributeError("Must run `get_data()` before plotting lightcurve.")

        # Assign default value to `lc` if none is provided by user:
        if lc is None:
            if not hasattr(self, "lc"):
                raise AttributeError(
                    "Must run `to_lightcurve()` before plotting lightcurve, if you have not provided `lc`."
                )
            lc = self.lc

        # Make a copy of dictionary
        lc = deepcopy(lc)

        # Only plot lightcurve for user specified `method`
        if method != "all":
            if method not in lc:
                raise ValueError(f"Method '{method}' is not in `lc`.")
            lc = {method: lc[method]}

        # Ensure lc["aperture"] is at least 1D
        if "aperture" in lc:
            lc["aperture"] = np.atleast_1d(lc["aperture"])  # type: ignore

        # Initialise figure
        n_axes = (len(lc["aperture"]) if "aperture" in lc else 0) + (
            1 if "psf" in lc else 0
        )
        if n_axes == 0:
            raise ValueError(
                "Must have at least one `aperture` or `psf` lightcurve in `lc`."
            )
        fig, ax = plt.subplots(
            n_axes,
            1,
            figsize=(8, 4 * n_axes),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        ax = np.atleast_1d(ax)

        # Run through each available lightcurve and plot
        for key in lc:
            if key == "aperture":
                # Run through each available aperture lightcurve
                for i, lc_ap in enumerate(lc[key]):
                    # Define aperture lightcurve quality mask.
                    quality_mask = self.create_lc_quality_mask(
                        spoc_quality=self.quality,
                        lc_quality=lc_ap["quality"],
                        bad_spoc_bits=bad_spoc_bits,
                        bad_lc_bits=bad_ap_bits,
                    )

                    ax[i].errorbar(
                        lc_ap["time"][quality_mask],
                        lc_ap["flux"][quality_mask],
                        yerr=lc_ap["flux_err"][quality_mask] if plot_err else None,
                        color="deeppink",
                        marker="o",
                        ms=2,
                        ls="",
                    )
                    if plot_bad_quality:
                        ax[i].errorbar(
                            lc_ap["time"][~quality_mask],
                            lc_ap["flux"][~quality_mask],
                            yerr=lc_ap["flux_err"][~quality_mask] if plot_err else None,
                            color="black",
                            marker="x",
                            ms=6,
                            ls="",
                            label="Bad quality data",
                        )
                        ax[i].legend()
                    ax[i].tick_params(
                        axis="x", which="both", labelbottom=True, bottom=True
                    )
                    ax[i].set_ylabel("Flux [e-/s]")
                    ax[i].grid(ls=":")
                    ax[i].set_axisbelow(True)
                    ax[i].set_title(
                        "Aperture{0} photometry (method = '{1}', {3} = {2})".format(
                            i if len(lc[key]) > 1 else "",
                            lc_ap["ap_method"],
                            lc_ap["ap_param"],
                            "threshold"
                            if lc_ap["ap_method"] in ["prf", "threshold"]
                            else "R value",
                        )
                    )

            elif key == "psf":
                # Define PSF lightcurve quality mask.
                quality_mask = self.create_lc_quality_mask(
                    spoc_quality=lc[key]["spoc_quality"],
                    lc_quality=lc[key]["quality"],
                    bad_spoc_bits=bad_spoc_bits,
                    bad_lc_bits=bad_psf_bits,
                )

                ax[-1].errorbar(
                    lc[key]["time"][quality_mask],
                    lc[key]["flux"][quality_mask],
                    yerr=lc[key]["flux_err"][quality_mask] if plot_err else None,
                    xerr=(
                        lc[key]["time_lerr"][quality_mask],
                        lc[key]["time_uerr"][quality_mask],
                    )
                    if plot_err
                    else None,
                    color="mediumorchid",
                    marker="o",
                    ms=2,
                    ls="",
                )
                if plot_bad_quality:
                    ax[-1].errorbar(
                        lc[key]["time"][~quality_mask],
                        lc[key]["flux"][~quality_mask],
                        yerr=lc[key]["flux_err"][~quality_mask] if plot_err else None,
                        xerr=(
                            lc[key]["time_lerr"][~quality_mask],
                            lc[key]["time_uerr"][~quality_mask],
                        )
                        if plot_err
                        else None,
                        color="black",
                        marker="x",
                        ms=6,
                        ls="",
                        label="Bad quality data",
                    )
                    ax[-1].legend()
                ax[-1].set_ylabel("Flux [e-/s]")
                ax[-1].grid(ls=":")
                ax[-1].set_axisbelow(True)
                ax[-1].set_title("PSF photometry")

        if ylim is not None:
            ax[-1].set_ylim(ylim)
        if xlim is not None:
            ax[-1].set_xlim(xlim)
        ax[-1].set_xlabel("Time [BJD - 2457000]")

        fig.suptitle(
            f"Asteroid {self.target} in Sector {self.sector} Camera {self.camera} CCD {self.ccd}"
        )

        # Save figure
        if save:
            # Create default file name
            if file_name is None:
                file_name = "tess-{0}-s{1:04}-{2}-{3}-shape{4}x{5}_lc.png".format(
                    str(self.target).replace(" ", "").replace("/", ""),
                    self.sector,
                    self.camera,
                    self.ccd,
                    *self.shape,
                )
            # Check outdir
            if len(outdir) > 0 and not outdir.endswith("/"):
                outdir += "/"
            plt.savefig(outdir + file_name)
            logger.info("Created file: {0}".format(outdir + file_name))

        plt.close(fig)
        return fig

    @staticmethod
    def from_name(
        target: str,
        sector: int,
        camera: Optional[int] = None,
        ccd: Optional[int] = None,
        time_step: float = 0.1,
    ):
        """
        Initialises MovingTPF from target name and TESS sector. Uses JPL/Horizons to retrieve ephemeris of target.
        Specifying a camera and CCD will only use the ephemeris from that camera/ccd.

        Parameters
        ----------
        target : str
            JPL/Horizons target ID of e.g. asteroid, comet.
        sector : int
            TESS sector number.
        camera : int
            TESS camera. Must be defined alongside `ccd`.
            If `None`, full ephemeris will be used to initialise MovingTPF.
        ccd : int
            TESS CCD. Must be defined alongside `camera`.
            If `None`, full ephemeris will be used to initialise MovingTPF.
        time_step : float
            Resolution of ephemeris, in days.

        Returns
        -------
        MovingTPF :
            Initialised MovingTPF with ephemeris and orbital elements from JPL/Horizons.
            Target ephemeris has columns ['time', 'sector', 'camera', 'ccd', 'column', 'row', 'vmag', 'hmag'
            'sun_distance', 'obs_distance', 'sto_angle'].

            - 'time' : float with units (JD - 2457000) in TDB at spacecraft.
            - 'sector', 'camera', 'ccd' : int
            - 'column', 'row' : float. These are one-indexed, where the lower left pixel of the FFI is (1,1).
            - 'vmag' : float. Visual magnitude.
            - 'hmag' : float. Absolute magntiude.
            - 'sun_distance' : float. Distance from Sun, in AU.
            - 'obs_distance' : float. Distance from observer (TESS), in AU.
            - 'sto_angle' : float. Sun-Target-Observer phase angle, in degrees. Closely approximates true phase angle. See Horizons documentation for definition: https://ssd.jpl.nasa.gov/horizons/manual.html#obsquan
        """

        # Get target ephemeris and orbital elements using tess-ephem
        logger.info(
            "Retrieving ephemeris for target {0} in sector {1}, camera {2}, ccd {3}.".format(
                target, sector, camera, ccd
            )
        )
        df_ephem, metadata = ephem(
            target, sector=sector, time_step=time_step, orbital_elements=True
        )

        # Check whether target was observed in sector.
        if len(df_ephem) == 0:
            raise ValueError(
                "Target {} was not observed in sector {}. Try using the utils function `target_observability()` to find out if/when TESS observed this target.".format(
                    target, sector
                )
            )

        # Filter ephemeris using camera/ccd.
        if camera is not None and ccd is not None:
            df_ephem = df_ephem[
                np.logical_and(df_ephem["camera"] == camera, df_ephem["ccd"] == ccd)
            ].copy()
            if len(df_ephem) == 0:
                raise ValueError(
                    "Target {} was not observed in sector {}, camera {}, ccd {}.".format(
                        target, sector, camera, ccd
                    )
                )

        # Add column for time in format (JD - 2457000) and scale TDB.
        # Note: tess-ephem returns time in UTC at spacecraft, so we convert to TDB scale.
        df_ephem["time"] = [t.value - 2457000 for t in df_ephem.index.values]
        df_ephem["time"] += df_ephem["tdb-ut"] / 24 / 3600
        df_ephem = df_ephem[
            [
                "time",
                "sector",
                "camera",
                "ccd",
                "ra",
                "dec",
                "column",
                "row",
                "vmag",
                "hmag",
                "sun_distance",
                "obs_distance",
                "sto_angle",
            ]
        ].reset_index(drop=True)

        # Rename keys in metadata dictionary
        metadata["inclination"] = metadata.pop("orbital_inclination")
        metadata["perihelion"] = metadata.pop("perihelion_distance")

        # Add date of query to metadata dictionary
        metadata["ephem_date"] = datetime.now().strftime("%Y-%m-%d")

        return MovingTPF(
            target=target, ephem=df_ephem, barycentric=False, metadata=metadata
        )
