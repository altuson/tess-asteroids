import time
import warnings
from datetime import datetime
from typing import Optional, Tuple, Union

import lkprf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.time import Time
from fbpca import pca
from lkspacecraft import TESSSpacecraft
from lkspacecraft.spacecraft import BadEphemeris
from numpy.polynomial import Polynomial
from scipy import ndimage, stats
from scipy.interpolate import CubicSpline
from tess_ephem import ephem
from tesscube import TESSCube
from tesscube.fits import get_wcs_header_by_extension
from tesscube.query import async_get_primary_hdu
from tesscube.utils import _sync_call, convert_coordinates_to_runs

from . import __version__, logger, straps
from .utils import animate_cube, compute_moments, inside_ellipse, make_wcs_header


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
        Target ephemeris with columns ['time','sector','camera','ccd','column','row']. Optional columns: ['vmag'].
            'time' : float in format (JD - 2457000). See also `time_scale` below.
            'sector', 'camera', 'ccd' : int
            'column', 'row' : float. These must be one-indexed, where the lower left pixel of the FFI is (1,1).
            'vmag' : float, optional. Visual magnitude.
    time_scale : str
        Time scale of input 'time'. One of ['tdb', 'utc']. Default is 'tdb'.
        If 'tdb', the input 'time' must be in TDB measured at the solar system barycenter from the TESS FFI header.
            This is the scale used for the 'TSTART'/'TSTOP' keywords in SPOC FFI headers and the 'TIME' column in SPOC
            TPFs and LCFs.
        If 'utc', the input 'time' must be in UTC measured at the spacecraft. This can be recovered from the SPOC data
            products: for FFIs subtract header keyword 'BARYCORR' from 'TSTART'/'TSTOP' and for TPFs/LCFs subtract the
            'TIMECORR' column from the 'TIME' column.
    """

    def __init__(self, target: str, ephem: pd.DataFrame, time_scale: str = "tdb"):
        self.target = target
        self.ephem = ephem
        self.time_scale = time_scale

        # Check self.ephem['time'] has correct units
        if min(self.ephem["time"]) >= 2457000:
            raise ValueError("ephem['time'] must have units (JD - 2457000).")

        # Check self.time_scale has expected value
        if self.time_scale not in ["tdb", "utc"]:
            raise ValueError(
                "`time_scale` must be either `utc` or `tdb`. Not `{0}`".format(
                    self.time_scale
                )
            )

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

        # Initialise tesscube
        self.cube = TESSCube(sector=self.sector, camera=self.camera, ccd=self.ccd)

        # Retrieve original primary header of FFI cube
        self.primary_hdu = _sync_call(
            async_get_primary_hdu, object_key=self.cube.object_key
        ).header

        logger.info("Initialised MovingTPF for target {0}.".format(self.target))

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
            One of ['threshold', 'prf', 'ellipse'].
        save : bool, default=False
            If True, save the TPF HDUList to a FITS file.
        outdir : str
            If `save`, this is the directory into which the file will be saved.
        file_name : str
            If `save`, this is the filename that will be used. Format must be `.fits`.
            If no filename is given, a default one will be generated.
        **kwargs
            Keyword arguments to be passed to `create_pixel_quality()`, `background_correction()`,
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
        method: str = "aperture",
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
            Method to extract lightcurve. One of `aperture` or `psf`.
        save : bool
            If True, save the lightcurve HDUList to a FITS file.
        outdir : str
            If `save`, this is the directory into which the file will be saved.
        file_name : str
            If `save`, this is the filename that will be used. Format must be `.fits`.
            If no filename is given, a default one will be generated.
        **kwargs
            Keyword arguments to be passed to `to_lightcurve()` and `to_fits()`.

        Returns
        -------
        """
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
        # Shape needs to be >=3 in both dimensions, otherwise make_wcs_header() errors.
        self.shape = shape

        # Use interpolation to get target (row,column) at cube time.
        # If input ephem uses UTC time scale, convert cube time to UTC for interpolation.
        column_interp = CubicSpline(
            self.ephem["time"].astype(float),
            self.ephem["column"].astype(float),
            extrapolate=False,
        )(
            self.cube.time
            if self.time_scale == "tdb"
            else self.cube.time - self.cube.last_hdu.data["BARYCORR"]
        )
        row_interp = CubicSpline(
            self.ephem["time"].astype(float),
            self.ephem["row"].astype(float),
            extrapolate=False,
        )(
            self.cube.time
            if self.time_scale == "tdb"
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
        # Warn user if there are pixels outside of FFI science array.
        if (
            sum(
                ~np.logical_and(
                    [r.all() for r in row[:, :, 0] <= 2048],
                    [
                        c.all()
                        for c in np.logical_and(
                            column[:, 0, :] >= 45, column[:, 0, :] <= 2092
                        )
                    ],
                )
            )
            > 0
        ):
            logger.warning(
                "Some of the requested pixels are outside of the FFI science array (1<=row<=2048, 45<=col<=2092), but they will be included in your TPF."
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

        # Convert (row,column) ephemeris to (ra,dec) using average WCS from observing sector.
        # Note: if MovingTPF was initialised from_name, then tess-ephem
        # internally converted (ra,dec) to (row,column) using the average WCS
        # from the observing sector. `self.coords` does not recover these original values
        # because the ephemeris has been interpolated.
        # Note: pixel_to_world() assumes zero-indexing so subtract one from (row,col).
        self.coords = np.asarray(
            [
                self.cube.wcs.pixel_to_world(
                    self.ephemeris[t, 1] - 1, self.ephemeris[t, 0] - 1
                )
                for t in range(len(self.time_original))
            ]
        )

        # Calculate UTC to TDB correction using position of object.
        # If SPICE kernels are missing, no correction will be computed.
        try:
            time_utc = self.time_original - self.timecorr_original
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
            for t in range(len(time_utc)):
                timecorr.append(
                    tess.get_barycentric_time_correction(
                        time=Time(time_utc[t] + 2457000, scale="utc", format="jd"),
                        ra=self.coords[t].ra.value,
                        dec=self.coords[t].dec.value,
                    )[0]
                    / 60
                    / 60
                    / 24
                )
            self.timecorr = np.asarray(timecorr)
            self.time = time_utc + self.timecorr
        except BadEphemeris:
            logger.warning(
                "UTC to TDB correction was not calculated due to missing SPICE kernels."
            )
            self.timecorr = self.timecorr_original
            self.time = self.time_original

    def reshape_data(self):
        """
        Reshape flux data into cube with shape (len(self.time), self.shape).

        Parameters
        ----------

        Returns
        -------
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

    def background_correction(self, method: str = "rolling", **kwargs):
        """
        Apply background correction to reshaped flux data.

        Parameters
        ----------
        method : str
            Method used for background correction. One of [`rolling`,`linear_model`].
        **kwargs
            Keyword arguments to be passed to `_bg_rolling_median()` and `_bg_linear_model()`.

        Returns
        -------
        """
        if not hasattr(self, "all_flux") or not hasattr(self, "flux"):
            raise AttributeError(
                "Must run `get_data()` and `reshape_data()` before computing background."
            )

        # Initialise masks that will flag NaNs in SL or BG linear model, only if `linear_model` is
        # the most recent method run.
        self.sl_nan_mask = np.zeros_like(self.time, dtype=bool)
        self.lm_nan_mask = np.zeros_like(self.all_flux, dtype=bool)

        # Get background via chosen method
        if method == "rolling":
            self.bg, self.bg_err = self._bg_rolling_median(**kwargs)

        elif method == "linear_model":
            self.bg, self.bg_err, _, _, _, _ = self._bg_linear_model(**kwargs)

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
            Array with same shape as self.flux.

        bg_err : ndarray
            Error on background flux estimate.
            Array with same shape as self.flux.
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
        target_threshold: float = 0.01,
        include_stars: bool = True,
        star_flux_threshold: float = 1.1,
        star_gradient_threshold: float = 5,
        **kwargs,
    ):
        """
        Creates a boolean mask, with the same shape as `self.all_flux` (ntimes, npixels), that masks stationary sources
        (e.g. stars) and the moving target.

        The moving target mask is created using the PRF model and selecting all pixels that contain more than the
        `target_threshold` fraction of the object's flux. This mask is defined per time.

        The star mask is created using two condiditons: i) a high flux value and ii) a high flux gradient. This mask is
        constant across all times. This is only included in the mask if `include_stars` is `True`.

        Parameters
        ----------
        target_threshold : float
            Pixels where the PRF model is greater than this threshold are included in the target mask. Must be between 0 and 1
            because the PRF model is normalised so that all values sum to one.
        include_stars : bool
            If `True`, returns a mask for moving target and stars. If `False`, returns a mask for moving target only.
        star_flux_threshold : float
            Used to define the threshold above which a pixel has a high flux.
        star_gradient_threshold : float
            Used to define the threshold above which a pixel has a high flux gradient.
        kwargs : dict
            Keywords arguments passed to `self._create_target_prf_model`, e.g `time_step`.

        Returns
        -------
        source_mask : ndarray
            Boolean mask with shape (ntimes, npixels). The mask is `True` when the moving target or a star is present.
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
        target_mask = (
            self._create_target_prf_model(all_flux=True, **kwargs) >= target_threshold
        )
        if not include_stars:
            return target_mask

        # Create mask for stationary sources e.g. stars (high flux values AND high flux gradients).

        # Median of each pixel across all times.
        med = np.nanmedian(self.all_flux, axis=0)

        # Mask pixels whose average flux value over all time is some fraction greater than average flux value over all
        # pixels and all times.
        star_flux_mask = med >= star_flux_threshold * np.nanmedian(self.all_flux)

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
        med_gradient = np.gradient(med_reshaped)
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

        return np.asarray(
            [np.logical_or(targ_mask, star_mask) for targ_mask in target_mask]
        )

    def _create_scattered_light_model(
        self,
        method: str = "all_time",
        ncomponents: int = 5,
        niter: int = 5,
        poly_deg: int = 3,
        window_length: float = 1,
        diagnostic_plot: bool = False,
        **kwargs,
    ):
        """
        Use PCA to model scattered light. Either compute the PCA components for all times at once (`all_time`) or
        use a for loop to go through each frame and compute PCA components in a time window (`per_time`).
        The latter is preferred, but runtime is longer (see below). The resulting scattered light model has the same shape
        as `self.all_flux`.

        e.g. asteroid 1980 VR1 in sector 1, camera 1, ccd 1 (ntimes = 905, npixels = 7917) run on MacBook Pro 2023.
        `all_time` method takes about 0.1 seconds to run (~80% is computation of PCA), `per_time` method takes about
        31 seconds to run (~86% is computation of PCA).

        e.g. asteroid 1998 YT6 in sector 6, camera 1, ccd 1 (ntimes = 967, npixels = 14056) run on MacBook Pro 2023.
        `all_time` method takes about 0.1 seconds to run (~75% is computation of PCA), `per_time` method takes about
        20 minutes to run (~95% is computation of PCA).

        Parameters
        ----------
        method : str
            Method used to compute scattered light model. One of [`all_time`, `per_time`].
            If `all_time`, the PCA components are computed for all times at once.
            If `per_time`, the PCA components are computed in a time window around each frame.
        ncomponents : int
            Number of PCA components.
        niter : int
            Number of iterations that will be run to compute the PCA components.
        poly_deg : int
            Polynomial degree for the cartesian design matrix.
        window_length : float
            If method is `per_time`, this is used to define the time window around each frame for the PCA computation.
            It is defined in days.
        diagnostic_plot : bool
            If True, shows two diagnostic plots to check the scattered light model.
        kwargs : dict
            Keywords arguments passed to `self._create_pca_source_mask`, e.g `target_threshold`, `star_flux_threshold`.

        Returns
        -------
        sl_model : ndarray
            Scattered light model, with same shape as `self.all_flux`.
        sl_model_err : ndarray
            Error on scattered light model, with same shape as `self.all_flux`.
        """

        # Parameter logic checks
        if window_length <= 0:
            raise ValueError(
                f"`window_length` must be greater than zero. Not '{window_length}'"
            )
        if poly_deg <= 0:
            raise ValueError(f"`poly_deg` must be greater than zero. Not '{poly_deg}'")

        # Create mask for moving target and stars.
        source_mask = self._create_source_mask(include_stars=True, **kwargs)

        # Create design matrix - a `poly_deg` degree polynomial in row and column.
        row, col = self.pixels.T
        R = row - np.nanmean(row)
        C = col - np.nanmean(col)
        X = np.vstack(
            [
                R.ravel() ** idx * C.ravel() ** jdx
                for idx in range(poly_deg)
                for jdx in range(poly_deg)
            ]
        ).T

        logger.info(
            "Started computation of scattered light model using {0} method.".format(
                method
            )
        )
        start_time = time.time()
        if method == "all_time":
            # Note: This masks all moving target pixels at all times, even when it is not present.
            # This is a limitation of computing the PCA at all times.
            source_mask = source_mask.any(axis=0)

            # Get the PCA components for all pixels at all times, excluding pixels with sources.
            # This will error if `ncomponents` is greater than the smallest dimension of the input matrix.
            U, s, V = pca(
                self.all_flux[:, ~source_mask], k=ncomponents, raw=True, n_iter=niter
            )

            # Compute best-fitting weights
            try:
                w = np.linalg.solve(
                    X[~source_mask].T.dot(X[~source_mask]), X[~source_mask].T.dot(V.T)
                )
            # If no solution is found, return nan at all times.
            except np.linalg.LinAlgError:
                sl_model = np.full(self.all_flux.shape, np.nan)
                sl_model_err = np.full(self.all_flux.shape, np.nan)

                # Update SL NaN mask
                if hasattr(self, "sl_nan_mask"):
                    self.sl_nan_mask = np.ones_like(self.time, dtype=bool)

                logger.warning(
                    "When computing the scattered light model, no solution was found. The scattered light model was set to nan at all times."
                )
            # If solution is found, continue:
            else:
                # Create model that can predict V for pixels not included in the PCA.
                V_model = X.dot(w).T

                # Compute scattered light model and error.
                # >>>>> Add error on SL model <<<<<
                sl_model = U.dot(np.diag(s)).dot(V_model)
                sl_model_err = np.full(sl_model.shape, np.nan)

        elif method == "per_time":
            # Define good quality data using SPOC quality flags.
            spoc_quality_mask = self.quality & (1 | 4 | 16 | 32 | 16384) == 0

            # Initialise scattered light model and error.
            sl_model = np.zeros(self.all_flux.shape)
            sl_model_err = np.zeros(self.all_flux.shape)

            # Run through each time
            for t in range(len(self.time)):
                # Indices that define a time window around the current frame.
                tmin, tmax = (
                    self.time[t] - 0.5 * window_length,
                    self.time[t] + 0.5 * window_length,
                )
                t_window = np.where(
                    np.logical_and(self.time >= tmin, self.time <= tmax)
                )[0]

                # Remove bad quality SPOC cadences, but ensure current time is always included.
                t_window = np.unique(
                    np.append(t_window[spoc_quality_mask[t_window]], t)
                )

                # Mask sources i.e. moving target and stars. This masks pixels associated with the moving target only
                # at the current time.
                sources = source_mask[t_window].any(axis=0)

                # Get the PCA components for pixels in time window, excluding pixels with sources.
                # This will return an AssertionError if `ncomponents` is greater than the smallest dimension of the
                # input matrix (i.e. too few times or pixels). In this case, set scattered light model to nan.
                try:
                    U, s, V = pca(
                        self.all_flux[t_window][:, ~sources],
                        k=ncomponents,
                        raw=True,
                        n_iter=niter,
                    )
                except AssertionError:
                    sl_model[t] = np.nan
                    sl_model_err[t] = np.nan

                    # Update SL NaN mask
                    if hasattr(self, "sl_nan_mask"):
                        self.sl_nan_mask[t] = True

                    logger.warning(
                        "At cadence number {0}, the PCA failed with an AssertionError. This means either niter < 0, ncomponents <= 0 or ncomponents is greater than the smallest dimension of the input matrix (i.e. masking of `self.all_flux` has removed too much data). The corresponding scattered light model was set to nan.".format(
                            self.cadence_number[t]
                        )
                    )
                    continue

                # Compute best-fitting weights
                try:
                    w = np.linalg.solve(
                        X[~sources].T.dot(X[~sources]), X[~sources].T.dot(V.T)
                    )
                except np.linalg.LinAlgError:
                    sl_model[t] = np.nan
                    sl_model_err[t] = np.nan

                    # Update SL NaN mask
                    if hasattr(self, "sl_nan_mask"):
                        self.sl_nan_mask[t] = True

                    logger.warning(
                        "When computing the scattered light model for cadence number {0}, no solution was found. The corresponding scattered light model was set to nan.".format(
                            self.cadence_number[t]
                        )
                    )
                    continue
                # Create model that can predict V for pixels not included in PCA.
                V_model = X.dot(w).T

                # Compute scattered light model and error.
                # >>>>> Add error on SL model <<<<<
                sl_model[t] = U[t_window == t].dot(np.diag(s)).dot(V_model)
                sl_model_err[t] = np.nan

        else:
            raise ValueError(
                "`method` must be one of: [`per_time`, `all_time`]. Not `{0}`".format(
                    method
                )
            )
        logger.info(
            "Finished computation of scattered light model in {0:.2f} sec.".format(
                time.time() - start_time
            )
        )

        if diagnostic_plot:
            logger.info("Making diagnostic plots for scattered light model...")
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
            plt.show()
            plt.close(fig)

            # Plot two: time-series of SL model for each pixel.
            # Note: setting aspect="auto" means pixels are not square in these visualisations.
            fig, ax = plt.subplots()
            im = ax.imshow(
                sl_model.T, origin="lower", aspect="auto", interpolation="none"
            )
            ax.set(xlabel="Time Index", ylabel="Pixel Index")
            # Add colorbar
            cbar = fig.colorbar(im, ax=ax, location="right")
            cbar.set_label("Flux [$e^-/s$]")
            plt.show()
            plt.close(fig)

        return np.asarray(sl_model), np.asarray(sl_model_err)

    def _bg_linear_model(
        self,
        sl_method: str = "all_time",
        sl_poly_deg: int = 3,
        sl_window_length: float = 1,
        window_length: float = 1,
        poly_deg: int = 3,
        sigma: float = 5,
        **kwargs,
    ):
        """

        Parameters
        ----------

        Returns
        -------
        bg : ndarray
            Background flux estimate.
            Array with same shape as self.flux.
        bg_err : ndarray
            Error on background flux estimate.
            Array with same shape as self.flux.
        sl_model : ndarray
        sl_model_err : ndarray
        linear_model : ndarray
        linear_model_err : ndarray
        """

        if not hasattr(self, "all_flux"):
            raise AttributeError("Must run `get_data()` before computing background.")

        # Parameter logic checks
        if window_length <= 0:
            raise ValueError(
                f"`window_length` must be greater than zero. Not '{window_length}'"
            )
        if poly_deg <= 0:
            raise ValueError(f"`poly_deg` must be greater than zero. Not '{poly_deg}'")
        if sigma <= 0:
            raise ValueError(f"`sigma` must be greater than zero. Not '{sigma}'")

        # Remove scattered light from flux
        sl_model, sl_model_err = self._create_scattered_light_model(
            method=sl_method,
            poly_deg=sl_poly_deg,
            window_length=sl_window_length,
            **kwargs,
        )
        sl_corr_flux = self.all_flux - sl_model
        sl_corr_flux_err = np.sqrt(
            np.nansum([self.all_flux_err**2, sl_model_err**2], axis=0)
        )

        # Initialise arrays
        linear_model = np.zeros(self.all_flux.shape)
        linear_model_err = np.zeros(self.all_flux.shape)

        # Define good quality data using SPOC quality flags.
        spoc_quality_mask = self.quality & (1 | 4 | 16 | 32 | 16384) == 0

        # Create mask for moving target.
        source_mask = self._create_source_mask(include_stars=False, **kwargs)

        # Scale time between (-0.5, 0.5) for linear modelling.
        scaled_time = (self.time - np.nanmedian(self.time)) / np.ptp(self.time)

        logger.info("Started computation of background linear model.")
        start_time = time.time()
        for t in range(len(self.time)):
            # Calculate indices that define a time window around the current frame.
            tmin, tmax = (
                self.time[t] - 0.5 * window_length,
                self.time[t] + 0.5 * window_length,
            )
            t_window = np.where(np.logical_and(self.time >= tmin, self.time <= tmax))[0]
            adx, bdx = min(t_window), max(t_window) + 1

            # Create design matrix - a `poly_deg` degree polynomial in time.
            X = np.vstack([(scaled_time[adx:bdx]) ** idx for idx in range(poly_deg)]).T

            # Initialise priors on LM components (broad normal distributions).
            prior_mu = np.zeros(X.shape[1])
            prior_sigma = np.ones(X.shape[1]) * 1e4

            # Initialise BG linear model distribution.
            linear_model_dist = np.zeros((self.all_flux.shape[1], 100))

            # Loop through each relevant pixel for fitting i.e. pixels inside movingTPF region.
            for pdx in np.where(self.target_mask[t])[0]:
                # Mask times with bad SPOC quality or where the moving target is present in the pixel.
                k = np.logical_and(
                    ~source_mask[adx:bdx][:, pdx], spoc_quality_mask[adx:bdx]
                )

                # Prime while loop
                n_clip = np.inf

                while n_clip != 0:
                    # If there are no times with which to fit, return nan.
                    if not k.any():
                        linear_model_dist[pdx] = np.nan
                        break

                    # Update priors: first component should be close to the median of the pixel flux value, but it can't be negative.
                    # If it is negative, leave prior unchanged.
                    prior_mu[0] = np.nanmedian(sl_corr_flux[adx:bdx][k, pdx])
                    if prior_mu[0] <= 0:
                        prior_mu[0] = 0
                    else:
                        prior_sigma[0] = prior_mu[0] ** 0.5

                    # Use weighted Bayesian LS.
                    sigma_w_inv = X[k].T.dot(
                        X[k] / sl_corr_flux_err[adx:bdx][k, pdx, None] ** 2
                    ) + np.diag(1 / prior_sigma**2)
                    B = (
                        X[k].T.dot(
                            sl_corr_flux[adx:bdx][k, pdx]
                            / sl_corr_flux_err[adx:bdx][k, pdx] ** 2
                        )
                        + prior_mu / prior_sigma**2
                    )

                    # Find the best-fitting weights and errors and turn it into a distribution
                    w = np.linalg.solve(sigma_w_inv, B)
                    werr = np.linalg.inv(sigma_w_inv)
                    wdist = np.random.multivariate_normal(
                        w, werr, size=linear_model_dist.shape[1]
                    )
                    linear_model_dist[pdx] = X[np.arange(adx, bdx) == t].dot(wdist.T)[0]

                    # Mask significant outliers.
                    n_clip_prev = np.sum(~k)
                    k = np.logical_and(
                        k,
                        np.abs(sl_corr_flux[adx:bdx][:, pdx] - X.dot(w))
                        / (sl_corr_flux_err[adx:bdx][:, pdx])
                        < sigma,
                    )
                    n_clip = np.sum(~k) - n_clip_prev

            # Compute BG linear model and error from distribution.
            linear_model[t], linear_model_err[t] = (
                np.nanmean(linear_model_dist, axis=1),
                np.nanstd(linear_model_dist, ddof=1, axis=1)
                / np.sqrt(linear_model_dist.shape[1]),
            )

            # If there is no `linear_model`, replace with median and MAD and flag as bad quality.
            for pix in np.where(np.isnan(linear_model[t]))[0]:
                linear_model[t, pix] = np.nanmedian(sl_corr_flux[adx:bdx][:, pix])
                linear_model_err[t, pix] = stats.median_abs_deviation(
                    sl_corr_flux[adx:bdx][:, pix]
                )
                # Update LM NaN mask
                if hasattr(self, "lm_nan_mask"):
                    self.lm_nan_mask[t, pix] = True
        logger.info(
            "Finished computation of background linear model in {0:.2f} sec.".format(
                time.time() - start_time
            )
        )

        # Reshape arrays to match `self.flux`
        sl_model_reshaped = []
        sl_model_err_reshaped = []
        linear_model_reshaped = []
        linear_model_err_reshaped = []
        for t in range(len(self.time)):
            sl_model_reshaped.append(
                sl_model[t][self.target_mask[t]].reshape(self.shape)
            )
            sl_model_err_reshaped.append(
                sl_model_err[t][self.target_mask[t]].reshape(self.shape)
            )
            linear_model_reshaped.append(
                linear_model[t][self.target_mask[t]].reshape(self.shape)
            )
            linear_model_err_reshaped.append(
                linear_model_err[t][self.target_mask[t]].reshape(self.shape)
            )

        # Combine LM and SL model to create global BG model
        bg = np.asarray(sl_model_reshaped) + np.asarray(linear_model_reshaped)
        bg_err = np.sqrt(
            np.nansum(
                [
                    np.asarray(linear_model_err_reshaped) ** 2,
                    np.asarray(sl_model_err_reshaped) ** 2,
                ],
                axis=0,
            )
        )

        return (
            bg,
            bg_err,
            np.asarray(sl_model_reshaped),
            np.asarray(sl_model_err_reshaped),
            np.asarray(linear_model_reshaped),
            np.asarray(linear_model_err_reshaped),
        )

    def create_aperture(self, method: str = "prf", **kwargs):
        """
        Creates an aperture mask using a method ['threshold', 'prf', 'ellipse'].
        It creates the `self.aperture_mask` attribute with the 3D mask.

        Parameters
        ----------
        method : str
            Method used for aperture estimation. One of ['threshold', 'prf', 'ellipse'].
        kwargs : dict
            Keywords arguments passed to aperture mask method, e.g
            `self._create_threshold_mask` takes `threshold` and `reference_pixel`.

        Returns
        -------
        """
        # Initialise mask that will flag NaNs in PRF model, only if PRF aperture is
        # the most recent method run. This is used by _create_lc_quality().
        self.prf_nan_mask = np.zeros_like(self.time, dtype=bool)

        # Get mask via chosen method
        if method == "threshold":
            self.aperture_mask = self._create_threshold_aperture(**kwargs)
        elif method == "prf":
            self.aperture_mask = self._create_prf_aperture(**kwargs)
        elif method == "ellipse":
            self.aperture_mask = self._create_ellipse_aperture(
                return_params=False, **kwargs
            )
        else:
            raise ValueError(
                f"Method must be one of: ['threshold', 'prf', 'ellipse']. Not '{method}'"
            )

        self.ap_method = method

        logger.info("Created aperture using method {0}.".format(self.ap_method))

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
        mad = stats.median_abs_deviation(self.corr_flux.ravel())

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

        return aperture_mask

    def _create_prf_aperture(self, threshold: Union[str, float] = 0.01, **kwargs):
        """
        Creates an aperture mask from the PRF model.

        Parameters
        ----------
        threshold : float or 'optimal'
            If float, must be in the range [0,1). Only pixels where the prf model >= `threshold`
            will be included in the aperture.
            If 'optimal', computes optimal value for threshold.
        **kwargs
            Keyword arguments to be passed to `_create_target_prf_model()`.

        Returns
        -------
        aperture_mask : ndarray
            Boolean numpy array, where pixels inside the aperture are 'True'.
            Shape is (ntimes, nrows, ncols).
        """

        # Create PRF model
        self.prf_model = self._create_target_prf_model(**kwargs)

        # Use PRF model to define aperture
        if threshold == "optimal":
            raise NotImplementedError(
                "Computation of optimal PRF aperture not implemented yet."
            )
        elif isinstance(threshold, float) and threshold < 1 and threshold >= 0:
            aperture_mask = self.prf_model >= threshold  # type: ignore
            # No pixels above threshold
            if (~aperture_mask).all():
                logger.warning(
                    f"When computing the PRF aperture, none of the pixels in the PRF model were greater than the threshold ({threshold})."
                )
        else:
            raise ValueError(
                f"Threshold must be either 'optimal' or a float between 0 and 1. Not '{threshold}'"
            )

        return aperture_mask

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
        """

        if not hasattr(self, "all_flux"):
            raise AttributeError("Must run `get_data()` before creating PRF model.")

        # If input ephemeris is in UTC, convert tstart and tstop to UTC.
        if self.time_scale == "tdb":
            tstart = self.tstart
            tstop = self.tstop
        else:
            tstart = self.tstart - self.timecorr_original
            tstop = self.tstop - self.timecorr_original

        # Initialise PRF - don't specify sector => uses post-sector4 models in all cases.
        prf = lkprf.TESSPRF(camera=self.camera, ccd=self.ccd)  # , sector=self.sector)

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
            resolution = 0.1 if track_length > 0.1 else track_length
            # Time resolution at which to evaluate PRF model
            time_step = (cadence / track_length) * resolution
            logger.info(
                "_create_target_prf_model() calculated a time_step of {0} minutes.".format(
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
            logger.warning("Warning from prf.evaluate(): {0}".format(w))

        # If first/last frame contains nans, replace PRF model with following/preceding frame.
        if np.isnan(prf_model).any():
            nan_ind = np.unique(np.where(np.isnan(prf_model))[0])

            # Update mask of NaNs in PRF model.
            if hasattr(self, "prf_nan_mask"):
                self.prf_nan_mask[nan_ind] = True

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

        return np.asarray(prf_model)

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
        ellipse_parameters: ndarray
            If `return_params`, will return centroid and ellipse parameters
            [X_cent, Y_cent, A, B, theta_deg] with shape (5, n_times).
        """
        # create a threshold mask to select pixels to use for moments
        threshold_mask = self._create_threshold_aperture(
            threshold=3.0, reference_pixel="center"
        )

        X, Y, X2, Y2, XY = compute_moments(self.flux, threshold_mask)

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
            return aperture_mask, np.array(
                [self.corner[:, 1] + X, self.corner[:, 0] + Y, A, B, theta_deg]
            ).T
        else:
            return aperture_mask

    def create_pixel_quality(
        self, sat_level: float = 1e5, sat_buffer_rad: int = 1, **kwargs
    ):
        """
        Create 3D pixel quality mask. The mask is a bit-wise combination of
        the following flags:

        Bit - Description
        ----------------
        1 - pixel is outside of science array
        2 - pixel is in a strap column
        3 - pixel is saturated
        4 - pixel is within `sat_buffer_rad` pixels of a saturated pixel
        5 - pixel has no scattered light correction. Only relevant if `linear_model` background correction was used.
        6 - pixel had no background linear model, value was infilled. Only relevant if `linear_model` background correction was used.

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
                "Must run `get_data()`, `reshape_data()` and `background_correction()` before creating pixel quality mask."
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

        # >>>>> ADD A MASK FOR OTHER SATURATION FEATURES <<<<<

        # Combine masks
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
                # Pixel did not have a background linear model, so value was replaced with median flux in
                # time window. It is only meaningful if the `linear_model` background correction was used.
                "lm_nan_mask": {
                    "bit": 6,
                    "value": self.lm_nan_mask[t][self.target_mask[t]].reshape(
                        self.shape
                    ),
                },
            }
            # Compute bit-wise mask
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
            Keyword arguments, e.g `self._aperture_photometry` takes `bad_bits`.

        Returns
        -------
        """

        # Initialise lightcurve dictionary
        if not hasattr(self, "lc"):
            self.lc = {}

        # Get lightcurve and quality mask via aperture photometry
        if method == "aperture":
            (
                flux,
                flux_err,
                bg,
                bg_err,
                col_cen,
                row_cen,
                col_cen_err,
                row_cen_err,
                flux_fraction,
            ) = self._aperture_photometry(**kwargs)
            quality = self._create_lc_quality()
            self.lc["aperture"] = {
                "time": self.time,
                "flux": flux,
                "flux_err": flux_err,
                "bg": bg,
                "bg_err": bg_err,
                "col_cen": col_cen,
                "col_cen_err": col_cen_err,
                "row_cen": row_cen,
                "row_cen_err": row_cen_err,
                "quality": quality,
                "flux_fraction": flux_fraction,
            }

        elif method == "psf":
            raise NotImplementedError("PSF photometry is not yet implemented.")
        else:
            raise ValueError(
                f"Method must be one of: ['aperture', 'psf']. Not '{method}'"
            )

    def _aperture_photometry(self, bad_bits: list = [1, 3], **kwargs):
        """
        Gets flux and BG flux inside aperture and computes flux-weighted centroid.

        Parameters
        ----------
        bad_bits : list
            Bits to mask during computation of aperture flux, BG flux and centroid. These bits correspond
            to the `self.pixel_quality` flags. By default, bits 1 (non-science pixel) and 3 (saturated pixel)
            are masked.

        Returns
        -------
        ap_flux, ap_flux_err, ap_bg, ap_bg_err, col_cen, row_cen, col_cen_err, row_cen_err, flux_fraction : ndarrays
            Sum of flux inside aperture and error (ap_flux, ap_flux_err), sum of background flux inside
            aperture and error (ap_bg, ap_bg_err), flux-weighted centroids inside aperture and errors
            (col_cen, row_cen, col_cen_err, row_cen_err) and fraction of PRF model flux inside aperture (flux_fraction).
            The row and column centroids are one-indexed and correspond to the position in the full FFI, where the
            lower left pixel has the value (1,1). Flux fraction will be nan unless `prf` aperture is used.
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

        # Compute `value` to mask bad bits.
        self.bad_bit_value = 0
        for bit in bad_bits:
            self.bad_bit_value += 2 ** (bit - 1)
        self.bad_bits = ",".join([str(bit) for bit in bad_bits])

        mask = []
        ap_flux = []
        ap_flux_err = []
        ap_bg = []
        ap_bg_err = []
        flux_fraction = []

        for t in range(len(self.time)):
            # Combine aperture mask with masking of bad bits.
            mask.append(
                np.logical_and(
                    self.aperture_mask[t],
                    self.pixel_quality[t] & self.bad_bit_value == 0,
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
            # If aperture method is not `prf`, this will be nan.
            if self.ap_method == "prf":
                flux_fraction.append(np.nansum(self.prf_model[t][mask[-1]], axis=0))
                if np.isnan(self.prf_model[t][mask[-1]]).all():
                    flux_fraction[-1] = np.nan
            else:
                flux_fraction.append(np.nan)

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

        return (
            np.asarray(ap_flux),
            np.asarray(ap_flux_err),
            np.asarray(ap_bg),
            np.asarray(ap_bg_err),
            col_cen,
            row_cen,
            col_cen_err,
            row_cen_err,
            np.asarray(flux_fraction),
        )

    def _create_lc_quality(self, method: str = "aperture"):
        """
        Creates quality mask for lightcurve. This is defined independently of SPOC quality flags.
        For `aperture` method, the mask is a bit-wise combination of the following flags:

        Bit - Description
        ----------------
        1 - no pixels inside aperture.
        2 - at least one non-science pixel inside aperture.
        3 - at least one pixel inside aperture is in a strap column.
        4 - at least one saturated pixel inside aperture.
        5 - at least one pixel inside aperture is 4-adjacent to a saturated pixel.
        6 - all pixels inside aperture are `bad_bits`.
        7 - PRF model contained nans.
            Only relevant if `prf` aperture was used.
        8 - at least one pixel inside aperture does not have scattered light correction.
            Only relevant if `linear_model` background correction was used.
        9 - at least one pixel inside aperture had no background linear model, value was infilled.
            Only relevant if `linear_model` background correction was used.

        Parameters
        ----------
        method : str
            Photometric extraction method. One of `aperture` or `psf`.

        Returns
        -------
        lc_quality : ndarray
            Lightcurve quality mask with length [ntimes].
        """

        if (
            not hasattr(self, "all_flux")
            or not hasattr(self, "flux")
            or not hasattr(self, "corr_flux")
            or not hasattr(self, "pixel_quality")
        ):
            raise AttributeError(
                "Must run `get_data()`, `reshape_data()`, `background_correction()` and `create_pixel_quality()` before creating lightcurve quality."
            )

        if method == "aperture":
            if not hasattr(self, "aperture_mask") or not hasattr(self, "bad_bit_value"):
                raise AttributeError(
                    "With `aperture` method, must run `create_aperture()` and `_aperture_photometry()` before creating lightcurve quality."
                )

            # Define masks
            masks = {
                # No pixels in aperture
                "no_pixel_mask": {
                    "bit": 1,
                    "value": np.array(
                        [self.aperture_mask[t].sum() for t in range(len(self.time))]
                    )
                    == 0,
                },
                # Non-science pixel in aperture
                "science_mask": {
                    "bit": 2,
                    "value": [
                        (self.pixel_quality[t][self.aperture_mask[t]] & 1 != 0).any()
                        for t in range(len(self.time))
                    ],
                },
                # Strap in aperture
                "strap_mask": {
                    "bit": 3,
                    "value": [
                        (self.pixel_quality[t][self.aperture_mask[t]] & 2 != 0).any()
                        for t in range(len(self.time))
                    ],
                },
                # Saturated pixel in aperture
                "sat_mask": {
                    "bit": 4,
                    "value": [
                        (self.pixel_quality[t][self.aperture_mask[t]] & 4 != 0).any()
                        for t in range(len(self.time))
                    ],
                },
                # Pixel in aperture is 4-adjacent to saturated pixel
                "sat_buffer_mask": {
                    "bit": 5,
                    "value": [
                        (self.pixel_quality[t][self.aperture_mask[t]] & 8 != 0).any()
                        for t in range(len(self.time))
                    ],
                },
                # All pixels in aperture are `bad_bits`, as defined by user in _aperture_photometry()
                "bad_bit_mask": {
                    "bit": 6,
                    "value": [
                        (
                            self.pixel_quality[t][self.aperture_mask[t]]
                            & self.bad_bit_value
                            != 0
                        ).all()
                        for t in range(len(self.time))
                    ],
                },
                # PRF model contained nans and was replaced with preceding/following frame.
                # This will only be meaningful if the `prf` aperture was used.
                "prf_nan_mask": {"bit": 7, "value": self.prf_nan_mask},
                # Pixel in aperture with no scattered light correction
                # Only relevant if `linear_model` background correction was used.
                "sl_nan_mask": {
                    "bit": 8,
                    "value": [
                        (self.pixel_quality[t][self.aperture_mask[t]] & 16 != 0).any()
                        for t in range(len(self.time))
                    ],
                },
                # Pixel in aperture with no background linear model, value was infilled.
                # Only relevant if `linear_model` background correction was used.
                "lm_nan_mask": {
                    "bit": 9,
                    "value": [
                        (self.pixel_quality[t][self.aperture_mask[t]] & 32 != 0).any()
                        for t in range(len(self.time))
                    ],
                },
                # Add flag for negative pixels in aperture?
            }

        elif method == "psf":
            raise NotImplementedError("PSF photometry is not yet implemented.")

        else:
            raise ValueError(
                f"Method must be one of: ['aperture', 'psf']. Not '{method}'"
            )

        # Compute bit-wise mask
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
            Type of file to be converted to FITS. One of ['tpf', 'lc'].
        save : bool
            If True, write the HDUList to a file.
        overwrite : bool
            If `save`, this determines whether to overwrite an existing file with the
            same name.
        outdir : str
            If `save`, this is the directory into which the file will be saved.
        file_name : str
            If `save`, this is the filename that will be used. Format must be `.fits`.
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

    def _make_primary_hdu(self, file_type: str):
        """
        Make primary header for moving TPF and lightcurve file. Initialises using tesscube
        and updates/adds keywords, e.g. properties of target, settings used to create files.

        Parameters
        ----------
        file_type : str
            Type of file to create a primary HDU for. One of ['tpf', 'lc'].

        Returns
        -------
        hdulist : astropy.io.fits.PrimaryHDU
            Primary HDU to use in moving TPF or lightcurve file.
        """

        # Attribute checks
        if (
            not hasattr(self, "time")
            or not hasattr(self, "bg_method")
            or not hasattr(self, "ap_method")
        ):
            raise AttributeError(
                "Must run `get_data()`, `background_correction()` and `create_aperture()` before creating primary header."
            )

        # Get primary hdu from tesscube
        hdu = self.cube.output_primary_ext

        # Update existing keywords
        hdu.header.set("DATE", datetime.now().strftime("%Y-%m-%d"))
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
        # Future: update TESSMAG with measured value.

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

        # Add keywords to describe how file was created
        hdu.header.set(
            "SHAPE",
            "({0},{1})".format(*self.shape),
            comment="shape of TPF (row, column)",
            after="CCD",
        )
        hdu.header.set(
            "BG_CORR",
            self.bg_method,
            comment="method used for background correction",
            after="SHAPE",
        )
        hdu.header.set(
            "AP_TYPE",
            self.ap_method,
            comment="method used to create aperture",
            after="BG_CORR",
        )
        hdu.header.set(
            "AP_NPIX",
            np.nanmedian([np.nansum(mask) for mask in self.aperture_mask]),
            comment="average number of pixels in aperture",
            after="AP_TYPE",
        )
        if file_type == "lc" and hasattr(self, "bad_bits"):
            hdu.header.set(
                "BAD_BITS",
                f"[{self.bad_bits}]",
                comment="bits excluded during aperture photometry",
                after="AP_NPIX",
            )

        # Add keywords for object properties
        hdu.header.set(
            "VMAG",
            round(
                np.nanmean(
                    self.ephem.loc[
                        np.logical_and(
                            self.ephem["time"]
                            >= (
                                self.time_original[0]
                                if self.time_scale == "tdb"
                                else self.time_original[0] - self.timecorr_original[0]
                            ),
                            self.ephem["time"]
                            <= (
                                self.time_original[-1]
                                if self.time_scale == "tdb"
                                else self.time_original[-1] - self.timecorr_original[-1]
                            ),
                        ),
                        "vmag",
                    ]
                ),
                3,
            )
            if "vmag" in self.ephem
            else 0.0,
            comment="V magnitude",
            after="TICVER",
        )
        hdu.header.set("HMAG", 0.0, comment="H absolute magnitude", after="VMAG")
        hdu.header.set("PERIHEL", 0.0, comment="[AU] perihelion", after="HMAG")
        hdu.header.set("ORBECC", 0.0, comment="orbit eccentricity", after="PERIHEL")
        hdu.header.set("ORBINC", 0.0, comment="[deg] orbit inclination", after="ORBECC")
        hdu.header.set(
            "RARATE", 0.0, comment='["/h] right ascension rate', after="ORBINC"
        )
        hdu.header.set("DECRATE", 0.0, comment='["/h] declination rate', after="RARATE")

        return hdu

    def _make_tpf_hdulist(self):
        """
        Make HDUList for moving TPF, using similar format to SPOC.

        Parameters
        ----------

        Returns
        -------
        hdulist : astropy.io.fits.HDUList
            HDUList for moving TPF.
        """
        if (
            not hasattr(self, "all_flux")
            or not hasattr(self, "flux")
            or not hasattr(self, "corr_flux")
            or not hasattr(self, "aperture_mask")
            or not hasattr(self, "pixel_quality")
        ):
            raise AttributeError(
                "Must run `get_data()`, `reshape_data()`, `background_correction()`, `create_pixel_quality()` and `create_aperture()` before saving TPF."
            )

        # Compute WCS header
        wcs_header = make_wcs_header(self.shape)

        # Offset between expected target position and center of TPF
        pos_corr1 = self.ephemeris[:, 1] - self.corner[:, 1] - 0.5 * (self.shape[1] - 1)
        pos_corr2 = self.ephemeris[:, 0] - self.corner[:, 0] - 0.5 * (self.shape[0] - 1)

        # Define SPOC-like FITS columns
        tform = str(self.corr_flux[0].size) + "E"
        dims = str(self.corr_flux[0].shape[::-1])
        cols = [
            # Times in TDB at SS barycenter.
            fits.Column(
                name="TIME",
                format="D",
                unit="BJD - 2457000, days",
                disp="D14.7",
                array=self.time,
            ),
            # Correction used to convert UTC at spacecraft to TDB at barycenter.
            fits.Column(
                name="TIMECORR",
                format="E",
                unit="d",
                disp="E14.7",
                array=self.timecorr,
            ),
            # Cadence number, as defined by tesscube.
            fits.Column(name="CADENCENO", format="I", array=self.cadence_number),
            # RAW_CNTS is included to give the files the same structure as the SPOC files
            fits.Column(
                name="RAW_CNTS",
                format=tform,
                dim=dims,
                unit="e-/s",
                disp="E14.7",
                array=np.zeros_like(self.corr_flux),
            ),
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
            cols,
            header=fits.Header(
                [
                    *self.cube.output_first_header.cards,
                    *get_wcs_header_by_extension(wcs_header, ext=4).cards,
                    *get_wcs_header_by_extension(wcs_header, ext=5).cards,
                    *get_wcs_header_by_extension(wcs_header, ext=6).cards,
                    *get_wcs_header_by_extension(wcs_header, ext=7).cards,
                    *get_wcs_header_by_extension(wcs_header, ext=8).cards,
                ]
            ),
        )
        table_hdu_spoc.header["EXTNAME"] = "PIXELS"

        # Create HDU containing average aperture
        # Aperture has values 0 and 2, where 0/2 indicates the pixel is outside/inside the aperture.
        # This format is used to be consistent with the aperture HDU from SPOC.
        aperture_hdu_average = fits.ImageHDU(
            data=np.nanmedian(self.aperture_mask, axis=0).astype("int32") * 2,
            header=fits.Header(
                [*self.cube.output_secondary_header.cards, *wcs_header.cards]
            ),
        )
        aperture_hdu_average.header["EXTNAME"] = "APERTURE"
        aperture_hdu_average.header.set(
            "NPIXSAP", None, "Number of pixels in optimal aperture"
        )
        aperture_hdu_average.header.set(
            "NPIXMISS", None, "Number of op. aperture pixels not collected"
        )

        # Define extra FITS columns
        cols = [
            # Original TESS FFI timestamps, in TDB at SS barycenter.
            # This was calculated for the center of the FFI, not the target position.
            fits.Column(
                name="ORIGINAL_TIME",
                format="D",
                unit="BJD - 2457000, days",
                disp="D14.7",
                array=self.time_original,
            ),
            # Original correction used to convert UTC at spacecraft to TDB at barycenter.
            # This was calculated for the center of the FFI, not the target position.
            fits.Column(
                name="ORIGINAL_TIMECORR",
                format="E",
                unit="d",
                disp="E14.7",
                array=self.timecorr_original,
            ),
            # Predicted position of target in world coordinates.
            fits.Column(
                name="RA",
                format="E",
                unit="deg",
                disp="E14.7",
                array=[coord.ra.value for coord in self.coords],
            ),
            fits.Column(
                name="DEC",
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
            # 3D pixel quality mask
            fits.Column(
                name="PIXEL_QUALITY",
                format=str(self.corr_flux[0].size) + "I",
                dim=dims,
                disp="B16.16",
                array=self.pixel_quality,
            ),
            # Aperture as a function of time.
            # Aperture has values 0 and 2, where 0/2 indicates the pixel is outside/inside the aperture.
            # This format is used to be consistent with the aperture HDU from SPOC.
            fits.Column(
                name="APERTURE",
                format=str(self.corr_flux[0].size) + "J",
                dim=dims,
                array=self.aperture_mask.astype("int32") * 2,
            ),
        ]

        # Create table HDU for extra columns
        table_hdu_extra = fits.BinTableHDU.from_columns(cols)
        table_hdu_extra.header["EXTNAME"] = "EXTRAS"

        # Return hdulist
        return fits.HDUList(
            [
                self._make_primary_hdu(file_type="tpf"),
                table_hdu_spoc,
                aperture_hdu_average,
                table_hdu_extra,
            ]
        )

    def _make_lc_hdulist(self):
        """
        Make HDUList for lightcurve file.

        Parameters
        ----------

        Returns
        -------
        hdulist : astropy.io.fits.HDUList
            HDUList for lightcurve file.
        """

        # Attribute checks
        if not hasattr(self, "lc") or len(self.lc) == 0:
            raise AttributeError("Must run `to_lightcurve()` before saving LC.")

        # Define columns for lightcurve
        cols = [
            # Times in TDB at SS barycenter.
            fits.Column(
                name="TIME",
                format="D",
                unit="BJD - 2457000, days",
                disp="D14.7",
                array=self.time,
            ),
            # Correction used to convert UTC at spacecraft to TDB at barycenter.
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
            # Original correction used to convert UTC at spacecraft to TDB at barycenter.
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
                format="D",
                unit="e-/s",
                disp="D14.7",
                array=self.lc["aperture"]["flux"]
                if "aperture" in self.lc
                else np.full(len(self.time), np.nan),
            ),
            fits.Column(
                name="FLUX_ERR",
                format="D",
                unit="e-/s",
                disp="D14.7",
                array=self.lc["aperture"]["flux_err"]
                if "aperture" in self.lc
                else np.full(len(self.time), np.nan),
            ),
            # Sum of BG flux inside aperture and err
            fits.Column(
                name="FLUX_BKG",
                format="D",
                unit="e-/s",
                disp="D14.7",
                array=self.lc["aperture"]["bg"]
                if "aperture" in self.lc
                else np.full(len(self.time), np.nan),
            ),
            fits.Column(
                name="FLUX_BKG_ERR",
                format="D",
                unit="e-/s",
                disp="D14.7",
                array=self.lc["aperture"]["bg_err"]
                if "aperture" in self.lc
                else np.full(len(self.time), np.nan),
            ),
            # Column centroid and err
            fits.Column(
                name="MOM_CENTR1",
                format="E",
                unit="pixel",
                disp="E14.7",
                array=self.lc["aperture"]["col_cen"]
                if "aperture" in self.lc
                else np.full(len(self.time), np.nan),
            ),
            fits.Column(
                name="MOM_CENTR1_ERR",
                format="E",
                unit="pixel",
                disp="E14.7",
                array=self.lc["aperture"]["col_cen_err"]
                if "aperture" in self.lc
                else np.full(len(self.time), np.nan),
            ),
            # Row centroid and err
            fits.Column(
                name="MOM_CENTR2",
                format="E",
                unit="pixel",
                disp="E14.7",
                array=self.lc["aperture"]["row_cen"]
                if "aperture" in self.lc
                else np.full(len(self.time), np.nan),
            ),
            fits.Column(
                name="MOM_CENTR2_ERR",
                format="E",
                unit="pixel",
                disp="E14.7",
                array=self.lc["aperture"]["row_cen_err"]
                if "aperture" in self.lc
                else np.full(len(self.time), np.nan),
            ),
            # Quality from _create_lc_quality()
            fits.Column(
                name="AP_QUALITY",
                format="I" if "aperture" in self.lc else "E",
                disp="B16.16",
                array=self.lc["aperture"]["quality"]
                if "aperture" in self.lc
                else np.full(len(self.time), np.nan),
            ),
            # Fraction of PRF model flux inside aperture. If `prf`
            # aperture was not used, this will be nan.
            fits.Column(
                name="FLUX_FRACTION",
                format="E",
                disp="E14.7",
                array=self.lc["aperture"]["flux_fraction"]
                if "aperture" in self.lc
                else np.full(len(self.time), np.nan),
            ),
            # --------------
            # PSF photometry
            # Flux and err
            fits.Column(
                name="PSF_FLUX",
                format="D",
                unit="e-/s",
                disp="D14.7",
                array=self.lc["psf"]["flux"]
                if "psf" in self.lc
                else np.full(len(self.time), np.nan),
            ),
            fits.Column(
                name="PSF_FLUX_ERR",
                format="D",
                unit="e-/s",
                disp="D14.7",
                array=self.lc["psf"]["flux_err"]
                if "psf" in self.lc
                else np.full(len(self.time), np.nan),
            ),
            # Column centroid and err
            fits.Column(
                name="PSF_CENTR1",
                format="E",
                unit="pixel",
                disp="E14.7",
                array=self.lc["psf"]["col_cen"]
                if "psf" in self.lc
                else np.full(len(self.time), np.nan),
            ),
            fits.Column(
                name="PSF_CENTR1_ERR",
                format="E",
                unit="pixel",
                disp="E14.7",
                array=self.lc["psf"]["col_cen_err"]
                if "psf" in self.lc
                else np.full(len(self.time), np.nan),
            ),
            # Row centroid and err
            fits.Column(
                name="PSF_CENTR2",
                format="E",
                unit="pixel",
                disp="E14.7",
                array=self.lc["psf"]["row_cen"]
                if "psf" in self.lc
                else np.full(len(self.time), np.nan),
            ),
            fits.Column(
                name="PSF_CENTR2_ERR",
                format="E",
                unit="pixel",
                disp="E14.7",
                array=self.lc["psf"]["row_cen_err"]
                if "psf" in self.lc
                else np.full(len(self.time), np.nan),
            ),
            # Quality from _create_lc_quality()
            fits.Column(
                name="PSF_QUALITY",
                format="I" if "psf" in self.lc else "E",
                disp="B16.16",
                array=self.lc["psf"]["quality"]
                if "psf" in self.lc
                else np.full(len(self.time), np.nan),
            ),
            # --------------
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
            # Predicted position of target in world coordinates.
            fits.Column(
                name="RA",
                format="E",
                unit="deg",
                disp="E14.7",
                array=[coord.ra.value for coord in self.coords],
            ),
            fits.Column(
                name="DEC",
                format="E",
                unit="deg",
                disp="E14.7",
                array=[coord.dec.value for coord in self.coords],
            ),
        ]

        # Create table HDU
        table_hdu = fits.BinTableHDU.from_columns(cols)
        table_hdu.header["EXTNAME"] = "LIGHTCURVE"

        # Return hdulist
        return fits.HDUList([self._make_primary_hdu(file_type="lc"), table_hdu])

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
                str(self.target).replace(" ", ""),
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
        hdulist.writeto(outdir + file_name, overwrite=overwrite)
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

        Parameters:
        ----------
        show_aperture : bool
            If True, the aperture used for photometry is displayed in the animation. Default is True.
        show_ephemeris : bool
            If True, the predicted position of the target is included in the animation. Default is True.
        step : int or None
            Spacing between frames, i.e. plot every nth frame.  If `None`, the spacing will be determined such
            that about 50 frames are shown. Showing more frames will increase the runtime and, if `save`, the
            file size.
        save : bool
            If True, save the animation. Default is False.
        outdir : str
            If `save`, this is the directory into which the file will be saved.
        file_name : str or None
            If `save`, this is the filename that will be used. Format must be `.gif`.
            If no filename is given, a default one will be generated.
        kwargs:
            Keyword arguments passed to `utils.animate_cube` such as [`interval`, `repeat_delay`, `cnorm`,
            `vmin`, `vmax`].

        Returns:
        --------
        Animation in HTML format. If in notebook environment, this allows animation to be displayed.
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
                        str(self.target).replace(" ", ""),
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
                "ipython needs to be installed for animate() to work (e.g., `pip install ipython`)"
            )

    @staticmethod
    def from_name(
        target: str,
        sector: int,
        camera: Optional[int] = None,
        ccd: Optional[int] = None,
        time_step: float = 1.0,
    ):
        """
        Initialises MovingTPF from target name and TESS sector. Specifying a camera and
        CCD will only use the ephemeris from that camera/ccd.

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
            Initialised MovingTPF with ephemeris from JPL/Horizons.
            Target ephemeris has columns ['time','sector','camera','ccd','column','row','vmag'].
                'time' : float with units (JD - 2457000) in UTC at spacecraft.
                'sector', 'camera', 'ccd' : int
                'column', 'row' : float. These are one-indexed, where the lower left pixel of the FFI is (1,1).
                'vmag' : float. Visual magnitude.
        """

        # Get target ephemeris using tess-ephem
        logger.info("Retrieving ephemeris for target {0}.".format(target))
        df_ephem = ephem(target, sector=sector, time_step=time_step, verbose=True)

        # Check whether target was observed in sector.
        if len(df_ephem) == 0:
            raise ValueError(
                "Target {} was not observed in sector {}.".format(target, sector)
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

        # Add column for time in units (JD - 2457000)
        # >>>>> Note: tess-ephem returns time in UTC at spacecraft. <<<<<
        df_ephem["time"] = [t.value - 2457000 for t in df_ephem.index.values]
        df_ephem = df_ephem[
            ["time", "sector", "camera", "ccd", "column", "row", "vmag"]
        ].reset_index(drop=True)

        return MovingTPF(target=target, ephem=df_ephem, time_scale="utc")
