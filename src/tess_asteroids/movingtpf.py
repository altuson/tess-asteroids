import time
from typing import Optional, Tuple, Union

import lkprf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.wcs.utils import fit_wcs_from_points
from numpy.polynomial import Polynomial
from scipy import ndimage, stats
from tess_ephem import ephem
from tesscube import TESSCube
from tesscube.fits import get_wcs_header_by_extension
from tesscube.utils import _sync_call, convert_coordinates_to_runs

from . import logger, straps
from .utils import animate_cube, compute_moments, inside_ellipse


class MovingTPF:
    """
    Create a TPF for a moving target (e.g. asteroid) from a TESS FFI.

    Includes methods to efficiently retrieve the data, correct the background,
    define an aperture mask and save a TPF in the SPOC format.

    Parameters
    ----------
    target : str
        Target ID. This is only used when saving the TPF.
    ephem : DataFrame
        Target ephemeris with columns ['time','sector','camera','ccd','column','row'].
            'time' : float with units (BJD - 2457000).
            'sector', 'camera', 'ccd' : int
            'column', 'row' : float. These must be one-indexed, where the lower left pixel of the FFI is (1,1).
    """

    def __init__(self, target: str, ephem: pd.DataFrame):
        self.target = target
        self.ephem = ephem

        # Check self.ephem['time'] has correct units
        if min(self.ephem["time"]) >= 2457000:
            raise ValueError("ephem['time'] must have units (BJD - 2457000).")

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

    def make_tpf(
        self,
        shape: Tuple[int, int] = (11, 11),
        bg_method: str = "rolling",
        ap_method: str = "prf",
        file_name: Optional[str] = None,
        save_loc: str = "",
        **kwargs,
    ):
        """
        Performs all steps to create and save a SPOC-like TPF for a moving target.

        Parameters
        ----------
        shape : Tuple(int,int)
            Defined as (row,column), in pixels.
            Defines the pixels that will be retrieved, centred on the target, at each timestamp.
        bg_method : str
            Method used for background correction.
            One of `rolling`.
        ap_method : str
            Method used to create aperture.
            One of ['threshold', 'prf', 'ellipse'].
        file_name : str
            Name of saved TPF.
        save_loc : str
            Directory into which the files will be saved.
        **kwargs
            Keyword arguments to be passed to `create_pixel_quality()`, `background_correction()`,
            `create_aperture()` and `save_data()`.

        Returns
        -------
        """
        # >>>>> ADD REFINE_COORDINATES() WHEN IMPLEMENTED <<<<<
        self.get_data(shape=shape)
        self.reshape_data()
        self.create_pixel_quality(**kwargs)
        self.background_correction(method=bg_method, **kwargs)
        self.create_aperture(method=ap_method, **kwargs)
        self.save_data(file_name_tpf=file_name, save_loc=save_loc, **kwargs)

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
            Defined as (row,column), in pixels.
            Defines the pixels that will be retrieved, centred on the target, at each timestamp.

        Returns
        -------
        """
        # Shape needs to be >=3 in both dimensions, otherwise _make_wcs_header() errors.
        self.shape = shape

        # Use interpolation to get target (row,column) at cube time.
        column_interp = np.interp(
            self.cube.time,
            self.ephem["time"].astype(float),
            self.ephem["column"].astype(float),
            left=np.nan,
            right=np.nan,
        )
        row_interp = np.interp(
            self.cube.time,
            self.ephem["time"].astype(float),
            self.ephem["row"].astype(float),
            left=np.nan,
            right=np.nan,
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
        self.time = self.cube.time[nan_mask][
            bound_mask
        ]  # FFI timestamps of each frame in the data cube.
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
        for t in range(len(self.time)):
            target_mask.append(
                np.logical_and(
                    np.isin(self.pixels.T[0], row[t].ravel()),
                    np.isin(self.pixels.T[1], column[t].ravel()),
                ),
            )
        self.target_mask = np.asarray(target_mask)

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
            Method used for background correction. One of `rolling`.
        **kwargs
            Keyword arguments to be passed to `_bg_rolling_median()`.

        Returns
        -------
        """
        if not hasattr(self, "all_flux") or not hasattr(self, "flux"):
            raise AttributeError(
                "Must run `get_data()` and `reshape_data()` before computing background."
            )

        # Get background via chosen method
        if method == "rolling":
            self.bg, self.bg_err = self._bg_rolling_median(**kwargs)

        else:
            raise ValueError(
                "`method` must be one of: `rolling`. Not `{0}`".format(method)
            )

        # Apply background correction
        self.corr_flux = self.flux - self.bg
        self.corr_flux_err = np.sqrt(self.flux_err**2 + self.bg_err**2)

    def _bg_rolling_median(self, nframes: int = 25):
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

    def _create_threshold_aperture(
        self,
        threshold: float = 3.0,
        reference_pixel: Union[str, Tuple[float, float]] = "center",
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
        self._create_target_prf_model(**kwargs)

        # Use PRF model to define aperture
        if threshold == "optimal":
            raise NotImplementedError(
                "Computation of optimal PRF aperture not implemented yet."
            )
        elif isinstance(threshold, float) and threshold < 1 and threshold >= 0:
            aperture_mask = self.prf_model >= threshold  # type: ignore
        else:
            raise ValueError(
                f"Threshold must be either 'optimal' or a float between 0 and 1. Not '{threshold}'"
            )

        return aperture_mask

    def _create_target_prf_model(self, time_step: Optional[float] = None):
        """
        Creates a PRF model of the target as a function of time, using the `lkprf` package.
        Since the target is moving, the PRF model per time is made by summing models on a high
        resolution time grid during the exposure. This function creates the `self.prf_model`
        attribute. The PRF model has shape (ntimes, nrows, ncols), each value represents the
        fraction of the total flux in that pixel at that time and at each time all values
        sum to one.

        Parameters
        ----------
        time_step : float
            Resolution of time grid used to build PRF model, in minutes. A smaller time_step
            will increase the runtime, but the PRF model will better match the extended shape
            of the moving target.
            If `None`, a value will be computed based upon the average speed of the target
            during the observation.

        Returns
        -------
        """

        if not hasattr(self, "all_flux"):
            raise AttributeError("Must run `get_data()` before creating PRF model.")

        # Initialise PRF - don't specify sector => uses post-sector4 models in all cases.
        prf = lkprf.TESSPRF(camera=self.camera, ccd=self.ccd)  # , sector=self.sector)

        # If no time_step is given, compute a value based upon the average target speed.
        # Note: some asteroids significantly change speed during the sector. Our tests
        # have shown that defining time_step for the average speed does not signficiantly
        # affect their PRF models.
        if time_step is None:
            # Average cadence, in minutes
            cadence = np.nanmean(self.tstop - self.tstart) * 24 * 60
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
            self.tstart[0],
            self.tstop[-1],
            int(np.ceil((self.tstop[-1] - self.tstart[0]) * 24 * 60 / time_step)),
        )
        column_interp = np.interp(
            high_res_time,
            self.ephem["time"].astype(float),
            self.ephem["column"].astype(float),
            left=np.nan,
            right=np.nan,
        )
        row_interp = np.interp(
            high_res_time,
            self.ephem["time"].astype(float),
            self.ephem["row"].astype(float),
            left=np.nan,
            right=np.nan,
        )

        # Build PRF model at each timestamp
        prf_model = []
        for t in range(len(self.time)):
            # Find indices in `high_res_time` between corresponding tstart/tstop.
            inds = np.where(
                np.logical_and(
                    high_res_time >= self.tstart[t], high_res_time <= self.tstop[t]
                )
            )[0]

            # Get PRF model throughout exposure, sum and normalise.
            # If `row_interp` or `col_interp` contain nans (i.e. outside range of interpolation),
            # then prf.evaluate breaks. In that case, manually define model with nans.
            try:
                model = prf.evaluate(
                    targets=[
                        (r, c) for r, c in zip(row_interp[inds], column_interp[inds])
                    ],
                    origin=(self.corner[t][0], self.corner[t][1]),
                    shape=self.shape,
                )
                model = sum(model) / np.sum(model)
            except ValueError:
                model = np.full(self.shape, np.nan)

            prf_model.append(model)

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

        self.prf_model = np.asarray(prf_model)

    def _create_ellipse_aperture(
        self,
        R: float = 3.0,
        smooth: bool = True,
        return_params: bool = False,
        plot: bool = False,
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

    def create_pixel_quality(self, sat_level: float = 1e5, sat_buffer_rad: int = 1):
        """
        Create 3D pixel quality mask. The mask is a bit-wise combination of
        the following flags:

        Bit - Description
        ----------------
        1 - pixel is outside of science array
        2 - pixel is in a strap column
        3 - pixel is saturated
        4 - pixel is within `sat_buffer_rad` pixels of a saturated pixel

        Parameters
        ----------
        sat_level : float
            Flux (e-/s) above which to consider a pixel saturated.
        sat_buffer_rad : int
            Approximate radius of saturation buffer (in pixels) around each saturated pixel.

        Returns
        -------
        """
        if not hasattr(self, "pixels") or not hasattr(self, "flux"):
            raise AttributeError(
                "Must run `get_data()` and `reshape_data()` before creating pixel quality mask."
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
            flux, flux_err, bg, bg_err, col_cen, row_cen, col_cen_err, row_cen_err = (
                self._aperture_photometry(**kwargs)
            )
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
            }

        elif method == "psf":
            raise NotImplementedError("PSF photometry is not yet implemented.")
        else:
            raise ValueError(
                f"Method must be one of: ['aperture', 'psf']. Not '{method}'"
            )

    def _aperture_photometry(self, bad_bits: list = [1, 3]):
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
        ap_flux, ap_flux_err, ap_bg, ap_bg_err, col_cen, row_cen, col_cen_err, row_cen_err : ndarrays
            Sum of flux inside aperture and error (ap_flux, ap_flux_err), sum of background flux inside
            aperture and error (ap_bg, ap_bg_err) and flux-weighted centroids inside aperture and errors
            (col_cen, row_cen, col_cen_err, row_cen_err). The row and column centroids are one-indexed and
            correspond to the position in the full FFI, where the lower left pixel has the value (1,1).
        """

        if (
            not hasattr(self, "all_flux")
            or not hasattr(self, "flux")
            or not hasattr(self, "pixel_quality")
            or not hasattr(self, "corr_flux")
            or not hasattr(self, "aperture_mask")
        ):
            raise AttributeError(
                "Must run `get_data()`, `reshape_data()`, `create_pixel_quality()`, `background_correction()` and `create_aperture()` before doing aperture photometry."
            )

        # Compute `value` to mask bad bits.
        self.bad_bit_value = 0
        for bit in bad_bits:
            self.bad_bit_value += 2 ** (bit - 1)

        mask = []
        ap_flux = []
        ap_flux_err = []
        ap_bg = []
        ap_bg_err = []

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
        7 - PRF model contained nans. Only relevant if `prf` aperture was used.

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
            or not hasattr(self, "pixel_quality")
            or not hasattr(self, "corr_flux")
        ):
            raise AttributeError(
                "Must run `get_data()`, `reshape_data()`, `create_pixel_quality()` and `background_correction()` before creating lightcurve quality."
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
                        ).any()
                        for t in range(len(self.time))
                    ],
                },
                # PRF model contained nans and was replaced with preceding/following frame.
                # This will only be meaningful if the `prf` aperture was used.
                "prf_nan_mask": {"bit": 7, "value": self.prf_nan_mask},
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

    def save_data(
        self,
        save_tpf: bool = True,
        save_table: bool = False,
        file_name_tpf: Optional[str] = None,
        save_loc: str = "",
    ):
        """
        Save retrieved pixel data. Can either save all pixels at all times or save a TPF with the same
        format produced by SPOC.

        Parameters
        ----------
        save_tpf : bool
            If True, save a SPOC-like TPF file where each frame is centred on the moving target.
        save_table : bool
            If True, save a table of all retrieved pixel fluxes at all times.
        file_name_tpf : str
            If save_tpf, this is the filename that will be used for the TPF.
            If no filename is given, a default one will be generated.
        save_loc : str
            Directory into which the files will be saved.

        Returns
        -------
        """
        if save_tpf:
            # Save TPF with SPOC format
            self._save_tpf(file_name=file_name_tpf, save_loc=save_loc)

        if save_table:
            # Save table of all flux data
            self._save_table()

    def _save_tpf(self, file_name: Optional[str] = None, save_loc: str = ""):
        """
        Save data and aperture in format that matches SPOC TPFs.

        Parameters
        ----------
        file_name : str
            This is the filename that will be used for the TPF.
            If no filename is given, a default one will be generated.
        save_loc : str
            Directory into which the TPF will be saved.

        Returns
        -------
        """
        if (
            not hasattr(self, "all_flux")
            or not hasattr(self, "flux")
            or not hasattr(self, "corr_flux")
            or not hasattr(self, "aperture_mask")
            or not hasattr(self, "pixel_quality")
        ):
            raise AttributeError(
                "Must run `get_data()`, `reshape_data()`, `create_pixel_quality()`, `background_correction()` and `create_aperture()` before saving TPF."
            )

        # Create default file name
        if file_name is None:
            file_name = "tess-{0}-s{1:04}-{2}-{3}-shape{4}x{5}-moving_tp.fits".format(
                str(self.target).replace(" ", ""),
                self.sector,
                self.camera,
                self.ccd,
                *self.shape,
            )

        if not file_name.endswith(".fits"):
            raise ValueError(
                "`file_name` must be a .fits file. Not `{0}`".format(file_name)
            )
        if len(save_loc) > 0 and not save_loc.endswith("/"):
            save_loc += "/"

        # Compute WCS header
        wcs_header = self._make_wcs_header()

        # Offset between expected target position and center of TPF
        pos_corr1 = self.ephemeris[:, 1] - self.corner[:, 1] - 0.5 * (self.shape[1] - 1)
        pos_corr2 = self.ephemeris[:, 0] - self.corner[:, 0] - 0.5 * (self.shape[0] - 1)

        # Compute TIMECORR
        # >>>>> UPDATE IN FUTURE <<<<<
        time_corr = np.zeros_like(self.time)

        # Define SPOC-like FITS columns
        tform = str(self.corr_flux[0].size) + "E"
        dims = str(self.corr_flux[0].shape[::-1])
        cols = [
            fits.Column(
                name="TIME",
                format="D",
                unit="BJD - 2457000, days",
                disp="D14.7",
                array=self.time,
            ),
            fits.Column(
                name="TIMECORR",
                format="E",
                unit="d",
                disp="E14.7",
                array=time_corr,
            ),
            fits.Column(name="CADENCENO", format="I", array=self.cadence_number),
            # This is included to give the files the same structure as the SPOC files
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

        # Create hdulist and save to file
        self.hdulist = fits.HDUList(
            [
                self.cube.output_primary_ext,
                table_hdu_spoc,
                aperture_hdu_average,
                table_hdu_extra,
            ]
        )
        self.hdulist.writeto(save_loc + file_name, overwrite=True)
        logger.info("Saved TPF to {0}".format(save_loc + file_name))

    def _make_wcs_header(self):
        """
        Make a dummy WCS header for the TPF file. In reality, there is a WCS per timestamp
        that needs to be accounted for.

        Parameters
        ----------

        Returns
        -------
        wcs_header : astropy.io.fits.header.Header
            Dummy WCS header to use in the TPF.
        """
        if not hasattr(self, "shape"):
            raise AttributeError("Must run `get_data()` before making WCS header.")

        # TPF corner (row,column)
        corner = (1, 1)

        # Make a dummy WCS where each pixel in TPF is assigned coordinates 1,1
        row, column = np.meshgrid(
            np.arange(corner[0], corner[0] + self.shape[0]),
            np.arange(corner[1], corner[1] + self.shape[1]),
        )
        coord = SkyCoord(np.full([len(row.ravel()), 2], (1, 1)), unit="deg")
        wcs = fit_wcs_from_points((column.ravel(), row.ravel()), coord)

        # Turn WCS into header
        wcs_header = wcs.to_header(relax=True)

        # Add the physical WCS keywords
        wcs_header.set("CRVAL1P", corner[1], "value at reference CCD column")
        wcs_header.set("CRVAL2P", corner[0], "value at reference CCD row")

        wcs_header.set(
            "WCSNAMEP", "PHYSICAL", "name of world coordinate system alternate P"
        )
        wcs_header.set("WCSAXESP", 2, "number of WCS physical axes")

        wcs_header.set("CTYPE1P", "RAWX", "physical WCS axis 1 type CCD col")
        wcs_header.set("CUNIT1P", "PIXEL", "physical WCS axis 1 unit")
        wcs_header.set("CRPIX1P", 1, "reference CCD column")
        wcs_header.set("CDELT1P", 1.0, "physical WCS axis 1 step")

        wcs_header.set("CTYPE2P", "RAWY", "physical WCS axis 2 type CCD col")
        wcs_header.set("CUNIT2P", "PIXEL", "physical WCS axis 2 unit")
        wcs_header.set("CRPIX2P", 1, "reference CCD row")
        wcs_header.set("CDELT2P", 1.0, "physical WCS axis 2 step")

        return wcs_header

    def _save_table(self):
        """
        Save all data in table.

        Parameters
        ----------

        Returns
        -------
        """
        raise NotImplementedError("_save_table() is not yet implemented.")

    def animate_tpf(
        self,
        show_aperture: bool = True,
        show_ephemeris: bool = True,
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

        file_name : str or None
            If provided, the animation will be saved to the specified file. The format of the file has to be GIF.
            If None, the animation will not be saved. Default is None.

        kwargs:
            Keyword arguments passed to `utils.animate_cube` such as [`interval`, `repeat_delay`, `cnorm`].

        Returns:
        --------
        """

        if (
            not hasattr(self, "all_flux")
            or not hasattr(self, "flux")
            or not hasattr(self, "corr_flux")
            or not hasattr(self, "aperture_mask")
        ):
            raise AttributeError(
                "Must run `get_data()`, `reshape_data()`, `background_correction()` and `create_aperture()` before animating."
            )
        ani = animate_cube(
            self.corr_flux,
            aperture_mask=self.aperture_mask if show_aperture else None,
            corner=self.corner,
            ephemeris=self.ephemeris if show_ephemeris else None,
            cadenceno=self.cadence_number,
            time=self.time,
            suptitle=f"Asteroid {self.target} in Sector {self.sector} Camera {self.camera} CCD {self.ccd}",
            **kwargs,
        )
        if file_name is not None and isinstance(file_name, str):
            ani.save(file_name, writer="pillow")
        try:
            from IPython.display import HTML

            return HTML(ani.to_jshtml())
        except ModuleNotFoundError:
            # To make installing the package easier, ipython is not a dependency,
            # because we can assume it is installed when notebook-specific features are called
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
            Initialised MovingTPF.
        df_ephem : DataFrame
            Target ephemeris with columns ['time','sector','camera','ccd','column','row'].
                'time' : float with units (JD - 2457000).
                'sector', 'camera', 'ccd' : int
                'column', 'row' : float. These are one-indexed, where the lower left pixel of the FFI is (1,1).
        """

        # Get target ephemeris using tess-ephem
        df_ephem = ephem(target, sector=sector, time_step=time_step)

        # Check whether target was observed in sector.
        if len(df_ephem) == 0:
            raise ValueError(
                "Target {} was not observed in sector {}.".format(target, sector)
            )

        # Filter ephemeris using camera/ccd.
        if camera is not None and ccd is not None:
            df_ephem = df_ephem[
                np.logical_and(df_ephem["camera"] == camera, df_ephem["ccd"] == ccd)
            ]
            if len(df_ephem) == 0:
                raise ValueError(
                    "Target {} was not observed in sector {}, camera {}, ccd {}.".format(
                        target, sector, camera, ccd
                    )
                )

        # Add column for time in units (JD - 2457000)
        # >>>>> Note: tess-ephem returns time in JD, not BJD. <<<<<
        df_ephem["time"] = [t.value - 2457000 for t in df_ephem.index.values]
        df_ephem = df_ephem[
            ["time", "sector", "camera", "ccd", "column", "row"]
        ].reset_index(drop=True)

        return MovingTPF(target=target, ephem=df_ephem), df_ephem
