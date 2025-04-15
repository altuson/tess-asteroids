import logging
import os

import lightkurve as lk
import numpy as np
import pandas as pd
from astropy.io import fits

from tess_asteroids import MovingTPF, __version__


def test_from_name():
    """
    Check that static method from_name() gives expected ephemeris for asteroids 1998 YT6 and 1994 EL3.
    """
    target = MovingTPF.from_name("1998 YT6", sector=6)

    assert target.sector == 6
    assert target.camera == 1
    assert target.ccd == 1

    # Check the time scale is UTC
    assert target.time_scale == "utc"

    # Bounds taken from tesswcs pointings.csv file for sector 6.
    assert min(target.ephem["time"]) >= 2458463.5 - 2457000
    assert max(target.ephem["time"]) <= 2458490.5 - 2457000

    # Asteroid 1994 EL3 is observed by camera 1, CCDs 1 and 2 during sector 6.
    target = MovingTPF.from_name("1994 EL3", sector=6, camera=1, ccd=1)
    assert target.sector == 6
    assert target.camera == 1
    assert target.ccd == 1


def test_data_logic(caplog):
    """
    Check that get_data(), reshape_data() and background_correction() give expected output
    in terms of variable shapes and warnings.
    """

    # Create artificial track
    time = np.linspace(2458790.5, 2458795.5, 100) - 2457000
    track = pd.DataFrame(
        {
            "time": time,
            "sector": np.full(len(time), 18),
            "camera": np.full(len(time), 3),
            "ccd": np.full(len(time), 2),
            "column": np.linspace(-10, 100, len(time)),
            "row": np.linspace(-10, 100, len(time)),
        }
    )

    shape = (12, 10)
    target = MovingTPF("simulated_track", track)

    # Check get_data() returns expected warnings
    with caplog.at_level(logging.WARNING):
        target.get_data(shape=shape)
    assert "Some of the requested pixels are outside of the FFI bounds" in caplog.text
    assert (
        "Some of the requested pixels are outside of the FFI science array"
        in caplog.text
    )

    # Check attributes have the same length as target.time
    assert (
        len(target.all_flux) == len(target.time)
        and len(target.all_flux_err) == len(target.time)
        and len(target.quality) == len(target.time)
        and len(target.cadence_number) == len(target.time)
        and len(target.corner) == len(target.time)
        and len(target.ephemeris) == len(target.time)
        and len(target.target_mask) == len(target.time)
        and len(target.coords) == len(target.time)
        and len(target.time_original) == len(target.time)
        and len(target.timecorr) == len(target.time)
    )

    # Check the ephemeris and corner have expected shape
    assert np.shape(target.corner)[1] == 2
    assert np.shape(target.ephemeris)[1] == 2

    # Check conversion between time and time_original
    assert (
        (target.time - target.timecorr)
        == (target.time_original - target.timecorr_original)
    ).all()

    # Check magnitude of time correction derived by lkspacecraft
    # Maximum Earth to SS barycenter is 500sec, with an
    # additional offset to account for TESS.
    assert (np.abs(target.timecorr * 24 * 3600) < 510).all()

    # Check the reshaped flux data has expected shape
    target.reshape_data()
    assert np.shape(target.flux) == (len(target.time), *shape)
    assert np.shape(target.flux_err) == (len(target.time), *shape)

    # Check the background correction was applied correctly, including
    # nan fluxes (non-science pixels).
    target.background_correction()
    assert np.array_equal(target.corr_flux, target.flux - target.bg, equal_nan=True)
    assert np.array_equal(
        target.corr_flux_err,
        np.sqrt(target.flux_err**2 + target.bg_err**2),
        equal_nan=True,
    )
    assert np.shape(target.corr_flux) == (len(target.time), *shape)
    assert np.shape(target.corr_flux_err) == (len(target.time), *shape)


def test_bg_linear_model():
    """ """
    # Initialise MovingTPF for 1980 VR1 and get data.
    target = MovingTPF.from_name("1980 VR1", sector=1, camera=1, ccd=1)
    target.get_data()
    target.reshape_data()

    for method in ["all_time", "per_time"]:
        # Background correction using `all_time` SL correction
        bg, bg_err, sl, sl_err, linear, linear_err = target._bg_linear_model(
            sl_method=method
        )

        # Check the components have the expected shape
        assert np.shape(bg) == np.shape(target.flux)
        assert np.shape(bg_err) == np.shape(target.flux)
        assert np.shape(sl) == np.shape(target.flux)
        assert np.shape(sl_err) == np.shape(target.flux)
        assert np.shape(linear) == np.shape(target.flux)
        assert np.shape(linear_err) == np.shape(target.flux)

        # Check method is recorded correctly.
        assert target.sl_method == method

        # Check models are summer for global model.
        assert np.array_equal(bg, sl + linear)

    # Check SL and LM quality masks:
    target.background_correction(sl_method="per_time", ncomponents=8000)
    assert np.shape(target.sl_nan_mask) == np.shape(target.time)
    assert np.array_equal(target.sl_nan_mask, np.ones_like(target.time, dtype=bool))
    assert np.shape(target.lm_nan_mask) == np.shape(target.all_flux)


def test_create_threshold_aperture():
    """
    Test threshold mask method that creates an aperture mask at each frame
    of flux pixels > threshold * STD.
    We test that the return mask has the same shape as `self.flux` and
    that expected median number of pixels in the mask for the test asteroid
    is a fixed numer (7).
    """
    # Make TPF for asteroid 1998 YT6
    target = MovingTPF.from_name("1998 YT6", sector=6)
    target.get_data(shape=(11, 11))
    target.reshape_data()
    target.background_correction(method="rolling")

    aperture_mask = target._create_threshold_aperture(
        threshold=3.0, reference_pixel="center"
    )

    assert aperture_mask.shape == target.flux.shape
    assert (
        np.median(aperture_mask.reshape((target.time.shape[0], -1)).sum(axis=1)) == 7.0
    )


def test_create_prf_aperture(caplog):
    """
    Test the PRF aperture. When running `_create_prf_aperture()`, it internally calls
    `_create_target_prf_model()` first. These tests check the properties of both the model
    and the aperture, including their shape and expected values.
    """

    # Make TPF for asteroid 1998 YT6
    target = MovingTPF.from_name("1998 YT6", sector=6)
    target.get_data(shape=(11, 11))

    # Make aperture from PRF model
    with caplog.at_level(logging.WARNING):
        aperture_mask = target._create_prf_aperture()

    # Check shape of the PRF model and aperture
    assert target.prf_model.shape == (len(target.time), *target.shape)
    assert aperture_mask.shape == (len(target.time), *target.shape)

    # Check that each frame of the PRF model sums to one
    # Note: rounding errors result in some values being very close, but not equal,
    # to one. Therefore, round to 10dp before asserting sum is one.
    assert all(
        [
            round(np.sum(target.prf_model[i]), 10) == 1
            for i in range(len(target.prf_model))
        ]
    )

    # Check that the last frame of the PRF model contained NaNs.
    # Previous testing has revealed this should be the case.
    assert "The PRF model contained nans in the last frame" in caplog.text

    # Check median number of pixels in the aperture across all frames equals 15.0.
    # Previous testing has revealed this should be the case.
    assert (
        np.median(aperture_mask.reshape((target.time.shape[0], -1)).sum(axis=1)) == 15.0
    )


def test_create_ellipse_aperture():
    """
    Test ellipse aperture. The ellipse is calculated by computing the first- and second-order
    moments from the flux image. The method returns a mask with pixels inside the
    ellipse, x/y centroid and the ellipse parameters (semi-major/minor axis and rotation angle).
    We test for array integrity and expected returned values.
    """
    # Make TPF for asteroid 1998 YT6
    target = MovingTPF.from_name("1998 YT6", sector=6)
    target.get_data(shape=(11, 11))
    target.reshape_data()
    target.background_correction(method="rolling")

    ellip_mask, params = target._create_ellipse_aperture(return_params=True)

    # aperture mask
    # for the test source the median number of pixels across all frames is 10
    assert ellip_mask.shape == target.flux.shape
    assert np.median(ellip_mask.reshape((target.time.shape[0], -1)).sum(axis=1)) == 10
    # ellipse parameters
    # for the test source the centroid values should be within 0.2 pixels
    # from the asteroid ephemeris
    assert params.shape == (target.time.shape[0], 5)
    assert np.isfinite(params).all()
    assert np.mean(params[:, 0] - target.ephemeris[:, 1]) < 0.2
    assert np.mean(params[:, 1] - target.ephemeris[:, 0]) < 0.2
    # angle is measure from the semi-major axis with + or - values
    assert ((params[:, 4] >= -180) & (params[:, 4] <= 180)).all()


def test_pixel_quality():
    """
    Test 3D pixel quality mask. Checks that the shape of the mask is the same as self.flux
    and that the values are as expected from a manual inspection.
    """

    # Create artificial track for a region that includes a saturated star, straps and non-science pixels.
    # Note: the track must have length > 1, but we use two timestamps very close together so resulting
    # moving TPF only has one frame.
    time = np.linspace(2458328.50, 2458328.52, 2) - 2457000
    track = pd.DataFrame(
        {
            "time": time,
            "sector": np.full(len(time), 1),
            "camera": np.full(len(time), 2),
            "ccd": np.full(len(time), 3),
            "column": np.linspace(1774, 1775, len(time)),
            "row": np.linspace(2020, 2021, len(time)),
        }
    )
    shape = (70, 70)
    target = MovingTPF("simulated_track", track)
    target.get_data(shape=shape)
    target.reshape_data()
    target.background_correction(method="rolling")
    target.create_pixel_quality(sat_buffer_rad=1)

    # Check that pixel quality mask has expected shape.
    assert np.shape(target.pixel_quality) == (len(target.time), *shape)

    # Check that pixel quality mask has expected values.
    # The expected values were determined manually using the first frame of the TPF and the strap table.
    pixels = np.zeros_like(target.flux[0])

    # Flag for non-science pixels
    pixels[63:, :] += 1

    # Flag for straps
    pixels[:, 11] += 2
    pixels[:, 12] += 2
    pixels[:, 15] += 2
    pixels[:, 16] += 2
    pixels[:, 19] += 2
    pixels[:, 20] += 2

    # Flag for saturation
    pixels[31:41, 33] += 4
    pixels[24:45, 34] += 4
    pixels[33:38, 35] += 4

    # Flag for saturation buffer
    pixels[31:41, 32] += 8
    pixels[41:45, 33] += 8
    pixels[24:31, 33] += 8
    pixels[23, 34] += 8
    pixels[45, 34] += 8
    pixels[24:33, 35] += 8
    pixels[38:45, 35] += 8
    pixels[33:38, 36] += 8

    assert np.array_equal(pixels, target.pixel_quality[0])


def test_make_tpf():
    """
    Check that make_tpf() correctly saves the TPF, that the file has the expected attributes
    and that it can be opened with lightkurve.
    """

    # Make TPF for asteroid 1998 YT6
    target = MovingTPF.from_name("1998 YT6", sector=6)
    target.make_tpf(save=True, outdir="tests")

    # Check the file exists
    assert os.path.exists("tests/tess-1998YT6-s0006-1-1-shape11x11-moving_tp.fits")

    # Open the file with astropy and check attributes
    with fits.open("tests/tess-1998YT6-s0006-1-1-shape11x11-moving_tp.fits") as hdul:
        assert "BAD_BITS" not in hdul[0].header.keys()
        assert hdul[0].header["PROCVER"].strip() == __version__
        assert hdul[0].header["AP_TYPE"].strip() == "prf"
        assert hdul[0].header["BG_CORR"].strip() == "linear_model"
        assert hdul[0].header["SL_CORR"].strip() == "all_time"
        assert hdul[0].header["VMAG"] > 0
        assert "SPOCDATE" in hdul[0].header.keys()
        assert "TIME" in hdul[1].columns.names
        assert "TIMECORR" in hdul[1].columns.names
        assert "APERTURE" in hdul[3].columns.names
        assert "PIXEL_QUALITY" in hdul[3].columns.names
        assert "CORNER1" in hdul[3].columns.names
        assert "CORNER2" in hdul[3].columns.names
        assert "RA" in hdul[3].columns.names
        assert "DEC" in hdul[3].columns.names
        assert "ORIGINAL_TIME" in hdul[3].columns.names
        assert "ORIGINAL_TIMECORR" in hdul[3].columns.names
        assert len(hdul[3].data["APERTURE"]) == len(target.time)
        assert np.allclose(target.corr_flux, hdul[1].data["FLUX"], rtol=1e-07)

        # Check the UTC to TDB conversion has been applied.
        assert (hdul[1].data["TIME"] != hdul[3].data["ORIGINAL_TIME"]).all()

    # Check the file can be opened with lightkurve
    tpf = lk.read(
        "tests/tess-1998YT6-s0006-1-1-shape11x11-moving_tp.fits", quality_bitmask="none"
    )
    assert isinstance(tpf, lk.targetpixelfile.TessTargetPixelFile)
    assert hasattr(tpf, "pipeline_mask")
    assert len(tpf.time) == len(target.time)
    assert np.allclose(target.corr_flux, tpf.flux.value, rtol=1e-07)

    # Delete the file
    os.remove("tests/tess-1998YT6-s0006-1-1-shape11x11-moving_tp.fits")


def test_to_lightcurve():
    """
    Test the to_lightcurve() function with the method `aperture`.  This internally calls
    _aperture_photometry() and _create_lc_quality(). The tests check the expected length
    of the lightcurve, the expected value of the centroid and the expected values of the
    quality mask.
    """

    # Make TPF for asteroid 1998 YT6.
    target = MovingTPF.from_name("1998 YT6", sector=6)
    target.make_tpf()

    # Use aperture photometry to extract lightcurve from TPF.
    target.to_lightcurve(method="aperture")

    # Check the lightcurve has the same length as target.time
    assert len(target.lc["aperture"]["time"]) == len(target.time)
    assert len(target.lc["aperture"]["flux"]) == len(target.time)
    assert len(target.lc["aperture"]["quality"]) == len(target.time)
    assert len(target.lc["aperture"]["flux_fraction"]) == len(target.time)

    # Check the average centroid is within 1/2 a pixel of the center of the TPF.
    assert (
        (np.nanmean(target.lc["aperture"]["row_cen"] - target.corner[:, 0]) > 4.5)
        & (np.nanmean(target.lc["aperture"]["row_cen"] - target.corner[:, 0]) < 5.5)
    ).all()
    assert (
        (np.nanmean(target.lc["aperture"]["col_cen"] - target.corner[:, 1]) > 4.5)
        & (np.nanmean(target.lc["aperture"]["col_cen"] - target.corner[:, 1]) < 5.5)
    ).all()

    # Check the pixel quality has been correctly accounted for in quality mask.
    for t in range(len(target.time)):
        if target.pixel_quality[t][target.aperture_mask[t]].any() != 0:
            assert target.lc["aperture"]["quality"][t] > 0

    # Check flux fraction has expected values
    assert (target.lc["aperture"]["flux_fraction"] <= 1).all()
    assert np.isfinite(target.lc["aperture"]["flux_fraction"]).all()


def test_make_lc():
    """
    Check that make_lc() correctly saves the LCF, that the file has the expected attributes
    and that it can be opened with lightkurve.
    """

    # Make TPF for asteroid 1998 YT6
    target = MovingTPF.from_name("1998 YT6", sector=6)
    target.make_tpf(bg_method="rolling")
    target.make_lc(save=True, outdir="tests")

    # Check the file exists
    assert os.path.exists("tests/tess-1998YT6-s0006-1-1-shape11x11_lc.fits")

    # Open the file with astropy and check some attributes
    with fits.open("tests/tess-1998YT6-s0006-1-1-shape11x11_lc.fits") as hdul:
        # Check primary header
        assert "BAD_BITS" in hdul[0].header.keys()
        assert hdul[0].header["PROCVER"].strip() == __version__
        assert hdul[0].header["AP_TYPE"].strip() == "prf"
        assert hdul[0].header["BG_CORR"].strip() == "rolling"
        assert hdul[0].header["SL_CORR"].strip() == "n/a"
        assert hdul[0].header["VMAG"] > 0
        assert "SPOCDATE" in hdul[0].header.keys()

        # Check columns in lightcurve HDU
        assert "TIME" in hdul[1].columns.names
        assert "TIMECORR" in hdul[1].columns.names
        assert "ORIGINAL_TIME" in hdul[1].columns.names
        assert "ORIGINAL_TIMECORR" in hdul[1].columns.names
        assert "FLUX" in hdul[1].columns.names
        assert "FLUX_ERR" in hdul[1].columns.names
        assert np.isnan(hdul[1].data["PSF_FLUX"]).all()
        assert "MOM_CENTR1" in hdul[1].columns.names
        assert "RA" in hdul[1].columns.names
        assert "EPHEM1" in hdul[1].columns.names

        # Check the UTC to TDB conversion has been applied.
        assert (hdul[1].data["TIME"] != hdul[1].data["ORIGINAL_TIME"]).all()

    # Check the file can be opened with lightkurve
    lc = lk.io.tess.read_tess_lightcurve(
        "tests/tess-1998YT6-s0006-1-1-shape11x11_lc.fits",
        quality_bitmask="none",
    )
    assert isinstance(lc, lk.lightcurve.TessLightCurve)
    assert len(lc.time) == len(target.time)
    assert np.array_equal(target.lc["aperture"]["flux"], lc.flux.value)
    assert np.array_equal(target.lc["aperture"]["flux_err"], lc.flux_err.value)

    # Delete the file
    os.remove("tests/tess-1998YT6-s0006-1-1-shape11x11_lc.fits")
