"""
Utility functions
"""

import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import simple_norm
from matplotlib import animation, patches


def inside_ellipse(x, y, cxx, cyy, cxy, x0=0, y0=0, R=1):
    """
    Returns a boolean mask indicating positions inside a specified ellipse.
    The ellipse is defined by its center, the radius, and the quadratic coefficients
    (cxx, cyy, cxy).
    Pixels with distance <= R^2 from the center (x0, y0) are considered inside the ellipse.

    Parameters
    ----------
    x : array_like
        The x-coordinates of the points to be tested.
    y : array_like
        The y-coordinates of the points to be tested.
    cxx : float
        The coefficient for the x^2 term in the ellipse equation.
    cyy : float
        The coefficient for the y^2 term in the ellipse equation.
    cxy : float
        The coefficient for the xy term in the ellipse equation.
    x0 : float, optional
        The x-coordinate of the center of the ellipse (default is 0).
    y0 : float, optional
        The y-coordinate of the center of the ellipse (default is 0).
    R : float, optional
        The radius of the ellipse (default is 1).

    Returns
    -------
    mask : ndarray
        A boolean array of the same shape as x and y, where True indicates that the
        corresponding point (x, y) is inside the ellipse defined by the provided parameters.
    """
    return cxx * (x - x0) ** 2 + cyy * (y - y0) ** 2 + cxy * (x - x0) * (y - y0) <= R**2


def compute_moments(flux, mask=None):
    """
    Computes first and second order moments of a 2d distribution over time
    using a coordinate grid with the same shape as `flux` (nt, nrows, ncols).
    First order moments (X,Y) are the centroid positions. The X,Y centroids are in
    the range [0, 'ncols'), [0, 'nrows'), respectively i.e. they are zero-indexed.
    Second order moments (X2, Y2, XY) represent the spatial spread of the distribution.

    Parameters
    ----------
    flux: ndarray
        3D array with flux values as (nt, nrows, ncols).
    mask: ndarray, boolean
        Mask to select pixels used for computing moments. Shape could
        be 3D (nt, nrows, ncols) or 2D (nrows, ncols). If a 2D mask is given,
        it is used for all frames `nt`.

    Returns
    -------
    X, Y, X2, Y2, XY: ndarrays
        First (X, Y) and second (X2, Y2, XY) order moments. Each array has
        shape (nt).
    """
    # check if mask is None
    if mask is None:
        mask = np.ones_like(flux).astype(bool)
    # reshape 2D mask into 3D mask if necessary
    if len(mask.shape) == 2:
        mask = np.repeat([mask], flux.shape[0], axis=0)

    X = np.zeros(flux.shape[0], dtype=float)
    Y = np.zeros(flux.shape[0], dtype=float)
    X2 = np.zeros(flux.shape[0], dtype=float)
    Y2 = np.zeros(flux.shape[0], dtype=float)
    XY = np.zeros(flux.shape[0], dtype=float)

    # compute moments for each frame
    for nt in range(flux.shape[0]):
        # skip frame if no pixels are used
        if mask[nt].sum() == 0:
            continue
        # dummy pixel grid
        row, col = np.mgrid[0 : flux.shape[1], 0 : flux.shape[2]]

        # first order moments
        Y[nt] = np.average(row[mask[nt]], weights=flux[nt, mask[nt]])
        X[nt] = np.average(col[mask[nt]], weights=flux[nt, mask[nt]])
        # second order moments
        Y2[nt] = np.average(row[mask[nt]] ** 2, weights=flux[nt, mask[nt]]) - Y[nt] ** 2
        X2[nt] = np.average(col[mask[nt]] ** 2, weights=flux[nt, mask[nt]]) - X[nt] ** 2
        XY[nt] = (
            np.average(row[mask[nt]] * col[mask[nt]], weights=flux[nt, mask[nt]])
            - X[nt] * Y[nt]
        )
    return X, Y, X2, Y2, XY


def plot_img_aperture(
    img,
    aperture_mask=None,
    cbar=True,
    ax=None,
    corner=[0, 0],
    xy=None,
    title=None,
    vmin=None,
    vmax=None,
    cnorm=None,
):
    """
    Plots an image with an optional aperture mask overlay.

    This function displays an image, optionally overlaying an aperture mask, and
    provides several customization options such as color scaling, title, and axis control.

    Parameters:
    -----------
    img : 2D array
        The image data to be plotted, typically a 2D array or matrix representing pixel values.

    aperture_mask : 2D array, optional
        A binary mask (same shape as `img`) indicating the aperture region to be overlaid on the image.

    cbar : bool, optional, default=True
        Whether to display a color bar alongside the plot.

    ax : matplotlib.axes.Axes, optional
        The axes object where the plot will be drawn. If not provided, the current axes will be used.

    corner : list of two ints, optional, default=[0, 0]
        The (row, column) coordinates of the lower left corner of the image.

    xy : tuple of float, optional, default=None
        Plot the position x and y in the figure, used for showing the source position.

    title : str, optional, default=None
        Title of the plot. If None, no title will be shown.

    vmin : float, optional, default=None
        Minimum value for color scale. If None, the minimum value in the image is used.

    vmax : float, optional, default=None
        Maximum value for color scale. If None, the maximum value in the image is used.

    cnorm : optional, default=None
        Color normalization object (e.g. astropy.visualization.simple_norm). If provided,
        then `vmax` and `vmin` are not used.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots()
        fig.suptitle("Asteroid Tracks")

    extent = (
        corner[1] - 0.5,
        corner[1] + img.shape[1] - 0.5,
        corner[0] - 0.5,
        corner[0] + img.shape[0] - 0.5,
    )

    if vmin is None and vmax is None and cnorm is None:
        vmin, vmax = np.percentile(img.ravel(), [3, 97])

    im = ax.imshow(
        img,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        norm=cnorm,
        rasterized=True,
        origin="lower",
        extent=extent,
    )
    if cbar:
        plt.colorbar(im, location="right", shrink=0.8, label="Flux [-e/s]")
    if xy is not None:
        ax.scatter(xy[1], xy[0], marker="o", c="red", alpha=1, s=15)

    ax.set_aspect("equal", "box")
    ax.set_title(title)

    if aperture_mask is not None:
        row, col = np.mgrid[
            corner[0] : corner[0] + img.shape[0], corner[1] : corner[1] + img.shape[1]
        ]
        for i, pi in enumerate(row[:, 0]):
            for j, pj in enumerate(col[0, :]):
                if aperture_mask[i, j]:
                    # print("here")
                    rect = patches.Rectangle(
                        xy=(pj - 0.5, pi - 0.5),
                        width=1,
                        height=1,
                        color="tab:red",
                        fill=False,
                        # hatch="//",
                        alpha=0.8,
                    )
                    ax.add_patch(rect)

    ax.set_xlabel("Pixel Column")
    ax.set_ylabel("Pixel Row")

    return ax


def animate_cube(
    cube,
    aperture_mask=None,
    corner=[0, 0],
    track=None,
    cadenceno=None,
    time=None,
    interval=200,
    repeat_delay=1000,
    cnorm=False,
    suptitle="",
):
    """
    Creates an animated visualization of a 3D image cube, with an optional aperture mask and other customization options.

    This function animates the slices of a 3D image cube, optionally overlaying an aperture mask,
    and provides controls for animation speed, title, and tracking information.

    Parameters:
    -----------
    cube : 3D array
        A 3D array representing the image cube (e.g., a stack of 2D images over time).

    aperture_mask : 2D or 3D array, optional
        A binary mask (same shape or a 2D slice of `cube`) to overlay on each frame of the animation.
        If a 2D mask is passed, it will be repeated for all times.

    corner : list of two ints, optional, default=[0, 0]
        The (row, column) coordinates of the lower left corner of the image.

    track : list or tuple, optional, default=None
        A list or tuple of object positions to be displayed on the plot.
        For proper display of object position, if corner is [0, 0] then track needs to be relative to corner.
        If corner is provided, track needs to be absolute.
        If None, no tracking information is shown.

    cadenceno : int, optional, default=None
        The cadence number of the frames, used for information display.

    time : array-like, optional, default=None
        Array of time values corresponding to the slices in the cube.

    interval : int, optional, default=200
        The time interval (in milliseconds) between each frame of the animation.

    repeat_delay : int, optional, default=1000
        The time delay (in milliseconds) before the animation restarts once it finishes.

    cnorm : optional, default=False
        Weather to use asinh color normalization (from astropy.visualization.simple_norm).

    suptitle : str, optional, default=""
        A string to be used as the super title of the animation.
        It can be used to provide additional context or information about the animated data.

    Returns:
    --------
    ani : matplotlib.animation.FuncAnimation
        The animation object that can be displayed or saved.
    """

    fig, ax = plt.subplots()
    fig.suptitle(suptitle)

    if aperture_mask is None:
        aperture_mask = np.repeat([None], len(cube), axis=0)
    else:
        if aperture_mask.shape == cube.shape[1:]:
            aperture_mask = np.repeat([aperture_mask], len(cube), axis=0)

    if track is None:
        track = np.repeat([None], len(cube), axis=0)
    if cadenceno is None:
        cadenceno = np.repeat([None], len(cube), axis=0)
    if time is None:
        time = np.repeat([None], len(cube), axis=0)
    if cnorm:
        norm = simple_norm(cube.ravel(), "asinh", percent=98)
    else:
        vlo, lo, mid, hi, vhi = np.nanpercentile(cube, [0.2, 1, 50, 99, 99.8])

    if len(corner) == 2:
        corner = np.repeat([corner], len(cube), axis=0)

    nt = 0
    ax = plot_img_aperture(
        cube[nt],
        aperture_mask=aperture_mask[nt],
        cbar=True,
        ax=ax,
        corner=corner[nt],
        xy=track[nt],
        title=f"CAD {cadenceno[nt]} | BTJD {time[nt]:.4f}",
        vmin=lo if not cnorm else None,
        vmax=hi if not cnorm else None,
        cnorm=norm if cnorm else None,
    )

    def animate(nt):
        ax.clear()
        _ = plot_img_aperture(
            cube[nt],
            aperture_mask=aperture_mask[nt],
            cbar=False,
            ax=ax,
            corner=corner[nt],
            xy=track[nt],
            title=f"CAD {cadenceno[nt]} | BTJD {time[nt]:.4f}",
            vmin=lo if not cnorm else None,
            vmax=hi if not cnorm else None,
            cnorm=norm if cnorm else None,
        )

        return ()

    plt.close(ax.figure)

    # Create the animation
    ani = animation.FuncAnimation(
        fig,
        animate,
        frames=len(cube),
        interval=interval,
        blit=True,
        repeat_delay=repeat_delay,
        repeat=True,
    )

    return ani
