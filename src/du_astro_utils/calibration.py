#!/usr/bin/env python3
"""
Module to run calibration steps of an astronomy FITS image.

Created on Tue Feb 20 09:09:49 2024

@author: Joseph Chevalier
"""

# pylint : skip-file

import os
from datetime import datetime

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from scipy.ndimage import median_filter


def get_infos_from_image(fits_image_path):
    """
    Get useful information from image's header.

    Parameters
    ----------
    fits_image_path : str or path
        Path to the FITS image.

    Returns
    -------
    acq_datetime : datetime object
        Date and time of acquisition.
    telescope : str
        Telescope used for acquisition.
    acq_cam : str
        Camera used for acquisition.
    acq_filter : str
        Filter used for acquisition.
    expos_time : float
        Exposure duration in seconds.
    size_x : int
        Size of image in pixels along axis x.
    size_y : int
        Size of image in pixels along axis y.

    """
    hdu = fits.open(fits_image_path)
    hdu = hdu[0]
    # Place value of header keyword in variables
    telescope = hdu.header.get("TELESCOP").strip()
    acq_type = hdu.header.get("ACQTYPE").strip()
    epoch = (hdu.header.get("DATE-OBS").strip()).split(".")[0] + "+00:00"
    expos_time = hdu.header.get("EXPTIME")
    size_x = hdu.header.get("NAXIS1")
    size_y = hdu.header.get("NAXIS2")
    acq_cam = hdu.header.get("CAMMODEL").strip()
    acq_filter = hdu.header.get("INSTFILT").strip()

    # Print some informations regarding the frame
    print(f"{acq_type} ({size_x}x{size_y}) taken in band {acq_filter} with {acq_cam} on {telescope} on {epoch} ({expos_time}s exposure).")

    # Process useful information
    acq_datetime = datetime.fromisoformat(epoch)
    return acq_datetime, telescope, acq_cam, acq_filter, expos_time, size_x, size_y


def check_obs_night(date, date_ref, max_days=7):
    """
    Check if a calibration frame is compatible with a science frame, date-wise.

    Parameters
    ----------
    date : datetime object
        Calibration frame's date.
    date_ref : datetime object
        Science frame's date.
    max_days : int, optional
        Number of days allowed between frames. The default is 7.

    Returns
    -------
    bool
        `True` if frames were taken less than `max_days` apart, otherwise `False`.

    """
    time_delta = date - date_ref
    print(time_delta)
    return abs(time_delta.days) <= max_days


def load_bias_frames(path_to_bias_dir, aq_date, aq_cam, size_x, size_y, override_date_check=False, max_days=7):
    """
    Find and load BIAS frames compatible with a science image.

    Parameters
    ----------
    path_to_bias_dir : str or path
        Path to the directory containing BIAS frames.
    aq_date : datetime object
        Calibration frame's date.
    aq_cam : str
        Camera used for acquisition.
    size_x : int
        Size of image in pixels along axis x.
    size_y : int
        Size of image in pixels along axis y.
    override_date_check : bool, optional
        Whether to allow calibration frames that are not date-wise compatible with the science frame.\
            The default is False.
    max_days : int, optional
        If `override_date_check` is `False`, the maximum number of days allowed between science image and calibration frame.\
            The default is 7.

    Returns
    -------
    bias_frames_list : list
        List of paths to appropriate BIAS frames.

    """
    _path = os.path.abspath(path_to_bias_dir)
    bias_frames_list = []
    for _file in os.listdir(_path):
        bn, ext = os.path.splitext(os.path.basename(_file))
        _fpath = os.path.join(_path, _file)
        if os.path.isfile(_fpath) and "fits" in ext.lower():
            _dat, _scope, _cam, _filt, _dur, _x, _y = get_infos_from_image(_fpath)
            if (check_obs_night(_dat, aq_date, max_days) or override_date_check) and _x == size_x and _y == size_y:
                bias_frames_list.append(_fpath)
    return bias_frames_list


def load_dark_frames(path_to_darks_dir, aq_date, aq_cam, expos_time, size_x, size_y, override_date_check=False, max_days=7):
    """
    Find and load DARK frames compatible with a science image.

    Parameters
    ----------
    path_to_darks_dir : str or path
        Path to the directory containing DARK frames.
    aq_date : datetime object
        Calibration frame's date.
    aq_cam : str
        Camera used for acquisition.
    expos_time : float
        Exposure duration in seconds.
    size_x : int
        Size of image in pixels along axis x.
    size_y : int
        Size of image in pixels along axis y.
    override_date_check : bool, optional
        Whether to allow calibration frames that are not date-wise compatible with the science frame.\
            The default is False.
    max_days : int, optional
        If `override_date_check` is `False`, the maximum number of days allowed between science image and calibration frame.\
            The default is 7.

    Returns
    -------
    dark_frames_list : list
        List of paths to appropriate DARK frames.

    """
    _path = os.path.abspath(path_to_darks_dir)
    dark_frames_list = []
    for _file in os.listdir(_path):
        bn, ext = os.path.splitext(os.path.basename(_file))
        _fpath = os.path.join(_path, _file)
        if os.path.isfile(_fpath) and "fits" in ext.lower():
            _dat, _scope, _cam, _filt, _dur, _x, _y = get_infos_from_image(_fpath)
            if (check_obs_night(_dat, aq_date, max_days) or override_date_check) and _x == size_x and _y == size_y and _dur == expos_time:
                dark_frames_list.append(_fpath)
    return dark_frames_list


def load_flat_frames(path_to_flats_dir, aq_date, aq_cam, aq_filter, size_x, size_y, override_date_check=False, max_days=7):
    """
    Find and load FLAT frames compatible with a science image.

    Parameters
    ----------
    path_to_flats_dir : str or path
        Path to the directory containing FLAT frames.
    aq_date : datetime object
        Calibration frame's date.
    aq_cam : str
        Camera used for acquisition.
    aq_filter : str
        Filter used for acquisition.
    size_x : int
        Size of image in pixels along axis x.
    size_y : int
        Size of image in pixels along axis y.
    override_date_check : bool, optional
        Whether to allow calibration frames that are not date-wise compatible with the science frame.\
            The default is False.
    max_days : int, optional
        If `override_date_check` is `False`, the maximum number of days allowed between science image and calibration frame.\
            The default is 7.

    Returns
    -------
    flat_frames_list : list
        List of paths to appropriate FLAT frames.

    """
    _path = os.path.abspath(path_to_flats_dir)
    flat_frames_list = []
    for _file in os.listdir(_path):
        bn, ext = os.path.splitext(os.path.basename(_file))
        _fpath = os.path.join(_path, _file)
        if os.path.isfile(_fpath) and "fits" in ext.lower():
            _dat, _scope, _cam, _filt, _dur, _x, _y = get_infos_from_image(_fpath)
            if (check_obs_night(_dat, aq_date, max_days) or override_date_check) and _x == size_x and _y == size_y and _filt == aq_filter:
                flat_frames_list.append(_fpath)
    return flat_frames_list


def master_bias(bias_frames_list):
    """
    Computes and returns the MASTER BIAS frame.

    Parameters
    ----------
    bias_frames_list : list
        List of paths to appropriate BIAS frames.

    Returns
    -------
    dict
        Dictionary containing the path to the written master bias and the corresponding image data.

    """
    # Get parent directory and hdu info
    fits_open_hdu = fits.open(bias_frames_list[0])[0]
    bias_dir = os.path.dirname(bias_frames_list[0])
    _dat, _scope, camera, _filt, _dur, _x, _y = get_infos_from_image(bias_frames_list[0])

    # Load frames
    bias_array = np.empty((len(bias_frames_list), *fits_open_hdu.data.shape))
    for it, _file in enumerate(bias_frames_list):
        _open_hdu = fits.open(_file)[0]
        try:
            bias_array[it, :, :] = _open_hdu.data
        except ValueError:
            bias_array[it, :, :] = np.transpose(_open_hdu.data)

    # Master bias = median of the bias images
    master_bias_as_array = np.median(bias_array, axis=0)

    # Write appropriate FITS files
    date = _dat.date().isoformat()
    fits_open_hdu.data = master_bias_as_array
    mb_dir = os.path.join(os.path.abspath(bias_dir), "MASTER_BIAS")
    if not os.path.isdir(mb_dir):
        os.makedirs(mb_dir)
    write_path = os.path.join(mb_dir, f"master_bias_{date}_{camera}.fits")
    fits_open_hdu.writeto(write_path, overwrite=True)
    print(f"Master BIAS written to {write_path}.")

    return {"path": write_path, "data": master_bias_as_array}


def master_dark(dark_frames_list, use_bias=False, master_bias=""):
    """
    Computes and returns the MASTER DARK frame.

    Parameters
    ----------
    dark_frames_list : list
        List of paths to appropriate DARK frames.
    use_bias : bool, optional
        Whether to use the master bias in calibration. The default is False.
    master_bias : str or path, optional
        if `use_bias` : the location of the master bias fits file. The default is "".

    Returns
    -------
    dict
        Path and pixel values to the master dark.
    dict
        Path and map of the bad (hot) pixels.

    """
    # Get parent directory and hdu info
    fits_open_hdu = fits.open(dark_frames_list[0])[0]
    darks_dir = os.path.dirname(dark_frames_list[0])
    _dat, _scope, camera, _filt, exposure, _x, _y = get_infos_from_image(dark_frames_list[0])

    mb_data = np.zeros_like(fits_open_hdu.data)
    if use_bias:
        # Get parent directory and hdu info
        mb_hdu = fits.open(master_bias)[0]
        mb_data += mb_hdu.data

    # Load frames
    darks_array = np.empty((len(dark_frames_list), *fits_open_hdu.data.shape))
    for it, _file in enumerate(dark_frames_list):
        _open_hdu = fits.open(_file)[0]
        try:
            darks_array[it, :, :] = _open_hdu.data
        except ValueError:
            darks_array[it, :, :] = np.transpose(_open_hdu.data)

    # Master bias = median of the bias images
    master_dark_as_array = np.median(darks_array, axis=0) - mb_data

    # Sigma-clipped statistics to detect hot pixels
    bkg_mean, bkg_median, bkg_sigma = sigma_clipped_stats(master_dark_as_array, sigma=3.0)

    # Threshold for hot pixel detection
    threshold = 5

    # Hot pixel are above a given value
    hot_pix_loc = np.where(master_dark_as_array > bkg_median + threshold * bkg_sigma)
    hot_pixels_map = np.zeros_like(master_dark_as_array, dtype=int)
    hot_pixels_map[hot_pix_loc] = 1

    # Some statitics
    print(f"Number of pixels in the dark: {len(master_dark_as_array.flatten()):8d}")
    print(f"Number of hot pixels       : {len(master_dark_as_array[hot_pix_loc]):8d}")
    print(f"Fraction of hot pixels (%) : {100*len(master_dark_as_array[hot_pix_loc])/len(master_dark_as_array.flatten()):.2f}")

    # Cosmetic smoothing
    smoothed = median_filter(master_dark_as_array, size=(5, 5))
    master_dark_as_array[hot_pix_loc] = smoothed[hot_pix_loc]

    # Write appropriate FITS files
    date = _dat.date().isoformat()
    fits_open_hdu.data = master_dark_as_array
    md_dir = os.path.join(os.path.abspath(darks_dir), "MASTER_DARKS")
    if not os.path.isdir(md_dir):
        os.makedirs(md_dir)
    md_write_path = os.path.join(md_dir, f"master_dark_{date}_{camera}_{exposure:.3f}.fits")
    fits_open_hdu.writeto(md_write_path, overwrite=True)
    print(f"Master DARK written to {md_write_path}.")

    fits_open_hdu.data = hot_pixels_map
    hp_write_path = os.path.join(md_dir, f"bad_pixels_hot_{date}_{camera}.fits")
    fits_open_hdu.writeto(hp_write_path, overwrite=True)
    print(f"Hot pixels map written to {hp_write_path}.")

    return {"path": md_write_path, "data": master_dark_as_array}, {
        "path": hp_write_path,
        "data": hot_pixels_map,
    }


def master_flat(flat_frames_list, master_dark_path):
    """
    Computes and returns the MASTER FLAT frame.

    Parameters
    ----------
    flat_frames_list : list
        List of paths to appropriate FLAT frames.
    master_dark_path : str or path
        Path to the appropriate MASTER DARK frame.

    Returns
    -------
    dict
        Path and pixel values to the master flat.
    dict
        Path and map of the bad (dead) pixels.

    """
    # Get parent directory and hdu info
    fits_open_hdu = fits.open(flat_frames_list[0])[0]
    flats_dir = os.path.dirname(flat_frames_list[0])
    _dat, _scope, camera, band, exposure, _x, _y = get_infos_from_image(flat_frames_list[0])

    # Load master dark to compute exposure ratio
    md_hdu = fits.open(master_dark_path)[0]

    # Load frames
    flats_array = np.empty((len(flat_frames_list), *fits_open_hdu.data.shape))
    for it, _file in enumerate(flat_frames_list):
        flat_hdu = fits.open(_file)[0]
        # Remove master dark, rescaled as necessary to account for exposure variations
        exp_ratio = flat_hdu.header.get("EXPTIME") / md_hdu.header.get("EXPTIME")

        try:
            scaled_flat = flat_hdu.data - exp_ratio * md_hdu.data
        except ValueError:
            scaled_flat = np.transpose(flat_hdu.data) - exp_ratio * md_hdu.data
        flats_array[it, :, :] = scaled_flat / np.mean(scaled_flat)

    # Master bias = median of the bias images
    master_flat_as_array = np.median(flats_array, axis=0)
    master_flat_as_array /= np.mean(master_flat_as_array)

    # Sigma-clipped statistics to detect hot pixels
    bkg_mean, bkg_median, bkg_sigma = sigma_clipped_stats(master_flat_as_array, sigma=3.0)

    # Threshold for hot pixel detection
    threshold = 5

    # Hot pixel are above a given value
    dead_pix_loc = np.where(master_flat_as_array <= max(0.0, bkg_median - threshold * bkg_sigma))
    dead_pixels_map = np.zeros_like(master_flat_as_array, dtype=int)
    dead_pixels_map[dead_pix_loc] = 1

    # Some statitics
    print(f"Number of pixels in the flat: {len(master_flat_as_array.flatten()):8d}")
    print(f"Number of dead pixels       : {len(master_flat_as_array[dead_pix_loc]):8d}")
    print(f"Fraction of dead pixels (%) : {100*len(master_flat_as_array[dead_pix_loc])/len(master_flat_as_array.flatten()):.2f}")

    # Cosmetic smoothing
    smoothed = median_filter(master_flat_as_array, size=(5, 5))
    master_flat_as_array[dead_pix_loc] = smoothed[dead_pix_loc]

    # Write appropriate FITS files
    date = _dat.date().isoformat()
    fits_open_hdu.data = master_flat_as_array
    mf_dir = os.path.join(os.path.abspath(flats_dir), "MASTER_FLATS")
    if not os.path.isdir(mf_dir):
        os.makedirs(mf_dir)
    mf_write_path = os.path.join(mf_dir, f"master_flat_{date}_{camera}_{band}_{exposure:.3f}.fits")
    fits_open_hdu.writeto(mf_write_path, overwrite=True)
    print(f"Master FLAT written to {mf_write_path}.")

    fits_open_hdu.data = dead_pixels_map
    dp_write_path = os.path.join(mf_dir, f"bad_pixels_dead_{date}_{camera}.fits")
    fits_open_hdu.writeto(dp_write_path, overwrite=True)
    print(f"Dead pixels map written to {dp_write_path}.")

    return {"path": mf_write_path, "data": master_flat_as_array}, {
        "path": dp_write_path,
        "data": dead_pixels_map,
    }


def reduce_sci_image(fits_image, path_to_darks_dir, path_to_flats_dir, path_to_bias_dir="", use_bias=False):
    """
    Apply calibration frames to a science image to obtain a reduced science image to be used for analysis.
    If the master bias is used : $REDUCED = \frac{SCIENCE-mDARK-mBIAS}{mFLAT}$
    Else (default - the DARK ususally contains bias data) : $REDUCED = \frac{SCIENCE-mDARK}{mFLAT}$

    Parameters
    ----------
    fits_image : str or path
        Path to the FITS image to be reduced.
    path_to_darks_dir : str or path
        Path to the directory containing DARK frames.
    path_to_flats_dir : str or path
        Path to the directory containing FLAT frames.
    path_to_bias_dir : str or path, optional
        if `use_bias` : path to the directory containing BIAS frames. The default is "".
    use_bias : bool, optional
        Whether to use the master bias in calibration. The default is False.

    Returns
    -------
    dic_to_return : dict
        Dictionary containing the paths to the reduced science image and associated calibration frames,\
            and the reduced data (numpy array of pixel values).

    """
    # Get image directory, failename and extension
    sc_im_dir = os.path.abspath(os.path.dirname(fits_image))
    sc_im_name, sc_im_ext = os.path.splitext(os.path.basename(fits_image))

    # Get information from FITS header
    sc_date, sc_scope, sc_cam, sc_filter, sc_expos, sc_x, sc_y = get_infos_from_image(fits_image)

    # Ensure the date can be matched to an observation night D, i.e.. it is either :
    # - D and 12:00:00 <= HH:MM:SS <= 23:59:59
    # - D+1 and 00:00:00 <= HH:MM:SS <= 11:59:59
    # TBD

    # BIAS is included in DARK unless exposure time of DARK is not significant
    if use_bias:
        # Master bias
        # TBD: check if there is already one that works
        # path_to_bias_dir = os.path.join(sc_im_dir, '..', 'bias')
        bias_list = load_bias_frames(path_to_bias_dir, sc_date, sc_cam, sc_x, sc_y)
        MASTER_BIAS = master_bias(bias_list)

        # Master dark
        # TBD: check if there is already one that works
        # path_to_darks_dir = os.path.join(sc_im_dir, '..', 'darks')
        darks_list = load_dark_frames(path_to_darks_dir, sc_date, sc_cam, sc_expos, sc_x, sc_y)
        MASTER_DARK, HOT_PIXELS = master_dark(darks_list, use_bias=True, master_bias=MASTER_BIAS["path"])
        additive_corr = MASTER_DARK["data"] - MASTER_BIAS["data"]
    else:
        # Master dark
        # TBD: check if there is already one that works
        # path_to_darks_dir = os.path.join(sc_im_dir, '..', 'darks')
        darks_list = load_dark_frames(path_to_darks_dir, sc_date, sc_cam, sc_expos, sc_x, sc_y)
        MASTER_DARK, HOT_PIXELS = master_dark(darks_list)
        additive_corr = MASTER_DARK["data"]

    # Master flat
    # TBD: check if there is already one that works
    # path_to_flats_dir = os.path.join(sc_im_dir, '..', 'flats')
    flats_list = load_flat_frames(path_to_flats_dir, sc_date, sc_cam, sc_filter, sc_x, sc_y)
    MASTER_FLAT, DEAD_PIXELS = master_flat(flats_list, MASTER_DARK["path"])

    sc_hdu = fits.open(fits_image)[0]
    try:
        RED_SCIENCE = (sc_hdu.data - additive_corr) / MASTER_FLAT["data"]
    except ValueError:
        RED_SCIENCE = (sc_hdu.data - np.transpose(additive_corr)) / np.transpose(MASTER_FLAT["data"])

    # Clean bad pixels
    smoothed = median_filter(RED_SCIENCE, size=(5, 5))

    # Hot pixels
    try:
        hot_pixel = np.where(HOT_PIXELS["data"] == 1)
        RED_SCIENCE[hot_pixel] = smoothed[hot_pixel]
    except:
        print("Cannot clean hot pixels")

    # Dead pixels
    try:
        dead_pixel = np.where(DEAD_PIXELS["data"] == 1)
        RED_SCIENCE[dead_pixel] = smoothed[dead_pixel]
    except:
        print("Cannot clean dead pixels")

    # Write appropriate FITS files
    new_fn = f"{sc_im_name}_REDUCED{sc_im_ext}"
    red_hdu = sc_hdu.copy()
    red_hdu.data = RED_SCIENCE
    red_hdu.header["PROCTYPE"] = "RED     "
    red_hdu.header["FILENAME"] = new_fn
    red_hdu.header["CREATOR"] = "JOCHEVAL"
    red_hdu.header["MASTER_DARK"] = MASTER_DARK["path"]
    red_hdu.header["MASTER_FLAT"] = MASTER_FLAT["path"]
    red_hdu.header["HOT_PIXELS"] = HOT_PIXELS["path"]
    red_hdu.header["DEAD_PIXELS"] = DEAD_PIXELS["path"]
    if use_bias:
        red_hdu.header["MASTER_BIAS"] = MASTER_BIAS["path"]
    redim_dir = os.path.join(os.path.abspath(sc_im_dir), "REDUCED")
    if not os.path.isdir(redim_dir):
        os.makedirs(redim_dir)
    write_path = os.path.join(redim_dir, new_fn)
    red_hdu.writeto(write_path, overwrite=True)
    print(f"Calibrated image written to {write_path}.")

    dic_to_return = {
        "path": write_path,
        "data": RED_SCIENCE,
        "MASTER DARK": MASTER_DARK,
        "MASTER FLAT": MASTER_FLAT,
        "HOT PIXELS": HOT_PIXELS,
        "DEAD PIXELS": DEAD_PIXELS,
    }

    if use_bias:
        dic_to_return.update({"MASTER BIAS": MASTER_BIAS})

    return dic_to_return
