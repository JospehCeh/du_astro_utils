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
from astropy.nddata import CCDData
from astropy.stats import sigma_clipped_stats

# from scipy.ndimage import median_filter
from .utils import get_calib_dirs_photometry


def get_infos_from_image(fits_image_path, verbose=True):
    """
    Get useful information from image's header.

    Parameters
    ----------
    fits_image_path : str or path
        Path to the FITS image.
    verbose : bool, optional
        Whether to print statements. The default is True.

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
    acq_focus : float
        Focus set for acquisition.
    expos_time : float
        Exposure duration in seconds.
    size_x : int
        Size of image in pixels along axis x.
    size_y : int
        Size of image in pixels along axis y.

    """
    with fits.open(fits_image_path) as hdul:
        hdu = hdul[0]
        # Place value of header keyword in variables
        telescope = hdu.header.get("TELESCOP").strip()
        acq_type = hdu.header.get("ACQTYPE").strip()
        try:
            epoch = (hdu.header.get("DATE-OBS").strip()).split(".")[0] + "+00:00"
            # Process useful information
            acq_datetime = datetime.fromisoformat(epoch)
        except ValueError:
            epoch = (hdu.header.get("DATEFILE").strip()) + "+00:00"
            # Process useful information
            acq_datetime = datetime.fromisoformat(epoch)
        expos_time = hdu.header.get("EXPTIME")
        size_x = hdu.header.get("NAXIS1")
        size_y = hdu.header.get("NAXIS2")
        acq_cam = hdu.header.get("CAMMODEL").strip()
        acq_filter = hdu.header.get("INSTFILT").strip()
        try:
            acq_focus = hdu.header.get("FOCPOS")
        except AttributeError:
            acq_focus = -99
        if acq_focus is None:
            acq_focus = -99
        # Print some informations regarding the frame
        if verbose:
            print(f"{acq_type} ({size_x}x{size_y}) taken in band {acq_filter} with {acq_cam} on {telescope}@{acq_focus} on {epoch} ({expos_time}s exposure).")

    return acq_datetime, telescope, acq_cam, acq_filter, acq_focus, expos_time, size_x, size_y


def check_obs_night(date, date_ref, max_days=7, verbose=True):
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
    verbose : bool, optional
        Whether to print statements. The default is True.

    Returns
    -------
    tuple(bool, int)
        `True` if frames were taken less than `max_days` apart, otherwise `False`, number of days apart

    """
    time_delta = date - date_ref
    if verbose:
        print(time_delta)
    return abs(time_delta.days) <= max_days, time_delta.days


def load_bias_frames(path_to_bias_dir, aq_date, aq_cam, size_x, size_y, override_date_check=False, max_days=7, verbose=True):
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
    verbose : bool, optional
        Whether to print statements. The default is True.

    Returns
    -------
    bias_frames_list : list
        List of paths to appropriate BIAS frames.

    """
    _path = os.path.abspath(path_to_bias_dir)
    ndays_min = np.inf
    for _file in os.listdir(_path):
        bn, ext = os.path.splitext(os.path.basename(_file))
        _fpath = os.path.join(_path, _file)
        if os.path.isfile(_fpath) and "fits" in ext.lower():
            _dat, _scope, _cam, _filt, _focus, _dur, _x, _y = get_infos_from_image(_fpath, verbose=verbose)
            compat_date, ndays = check_obs_night(_dat, aq_date, max_days, verbose=verbose)
            if (compat_date or override_date_check) and _x == size_x and _y == size_y:
                if abs(ndays) < ndays_min:
                    bias_frames_list = [_fpath]
                elif abs(ndays) == ndays_min:
                    bias_frames_list.append(_fpath)
    return bias_frames_list


def load_dark_frames(path_to_darks_dir, aq_date, aq_cam, expos_time, size_x, size_y, override_date_check=False, max_days=7, verbose=True):
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
    verbose : bool, optional
        Whether to print statements. The default is True.

    Returns
    -------
    dark_frames_list : list
        List of paths to appropriate DARK frames.

    """
    _path = os.path.abspath(path_to_darks_dir)
    ndays_min = np.inf
    for _file in os.listdir(_path):
        bn, ext = os.path.splitext(os.path.basename(_file))
        _fpath = os.path.join(_path, _file)
        if os.path.isfile(_fpath) and "fits" in ext.lower():
            _dat, _scope, _cam, _filt, _focus, _dur, _x, _y = get_infos_from_image(_fpath, verbose=verbose)
            compat_date, ndays = check_obs_night(_dat, aq_date, max_days, verbose=verbose)
            if (compat_date or override_date_check) and _x == size_x and _y == size_y and _dur == expos_time:
                if abs(ndays) < ndays_min:
                    dark_frames_list = [_fpath]
                elif abs(ndays) == ndays_min:
                    dark_frames_list.append(_fpath)
    return dark_frames_list


def load_flat_frames(path_to_flats_dir, aq_date, aq_cam, aq_focus, aq_filter, size_x, size_y, override_date_check=False, max_days=7, verbose=True):
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
    aq_focus : float
        Focus set for acquisition.
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
    verbose : bool, optional
        Whether to print statements. The default is True.

    Returns
    -------
    flat_frames_list : list
        List of paths to appropriate FLAT frames.

    """
    _path = os.path.abspath(path_to_flats_dir)
    ndays_min = np.inf
    for _file in os.listdir(_path):
        bn, ext = os.path.splitext(os.path.basename(_file))
        _fpath = os.path.join(_path, _file)
        if os.path.isfile(_fpath) and "fits" in ext.lower():
            _dat, _scope, _cam, _filt, _focus, _dur, _x, _y = get_infos_from_image(_fpath, verbose=verbose)
            if len(_filt) > 0 and _filt[-1] == "+":
                _filt = _filt[:-1]
            if len(aq_filter) > 0 and aq_filter[-1] == "+":
                aq_filter = aq_filter[:-1]
            if "_" in aq_filter:
                aq_filter = "".join(aq_filter.split("_"))
            # print(_dat, _scope, _cam, _filt, _dur, _x, _y)
            compat_filt = (_filt == aq_filter) or (_filt.lower() in ["", "none"] and aq_filter.lower() in ["", "none"])
            compat_foc = abs(_focus - aq_focus) < 1.5
            compat_date, ndays = check_obs_night(_dat, aq_date, max_days, verbose=verbose)
            if (compat_date or override_date_check) and _x == size_x and _y == size_y and compat_filt and compat_foc:
                if abs(ndays) < ndays_min:
                    flat_frames_list = [_fpath]
                elif abs(ndays) == ndays_min:
                    flat_frames_list.append(_fpath)
    return flat_frames_list


def master_bias(bias_frames_list, overwrite=False, verbose=True):
    """
    Computes and returns the MASTER BIAS frame.

    Parameters
    ----------
    bias_frames_list : list
        List of paths to appropriate BIAS frames.
    overwrite : bool, optional
        Whether to overwrite an existing master bias (if it exists). The default is False.
    verbose : bool, optional
        Whether to print statements. The default is True.

    Returns
    -------
    dict
        Dictionary containing the path to the written master bias and the corresponding image data.

    """
    from du_astro_utils import C2PU_DATA_DIR, C2PU_RES_DIR

    # Get parent directory and hdu info
    with fits.open(bias_frames_list[0]) as frame0:
        fits_open_hdu = frame0[0]
        bias_dir = os.path.dirname(bias_frames_list[0])
        _dat, _scope, camera, _filt, _focus, _dur, _x, _y = get_infos_from_image(bias_frames_list[0], verbose=verbose)
        date = _dat.date().isoformat()

        # Check existing files
        compath = os.path.commonpath((C2PU_DATA_DIR, bias_dir))
        rel_path = os.path.relpath(bias_dir, start=compath)
        mb_dir = os.path.abspath(os.path.join(C2PU_RES_DIR, rel_path, "MASTER_BIAS"))
        if not os.path.isdir(mb_dir):
            os.makedirs(mb_dir)
        write_path = os.path.join(mb_dir, f"master_bias_{date}_{camera}.fits")
        make_calib = overwrite or not (os.path.isfile(write_path))

        if make_calib:
            # Load frames
            bias_array = np.empty((len(bias_frames_list), *fits_open_hdu.data.shape))
            for it, _file in enumerate(bias_frames_list):
                with fits.open(_file) as _hdul:
                    _open_hdu = _hdul[0]
                    try:
                        bias_array[it, :, :] = _open_hdu.data
                    except ValueError:
                        bias_array[it, :, :] = np.transpose(_open_hdu.data)

            # Master bias = mean of the bias images
            master_bias_as_array = np.mean(bias_array, axis=0)

            # Write appropriate FITS files
            fits_open_hdu.data = master_bias_as_array.astype(np.float32)
            fits_open_hdu.writeto(write_path, overwrite=True)
            if verbose:
                print(f"Master BIAS written to {write_path}.")
        else:
            if verbose:
                print(f"Using existing file : {write_path}.")
            with fits.open(write_path) as existing:
                master_bias_as_array = existing[0].data

    return {"path": write_path, "data": master_bias_as_array}


def master_dark(dark_frames_list, use_bias=False, master_bias="", overwrite=False, verbose=True):
    """
    Computes and returns the MASTER DARK frame.

    Parameters
    ----------
    dark_frames_list : list
        List of paths to appropriate DARK frames.
    use_bias : bool
        Whether to substract the master bias frame from the master dark. Necessary if the exposure of the dark differs from that of the science image. The default is False.
    master_bias : str or path, optional
        if `use_bias` : the location of the master bias fits file. The default is "".
    overwrite : bool, optional
        Whether to overwrite an existing master dark (if it exists). The default is False.
    verbose : bool, optional
        Whether to print statements. The default is True.

    Returns
    -------
    dict
        Path and pixel values to the master dark.
    dict
        Path and map of the bad (hot) pixels.

    """
    from du_astro_utils import C2PU_DATA_DIR, C2PU_RES_DIR

    # Get parent directory and hdu info
    with fits.open(dark_frames_list[0]) as frame0:
        fits_open_hdu = frame0[0]
        darks_dir = os.path.dirname(dark_frames_list[0])
        _dat, _scope, camera, _filt, _focus, exposure, _x, _y = get_infos_from_image(dark_frames_list[0], verbose=verbose)
        date = _dat.date().isoformat()

        # Check existing master darks
        compath = os.path.commonpath((C2PU_DATA_DIR, darks_dir))
        rel_path = os.path.relpath(darks_dir, start=compath)
        md_dir = os.path.abspath(os.path.join(C2PU_RES_DIR, rel_path, "MASTER_DARKS"))
        if not os.path.isdir(md_dir):
            os.makedirs(md_dir)
        md_write_path = os.path.join(md_dir, f"master_dark_{date}_{camera}_{exposure:.3f}.fits")
        hp_write_path = os.path.join(md_dir, f"bad_pixels_hot_{date}_{camera}.fits")
        make_calib = overwrite or not (os.path.isfile(md_write_path) and os.path.isfile(hp_write_path))

        if make_calib:
            # Get parent directory and hdu info
            if use_bias:
                with fits.open(master_bias) as mb_hdul:
                    mb_hdu = mb_hdul[0]
                    mb_data = mb_hdu.data
            else:
                mb_data = np.zeros_like(fits_open_hdu.data)

            # Load frames
            darks_array = np.empty((len(dark_frames_list), *fits_open_hdu.data.shape))
            for it, _file in enumerate(dark_frames_list):
                with fits.open(_file) as _hdul:
                    _open_hdu = _hdul[0]
                    darks_array[it, :, :] = _open_hdu.data

            # Master bias = median of the bias images
            try:
                master_dark_as_array = np.median(darks_array, axis=0) - mb_data
            except ValueError:
                master_dark_as_array = np.transpose(np.median(darks_array, axis=0)) - mb_data

            # Sigma-clipped statistics to detect hot pixels
            bkg_mean, bkg_median, bkg_sigma = sigma_clipped_stats(master_dark_as_array, sigma=3.0)

            # Threshold for hot pixel detection
            threshold = 5

            # Hot pixel are above a given value
            hot_pix_loc = np.where(master_dark_as_array > bkg_median + threshold * bkg_sigma)
            hot_pixels_map = np.zeros_like(master_dark_as_array)
            hot_pixels_map[hot_pix_loc] = 1

            md_clean = np.where(master_dark_as_array < 0, 0, master_dark_as_array)

            # Some statitics
            if verbose:
                print(f"Number of pixels in the dark: {len(master_dark_as_array.flatten()):8d}")
                print(f"Number of hot pixels       : {len(master_dark_as_array[hot_pix_loc]):8d}")
                print(f"Fraction of hot pixels (%) : {100*len(master_dark_as_array[hot_pix_loc])/len(master_dark_as_array.flatten()):.2f}")

            # Cosmetic smoothing
            # smoothed = median_filter(md_clean, size=(5, 5))
            # md_clean[hot_pix_loc] = smoothed[hot_pix_loc]

            # Write appropriate FITS files
            fits_open_hdu.data = md_clean.astype(np.float32)
            fits_open_hdu.writeto(md_write_path, overwrite=True)
            if verbose:
                print(f"Master DARK written to {md_write_path}.")

            fits_open_hdu.data = hot_pixels_map.astype(np.uint32)
            fits_open_hdu.writeto(hp_write_path, overwrite=True)
            if verbose:
                print(f"Hot pixels map written to {hp_write_path}.")
        else:
            if verbose:
                print(f"Using existing files : {md_write_path} and {hp_write_path}.")
            with fits.open(md_write_path) as exist0:
                md_clean = exist0[0].data
            with fits.open(hp_write_path) as exist1:
                hot_pixels_map = exist1[0].data

    return {"path": md_write_path, "data": md_clean}, {
        "path": hp_write_path,
        "data": hot_pixels_map,
    }


def median_dark(dark_frames_list, use_bias=False, master_bias="", overwrite=False, verbose=True):
    """
    Computes and returns the median (MASTER DARK) frame.

    Parameters
    ----------
    dark_frames_list : list
        List of paths to appropriate DARK frames.
    use_bias : bool
        Whether to substract the master bias frame from the master dark. Necessary if the exposure of the dark differs from that of the science image. The default is False.
    master_bias : str or path
        if `use_bias` : the location of the master bias fits file. The default is "".
    overwrite : bool, optional
        Whether to overwrite an existing master dark (if it exists). The default is False.
    verbose : bool, optional
        Whether to print statements. The default is True.

    Returns
    -------
    path
        Path to the master dark.

    """
    from du_astro_utils import C2PU_DATA_DIR, C2PU_RES_DIR

    # Get parent directory and hdu info
    with fits.open(dark_frames_list[0]) as frame0:
        fits_open_hdu = frame0[0]
        darks_dir = os.path.dirname(dark_frames_list[0])
        _dat, _scope, camera, _filt, _focus, exposure, _x, _y = get_infos_from_image(dark_frames_list[0], verbose=verbose)
        date = _dat.date().isoformat()

        # Check existing master darks
        compath = os.path.commonpath((C2PU_DATA_DIR, darks_dir))
        rel_path = os.path.relpath(darks_dir, start=compath)
        md_dir = os.path.abspath(os.path.join(C2PU_RES_DIR, rel_path, "MASTER_DARKS"))
        if not os.path.isdir(md_dir):
            os.makedirs(md_dir)
        md_write_path = os.path.join(md_dir, f"median_dark_{date}_{camera}_{exposure:.3f}.fits")
        make_calib = overwrite or not (os.path.isfile(md_write_path))

        if make_calib:
            # Get parent directory and hdu info
            # Load frames
            darks_array = np.empty((len(dark_frames_list), *fits_open_hdu.data.shape))
            for it, _file in enumerate(dark_frames_list):
                with fits.open(_file) as _hdul:
                    _open_hdu = _hdul[0]
                    darks_array[it, :, :] = _open_hdu.data

            if use_bias:
                with fits.open(master_bias) as mb_hdul:
                    mb_hdu = mb_hdul[0]
                    mb_data = mb_hdu.data
            else:
                mb_data = np.zeros_like(fits_open_hdu.data)

            # Master dark = median of the bias images
            try:
                master_dark_as_array = np.median(darks_array, axis=0) - mb_data
            except ValueError:
                master_dark_as_array = np.transpose(np.median(darks_array, axis=0)) - mb_data

            master_dark_as_array = np.where(master_dark_as_array < 0, 0, master_dark_as_array)
            # Write appropriate FITS files
            fits_open_hdu.data = master_dark_as_array.astype(np.float32)
            fits_open_hdu.writeto(md_write_path, overwrite=True)
            if verbose:
                print(f"Master DARK written to {md_write_path}.")
        else:
            if verbose:
                print(f"Using existing files : {md_write_path}.")
            with fits.open(md_write_path) as exist0:
                master_dark_as_array = exist0[0].data
    return {"path": md_write_path, "data": master_dark_as_array}


def master_flat(flat_frames_list, overwrite=False, verbose=True):
    """
    Computes and returns the MASTER FLAT frame.

    Parameters
    ----------
    flat_frames_list : list
        List of paths to appropriate FLAT frames.
    overwrite : bool, optional
        Whether to overwrite an existing master flat (if it exists). The default is False.
    verbose : bool, optional
        Whether to print statements. The default is True.

    Returns
    -------
    dict
        Path and pixel values to the master flat.
    dict
        Path and map of the bad (dead) pixels.

    """
    from du_astro_utils import C2PU_DATA_DIR, C2PU_RES_DIR, get_calib_dirs_photometry

    bias_dir, darks_dir, _ = get_calib_dirs_photometry(flat_frames_list[0])
    _dat, _scope, camera, band, focus, exposure, _x, _y = get_infos_from_image(flat_frames_list[0], verbose=verbose)
    date = _dat.date().isoformat()

    darks_list = load_dark_frames(darks_dir, _dat, camera, exposure, _x, _y, override_date_check=True, verbose=verbose)
    _, _, _, _, _, dark_expos, _, _ = get_infos_from_image(darks_list[0], verbose=verbose)

    # BIAS is included in DARK unless exposure time of DARK is different than that of the image
    if abs(dark_expos - exposure) > 0.5:
        print(dark_expos, exposure)
        # Master bias
        bias_list = load_bias_frames(bias_dir, _dat, camera, _x, _y, override_date_check=True, verbose=verbose)
        _MASTERBIAS = master_bias(bias_list, overwrite=False, verbose=verbose)
        _MASTERDARK, _ = master_dark(darks_list, use_bias=True, master_bias=_MASTERBIAS["path"], overwrite=False, verbose=verbose)
        mb_data = _MASTERBIAS["data"]
    else:
        _MASTERDARK, _ = master_dark(darks_list, use_bias=False, overwrite=False, verbose=verbose)
        mb_data = np.zeros_like(_MASTERDARK["data"])

    # Get parent directory and hdu info
    with fits.open(flat_frames_list[0]) as frame0:
        fits_open_hdu = frame0[0]
        flats_dir = os.path.dirname(flat_frames_list[0])

        # Check existing master flat
        compath = os.path.commonpath((C2PU_DATA_DIR, flats_dir))
        rel_path = os.path.relpath(flats_dir, start=compath)
        mf_dir = os.path.abspath(os.path.join(C2PU_RES_DIR, rel_path, "MASTER_FLATS"))
        if not os.path.isdir(mf_dir):
            os.makedirs(mf_dir)
        mf_write_path = os.path.join(mf_dir, f"master_flat_{date}_{camera}_{band}_{exposure:.3f}_foc{int(focus)}.fits")
        dp_write_path = os.path.join(mf_dir, f"bad_pixels_dead_{date}_{camera}.fits")
        make_calib = overwrite or not (os.path.isfile(mf_write_path) and os.path.isfile(dp_write_path))

        if make_calib:
            # Load frames
            flats_array = np.empty((len(flat_frames_list), *fits_open_hdu.data.shape))
            for it, _file in enumerate(flat_frames_list):
                with fits.open(_file) as _hdul:
                    flat_hdu = _hdul[0]
                    fl_exp = flat_hdu.header.get("EXPTIME")
                    # Remove master dark, rescaled as necessary to account for exposure variations
                    if fl_exp >= 1.0:
                        # Load master dark to compute exposure ratio
                        with fits.open(_MASTERDARK["path"]) as md_hdul:
                            md_hdu = md_hdul[0]
                            md_data = md_hdu.data
                            md_exp = md_hdu.header.get("EXPTIME")
                        exp_ratio = flat_hdu.header.get("EXPTIME") / md_exp
                        try:
                            scaled_flat = (flat_hdu.data - mb_data) - exp_ratio * md_data
                        except ValueError:
                            scaled_flat = (np.transpose(flat_hdu.data) - mb_data) - exp_ratio * md_data
                    else:
                        try:
                            scaled_flat = flat_hdu.data
                        except ValueError:
                            scaled_flat = np.transpose(flat_hdu.data)
                    _, norm, _ = sigma_clipped_stats(scaled_flat, sigma=5.0)
                    flats_array[it, :, :] = scaled_flat / norm

            # Master bias = median of the bias images
            master_flat_as_array = np.median(flats_array, axis=0)
            _, norm, _ = sigma_clipped_stats(master_flat_as_array, sigma=5.0)
            master_flat_as_array /= norm

            # Sigma-clipped statistics to detect hot pixels
            bkg_mean, bkg_median, bkg_sigma = sigma_clipped_stats(master_flat_as_array, sigma=3.0)
            # print(bkg_mean, bkg_median, bkg_sigma)

            # Threshold for hot pixel detection
            threshold = 5

            # Hot pixel are above a given value
            dead_pix_loc = np.where(master_flat_as_array <= max(0.0, bkg_median - threshold * bkg_sigma))
            dead_pixels_map = np.zeros_like(master_flat_as_array)
            dead_pixels_map[dead_pix_loc] = 1

            # _sel_too_low = master_flat_as_array < bkg_sigma
            master_flat_as_array[dead_pix_loc] = bkg_median - threshold * bkg_sigma
            # patch pour éviter de confondre des artefacts de bord d'image avec des sources lors de plate-solving

            # Some statitics
            if verbose:
                print(f"Number of pixels in the flat: {len(master_flat_as_array.flatten()):8d}")
                print(f"Number of dead pixels       : {len(master_flat_as_array[dead_pix_loc]):8d}")
                print(f"Fraction of dead pixels (%) : {100*len(master_flat_as_array[dead_pix_loc])/len(master_flat_as_array.flatten()):.2f}")

            # Cosmetic smoothing
            # smoothed = median_filter(master_flat_as_array, size=(5, 5))
            # master_flat_as_array[dead_pix_loc] = smoothed[dead_pix_loc]

            # Write appropriate FITS files
            fits_open_hdu.data = master_flat_as_array.astype(np.float32)
            fits_open_hdu.writeto(mf_write_path, overwrite=True)
            if verbose:
                print(f"Master FLAT written to {mf_write_path}.")

            fits_open_hdu.data = dead_pixels_map.astype(np.uint32)
            fits_open_hdu.writeto(dp_write_path, overwrite=True)
            if verbose:
                print(f"Dead pixels map written to {dp_write_path}.")
        else:
            if verbose:
                print(f"Using existing files : {mf_write_path} and {dp_write_path}.")
            with fits.open(mf_write_path) as exist0:
                master_flat_as_array = exist0[0].data
            with fits.open(dp_write_path) as exist1:
                dead_pixels_map = exist1[0].data

    return {"path": mf_write_path, "data": master_flat_as_array}, {
        "path": dp_write_path,
        "data": dead_pixels_map,
    }


def reduce_sci_image(fits_image, path_to_darks_dir, path_to_flats_dir, path_to_bias_dir, override_date_check=False, max_days=7, speedup=False, verbose=False, overwrite=False, write_tmp=True, overwrite_calibs=False):
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
    path_to_bias_dir : str or path
        if `use_bias` : path to the directory containing BIAS frames. The default is "".
    override_date_check : bool, optional
        Whether to allow calibration frames that are not date-wise compatible with the science frame.\
            The default is False.
    max_days : int, optional
        If `override_date_check` is `False`, the maximum number of days allowed between science image and calibration frame.\
            The default is 7.
    speedup : bool, optional
        Whether to restrict some ressource-consuming functions so that the code executes faster or doesn't crash on smaller machines.
        *E.g.* can be used in Jupyter notebooks to avoid kernel dying.
        Effects include : limiting the number of calibration frames to create MASTER frames ;\
            limiting the number of sources to estimate the FWHM of the image.
        The default is False.
    verbose : bool, optional
        Whether to print statements. The default is False.
    overwrite : bool, optional
        Whether to overwrite an existing reduced image (if it exists). The default is False.
    write_tmp : bool, optional
        Whether to write the reduced image in a temporary file. Overrides the overwrite parameter. The default is True.
    overwrite_calibs : bool, optional
        Whether to overwrite an existing calibration frame (if it exists). The default is False.

    Returns
    -------
    dic_to_return : dict
        Dictionary containing the paths to the reduced science image and associated calibration frames,\
            and the reduced data (numpy array of pixel values).

    """
    from du_astro_utils import C2PU_DATA_DIR, C2PU_RES_DIR

    # Get image directory, failename and extension
    sc_im_dir = os.path.abspath(os.path.dirname(fits_image))
    sc_im_name, sc_im_ext = os.path.splitext(os.path.basename(fits_image))

    # Get information from FITS header
    sc_date, sc_scope, sc_cam, sc_filter, sc_focus, sc_expos, sc_x, sc_y = get_infos_from_image(fits_image, verbose=verbose)

    #  Check existing files
    compath = os.path.commonpath((C2PU_DATA_DIR, sc_im_dir))
    rel_path = os.path.relpath(sc_im_dir, start=compath)
    redim_dir = os.path.abspath(os.path.join(C2PU_RES_DIR, rel_path, "REDUCED"))
    new_fn = f"{sc_im_name}_REDUCED{sc_im_ext}"
    if not os.path.isdir(redim_dir):
        os.makedirs(redim_dir)
    write_path = os.path.join(redim_dir, new_fn)
    make_reduced = not (os.path.isfile(write_path)) or overwrite or write_tmp
    if write_tmp:
        write_path = os.path.join(redim_dir, f"tmp_red{sc_im_ext}")

    if make_reduced:
        if speedup:
            max_cal_frames = 10
        # Master dark
        # path_to_darks_dir = os.path.join(sc_im_dir, '..', 'darks')
        darks_list = load_dark_frames(path_to_darks_dir, sc_date, sc_cam, sc_expos, sc_x, sc_y, override_date_check=override_date_check, max_days=max_days, verbose=verbose)
        if speedup:
            darks_list = np.random.choice(darks_list, max_cal_frames)
        _, _, _, _, _, dark_expos, _, _ = get_infos_from_image(darks_list[0], verbose=verbose)
        exp_ratio = sc_expos / dark_expos
        # BIAS is included in DARK unless exposure time of DARK is different than that of the image
        if abs(dark_expos - sc_expos) > 0.5:
            print(dark_expos, sc_expos)
            # Master bias
            bias_list = load_bias_frames(path_to_bias_dir, sc_date, sc_cam, sc_x, sc_y, override_date_check=override_date_check, max_days=max_days, verbose=verbose)
            if speedup:
                bias_list = np.random.choice(bias_list, max_cal_frames)
            MASTER_BIAS = master_bias(bias_list, overwrite=overwrite_calibs, verbose=verbose)
            MASTER_DARK, HOT_PIXELS = master_dark(darks_list, use_bias=True, master_bias=MASTER_BIAS["path"], overwrite=overwrite_calibs, verbose=verbose)
            mb_data = MASTER_BIAS["data"]
        else:
            MASTER_DARK, HOT_PIXELS = master_dark(darks_list, use_bias=False, overwrite=overwrite_calibs, verbose=verbose)
            mb_data = np.zeros_like(MASTER_DARK["data"])

        # Master flat
        # TBD: check if there is already one that works
        # path_to_flats_dir = os.path.join(sc_im_dir, '..', 'flats')
        flats_list = load_flat_frames(path_to_flats_dir, sc_date, sc_cam, sc_focus, sc_filter, sc_x, sc_y, override_date_check=override_date_check, max_days=max_days, verbose=verbose)
        # print(os.listdir(path_to_flats_dir), flats_list)
        if speedup:
            flats_list = np.random.choice(flats_list, max_cal_frames)
        MASTER_FLAT, DEAD_PIXELS = master_flat(flats_list, overwrite=overwrite_calibs, verbose=verbose)

        with fits.open(fits_image) as sc_hdul:
            sc_hdu = sc_hdul[0]
            with fits.open(MASTER_DARK["path"]) as md_hdul:
                exp_ratio = sc_hdu.header.get("EXPTIME") / md_hdul[0].header.get("EXPTIME")
            try:
                RED_SCIENCE = (sc_hdu.data - exp_ratio * MASTER_DARK["data"] - mb_data) / MASTER_FLAT["data"]
            except ValueError:
                RED_SCIENCE = (sc_hdu.data - np.transpose(exp_ratio * MASTER_DARK["data"]) - np.transpose(mb_data)) / np.transpose(MASTER_FLAT["data"])

            RED_SCIENCE = np.where(RED_SCIENCE < 0, 0, RED_SCIENCE)

            # Clean bad pixels
            # smoothed = median_filter(RED_SCIENCE, size=(5, 5))

            # Hot pixels
            # try:
            #    hot_pixel = np.where(HOT_PIXELS["data"] == 1)
            #    RED_SCIENCE[hot_pixel] = smoothed[hot_pixel]
            # except:
            #    print("Cannot clean hot pixels")

            # Dead pixels
            # try:
            #    dead_pixel = np.where(DEAD_PIXELS["data"] == 1)
            #    RED_SCIENCE[dead_pixel] = smoothed[dead_pixel]
            # except:
            #    print("Cannot clean dead pixels")

            # Write appropriate FITS files
            red_hdu = sc_hdu.copy()
            red_hdu.data = CCDData(RED_SCIENCE.astype(np.float32), mask=np.logical_or(DEAD_PIXELS["data"] == 1, HOT_PIXELS["data"] == 1), unit="adu")
            red_hdu.header["PROCTYPE"] = "RED     "
            red_hdu.header["FILENAME"] = new_fn
            red_hdu.header["CREATOR"] = "JOCHEVAL"
            red_hdu.header["MBIAS"] = MASTER_BIAS["path"]
            red_hdu.header["MDARK"] = MASTER_DARK["path"]
            red_hdu.header["MFLAT"] = MASTER_FLAT["path"]
            red_hdu.header["HPIXELS"] = HOT_PIXELS["path"]
            red_hdu.header["DPIXELS"] = DEAD_PIXELS["path"]
            red_hdu.writeto(write_path, overwrite=True)
            if verbose:
                print(f"Calibrated image written to {write_path}.")

            dic_to_return = {
                "path": write_path,
                "MASTER BIAS": MASTER_BIAS["path"],
                "MASTER DARK": MASTER_DARK["path"],
                "MASTER FLAT": MASTER_FLAT["path"],
                "HOT PIXELS": HOT_PIXELS["path"],
                "DEAD PIXELS": DEAD_PIXELS["path"],
            }

    else:
        if verbose:
            print(f"Calibrated image loaded from {write_path}.")
        with fits.open(write_path) as hudl:
            red_hdu = hudl[0]
            hdr = red_hdu.header
            dic_to_return = {
                "path": write_path,
                "MASTER BIAS": hdr.get("MBIAS"),
                "MASTER DARK": hdr.get("MDARK"),
                "MASTER FLAT": hdr.get("MFLAT"),
                "HOT PIXELS": hdr.get("HPIXELS"),
                "DEAD PIXELS": hdr.get("DPIXELS"),
            }

    return dic_to_return


def dedark_sci_image(fits_image, override_date_check=False, max_days=7, verbose=False, overwrite=False, write_tmp=False, overwrite_calibs=False):
    """
    Simply remove dark currents from a science image to obtain a reduced science image to be used for analysis.
    If the master bias is used : $REDUCED = \frac{SCIENCE-mDARK}$

    Parameters
    ----------
    fits_image : str or path
        Path to the FITS image to be reduced.
    override_date_check : bool, optional
        Whether to allow calibration frames that are not date-wise compatible with the science frame.\
            The default is False.
    max_days : int, optional
        If `override_date_check` is `False`, the maximum number of days allowed between science image and calibration frame.\
            The default is 7.
    verbose : bool, optional
        Whether to print statements. The default is False.
    overwrite : bool, optional
        Whether to overwrite an existing reduced image (if it exists). The default is False.
    write_tmp : bool, optional
        Whether to write the reduced image in a temporary file. Overrides the overwrite parameter. The default is False.
    overwrite_calibs : bool, optional
        Whether to overwrite an existing calibration frame (if it exists). The default is False.

    Returns
    -------
    dic_to_return : dict
        Dictionary containing the paths to the reduced science image and associated calibration frames.

    """
    from du_astro_utils import C2PU_DATA_DIR, C2PU_RES_DIR

    img_abs_path = os.path.abspath(fits_image)

    path_to_bias_dir, path_to_darks_dir, _ = get_calib_dirs_photometry(img_abs_path)

    # Get image directory, failename and extension
    sc_im_dir = os.path.abspath(os.path.dirname(img_abs_path))
    sc_im_name, sc_im_ext = os.path.splitext(os.path.basename(img_abs_path))

    # Get information from FITS header
    sc_date, sc_scope, sc_cam, sc_filter, sc_focus, sc_expos, sc_x, sc_y = get_infos_from_image(img_abs_path, verbose=verbose)

    #  Check existing files
    compath = os.path.commonpath((C2PU_DATA_DIR, sc_im_dir))
    rel_path = os.path.relpath(sc_im_dir, start=compath)
    redim_dir = os.path.abspath(os.path.join(C2PU_RES_DIR, rel_path, "REDUCED"))
    new_fn = f"{sc_im_name}_DEDARK{sc_im_ext}"
    if not os.path.isdir(redim_dir):
        os.makedirs(redim_dir)
    write_path = os.path.join(redim_dir, new_fn)
    make_reduced = not (os.path.isfile(write_path)) or overwrite or write_tmp
    if write_tmp:
        write_path = os.path.join(redim_dir, f"tmp_dedark{sc_im_ext}")

    if make_reduced:
        # Master dark
        # path_to_darks_dir = os.path.join(sc_im_dir, '..', 'darks')
        darks_list = load_dark_frames(path_to_darks_dir, sc_date, sc_cam, sc_expos, sc_x, sc_y, override_date_check=override_date_check, max_days=max_days, verbose=verbose)
        _, _, _, _, _, dark_expos, _, _ = get_infos_from_image(darks_list[0], verbose=verbose)
        exp_ratio = sc_expos / dark_expos
        # print(exp_ratio)

        # BIAS is included in DARK unless exposure time of DARK is different than that of the image
        if dark_expos != sc_expos:
            print(dark_expos, sc_expos)
            # Master bias
            bias_list = load_bias_frames(path_to_bias_dir, sc_date, sc_cam, sc_x, sc_y, override_date_check=override_date_check, max_days=max_days, verbose=verbose)
            MASTER_BIAS = master_bias(bias_list, overwrite=overwrite_calibs, verbose=verbose)
            MASTER_DARK = median_dark(darks_list, use_bias=True, master_bias=MASTER_BIAS["path"], overwrite=overwrite_calibs, verbose=verbose)
            with fits.open(fits_image) as sc_hdul:
                sc_hdu = sc_hdul[0].copy()
                try:
                    RED_SCIENCE = sc_hdu.data - exp_ratio * MASTER_DARK["data"] - MASTER_BIAS["data"]
                except ValueError:
                    RED_SCIENCE = sc_hdu.data - np.transpose(exp_ratio * MASTER_DARK["data"] + MASTER_BIAS["data"])
        else:
            MASTER_DARK = median_dark(darks_list, overwrite=overwrite_calibs, verbose=verbose)
            with fits.open(fits_image) as sc_hdul:
                sc_hdu = sc_hdul[0].copy()
                try:
                    RED_SCIENCE = sc_hdu.data - MASTER_DARK["data"]
                except ValueError:
                    RED_SCIENCE = sc_hdu.data - np.transpose(MASTER_DARK["data"])

        RED_SCIENCE = np.where(RED_SCIENCE < 0, 0, RED_SCIENCE)

        # Write appropriate FITS files
        red_hdu = sc_hdu
        red_hdu.data = RED_SCIENCE.astype(np.float32)
        red_hdu.header["PROCTYPE"] = "RED     "
        red_hdu.header["FILENAME"] = new_fn
        red_hdu.header["CREATOR"] = "JOCHEVAL"
        if dark_expos != sc_expos:
            red_hdu.header["MBIAS"] = MASTER_BIAS["path"]
        red_hdu.header["MDARK"] = MASTER_DARK["path"]
        red_hdu.writeto(write_path, overwrite=True)
        if verbose:
            print(f"Calibrated image written to {write_path}.")

        dic_to_return = {"path": write_path, "data": red_hdu.data}

    else:
        if verbose:
            print(f"Calibrated image loaded from {write_path}.")
        with fits.open(write_path) as hudl:
            red_hdu = hudl[0]
            dic_to_return = {"path": write_path, "data": red_hdu.data}
    return dic_to_return
