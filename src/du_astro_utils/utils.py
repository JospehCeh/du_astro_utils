#!/usr/bin/env python3
"""
Module with helpers for locating and using calibration frames.

Created on Tue Feb 20 15:27:27 2024

@author: Joseph Chevalier
"""

# pylint : skip-file

import os

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from tqdm import tqdm

DIR_PHOTOM = "Photometry"
DIR_SPECTRO = "Spectroscopy"
DIR_ASTER = "Asteroids"
DIR_CALIB = "CCD__BIAS_DARKS_FLATS"
DIR_EXPLTS = "ExoPInts"
DIR_CLUSTERS = "Clusters"
DIR_VARSTARS = "VariableStars"
DIR_GALCLUST = "Amas_Galaxies"

try:
    C2PU_DATA_DIR = os.environ["ARCHIVESC2PU"]
except KeyError:
    try:
        C2PU_DATA_DIR = input("Please type in the location of the root archives directory, e.g. /home/user/Archives_C2PU")
    except OSError:
        C2PU_DATA_DIR = "."


def get_calib_dirs_photometry(fits_image_path):
    """
    Returns the paths to calibration images in the archives.

    Parameters
    ----------
    fits_image_path : str or path
        DESCRIPTION.

    Returns
    -------
    bias_dir : str or path
        Path to BIAS images.
    darks_dir : str or path
        Path to DARK images.
    flats_dir : str or path
        Path to FLAT images.

    """
    with fits.open(fits_image_path) as hdul:
        hdu = hdul[0]
        expos_time = hdu.header.get("EXPTIME")
        acq_cam = hdu.header.get("CAMMODEL").strip()
        acq_filter = hdu.header.get("INSTFILT").strip()

        root_dir = os.path.join(C2PU_DATA_DIR, DIR_PHOTOM, DIR_CALIB)
        bias_dir = os.path.join(root_dir, "BIAS")
        darks_dir = os.path.join(root_dir, "DARKS")
        flats_dir = os.path.join(root_dir, "FLATS")

        for cam in os.listdir(bias_dir):
            if cam in acq_cam:
                bias_dir = os.path.join(bias_dir, cam)

        for cam in os.listdir(darks_dir):
            if cam in acq_cam:
                darks_dir = os.path.join(darks_dir, cam)

        for cam in os.listdir(flats_dir):
            if cam in acq_cam:
                flats_dir = os.path.join(flats_dir, cam)

        all_darks_durs = os.listdir(darks_dir)
        exp_in_s = np.array([float(dur[:-1]) for dur in all_darks_durs])
        best_dur_loc = np.argmin(np.abs(exp_in_s - expos_time))
        darks_dir = os.path.join(darks_dir, all_darks_durs[best_dur_loc])

        all_flats_filts = os.listdir(flats_dir)
        for _filt in all_flats_filts:
            if "none" in _filt.lower() and "focus" not in _filt.lower():
                if acq_filter.lower() == "" or acq_filter.lower() == "none":
                    flats_dir = os.path.join(flats_dir, _filt)
            elif (acq_filter.lower() in _filt.lower()) and acq_filter.lower() != "":
                flats_dir = os.path.join(flats_dir, _filt)

    return bias_dir, darks_dir, flats_dir


def propagate_wcs(aligned_files_list, ref_file=None, overwrite=False):
    """
    Propagates the WCS from the reference FITS image to all images in the list.
    Images should be aligned (*e.g.* using AstroImageJ utilities) and the reference image plate-solved
    (*e.g.* with [astrometry.net](nova.astrometry.net)) prior to using this function.

    Parameters
    ----------
    aligned_files_list : list of path or str
        List of paths to the FITS files to be updated. Can contain the reference file.
    ref_file : path or str, optional
        Path to the reference FITS file to source the WCS. If None (default), the first file of `aligned_files_list` will be used. The default is None.
    overwrite : bool
        Whether to overwrite the file or save to a new one. If False (default), the new file will be saved in a `solved` subdirectory. The default is False.

    Returns
    -------
    list of path
        List of paths to updated FITS files.
    """
    new_fits = []
    if ref_file is None:
        ref_file = aligned_files_list[0]

    ref_path = os.path.abspath(ref_file)
    with fits.open(ref_path) as ref_hdul:
        _ref_hdr = ref_hdul[0].header
        _ref_hdr["CTYPE1"] = f"{_ref_hdr['CTYPE1']}-SIP"
        _ref_hdr["CTYPE2"] = f"{_ref_hdr['CTYPE2']}-SIP"
        ref_wcs = WCS(_ref_hdr)
        ref_wcs_hdr = ref_wcs.to_header()
        try:
            _ = ref_wcs_hdr.pop("DATE-OBS")
            _ = ref_wcs_hdr.pop("MJD-OBS")
        except KeyError:
            pass
    for fitsf in tqdm(aligned_files_list):
        _path = os.path.abspath(fitsf)
        _dir, _file = os.path.split(_path)
        _fn, _ext = os.path.splitext(_file)
        with fits.open(_path) as _hdul:
            _hdu = _hdul[0].copy()
        _hdu.header.update(ref_wcs_hdr)
        if overwrite:
            _hdu.writeto(_path, overwrite=True)
            new_fits.append(_path)
        else:
            _ndir = os.path.join(_dir, "plate_solved")
            if not os.path.isdir(_ndir):
                os.makedirs(_ndir)
            _npath = os.path.join(_ndir, f"{_fn}_solved{_ext}")
            _hdu.write(_npath, overwrite=True)
            new_fits.append(_npath)
    return new_fits
