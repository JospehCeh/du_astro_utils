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
