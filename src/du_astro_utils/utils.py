#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 15:27:27 2024

@author: Joseph Chevalier
"""

from datetime import datetime
from astropy.io import fits
import os
import numpy as np
import glob

DIR_PHOTOM = 'Photometry'
DIR_SPECTRO = 'Spectroscopy'
DIR_ASTER = 'Asteroids'
DIR_CALIB = 'CCD__BIAS_DARKS_FLATS'
C2PU_DATA_DIR = '/home/etudiant/Documents/Archives_C2PU'

def get_calib_dirs_photometry(fits_image_path):
    hdu = fits.open(fits_image_path)[0]
    expos_time = hdu.header.get('EXPTIME')
    acq_cam = hdu.header.get('CAMMODEL').strip()
    acq_filter = hdu.header.get('INSTFILT').strip()
    
    root_dir = os.path.join(C2PU_DATA_DIR, DIR_PHOTOM, DIR_CALIB)
    bias_dir = os.path.join(root_dir, 'BIAS')
    darks_dir = os.path.join(root_dir, 'DARKS')
    flats_dir = os.path.join(root_dir, 'FLATS')
    
    for cam in os.listdir(bias_dir):
        if cam in acq_cam : bias_dir = os.path.join(bias_dir, cam)
    
    for cam in os.listdir(darks_dir):
        if cam in acq_cam : darks_dir = os.path.join(darks_dir, cam)
    
    for cam in os.listdir(flats_dir):
        if cam in acq_cam : flats_dir = os.path.join(flats_dir, cam)
    
    all_darks_durs = os.listdir(darks_dir)
    exp_in_s = np.array([float(dur[:-1]) for dur in all_darks_durs])
    best_dur_loc = np.argmin(np.abs(exp_in_s - expos_time))
    darks_dir = os.path.join(darks_dir, all_darks_durs[best_dur_loc])
    
    all_flats_filts = os.listdir(flats_dir)
    for _filt in all_flats_filts:
        if (acq_filter.lower() == "") and ("none" in _filt.lower()) : flats_dir = os.path.join(flats_dir, _filt)
        elif acq_filter.lower() in _filt.lower() : flats_dir = os.path.join(flats_dir, _filt)
        
    
    return bias_dir, darks_dir, flats_dir