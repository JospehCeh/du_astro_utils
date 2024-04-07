#!/usr/bin/env python3
"""
Module with photometry calibration and analysis routines.

Created on Tue Feb 20 15:28:18 2024

@author: Joseph Chevalier
"""

# pylint : skip-file

import numpy as np
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord
from astropy.io import ascii, fits
from astropy.nddata import Cutout2D
from astropy.nddata.utils import NoOverlapError
from astropy.stats import sigma_clipped_stats
from astropy.time import Time
from astropy.timeseries import TimeSeries
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel
from astroquery.imcce import Skybot
from astroquery.mast import Catalogs
from astroquery.vizier import Vizier
from photutils import DAOStarFinder
from photutils.aperture import (
    CircularAnnulus,
    CircularAperture,
    SkyCircularAnnulus,
    SkyCircularAperture,
    aperture_photometry,
)
from scipy.optimize import curve_fit


def gaussian(x, a, x0, sigma):
    """
    Computes a gaussian function (or bell curve).

    Parameters
    ----------
    x : float or array
        Points where the gaussian is computed.
    a : float
        Scaling factor.
    x0 : float
        Center of the gaussian.
    sigma : float
        Standard deviation of the gaussian.

    Returns
    -------
    float or array
        Values of the gaussian along x.

    """
    return (a / np.sqrt(2 * np.pi * sigma**2)) * np.exp(-((x - x0) ** 2) / (2 * sigma**2))


def snr(R, sigma, nu):
    """
    Computes theoretical signal-to-noise ratio with a simplified model.

    Parameters
    ----------
    R : float
        Instrument noise.
    sigma : float
        Signal noise.
    nu : float
        (Other?) noise.

    Returns
    -------
    snr : float
        Signal-to-noise ratio.

    """
    f_star = 1 - np.exp(-0.5 * (R / sigma) ** 2)
    snr = f_star / np.sqrt(f_star + np.pi * nu * R * R)
    return snr


def detect_sources(reduced_fits_image, detection_fwhm=4.0, detection_threshold_nsigmas=3.0, verbose=True):
    """
    Automatically detect sources in an image.

    Parameters
    ----------
    reduced_fits_image : str or path
        Path to the studied image in FITS format.
    detection_fwhm : float, optional
        Full-width at half-maximum of signal to considerate, in pixels. The default is 4.0.
    detection_threshold_nsigmas : float, optional
        Detection threshold in numbers of standard deviations. The default is 3.0.
    verbose : bool, optional
        Whether to print statements. The default is True.

    Returns
    -------
    sources : table
        Astropy table of sources detected, as returned by DAOStarFinder.

    """
    # extract basic info from FITS
    with fits.open(reduced_fits_image) as hdul:
        hdu = hdul[0]

        # Compute background statistics
        bkg_mean, bkg_median, bkg_sigma = sigma_clipped_stats(hdu.data, sigma=3.0)

        # Initializing the DAO Star Finder
        daofind = DAOStarFinder(fwhm=detection_fwhm, threshold=bkg_median + detection_threshold_nsigmas * bkg_sigma)

        # Search sources in the frame
        sources = daofind(hdu.data)
        if verbose:
            print(f"Number of sources detected: {len(sources):d}")

    return sources


def get_fwhm(reduced_fits_image, sources, src_siz=51):
    """
    Computes the Full-width at half-maximum property of an image.

    Parameters
    ----------
    reduced_fits_image : str or path
        Path to the studied image in FITS format.
    sources : Table
        Table of sources in the image.
    src_siz : float, optional
        Sources diameter in pixels. The default is 51.

    Returns
    -------
    float
        Full-width at half-maximum of sources described by a gaussian curve.

    """
    with fits.open(reduced_fits_image) as hdul:
        hdu = hdul[0]
        wcs = WCS(hdu.header)

        # Definition of cutout central position and size
        naxis1 = hdu.header.get("NAXIS1")
        naxis2 = hdu.header.get("NAXIS2")
        size = (naxis1 // 7, naxis2 // 7)

        # iteration on position
        x_y = [(1, 1), (1, 2), (2, 1), (2, 2)]
        x_y_loc = 0
        found = False
        stop = found or (x_y_loc > len(x_y))
        while not (stop):
            found = True
            nx, ny = x_y[x_y_loc]
            position = (nx * naxis1 // 3, ny * naxis2 // 3)
            # print(f"DEBUG: position {position}, size {size}")

            # Cutout
            cutout = Cutout2D(hdu.data, position=position, size=size, wcs=wcs)
            zoom_hdu = hdu.copy()
            zoom_hdu.data = cutout.data
            zoom_hdu.header.update(cutout.wcs.to_header())
            # zoom_wcs = WCS(zoom_hdu.header)

            # Selection and edition of the centroids
            zoom_sources = sources[(sources["xcentroid"] > (position[0] - size[0] / 2)) & (sources["xcentroid"] < (position[0] + size[0] / 2)) & (sources["ycentroid"] > (position[1] - size[1] / 2)) & (sources["ycentroid"] < (position[1] + size[1] / 2))]
            zoom_sources["xcentroid"] -= position[0] - size[0] / 2
            zoom_sources["ycentroid"] -= position[1] - size[1] / 2

            # Randomly choose a source far enough from edges
            # src_sel = np.random.randint(0, len(zoom_sources))
            src_sel = np.random.choice(len(zoom_sources), size=len(zoom_sources))

            count = 0
            sel = src_sel[count]
            src_pos = (zoom_sources[sel]["xcentroid"], zoom_sources[sel]["ycentroid"])
            while found and ((src_pos[0] < src_siz) or (src_pos[1] < src_siz) or ((zoom_hdu.data.shape[0] - src_pos[0]) < src_siz) or ((zoom_hdu.data.shape[1] - src_pos[1]) < src_siz)):
                count += 1
                try:
                    sel = src_sel[count]
                    src_pos = (zoom_sources[sel]["xcentroid"], zoom_sources[sel]["ycentroid"])
                except IndexError:
                    found = False
            x_y_loc += 1
            stop = found or (x_y_loc > len(x_y))

        # print(f"DEBUG : count {count}, source pos {src_pos}, source size {src_siz}")

        if found:
            try:
                # Cutout
                src_cut = Cutout2D(zoom_hdu.data, position=src_pos, size=src_siz)
                # xc = src_siz / 2.0
                # yc = src_siz / 2.0

                # Source profile
                x_arr = np.array([x for x in range(src_siz)])
                x_sum = np.sum(src_cut.data, axis=1)
                x_sum = x_sum - np.median(x_sum)

                # Adjust profile with Gaussian
                param, _ = curve_fit(gaussian, x_arr, x_sum, p0=[np.max(src_cut.data), src_siz / 2.0, 5])
                fwhm = np.abs(param[2]) * 2.355
            except NoOverlapError:
                fwhm = 4.0
        else:
            fwhm = 4.0
        # gaus_model = gaussian(x_arr, *param)
    return fwhm


def apert_photometry(reduced_fits_image, sources, fwhm):
    """
    Performs aperture photometry of the sources in the image ; writes the results in ASCII format.

    Parameters
    ----------
    reduced_fits_image : str or path
        Path to the studied image in FITS format.
    sources : Table
        Table of sources in the image.
    fwhm : float
        Full-width at half-maximum of sources described by a gaussian curve.

    Returns
    -------
    phot_table : Table
        Photometry of the sources in ADU counts.

    """
    with fits.open(reduced_fits_image) as hdul:
        hdu = hdul[0]
        epoch = Time(hdu.header.get("DATE-OBS"), format="isot")

        # Defining apertures
        aperture_radius = 1.0 * fwhm
        annulus_radius = [aperture_radius + 2, aperture_radius + 5]
        positions = np.transpose((sources["xcentroid"], sources["ycentroid"]))
        apertures = CircularAperture(positions, r=aperture_radius)
        phot_table = aperture_photometry(hdu.data, apertures)

        # Define annuli
        annulus_aperture = CircularAnnulus(positions, r_in=annulus_radius[0], r_out=annulus_radius[1])
        annulus_masks = annulus_aperture.to_mask(method="center")

        # For each source, compute the median (through sigma/clipping)
        bkg_median_arr = []
        for mask in annulus_masks:
            annulus_data = mask.multiply(hdu.data)
            annulus_data_1d = annulus_data[mask.data > 0]
            _, median_sigclip, _ = sigma_clipped_stats(annulus_data_1d)
            bkg_median_arr.append(median_sigclip)

        # Store background stat in phot_table
        bkg_median_arr = np.array(bkg_median_arr)
        phot_table["annulus_median"] = bkg_median_arr
        phot_table["aper_bkg"] = bkg_median_arr * apertures.area
        phot_table["aper_sum_bkgsub"] = phot_table["aperture_sum"] - phot_table["aper_bkg"]
        phot_table["noise"] = np.sqrt(phot_table["aper_sum_bkgsub"] + phot_table["aper_bkg"])  # photon noise: source + sky
        phot_table["SNR"] = phot_table["aper_sum_bkgsub"] / phot_table["noise"]

        time_ser = TimeSeries(time=np.full(phot_table["aper_sum_bkgsub"].shape, epoch), data=phot_table)

        name_csv = reduced_fits_image[:-8] + "-apert_phot.csv"
        ascii.write(time_ser, name_csv, format="csv", overwrite=True)
    return time_ser


def query_sso_photometry(reduced_fits_image, fwhm, cone_angle_deg=0.25, verbose=True):
    """
    Locates Solar System Objects in the image and performs forced photometry ;\
        writes the results in ASCII format.

    Parameters
    ----------
    reduced_fits_image : str or path
        Path to the studied image in FITS format.
    fwhm : float
        Full-width at half-maximum of sources described by a gaussian curve.
    cone_angle_deg : float, optional
        Angle of the cone within which SSO objects are searched, in degrees. The default is 0.25.
    verbose : bool, optional
        Whether to print statements. The default is True.

    Returns
    -------
    phot_ssos : Table
        Forced photometry of the detected Solar System Objects in ADU counts.

    """
    with fits.open(reduced_fits_image) as hdul:
        hdu = hdul[0]
        wcs = WCS(hdu.header)
        epoch = Time(hdu.header.get("DATE-OBS"), format="isot")
        bkg_mean, bkg_median, bkg_sigma = sigma_clipped_stats(hdu.data, sigma=3.0)

        # Query SkyBoT
        try:
            field = SkyCoord(wcs.wcs.crval[0], wcs.wcs.crval[1], unit=u.deg)
            ssos = Skybot.cone_search(field, cone_angle_deg * u.deg, epoch)
            if verbose:
                ssos.pprint()
                print(f"Number of SSOs predicted: {len(ssos):d}")
        # And tell us if no SSO are present in the field of view
        except TypeError:
            print("No SSO in the current Field of View")
            return None

        # Catalog to pixels
        catalog_x, catalog_y = skycoord_to_pixel(SkyCoord(ssos["RA"], ssos["DEC"]), wcs=wcs)

        # Defining apertures
        ssos_positions = SkyCoord(ra=ssos["RA"], dec=ssos["DEC"], unit="deg")
        positions = np.transpose((catalog_x, catalog_y))
        aperture_radius = 1.0 * fwhm
        apertures = CircularAperture(positions, r=aperture_radius)
        skyapertures_ssos = SkyCircularAperture(ssos_positions, apertures.to_sky(wcs=wcs).r)

        # Measure photometry for these apertures
        phot_ssos = aperture_photometry(hdu.data, skyapertures_ssos, wcs=wcs)
        ssos = ssos[~np.isnan(phot_ssos["aperture_sum"])]
        ssos_positions = ssos_positions[~np.isnan(phot_ssos["aperture_sum"])]
        phot_ssos = phot_ssos[~np.isnan(phot_ssos["aperture_sum"])]

        annulus_radius = [aperture_radius + 2, aperture_radius + 5]
        annulus_aperture = CircularAnnulus(positions, r_in=annulus_radius[0], r_out=annulus_radius[1])
        annulus_aperture_ssos = SkyCircularAnnulus(
            ssos_positions,
            r_in=annulus_aperture.to_sky(wcs=wcs).r_in,
            r_out=annulus_aperture.to_sky(wcs=wcs).r_out,
        )
        annulus_masks_ssos = annulus_aperture_ssos.to_pixel(wcs=wcs).to_mask(method="center")

        bkg_median = []
        for mask in annulus_masks_ssos:
            annulus_data_ssos = mask.multiply(hdu.data)
            annulus_data_1d = annulus_data_ssos[mask.data > 0]
            _, median_sigclip, _ = sigma_clipped_stats(annulus_data_1d)
            bkg_median.append(median_sigclip)

        bkg_median = np.array(bkg_median)
        phot_ssos["annulus_median"] = bkg_median
        phot_ssos["aper_bkg"] = bkg_median * apertures.area
        phot_ssos["aper_sum_bkgsub"] = phot_ssos["aperture_sum"] - phot_ssos["aper_bkg"]

        ssos = ssos[phot_ssos["aper_sum_bkgsub"] > 0]
        ssos_positions = ssos_positions[phot_ssos["aper_sum_bkgsub"] > 0]
        phot_ssos = phot_ssos[phot_ssos["aper_sum_bkgsub"] > 0]

        phot_ssos["noise"] = np.sqrt(phot_ssos["aper_sum_bkgsub"] + phot_ssos["aper_bkg"])  # photon noise: source + sky
        phot_ssos["SNR"] = phot_ssos["aper_sum_bkgsub"] / phot_ssos["noise"]

        phot_ssos["Name"] = ssos["Name"]
        phot_ssos["Number"] = ssos["Number"]
        phot_ssos["xyPosition"] = ssos_positions

        time_ser = TimeSeries(time=np.full(phot_ssos["aper_sum_bkgsub"].shape, epoch), data=phot_ssos)

        name_csv = reduced_fits_image[:-8] + "-forced_phot.csv"
        ascii.write(time_ser, name_csv, format="csv", overwrite=True)
    return time_ser


def query_named_sso_photometry(reduced_fits_image, fwhm, cone_angle_deg=0.25, verbose=True):
    """
    Identifies the target in a Solar System Object queryin the image and performs forced photometry ;\
        writes the results in ASCII format.

    Parameters
    ----------
    reduced_fits_image : str or path
        Path to the studied image in FITS format.
    fwhm : float
        Full-width at half-maximum of sources described by a gaussian curve.
    cone_angle_deg : float, optional
        Angle of the cone within which SSO objects are searched, in degrees. The default is 0.25.
    verbose : bool, optional
        Whether to print statements. The default is True.

    Returns
    -------
    phot_ssos : Table
        Forced photometry of the detected Solar System Objects in ADU counts.

    """
    with fits.open(reduced_fits_image) as hdul:
        hdu = hdul[0]
        wcs = WCS(hdu.header)
        epoch = Time(hdu.header.get("DATE-OBS"), format="isot")
        bkg_mean, bkg_median, bkg_sigma = sigma_clipped_stats(hdu.data, sigma=3.0)

        # Query SkyBoT
        target = hdu.header.get("OBJECT").lower()

        try:
            field = SkyCoord(wcs.wcs.crval[0], wcs.wcs.crval[1], unit=u.deg)
            ssos = Skybot.cone_search(field, cone_angle_deg * u.deg, epoch)
            if verbose:
                ssos.pprint()
            sel = [n.lower() in target for n in ssos["Name"]]
            targets = ssos[sel]
            if verbose:
                targets.pprint()
                print(f"Number of matching targets: {len(targets):d}")
        # And tell us if no SSO are present in the field of view
        except TypeError:
            print("No SSO in the current Field of View")
            return None

        # Catalog to pixels
        catalog_x, catalog_y = skycoord_to_pixel(SkyCoord(targets["RA"], targets["DEC"]), wcs=wcs)

        # Defining apertures
        ssos_positions = SkyCoord(ra=targets["RA"], dec=targets["DEC"], unit="deg")
        positions = np.transpose((catalog_x, catalog_y))
        aperture_radius = 1.0 * fwhm
        apertures = CircularAperture(positions, r=aperture_radius)
        skyapertures_ssos = SkyCircularAperture(ssos_positions, apertures.to_sky(wcs=wcs).r)

        # Measure photometry for these apertures
        phot_ssos = aperture_photometry(hdu.data, skyapertures_ssos, wcs=wcs)
        # targets = targets[~np.isnan(phot_ssos["aperture_sum"])]
        # ssos_positions = ssos_positions[~np.isnan(phot_ssos["aperture_sum"])]
        # phot_ssos = phot_ssos[~np.isnan(phot_ssos["aperture_sum"])]

        annulus_radius = [aperture_radius + 2, aperture_radius + 5]
        annulus_aperture = CircularAnnulus(positions, r_in=annulus_radius[0], r_out=annulus_radius[1])
        annulus_aperture_ssos = SkyCircularAnnulus(
            ssos_positions,
            r_in=annulus_aperture.to_sky(wcs=wcs).r_in,
            r_out=annulus_aperture.to_sky(wcs=wcs).r_out,
        )
        annulus_masks_ssos = annulus_aperture_ssos.to_pixel(wcs=wcs).to_mask(method="center")

        bkg_median = []
        for mask in annulus_masks_ssos:
            annulus_data_ssos = mask.multiply(hdu.data)
            annulus_data_1d = annulus_data_ssos[mask.data > 0]
            _, median_sigclip, _ = sigma_clipped_stats(annulus_data_1d)
            bkg_median.append(median_sigclip)

        bkg_median = np.array(bkg_median)
        phot_ssos["annulus_median"] = bkg_median
        phot_ssos["aper_bkg"] = bkg_median * apertures.area
        phot_ssos["aper_sum_bkgsub"] = phot_ssos["aperture_sum"] - phot_ssos["aper_bkg"]

        # targets = targets[phot_ssos["aper_sum_bkgsub"] > 0]
        # ssos_positions = ssos_positions[phot_ssos["aper_sum_bkgsub"] > 0]
        # phot_ssos = phot_ssos[phot_ssos["aper_sum_bkgsub"] > 0]

        phot_ssos["noise"] = np.sqrt(np.abs(phot_ssos["aper_sum_bkgsub"]) + np.abs(phot_ssos["aper_bkg"]))  # photon noise: source + sky
        phot_ssos["SNR"] = np.abs(phot_ssos["aper_sum_bkgsub"]) / np.abs(phot_ssos["noise"])

        phot_ssos["Name"] = targets["Name"]
        phot_ssos["Number"] = targets["Number"]
        # phot_ssos["Epoch"] = np.full_like(targets["Name"], epoch)
        phot_ssos["xyPosition"] = ssos_positions

        time_ser = TimeSeries(time=np.full(phot_ssos["aper_sum_bkgsub"].shape, epoch), data=phot_ssos)

        # try:
        #    time_ser = hstack([time_ser, phot_ssos])
        # except IndexError:
        #    time_ser = phot_ssos

        name_csv = reduced_fits_image[:-8] + f"{target}-forced_phot.csv"
        ascii.write(time_ser, name_csv, format="csv", overwrite=True)
    return time_ser


def query_panstarrs(reduced_fits_image, cone_angle_deg=0.25, mag_limit=19, source="Vizier"):
    """
    Identifies the target in a Solar System Object queryin the image and performs forced photometry ;\
        writes the results in ASCII format.

    Parameters
    ----------
    reduced_fits_image : str or path
        Path to the studied image in FITS format.
    cone_angle_deg : float, optional
        Angle of the cone within which SSO objects are searched, in degrees. The default is 0.25.
    mag_limit : int or float
        Highest (faintest) sources to consider in the catalog. The default is 19.
    source : str
        Source from which to query the PANSTARRS catalog from : whether 'Vizier' or 'MAST'. The default is 'Vizier'.

    Returns
    -------
    catalog : Table
        Calibrated sources from PANSTARRS in the input image.

    """

    # Read FITS
    with fits.open(reduced_fits_image) as hdul:
        hdu = hdul[0]
        wcs = WCS(hdu.header)
        field = SkyCoord(wcs.wcs.crval[0], wcs.wcs.crval[1], unit=u.deg)

        if "mast" in source.lower():
            # Query PanSTARRS catalog to MAST
            catalog = Catalogs.query_criteria(
                coordinates=field.to_string(),
                radius=cone_angle_deg,
                catalog="PANSTARRS",
                table="mean",
                data_release="dr2",
                nStackDetections=[("gte", 1)],
                rMeanPSFMag=[("lt", mag_limit), ("gt", 1)],
                rMeanPSFMagErr=[("lt", 0.02), ("gt", 0.0)],
                columns=["objName", "raMean", "decMean", "nDetections", "uMeanPSFMag", "uMeanPSFMagErr", "gMeanPSFMag", "gMeanPSFMagErr", "rMeanPSFMag", "rMeanPSFMagErr", "iMeanPSFMag", "iMeanPSFMagErr", "zMeanPSFMag", "zMeanPSFMagErr"],
            )
        elif "vizier" in source.lower():
            # Query PanSTARRS catalog to Vizier
            catalog = Vizier(row_limit=-1).query_region(field.to_string(), radius=Angle(cone_angle_deg, "deg"), catalog="II/349/ps1", column_filters={"imag": "1..21", "e_imag": "0..0.02"})
        else:
            raise ValueError(f"Unknown source {source} for catalog PANSTARRS. Acceptable sources are 'MAST' and 'Vizier'.\nPlease check input and re-run.")

    return catalog


# def calc_zeropoint(reduced_fits_image, sources, fwhm):
