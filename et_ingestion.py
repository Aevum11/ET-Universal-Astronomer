#!/usr/bin/env python3
"""
ET INGESTION - Universal Data Ingestion Engine
===============================================
Exception Theory Universal Astronomer v7.5

This module handles file ingestion with comprehensive format support.
In ET terms, this is the process of bringing external configurations
into the manifold for analysis.

From Rules of Exception Law:
"More D from Other D" - Descriptors can reveal other descriptors.
File loading is the initial substantiation of data descriptors.
"""

import os
import struct
import json
from typing import Tuple, Optional
import numpy as np

from .et_core import LOGGER


class UniversalIngestor:
    """
    Handles file ingestion with comprehensive format support.
    
    ET Derivation:
    The ingestor is the interface between external data and the manifold.
    It converts raw bytes (unsubstantiated configurations) into
    structured arrays (substantiated descriptors) ready for T-navigation.
    
    Supports: FITS, NetCDF, CSV, TXT, DAT, JSON, and binary formats.
    """
    
    _last_wavelengths = None
    _last_data_type = None
    _last_header = None
    MAX_DISPLAY_POINTS = 50000
    
    @staticmethod
    def reset():
        """Reset ingestor state."""
        UniversalIngestor._last_wavelengths = None
        UniversalIngestor._last_data_type = None
        UniversalIngestor._last_header = None
        LOGGER.log_system("UniversalIngestor reset")
    
    @staticmethod
    def downsample_for_display(data: np.ndarray, wavelengths: np.ndarray = None,
                                max_points: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Intelligently downsample large datasets for display.
        
        ET Derivation:
        Large datasets exceed finite display constraints (D).
        Downsampling preserves essential structure while respecting
        the finite nature of visual representation.
        """
        if max_points is None:
            max_points = UniversalIngestor.MAX_DISPLAY_POINTS
        
        n = len(data)
        if n <= max_points:
            if wavelengths is None:
                return data, np.arange(n, dtype=np.float64)
            return data, wavelengths
        
        step = n // max_points
        indices = np.arange(0, n, step)[:max_points]
        
        data_ds = data[indices]
        
        if wavelengths is not None and len(wavelengths) == n:
            wave_ds = wavelengths[indices]
        else:
            wave_ds = np.arange(len(data_ds), dtype=np.float64)
        
        LOGGER.log_system(f"Downsampled {n:,} → {len(data_ds):,} points (step={step})")
        
        return data_ds, wave_ds
    
    @staticmethod
    def _construct_wavelength_from_header(header, n_pixels: int) -> Optional[np.ndarray]:
        """
        Construct wavelength array from FITS header WCS keywords.
        
        ET Derivation:
        The header contains descriptor information that allows us to
        reconstruct the wavelength coordinate system from first principles.
        """
        wave = None
        method = None
        
        if 'CRVAL1' in header:
            crval1 = float(header['CRVAL1'])
            crpix1 = float(header.get('CRPIX1', 1))
            
            if 'CDELT1' in header:
                cdelt1 = float(header['CDELT1'])
                method = "CDELT1"
            elif 'CD1_1' in header:
                cdelt1 = float(header['CD1_1'])
                method = "CD1_1"
            else:
                cdelt1 = None
            
            if cdelt1 is not None and cdelt1 != 0:
                wave = crval1 + (np.arange(n_pixels, dtype=np.float64) - (crpix1 - 1)) * cdelt1
                
                ctype1 = header.get('CTYPE1', '').upper()
                dc_flag = header.get('DC-FLAG', None)
                
                if 'LOG' in ctype1 or dc_flag == 1:
                    wave = 10.0 ** wave
                    method += " (log-linear)"
                
                LOGGER.log_system(f"Wavelength from {method}: CRVAL1={crval1}, CDELT1={cdelt1}")
        
        if wave is None and 'COEFF0' in header and 'COEFF1' in header:
            coeff0 = float(header['COEFF0'])
            coeff1 = float(header['COEFF1'])
            log_wave = coeff0 + coeff1 * np.arange(n_pixels, dtype=np.float64)
            wave = 10.0 ** log_wave
            method = "COEFF0/COEFF1 (SDSS log-linear)"
            LOGGER.log_system(f"Wavelength from {method}")
        
        if wave is not None:
            wmin, wmax = np.min(wave), np.max(wave)
            
            if wmin > 0 and wmax > wmin:
                if 100 <= wmin <= 50000 and 100 <= wmax <= 100000:
                    return wave
                else:
                    if wmax < 1:
                        wave_ang = wave * 1e10
                        if 1000 <= np.min(wave_ang) <= 50000:
                            return wave_ang
                    elif wmax < 100:
                        wave_ang = wave * 10000
                        if 1000 <= np.min(wave_ang) <= 100000:
                            return wave_ang
                    elif wmax < 10000 and wmax > 100:
                        wave_ang = wave * 10
                        if 1000 <= np.min(wave_ang) <= 100000:
                            return wave_ang
                    return wave
        
        return None
    
    @staticmethod
    def _is_2d_image(header, data_shape) -> bool:
        """
        Determine if FITS data is a 2D image vs 1D spectrum.
        
        ET Derivation:
        2D images are manifold slices (spatial descriptors in 2 dimensions).
        1D spectra are manifold cuts (wavelength descriptor traversal).
        """
        if len(data_shape) != 2:
            return False
        
        if data_shape[0] <= 1 or data_shape[1] <= 1:
            return False
        
        ctype1 = str(header.get('CTYPE1', '')).upper()
        spatial_types = ['RA', 'DEC', 'GLON', 'GLAT', 'ELON', 'ELAT', 'PIXEL', 'LINEAR']
        spectral_types = ['WAVE', 'FREQ', 'VELO', 'AWAV', 'LAMBDA', 'LOG']
        
        for spec in spectral_types:
            if spec in ctype1:
                return False
        
        for spat in spatial_types:
            if spat in ctype1:
                return True
        
        if min(data_shape) > 10:
            ratio = max(data_shape) / min(data_shape)
            if ratio < 100:
                return True
        
        return False
    
    @staticmethod
    def _try_astropy_fits(filepath: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Attempt to load FITS using astropy."""
        try:
            from astropy.io import fits
            
            with fits.open(filepath) as hdul:
                LOGGER.log_system(f"FITS structure: {len(hdul)} extensions")
                
                # Check for 2D image data first
                if hdul[0].data is not None:
                    data_shape = hdul[0].data.shape
                    if UniversalIngestor._is_2d_image(hdul[0].header, data_shape):
                        LOGGER.log_system(f"Detected 2D image: {data_shape}")
                        image_data = hdul[0].data.astype(np.float64)
                        UniversalIngestor._last_data_type = '2D'
                        UniversalIngestor._last_header = dict(hdul[0].header)
                        return (None, image_data)
                
                # Look for binary table with spectral data
                for ext in hdul:
                    if hasattr(ext, 'columns') and ext.columns is not None:
                        col_names = [c.name.upper() for c in ext.columns]
                        
                        wave_col = None
                        for name in ['WAVELENGTH', 'WAVE', 'LAMBDA', 'LOGLAM']:
                            if name in col_names:
                                wave_col = name
                                break
                        
                        flux_col = None
                        for name in ['FLUX', 'SPEC', 'DATA', 'COUNTS', 'INTENSITY']:
                            if name in col_names:
                                flux_col = name
                                break
                        
                        if wave_col and flux_col:
                            wave_data = ext.data[wave_col].astype(np.float64).flatten()
                            flux_data = ext.data[flux_col].astype(np.float64).flatten()
                            
                            if 'LOG' in wave_col.upper():
                                wave_data = 10.0 ** wave_data
                            
                            flux_data = np.where(np.isfinite(flux_data), flux_data, np.nan)
                            wave_data = np.where(np.isfinite(wave_data), wave_data, np.nan)
                            
                            UniversalIngestor._last_data_type = '1D'
                            return (wave_data, flux_data)
                
                # Primary HDU image data
                if hdul[0].data is not None:
                    data = hdul[0].data.astype(np.float64).flatten()
                    data = np.where(np.isfinite(data), data, np.nan)
                    
                    wave = UniversalIngestor._construct_wavelength_from_header(
                        hdul[0].header, len(data))
                    
                    if wave is None:
                        wave = np.arange(len(data), dtype=np.float64)
                    
                    UniversalIngestor._last_data_type = '1D'
                    return (wave, data)
            
            return None
            
        except ImportError:
            LOGGER.log_system("astropy not available, using manual parser")
            return None
        except Exception as e:
            LOGGER.log_system(f"FITS error: {e}")
            return None
    
    @staticmethod
    def _try_netcdf(filepath: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Attempt to load NetCDF files."""
        try:
            import netCDF4 as nc
            
            ds = nc.Dataset(filepath, 'r')
            LOGGER.log_system(f"NetCDF variables: {list(ds.variables.keys())}")
            
            # Check for 2D image
            image_names = ['image', 'IMAGE', 'data', 'DATA', 'counts', 'COUNTS',
                           'intensity', 'INTENSITY', 'band', 'BAND']
            for img_name in image_names:
                if img_name in ds.variables:
                    img_var = ds.variables[img_name]
                    img_shape = img_var.shape
                    if len(img_shape) == 2 and min(img_shape) > 10:
                        image_data = img_var[:].astype(np.float64)
                        if hasattr(image_data, 'data'):
                            image_data = np.array(image_data.data)
                        ds.close()
                        UniversalIngestor._last_data_type = '2D'
                        UniversalIngestor._last_header = {}
                        return (None, image_data)
            
            # Find spectral data
            wave_names = ['wavelength', 'Vacuum Wavelength', 'wave', 'lambda', 'lam',
                          'WAVELENGTH', 'WAVE', 'LAMBDA', 'wl']
            wave_var = None
            wave_name = None
            
            for name in wave_names:
                if name in ds.variables:
                    wave_var = ds.variables[name]
                    wave_name = name
                    break
            
            if wave_var is None:
                for var_name in ds.variables.keys():
                    if 'wave' in var_name.lower():
                        wave_var = ds.variables[var_name]
                        wave_name = var_name
                        break
            
            flux_names = ['SSI', 'flux', 'FLUX', 'irradiance', 'IRRADIANCE',
                          'intensity', 'INTENSITY', 'data', 'DATA', 'counts', 'spec']
            flux_var = None
            
            for name in flux_names:
                if name in ds.variables:
                    flux_var = ds.variables[name]
                    break
            
            if wave_var is None or flux_var is None:
                ds.close()
                return None
            
            wave_data = wave_var[:].astype(np.float64)
            flux_data = flux_var[:].astype(np.float64)
            
            if len(wave_data.shape) > 1:
                wave_data = wave_data.flatten()
            if len(flux_data.shape) > 1:
                flux_data = flux_data.flatten()
            
            # Handle units
            wave_unit = getattr(wave_var, 'units', None)
            if wave_unit:
                if 'nm' in wave_unit.lower():
                    wave_data = wave_data * 10.0
                elif 'um' in wave_unit.lower() or 'micron' in wave_unit.lower():
                    wave_data = wave_data * 10000.0
                elif 'm' in wave_unit.lower() and 'nm' not in wave_unit.lower():
                    wave_data = wave_data * 1e10
            else:
                wmin, wmax = np.nanmin(wave_data), np.nanmax(wave_data)
                if wmax < 10:
                    wave_data = wave_data * 10000.0
                elif wmax < 3000:
                    wave_data = wave_data * 10.0
            
            flux_data = np.where(np.isfinite(flux_data), flux_data, np.nan)
            wave_data = np.where(np.isfinite(wave_data), wave_data, np.nan)
            
            if hasattr(flux_data, 'data'):
                flux_data = np.array(flux_data.data)
            if hasattr(wave_data, 'data'):
                wave_data = np.array(wave_data.data)
            
            ds.close()
            UniversalIngestor._last_data_type = '1D'
            return (wave_data, flux_data)
            
        except ImportError:
            LOGGER.log_system("netCDF4 not available")
            return None
        except Exception as e:
            LOGGER.log_system(f"NetCDF error: {e}")
            return None
    
    @staticmethod
    def load(filepath: str) -> np.ndarray:
        """
        Main entry point for file loading.
        
        ET Derivation:
        This is the substantiation operation - bringing external data
        into the manifold as analyzable descriptors.
        """
        ext = os.path.splitext(filepath)[1].lower()
        vals = []
        wavelengths = None
        
        LOGGER.log_system(f"Loading file: {filepath}")
        
        try:
            if ext in ['.csv', '.txt', '.dat', '.tsv']:
                with open(filepath, 'r', errors='ignore') as f:
                    for line in f:
                        parts = line.replace(',', ' ').replace(';', ' ').split()
                        for p in parts:
                            try:
                                val = np.float64(p)
                                vals.append(val)
                            except ValueError:
                                pass
                
            elif ext == '.json':
                with open(filepath, 'r') as f:
                    d = json.load(f)
                    
                    def flatten(x):
                        if isinstance(x, (int, float)):
                            return [np.float64(x)]
                        if isinstance(x, list):
                            return sum([flatten(i) for i in x], [])
                        if isinstance(x, dict):
                            return sum([flatten(i) for i in x.values()], [])
                        return []
                    
                    vals = flatten(d)
            
            elif ext == '.nc':
                result = UniversalIngestor._try_netcdf(filepath)
                
                if result is not None:
                    wavelengths, flux = result
                    
                    if wavelengths is None and flux is not None and len(flux.shape) == 2:
                        UniversalIngestor._last_wavelengths = None
                        return flux
                    
                    UniversalIngestor._last_wavelengths = wavelengths
                    vals = flux.tolist() if hasattr(flux, 'tolist') else list(flux)
            
            elif ext in ['.fits', '.fit']:
                result = UniversalIngestor._try_astropy_fits(filepath)
                
                if result is not None:
                    wavelengths, flux = result
                    
                    if wavelengths is None and flux is not None and len(flux.shape) == 2:
                        UniversalIngestor._last_wavelengths = None
                        return flux
                    
                    UniversalIngestor._last_wavelengths = wavelengths
                    vals = flux.tolist() if hasattr(flux, 'tolist') else list(flux)
            
            else:
                with open(filepath, 'rb') as f:
                    raw = f.read()
                    for i in range(0, len(raw) - 3, 4):
                        try:
                            vals.append(struct.unpack('<f', raw[i:i+4])[0])
                        except:
                            pass
            
        except Exception as e:
            LOGGER.log_system(f"Error loading file: {e}")
        
        if not vals:
            vals = [0.0]
        
        arr = np.array(vals, dtype=np.float64)
        
        if wavelengths is None:
            UniversalIngestor._last_wavelengths = None
        
        LOGGER.log_system(f"Loaded {len(arr)} data points")
        
        return arr
