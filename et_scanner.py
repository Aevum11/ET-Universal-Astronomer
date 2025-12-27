#!/usr/bin/env python3
"""
ET SCANNER - Manifold Scanner and Image Processing
===================================================
Exception Theory Universal Astronomer v7.5

This module provides 2D image processing and the manifold scanner.
In ET terms, this represents T (Traverser) - the indeterminate agency.

From Rules of Exception Law:
"T (Traverser) has cardinality [0/0] (indeterminate). T is agency."

The scanner traverses through the manifold, detecting binding events
and revealing the structure of data through active traversal.
"""

import numpy as np
import scipy.ndimage as ndimage

from .et_core import LOGGER
from .et_manifold import MANIFOLD


class ETConfig:
    """
    Configuration Matrix for the ET Manifold Scanner.
    
    ET Derivation:
    These are the Descriptors (D) that constrain how the Traverser (T)
    navigates the manifold. Each parameter is a finite constraint.
    """
    
    def __init__(self):
        self.saturation_threshold = None
        self.read_noise = 5.0
        self.plate_scale = None
        self.gain = 1.0
        self.flag_unstable = 1.5
        self.flag_smoothed = 0.5
        self.destripe = False
        self.fft_filter = False
        self.fft_auto_tune = True
        self.fft_threshold = 5.0
        self.auto_tune_gate = True
        self.force_float64 = True
        self.preservation_mode = True
        self.batch_size = 10


class ETManifoldScanner:
    """
    ET Manifold Scanner V21: The Unbound State.
    
    v7.5: Enhanced with dynamic manifold detection and export capabilities.
    
    ET Derivation:
    The scanner is the Traverser - it actively navigates through the
    descriptor space of the image, binding to features and revealing
    the underlying manifold structure.
    
    From Rules of Exception Law:
    "T's engagement is with (P°D) configurations. T actively navigates."
    """
    
    def __init__(self, config: ETConfig = None):
        self.cfg = config if config else ETConfig()
        self.scale = 1.0
        if self.cfg.plate_scale:
            self.scale = self.cfg.plate_scale
        self._cache_scale = -1.0
        self.V_FACTORS = None
        self.COV_FACTORS = None
        
        # Processing results for export
        self.last_results = {}
    
    def _update_geometry(self, header=None):
        """Update spatial geometry from header or config."""
        current_scale = 1.0
        if self.cfg.plate_scale:
            current_scale = self.cfg.plate_scale
        elif header:
            current_scale = self._parse_header_scale(header)
        
        if current_scale != self._cache_scale:
            self.scale = current_scale
            self.box_size = max(3, int(5 * self.scale))
            self.void_ds_factor = np.clip(0.25 / self.scale, 0.1, 0.5)
            self.V_FACTORS, self.COV_FACTORS = self._calculate_kernel_physics()
            self._cache_scale = current_scale
    
    def _parse_header_scale(self, header) -> float:
        """Parse plate scale from FITS header."""
        scale = 1.0
        keys = ['CDELT1', 'CD1_1', 'PIXSCAL', 'SECPIX', 'SCALE']
        for k in keys:
            if k in header:
                try:
                    val = abs(float(header[k]))
                    if val < 0.01:
                        val *= 3600.0
                    if val > 0:
                        return val
                except:
                    continue
        return scale
    
    def _calculate_kernel_physics(self):
        """
        ET-DERIVED: Calculate variance factors for different kernel types.
        
        The kernels represent different modes of T-navigation:
        - void: diffuse traversal (gaussian blur)
        - obj: point traversal (identity)
        - edge: gradient traversal (sharpening)
        """
        impulse = np.zeros((51, 51))
        impulse[25, 25] = 1.0
        
        k_void = ndimage.gaussian_filter(impulse, sigma=2.5)
        k_obj = impulse
        k_sharp = impulse + (impulse - ndimage.gaussian_filter(impulse, sigma=1.0)) * 0.7
        
        v_void = np.sum(k_void**2)
        v_obj = np.sum(k_obj**2)
        v_edge = np.sum(k_sharp**2)
        
        cov_void_obj = np.sum(k_void * k_obj) / np.sqrt(v_void * v_obj)
        cov_obj_edge = np.sum(k_obj * k_sharp) / np.sqrt(v_obj * v_edge)
        
        return {'void': v_void, 'obj': v_obj, 'edge': v_edge}, \
               {'void_obj': cov_void_obj, 'obj_edge': cov_obj_edge}
    
    def _fourier_filter(self, data: np.ndarray) -> np.ndarray:
        """
        Adaptive Fourier Manifold Filtering.
        
        ET Derivation:
        The Fourier domain reveals periodic structure in the manifold.
        Filtering removes descriptor noise while preserving T-binding signatures.
        """
        rows, cols = data.shape
        f_transform = np.fft.fft2(data)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        log_mag = np.log1p(magnitude)
        background_spec = ndimage.median_filter(log_mag, size=5)
        diff_spec = log_mag - background_spec
        
        if self.cfg.fft_auto_tune:
            std_spec = np.std(diff_spec)
            threshold = 5.0 + (std_spec * 2.0)
        else:
            threshold = self.cfg.fft_threshold
        
        mask_freq = diff_spec > threshold
        cy, cx = rows // 2, cols // 2
        y, x = np.ogrid[:rows, :cols]
        center_mask = ((x - cx)**2 + (y - cy)**2) < (min(rows, cols) * 0.05)**2
        mask_freq[center_mask] = False
        
        f_shift[mask_freq] *= 0.1
        img_back = np.fft.ifft2(np.fft.ifftshift(f_shift))
        return np.abs(img_back)
    
    def _destripe_tensor(self, data: np.ndarray) -> np.ndarray:
        """
        Remove row/column stripe patterns.
        
        ET Derivation:
        Stripes are systematic descriptor artifacts that mask T-binding.
        Removal reveals the true manifold structure.
        """
        row_med = np.median(data, axis=1, keepdims=True)
        col_med = np.median(data, axis=0, keepdims=True)
        stripes = row_med + col_med
        return data - stripes + np.mean(stripes)
    
    def _fast_noise_floor(self, data: np.ndarray):
        """
        Fast robust noise estimation.
        
        ET Derivation:
        The noise floor represents the base variance of the manifold.
        This is the "can be otherwise" floor for all measurements.
        """
        stride = int(25 * self.scale)
        sample = data[::stride, ::stride].flatten()
        mask = np.ones(sample.shape, dtype=bool)
        mean, std = np.median(sample), 0
        for _ in range(3):
            if np.sum(mask) < 10:
                break
            valid = sample[mask]
            diff = np.abs(valid - mean)
            std = 1.4826 * np.median(diff)
            if std == 0:
                std = 1e-5
            full_diff = np.abs(sample - mean)
            mask = mask & (full_diff < 3.0 * std)
            if np.sum(mask) < 10:
                break
            mean = np.mean(sample[mask])
        return std, mean
    
    def _manifold_healing(self, data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Interpolate bad pixels using manifold context.
        
        ET Derivation:
        Bad pixels are incoherent regions (Rule 16). We heal them by
        allowing the surrounding manifold structure to propagate inward,
        restoring traversability.
        """
        filled = data.copy()
        filled[mask] = ndimage.median_filter(data, size=3)[mask]
        for _ in range(3):
            blurred = ndimage.uniform_filter(filled, size=3)
            filled[mask] = blurred[mask]
        return filled
    
    def _adaptive_tensor(self, data: np.ndarray) -> np.ndarray:
        """
        ET-DERIVED: Adaptive structure tensor for coherence/edge detection.
        
        The structure tensor reveals the Descriptor-field (D-field):
        - High coherence = strong D binding
        - Low coherence = weak D binding (more T-navigation freedom)
        """
        Ix = np.diff(data, axis=1, append=0)
        Iy = np.diff(data, axis=0, append=0)
        Ix2_s = ndimage.uniform_filter(Ix**2, size=self.box_size)
        Iy2_s = ndimage.uniform_filter(Iy**2, size=self.box_size)
        Ixy_s = ndimage.uniform_filter(Ix*Iy, size=self.box_size)
        tr = Ix2_s + Iy2_s
        rad = np.sqrt(np.maximum(0, 4*Ixy_s**2 + (Ix2_s - Iy2_s)**2))
        coherence = rad / (tr + 1e-9)
        energy = tr
        p95 = np.percentile(energy, 95)
        if p95 > 0:
            energy /= p95
        return np.clip(coherence * np.sqrt(np.clip(energy, 0, 1)), 0, 1)
    
    def process_2d_slice(self, data: np.ndarray, mask: np.ndarray = None,
                         weight_map: np.ndarray = None, header=None,
                         scanner_gain: float = 1.2):
        """
        ET-DERIVED: Full 2D manifold processing with variance-propagated uncertainty.
        
        This is the main processing function that:
        1. Traverses the image manifold
        2. Detects binding strength (D-field)
        3. Calculates uncertainty from variance propagation
        4. Produces quality assessment
        
        From Rules of Exception Law:
        "E = (P°D°T)_substantiated" - Active existence requires all three.
        """
        self._update_geometry(header)
        dtype = np.float64 if self.cfg.force_float64 else np.float32
        data = np.asarray(data, dtype=dtype)
        epsilon = np.finfo(dtype).eps
        data += np.random.uniform(-0.5*epsilon, 0.5*epsilon, size=data.shape)
        
        LOGGER.log_image('processing', f"Processing 2D slice: shape={data.shape}")
        
        if self.cfg.fft_filter:
            data = self._fourier_filter(data)
            LOGGER.log_image('processing', "Applied FFT filter")
        if self.cfg.destripe:
            data = self._destripe_tensor(data)
            LOGGER.log_image('processing', "Applied destriping")
        
        is_bad = np.zeros_like(data, dtype=bool)
        if mask is not None:
            is_bad |= (mask > 0)
        if weight_map is not None:
            is_bad |= (weight_map <= 0)
        if np.any(is_bad):
            data_clean = self._manifold_healing(data, is_bad)
            LOGGER.log_image('processing', f"Healed {np.sum(is_bad)} bad pixels")
        else:
            data_clean = data
        
        noise_std, bg_mean = self._fast_noise_floor(data_clean)
        LOGGER.log_image('processing', f"Noise floor: std={noise_std:.4f}, bg={bg_mean:.4f}")
        
        snr_gate = 3.0
        if self.cfg.auto_tune_gate:
            dr = np.percentile(data_clean, 99.9) - bg_mean
            metric = dr / (noise_std + 1e-9)
            if metric > 1000:
                snr_gate = 2.5
            elif metric > 100:
                snr_gate = 3.0
            elif metric > 10:
                snr_gate = 4.0
            else:
                snr_gate = 5.0
        
        snr_map = (data_clean - bg_mean) / (noise_std + 1e-9)
        D_field = self._adaptive_tensor(data_clean)
        
        # Multi-layer blending using manifold-aligned layer counts
        fold = MANIFOLD.detected_fold
        layers = [fold, fold * 4, fold * 8]  # 12, 48, 96 or detected equivalents
        
        void_layer = ndimage.gaussian_filter(data_clean, sigma=layers[0]/4)
        obj_layer = data_clean
        edge_layer = data_clean + (data_clean - ndimage.gaussian_filter(data_clean, sigma=2)) * 0.5
        
        # Blending weights based on D-field
        w_void = 1.0 - D_field
        w_obj = D_field * (1 - np.clip((D_field - 0.5) * 2, 0, 1))
        w_edge = np.clip((D_field - 0.5) * 2, 0, 1)
        
        w_sum = w_void + w_obj + w_edge + 1e-9
        blended = (void_layer * w_void + obj_layer * w_obj + edge_layer * w_edge) / w_sum
        
        # Uncertainty map (variance-propagated)
        if self.V_FACTORS is None:
            self.V_FACTORS, self.COV_FACTORS = self._calculate_kernel_physics()
        
        unc_map = np.sqrt(
            (w_void/w_sum)**2 * self.V_FACTORS['void'] +
            (w_obj/w_sum)**2 * self.V_FACTORS['obj'] +
            (w_edge/w_sum)**2 * self.V_FACTORS['edge']
        ) * noise_std
        
        # Quality mask
        quality_mask = np.zeros_like(data, dtype=np.uint8)
        quality_mask[is_bad] = 2  # Bad pixels
        quality_mask[unc_map > noise_std * 2] = 1  # Uncertain
        
        # Apply scanner gain
        output = blended * scanner_gain
        
        LOGGER.log_image('processing', f"Complete: D-field mean={np.mean(D_field):.3f}")
        
        # Store results for export
        self.last_results = {
            'processed': output,
            'd_field': D_field,
            'uncertainty': unc_map,
            'quality': quality_mask,
            'snr_map': snr_map,
            'noise_std': noise_std,
            'bg_mean': bg_mean
        }
        
        return output, D_field, unc_map, quality_mask
    
    # Alias for backward compatibility
    def _process_2d_slice(self, *args, **kwargs):
        """Backward compatibility alias."""
        return self.process_2d_slice(*args, **kwargs)
