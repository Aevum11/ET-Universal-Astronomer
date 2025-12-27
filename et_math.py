#!/usr/bin/env python3
"""
ET MATH - The Mathematical Core
================================
Exception Theory Universal Astronomer v7.5

This module provides the immutable mathematical axioms of Exception Theory.
In ET terms, this represents D (Descriptor) - the finite constraints.

From Rules of Exception Law:
"D (Descriptor) has cardinality n (finite). D is the constraint."

MATHEMATICAL FOUNDATIONS:
- Rule 3: P (Point) is infinite substrate
- Rule 4: D (Descriptor) is finite constraint; time, space, causality are D
- Rule 5: T (Traverser) is indeterminate; consciousness, entanglement, gravity are T
- Rule 11: S = (P°D°T); Mathematics involves all three primitives
- Rule 16: Limits as Traversal; T IS the limit operation, |T| = [0/0]
"""

import math
from typing import List, Tuple, Dict, Optional
import numpy as np
import scipy.signal
import scipy.ndimage as ndimage

from .et_core import LOGGER
from .et_manifold import MANIFOLD


class ETMath:
    """
    The immutable axioms of Exception Theory applied to Data Analysis.
    
    v7.5: Now uses ManifoldGeometry for dynamic fold handling.
    
    MATHEMATICAL FOUNDATIONS (from Rules of Exception Law):
    - Rule 3: P (Point) is infinite substrate
    - Rule 4: D (Descriptor) is finite constraint; time, space, causality are D
    - Rule 5: T (Traverser) is indeterminate; consciousness, entanglement, gravity are T
    - Rule 11: S = (P°D°T); Mathematics involves all three primitives
    - Rule 16: Limits as Traversal; T IS the limit operation, |T| = [0/0]
    """
    
    # Reference to global manifold geometry
    _manifold = MANIFOLD
    
    # =========================================================================
    # DYNAMIC CONSTANTS (Now derived from ManifoldGeometry)
    # =========================================================================
    
    @classmethod
    def get_manifold_symmetry(cls, level: int = None) -> int:
        """Get manifold symmetry at specified level (None = detected)."""
        if level is None:
            return cls._manifold.detected_fold
        return cls._manifold.get_fold(level)
    
    @classmethod
    def get_manifold_resonance(cls, fold: int = None) -> float:
        """Get base variance = 1/fold."""
        return cls._manifold.get_variance(fold)
    
    @classmethod
    def get_resonance_threshold(cls, fold: int = None) -> float:
        """Get subliminal detection threshold."""
        return cls._manifold.get_resonance_threshold(fold)
    
    @classmethod
    def get_gaze_threshold(cls, fold: int = None) -> float:
        """Get conscious detection threshold."""
        return cls._manifold.get_gaze_threshold(fold)
    
    # Legacy static properties for backward compatibility
    MANIFOLD_SYMMETRY = property(lambda self: MANIFOLD.detected_fold)
    MANIFOLD_RESONANCE = property(lambda self: MANIFOLD.detected_variance)
    RESONANCE_THRESHOLD = property(lambda self: MANIFOLD.get_resonance_threshold())
    GAZE_THRESHOLD = property(lambda self: MANIFOLD.get_gaze_threshold())
    
    # Static values for direct access (will be updated by detection)
    @staticmethod
    def _get_static_manifold_symmetry():
        return MANIFOLD.detected_fold
    
    @staticmethod
    def _get_static_manifold_resonance():
        return MANIFOLD.detected_variance
    
    # =========================================================================
    # PHYSICAL CONSTANTS (PURE PHYSICS - unchanged)
    # =========================================================================
    
    # Gravitational constant in galactic units: kpc (km/s)² / M_sun
    G_GALACTIC = 4.302e-6
    
    # Gravitational constant in SI units: m³ kg⁻¹ s⁻²
    G_SI = 6.67430e-11
    
    # Binding threshold for spectral line detection
    BINDING_THRESHOLD = 0.80
    
    # =========================================================================
    # RESOLUTION CLASSIFICATION SYSTEM (v6.7)
    # =========================================================================
    
    RES_BUCKET_THRESHOLD = 1.0    # >1Å = low resolution ("bucket")
    RES_CUP_THRESHOLD = 0.1       # <0.1Å = high resolution ("tiny cup")
    
    @staticmethod
    def classify_resolution(wavelengths: np.ndarray) -> dict:
        """Classify spectral resolution and return adaptive parameters."""
        if wavelengths is None or len(wavelengths) < 2:
            return {
                'class': 'UNKNOWN',
                'step_angstroms': float('nan'),
                'detection_threshold': 0.05,
                'resolution_factor': 0.25,
                'expected_accuracy': 0.0,
                'description': 'No wavelength calibration'
            }
        
        steps = np.abs(np.diff(wavelengths))
        step = float(np.median(steps[steps > 0])) if np.any(steps > 0) else 0.0
        
        if step > 5.0:
            res_class = 'BUCKET'
            detection_thresh = 0.03
            res_factor = max(0.05, 1.0 / step)
            expected_accuracy = max(0.10, 0.35 - (step / 100))
            description = f"LOW RES ({step:.1f}Å) - Absorption diluted ~{step:.0f}×"
            
        elif step > ETMath.RES_BUCKET_THRESHOLD:
            res_class = 'BUCKET'
            detection_thresh = 0.04
            res_factor = max(0.2, 1.0 / step)
            expected_accuracy = max(0.25, 0.55 - step * 0.06)
            description = f"LOW-MED RES ({step:.2f}Å) - Lines partially resolved"
            
        elif step > ETMath.RES_CUP_THRESHOLD:
            res_class = 'BOWL'
            detection_thresh = 0.05
            res_factor = max(0.5, 1.0 - step * 0.5)
            expected_accuracy = 0.55 + (1.0 - step) * 0.30
            description = f"MED RES ({step:.3f}Å) - Most lines well resolved"
            
        elif step > 0.01:
            res_class = 'CUP'
            detection_thresh = 0.05
            res_factor = 0.90 + step
            expected_accuracy = 0.85 + (0.1 - step)
            description = f"HIGH RES ({step:.4f}Å) - Full line profiles visible"
            
        else:
            res_class = 'MICRO'
            detection_thresh = 0.05
            res_factor = 1.0
            expected_accuracy = 0.95
            description = f"ULTRA RES ({step:.5f}Å) - Sub-line structure visible"
        
        return {
            'class': res_class,
            'step_angstroms': step,
            'detection_threshold': detection_thresh,
            'resolution_factor': res_factor,
            'expected_accuracy': expected_accuracy,
            'description': description
        }
    
    # Koide formula constant (ET predicts this from 2/3 ratio)
    KOIDE_CONSTANT = 2.0 / 3.0
    
    # Dark matter/energy geometric ratios (from ET manifold structure)
    DARK_ENERGY_RATIO = 2.0 / 3.0      # ~68% of universe
    DARK_MATTER_RATIO = 1.0 / 4.0      # ~25% (approximate)
    BARYONIC_RATIO = 1.0 / 12.0        # ~8% (1/12 of visible)
    
    # =========================================================================
    # DYNAMICALLY DISCOVERED ABSORPTION LINES
    # =========================================================================
    
    # Known absorption lines (conventional astrophysics data for comparison)
    CHEMO_RESONANCE_MAP = {
        6562.8: ("Hydrogen (H-α)", 0.82),
        4861.3: ("Hydrogen (H-β)", 0.75),
        4340.5: ("Hydrogen (H-γ)", 0.68),
        5889.9: ("Sodium (Na D₁)", 0.70),
        5895.9: ("Sodium (Na D₂)", 0.68),
        5167.3: ("Magnesium (Mg b₁)", 0.65),
        5172.7: ("Magnesium (Mg b₂)", 0.63),
        5183.6: ("Magnesium (Mg b₄)", 0.61),
        4300.0: ("Carbon (G-Band)", 0.60),
        3933.7: ("Calcium (Ca K)", 0.88),
        3968.5: ("Calcium (Ca H)", 0.85),
        8542.1: ("Calcium (Ca IR)", 0.55),
        6300.3: ("Oxygen [OI]", 0.15),
        7600.0: ("Oxygen (Telluric A)", 0.0),
        6867.0: ("Oxygen (Telluric B)", 0.0),
        4271.8: ("Iron (Fe I)", 0.45),
        5270.4: ("Iron (Fe I)", 0.50),
    }
    
    @classmethod
    def discover_absorption_lines(cls, wavelengths: np.ndarray, binding: np.ndarray,
                                   threshold: float = None) -> Dict[float, Tuple[str, float]]:
        """
        Dynamically discover absorption lines from data.
        
        ET Derivation: Strong binding regions indicate T-mediated absorption.
        Lines are found, not forced, then matched to known patterns.
        """
        if threshold is None:
            # Use manifold-derived threshold
            threshold = cls.get_manifold_resonance() * 3  # ~0.25 for 12-fold
        
        discovered = {}
        
        if wavelengths is None or len(wavelengths) < 10 or len(binding) < 10:
            return discovered
        
        # Find local maxima in binding (absorption features)
        binding_smooth = ndimage.uniform_filter1d(binding, size=5)
        
        # Use gradient to find peaks
        d1 = np.gradient(binding_smooth)
        d2 = np.gradient(d1)
        
        # Peaks where first derivative crosses zero and second derivative is negative
        for i in range(2, len(binding_smooth) - 2):
            if binding_smooth[i] > threshold:
                if d1[i-1] > 0 and d1[i+1] < 0:  # Peak
                    if d2[i] < 0:  # Confirmed maximum
                        wv = wavelengths[i]
                        depth = binding_smooth[i]
                        
                        # Try to match to known line
                        matched = False
                        for known_wv, (name, _) in cls.CHEMO_RESONANCE_MAP.items():
                            if abs(wv - known_wv) < 5.0:  # Within 5Å
                                discovered[wv] = (name, depth)
                                matched = True
                                break
                        
                        if not matched:
                            # Unknown line - classify by wavelength region
                            if wv < 4000:
                                region = "UV"
                            elif wv < 5000:
                                region = "Blue"
                            elif wv < 6000:
                                region = "Green"
                            elif wv < 7000:
                                region = "Red"
                            else:
                                region = "IR"
                            discovered[wv] = (f"Unknown ({region} @ {wv:.1f}Å)", depth)
        
        LOGGER.log_manifold(f"Discovered {len(discovered)} absorption features")
        return discovered
    
    # =========================================================================
    # EFFICIENT VARIANCE CALCULATIONS (Memory Variance Solution)
    # =========================================================================
    
    @staticmethod
    def efficient_variance(data: np.ndarray, window: int = None) -> np.ndarray:
        """
        Efficient running variance calculation using Welford's algorithm.
        
        ET Derivation: Variance measures "can be otherwise" degree.
        Efficient calculation enables real-time manifold analysis.
        """
        n = len(data)
        if n < 2:
            return np.zeros(n)
        
        if window is None:
            # Use manifold-aligned window
            window = max(MANIFOLD.base_symmetry, n // 12)
        
        window = min(window, n)
        
        # Use cumulative sums for O(n) calculation
        cumsum = np.cumsum(data)
        cumsum_sq = np.cumsum(data ** 2)
        
        # Pad for windowed calculation
        cumsum = np.concatenate([[0], cumsum])
        cumsum_sq = np.concatenate([[0], cumsum_sq])
        
        # Calculate windowed variance
        variance = np.zeros(n)
        half_win = window // 2
        
        for i in range(n):
            start = max(0, i - half_win)
            end = min(n, i + half_win + 1)
            count = end - start
            
            if count > 1:
                sum_val = cumsum[end] - cumsum[start]
                sum_sq = cumsum_sq[end] - cumsum_sq[start]
                mean = sum_val / count
                variance[i] = (sum_sq / count) - (mean ** 2)
            else:
                variance[i] = 0.0
        
        return variance
    
    @staticmethod
    def efficient_cv_squared(data: np.ndarray) -> float:
        """
        Efficient coefficient of variation squared.
        
        CV² = Var(X) / μ² - the scale-invariant variance measure.
        This is the key metric for manifold alignment detection.
        """
        clean = data[np.isfinite(data)]
        if len(clean) < 2:
            return 0.0
        
        mean = np.mean(clean)
        if abs(mean) < 1e-12:
            return 0.0
        
        var = np.var(clean)
        return var / (mean ** 2)
    
    # =========================================================================
    # PURE NEWTONIAN PHYSICS (BASE MATH - NO ET MODIFICATIONS)
    # =========================================================================
    
    @staticmethod
    def newtonian_orbital_velocity(M: float, r: float) -> float:
        """Pure Newtonian orbital velocity: v = √(GM/r)"""
        if r <= 0:
            return 0.0
        return math.sqrt(ETMath.G_GALACTIC * M / r)
    
    @staticmethod
    def newtonian_gravitational_acceleration(M: float, r: float) -> float:
        """Pure Newtonian gravitational acceleration: a = GM/r²"""
        if r <= 0:
            return 0.0
        return ETMath.G_GALACTIC * M / (r * r)
    
    @staticmethod
    def newtonian_escape_velocity(M: float, r: float) -> float:
        """Pure Newtonian escape velocity: v_esc = √(2GM/r)"""
        if r <= 0:
            return 0.0
        return math.sqrt(2.0 * ETMath.G_GALACTIC * M / r)
    
    @staticmethod
    def newtonian_orbital_period(a: float, M: float) -> float:
        """Pure Newtonian orbital period (Kepler's 3rd Law): T = 2π√(a³/GM)"""
        if M <= 0:
            return float('inf')
        return 2.0 * math.pi * math.sqrt(a**3 / (ETMath.G_GALACTIC * M))
    
    @staticmethod
    def newtonian_schwarzschild_radius(M: float) -> float:
        """Schwarzschild radius: r_s = 2GM/c²"""
        c_km_s = 299792.458
        G_km = 1.327e11
        return 2.0 * G_km * M / (c_km_s * c_km_s)
    
    # =========================================================================
    # ET-DERIVED PHYSICS (Calculations from P-D-T Primitives)
    # =========================================================================
    
    @staticmethod
    def et_binding_strength(flux: np.ndarray, wavelengths: np.ndarray = None) -> np.ndarray:
        """
        ET-DERIVED: Converts Flux (Descriptor intensity) to Binding Strength (T_gravity).
        
        v7.5: Uses efficient windowed calculation with manifold-aligned window.
        """
        if len(flux) == 0:
            return np.array([])
        
        # Handle negative flux (calibration artifacts)
        flux_safe = np.where(np.isfinite(flux), flux, np.nan)
        flux_min = np.nanmin(flux_safe)
        
        if flux_min < 0:
            offset = abs(flux_min) + 1e-10
            flux_safe = flux_safe + offset
            LOGGER.log_manifold(f"Binding: Applied offset {offset:.4e} for negative flux")
        
        # Determine window size based on wavelength or manifold
        if wavelengths is not None and len(wavelengths) > 1:
            wv_step = np.median(np.abs(np.diff(wavelengths)))
            window_points = max(5, int(100 / max(wv_step, 0.01)))
        else:
            # Use manifold-aligned window
            window_points = max(5, len(flux) // MANIFOLD.base_symmetry)
        
        # Efficient local continuum estimation
        try:
            from scipy.ndimage import maximum_filter1d, uniform_filter1d
            local_max = maximum_filter1d(flux_safe, size=window_points, mode='nearest')
            local_continuum = uniform_filter1d(local_max, size=window_points//2 + 1, mode='nearest')
        except ImportError:
            local_continuum = np.zeros_like(flux_safe)
            half_win = window_points // 2
            for i in range(len(flux_safe)):
                start = max(0, i - half_win)
                end = min(len(flux_safe), i + half_win + 1)
                local_continuum[i] = np.nanmax(flux_safe[start:end])
        
        local_continuum = np.maximum(local_continuum, flux_safe * 1.001)
        local_continuum = np.where(local_continuum > 0, local_continuum, 1.0)
        
        binding = 1.0 - (flux_safe / local_continuum)
        binding = np.clip(binding, 0.0, 1.0)
        
        LOGGER.log_manifold(f"Binding: Window={window_points}, Range={np.nanmin(binding)*100:.1f}%-{np.nanmax(binding)*100:.1f}%")
        
        return binding
    
    @staticmethod
    def et_variance_flow(data: np.ndarray) -> float:
        """
        ET-DERIVED: Calculates descriptor variance (degree of "can be otherwise").
        
        This is CV² = σ²/μ² - the normalized variance.
        """
        return ETMath.efficient_cv_squared(data)
    
    @staticmethod
    def et_jerk_intensity(data: np.ndarray) -> float:
        """
        ET-DERIVED: Calculates Agency Signature (3rd derivative intensity).
        
        3rd derivative = jerk = agency signature.
        """
        if len(data) < 5:
            return 0.0
        
        d_min, d_max = np.nanmin(data), np.nanmax(data)
        if d_max == d_min:
            return 0.0
        
        normalized = (data - d_min) / (d_max - d_min)
        
        d1 = np.gradient(normalized)
        d2 = np.gradient(d1)
        d3 = np.gradient(d2)
        
        return np.nanmean(np.abs(d3)) * 100.0
    
    @staticmethod
    def et_rotation_velocity(r: float, M_local: float, include_variance: bool = True) -> Tuple[float, float]:
        """
        ET-DERIVED: Galactic rotation velocity with non-local binding.
        
        Uses dynamically detected manifold variance.
        """
        if r <= 0:
            return (0.0, 0.0)
        
        v_newt = math.sqrt(ETMath.G_GALACTIC * M_local / r)
        
        if not include_variance:
            return (v_newt, v_newt)
        
        # Use detected manifold variance
        manifold_resonance = MANIFOLD.detected_variance
        COUPLING_SCALE = 1e9
        
        M_variance = (r * r) * manifold_resonance * COUPLING_SCALE
        M_effective = M_local + M_variance
        
        v_et = math.sqrt(ETMath.G_GALACTIC * M_effective / r)
        
        return (v_newt, v_et)
    
    @staticmethod
    def et_time_dilation(variance: float) -> float:
        """ET-DERIVED: Time dilation factor from descriptor variance."""
        if variance <= 0:
            return 1.0
        if variance >= 1.0:
            return 1.0
        return 1.0 / (variance + 1e-12)
    
    @staticmethod
    def et_gaze_pressure(T_intent: float, focus: float, distance: float) -> float:
        """ET-DERIVED: Calculate "feeling of being watched" pressure."""
        if distance <= 0:
            distance = 1e-12
        return (T_intent * focus) / (distance * distance)
    
    # =========================================================================
    # SIGNAL CLASSIFICATION
    # =========================================================================
    
    @staticmethod
    def classify_t_signature(binding_data: np.ndarray) -> dict:
        """
        Classifies T-types based on 'Rules of Exception Law' Rule 5.
        
        v7.5: Uses dynamically detected manifold parameters.
        """
        results = {
            "T_Gravity": 0.0,
            "T_Agency": 0.0,
            "Manifold_Align": False,
            "Manifold_Fold": MANIFOLD.detected_fold,
            "Variance": 0.0,
            "Type": "UNKNOWN"
        }
        
        if len(binding_data) < 10:
            return results
        
        clean = binding_data[np.isfinite(binding_data)]
        if len(clean) < 10:
            return results
        
        # Detect manifold fold from binding data
        detected_fold = MANIFOLD.detect_fold_from_data(clean)
        results["Manifold_Fold"] = detected_fold
        
        # T_Gravity: Fraction of deep binding points
        deep_points = np.sum(clean > ETMath.BINDING_THRESHOLD)
        results["T_Gravity"] = deep_points / len(clean)
        
        # T_Agency: Jerk intensity
        d1 = np.gradient(clean)
        d2 = np.gradient(d1)
        d3 = np.gradient(d2)
        results["T_Agency"] = np.nanstd(d3) * 100.0
        
        # Variance and Manifold Alignment
        signal_variance = np.nanvar(clean)
        results["Variance"] = signal_variance
        
        # Check alignment with detected fold
        expected_variance = 1.0 / detected_fold
        deviation = abs(signal_variance - expected_variance)
        if deviation < (expected_variance * 0.15):
            results["Manifold_Align"] = True
        
        # Classification
        if results["T_Gravity"] > 0.01:
            results["Type"] = "TYPE-G (GRAVITY/STAR)"
        elif results["T_Agency"] > 5.0:
            results["Type"] = "TYPE-I (AGENCY/NAV)"
        elif results["Manifold_Align"]:
            results["Type"] = f"TYPE-E (RESONANT @ {detected_fold}-fold)"
        else:
            results["Type"] = "TYPE-D (CHAOS)"
        
        return results
    
    # =========================================================================
    # SIGNAL DECOMPOSITION
    # =========================================================================
    
    @staticmethod
    def decompose_signal(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Separates signal into: Gravity Wave, Agency Wave, Chaos Field."""
        n = len(data)
        if n < 5:
            return data.copy(), np.zeros_like(data), np.zeros_like(data)
        
        window = n // 5
        if window % 2 == 0:
            window += 1
        window = max(5, min(window, n - 2))
        if window >= n:
            window = n - 1 if n % 2 == 1 else n - 2
            if window < 5:
                window = 5 if n >= 5 else n
        
        gravity = np.zeros_like(data, dtype=float)
        try:
            if n >= window and window >= 5:
                gravity = scipy.signal.savgol_filter(data, window, min(3, window-2))
            else:
                kernel = np.ones(min(3, n)) / min(3, n)
                gravity = np.convolve(data, kernel, mode='same')
        except Exception:
            gravity = data.copy()
        
        residual = data - gravity
        
        try:
            d2 = np.gradient(np.gradient(residual))
            threshold = np.nanstd(d2) * 1.5
            agency = np.where(np.abs(d2) > threshold, residual, 0.0)
        except Exception:
            agency = np.zeros_like(residual)
        
        chaos = residual - agency
        
        return gravity, agency, chaos
    
    # =========================================================================
    # ORBIT PREDICTION
    # =========================================================================
    
    @staticmethod
    def predict_orbit_newtonian(pos: np.ndarray, vel: np.ndarray, mass: float,
                                 steps: int = 100, dt: float = 0.01) -> np.ndarray:
        """PURE NEWTONIAN orbit prediction."""
        G = ETMath.G_GALACTIC
        trajectory = [pos.copy()]
        p = pos.copy().astype(float)
        v = vel.copy().astype(float)
        
        for _ in range(steps):
            r = np.linalg.norm(p)
            if r < 1e-6:
                r = 1e-6
            a = -G * mass * p / (r ** 3)
            v += a * dt
            p += v * dt
            trajectory.append(p.copy())
        
        return np.array(trajectory)
    
    @staticmethod
    def predict_orbit_et(pos: np.ndarray, vel: np.ndarray, mass: float,
                          steps: int = 100, dt: float = 0.01) -> np.ndarray:
        """ET-CORRECTED orbit prediction with manifold variance field."""
        G = ETMath.G_GALACTIC
        trajectory = [pos.copy()]
        p = pos.copy().astype(float)
        v = vel.copy().astype(float)
        
        # Use detected manifold variance
        manifold_resonance = MANIFOLD.detected_variance
        COUPLING_SCALE = 1e9
        
        for _ in range(steps):
            r = np.linalg.norm(p)
            if r < 1e-6:
                r = 1e-6
            
            # Newtonian acceleration
            a_newton = -G * mass * p / (r ** 3)
            
            # ET variance field contribution
            M_variance = (r * r) * manifold_resonance * COUPLING_SCALE
            a_variance = -G * M_variance * p / (r ** 3)
            
            a_total = a_newton + a_variance * 0.1  # Subtle contribution
            
            v += a_total * dt
            p += v * dt
            trajectory.append(p.copy())
        
        return np.array(trajectory)
    
    # =========================================================================
    # GALACTIC ROTATION CURVE
    # =========================================================================
    
    @staticmethod
    def solve_rotation_curve(r: np.ndarray, m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Newtonian and ET rotation curves.
        
        v7.5: Uses dynamically detected manifold parameters.
        """
        v_newt = np.zeros_like(r)
        v_et = np.zeros_like(r)
        
        manifold_resonance = MANIFOLD.detected_variance
        COUPLING_SCALE = 1e9
        
        for i in range(len(r)):
            if r[i] > 0:
                v_newt[i] = math.sqrt(ETMath.G_GALACTIC * m[i] / r[i])
                
                M_variance = (r[i] ** 2) * manifold_resonance * COUPLING_SCALE
                M_effective = m[i] + M_variance
                v_et[i] = math.sqrt(ETMath.G_GALACTIC * M_effective / r[i])
        
        return v_newt, v_et
    
    # =========================================================================
    # WRAPPER METHODS
    # =========================================================================
    
    @staticmethod
    def calculate_binding(flux_stream: np.ndarray, wavelengths: np.ndarray = None,
                          continuum_relative: bool = True) -> np.ndarray:
        """Wrapper for et_binding_strength."""
        if continuum_relative:
            return ETMath.et_binding_strength(flux_stream, wavelengths)
        else:
            if len(flux_stream) == 0:
                return np.array([])
            valid = flux_stream[np.isfinite(flux_stream)]
            if len(valid) == 0:
                return np.zeros_like(flux_stream)
            f_min, f_max = np.nanmin(flux_stream), np.nanmax(flux_stream)
            if f_max == f_min:
                return np.zeros_like(flux_stream)
            norm = (flux_stream - f_min) / (f_max - f_min)
            return 1.0 - norm
    
    @staticmethod
    def variance_flow(data: np.ndarray) -> float:
        return ETMath.et_variance_flow(data)
    
    @staticmethod
    def jerk_intensity(data: np.ndarray) -> float:
        return ETMath.et_jerk_intensity(data)
    
    @staticmethod
    def predict_orbit(pos: np.ndarray, vel: np.ndarray, mass: float,
                      steps: int = 100, dt: float = 0.01) -> np.ndarray:
        return ETMath.predict_orbit_et(pos, vel, mass, steps, dt)
    
    @staticmethod
    def radial_velocity(wavelength_observed: float, wavelength_rest: float) -> float:
        """Calculate radial velocity from Doppler shift."""
        c = 299792.458
        if wavelength_rest <= 0:
            return 0.0
        return c * (wavelength_observed - wavelength_rest) / wavelength_rest
    
    @staticmethod
    def absolute_magnitude(apparent_mag: float, distance_pc: float) -> float:
        """Calculate absolute magnitude."""
        if distance_pc <= 0:
            return float('nan')
        return apparent_mag - 5.0 * math.log10(distance_pc) + 5.0
    
    @staticmethod
    def luminosity_from_magnitude(absolute_mag: float) -> float:
        """Calculate luminosity relative to Sun."""
        M_SUN = 4.83
        return 10.0 ** ((M_SUN - absolute_mag) / 2.5)
    
    @staticmethod
    def stellar_temperature(b_minus_v: float) -> float:
        """Estimate stellar temperature from B-V color index."""
        term1 = 1.0 / (0.92 * b_minus_v + 1.7)
        term2 = 1.0 / (0.92 * b_minus_v + 0.62)
        return 4600.0 * (term1 + term2)
    
    @staticmethod
    def mass_from_luminosity(luminosity_solar: float) -> float:
        """Estimate stellar mass from luminosity."""
        if luminosity_solar <= 0:
            return 0.0
        return luminosity_solar ** (1.0 / 3.5)
