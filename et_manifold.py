#!/usr/bin/env python3
"""
ET MANIFOLD - Dynamic Manifold Geometry
========================================
Exception Theory Universal Astronomer v7.5

This module provides the manifold geometry calculations.
In ET terms, this represents P (Point) - the infinite substrate.

From Rules of Exception Law:
"P (Point) has cardinality Ω (Absolute Infinity). P is the substrate."

The manifold has 12-fold symmetry from:
- 3 Primitives (P, D, T) × 4 Logic States (0, 1, 2, +1) = 12

Higher folds emerge from power set structure:
- 12 × 2^n for n ∈ {0, 1, 2, 3, ...}
- Gives: 12, 24, 48, 96, 192...
"""

from typing import List, Tuple
import numpy as np

from .et_core import LOGGER, set_manifold


class ManifoldGeometry:
    """
    Dynamic Manifold Geometry Calculator.
    
    ET DERIVATION:
    The manifold has 12-fold symmetry from:
    - 3 Primitives (P, D, T) × 4 Logic States (0, 1, 2, +1) = 12
    - This creates fundamental interaction types
    
    Higher folds emerge from power set structure:
    - 12 × 2^n for n ∈ {0, 1, 2, 3, ...}
    - Gives: 12, 24, 48, 96, 192...
    
    The variance at each fold level:
    - 1/12, 1/24, 1/48, 1/96...
    """
    
    # Base constants discovered from ET structure
    PRIMITIVES = 3      # P, D, T
    LOGIC_STATES = 4    # 0, 1, 2, +1 (from Rules of Exception Law)
    
    def __init__(self):
        self._fold_cache = {}
        self._variance_cache = {}
        self._resonance_cache = {}
        self.detected_fold = 12  # Default, will be updated by analysis
        self.detected_variance = 1.0 / 12.0
        LOGGER.log_manifold("ManifoldGeometry initialized with base 12-fold symmetry")
    
    @property
    def base_symmetry(self) -> int:
        """Base manifold symmetry: 3 × 4 = 12"""
        return self.PRIMITIVES * self.LOGIC_STATES
    
    @property
    def fold(self) -> int:
        """Current detected fold (convenience property)."""
        return self.detected_fold
    
    @property
    def base_variance(self) -> float:
        """Base variance at current fold: 1/fold."""
        return 1.0 / self.detected_fold
    
    @property
    def resonance_threshold(self) -> float:
        """Resonance threshold: (fold + 1) / fold."""
        return (self.detected_fold + 1) / self.detected_fold
    
    @property
    def gaze_threshold(self) -> float:
        """Gaze threshold: 1 + 2.4 / fold (observer effect boundary)."""
        return 1.0 + 2.4 / self.detected_fold
    
    @property
    def higher_folds(self) -> List[int]:
        """List of higher fold values: 12, 24, 48, 96, 192..."""
        return [self.get_fold(i) for i in range(10)]
    
    def get_fold(self, level: int = 0) -> int:
        """
        Get fold value at specified level.
        
        Level 0: 12 (base)
        Level 1: 24 (12 × 2)
        Level 2: 48 (12 × 4)
        Level 3: 96 (12 × 8)
        ...
        
        ET Derivation: Higher levels represent deeper manifold structure,
        where descriptor combinations double at each level (power set).
        """
        if level in self._fold_cache:
            return self._fold_cache[level]
        
        fold = self.base_symmetry * (2 ** level)
        self._fold_cache[level] = fold
        LOGGER.log_manifold(f"Computed fold at level {level}: {fold}")
        return fold
    
    def get_variance(self, fold: int = None) -> float:
        """
        Get base variance for given fold.
        
        Variance = 1/fold (the minimal descriptor wiggle room)
        
        ET Derivation: Each fold represents a partition of the manifold.
        The variance is the probability of being in any single partition.
        """
        if fold is None:
            fold = self.detected_fold
        
        if fold in self._variance_cache:
            return self._variance_cache[fold]
        
        variance = 1.0 / fold
        self._variance_cache[fold] = variance
        return variance
    
    def get_resonance_threshold(self, fold: int = None) -> float:
        """
        Get resonance threshold (subliminal detection).
        
        Threshold = 1 + 1/fold = (fold + 1)/fold
        
        ET Derivation: When T adds minimal intent (1/fold weight),
        the field becomes (fold + 1)/fold ≈ 1.0833 for fold=12.
        """
        if fold is None:
            fold = self.detected_fold
        
        if fold in self._resonance_cache:
            return self._resonance_cache[fold]
        
        threshold = (fold + 1.0) / fold
        self._resonance_cache[fold] = threshold
        return threshold
    
    def get_gaze_threshold(self, fold: int = None) -> float:
        """
        Get gaze threshold (conscious detection).
        
        For 12-fold: 1.20 (20% above baseline)
        Scales proportionally for other folds.
        
        ET Derivation: Conscious detection requires stronger T binding.
        The 20% comes from biological signal variance in perception.
        """
        if fold is None:
            fold = self.detected_fold
        
        # Base gaze threshold at fold=12 is 1.20
        # This represents 20% above baseline = 1 + 2.4/12
        base_addition = 2.4 / 12.0  # 0.20 at fold=12
        
        # Scale proportionally to fold
        scaled_addition = base_addition * (12.0 / fold)
        return 1.0 + scaled_addition
    
    def detect_fold_from_data(self, data: np.ndarray) -> int:
        """
        Dynamically detect manifold fold from data characteristics.
        
        ET Derivation: The variance structure of data reveals its
        manifold alignment. Data naturally clusters at fold boundaries.
        """
        if len(data) < 20:
            LOGGER.log_manifold("Insufficient data for fold detection, using default 12")
            return 12
        
        clean_data = data[np.isfinite(data)]
        if len(clean_data) < 20:
            return 12
        
        # Calculate coefficient of variation squared (normalized variance)
        mean_val = np.mean(clean_data)
        if abs(mean_val) < 1e-12:
            mean_val = 1.0
        
        variance = np.var(clean_data)
        cv_squared = variance / (mean_val ** 2)
        
        # Find closest manifold fold based on variance alignment
        folds_to_check = [12, 24, 48, 96, 192]
        best_fold = 12
        best_alignment = float('inf')
        
        for fold in folds_to_check:
            expected_variance = 1.0 / fold
            alignment_error = abs(cv_squared - expected_variance)
            
            if alignment_error < best_alignment:
                best_alignment = alignment_error
                best_fold = fold
        
        # Also check if data length suggests a fold
        n = len(clean_data)
        for fold in folds_to_check:
            if n % fold == 0:
                # Data length is divisible by fold - stronger alignment
                expected_variance = 1.0 / fold
                alignment_error = abs(cv_squared - expected_variance) * 0.8  # 20% bonus
                if alignment_error < best_alignment:
                    best_alignment = alignment_error
                    best_fold = fold
        
        self.detected_fold = best_fold
        self.detected_variance = 1.0 / best_fold
        
        LOGGER.log_manifold(f"Detected manifold fold: {best_fold} (variance: {self.detected_variance:.6f})")
        LOGGER.log_manifold(f"Data CV²: {cv_squared:.6f}, Expected: {self.detected_variance:.6f}")
        
        return best_fold
    
    def get_fold_sequence(self, max_level: int = 4) -> List[int]:
        """Get sequence of fold values: [12, 24, 48, 96, ...]"""
        return [self.get_fold(level) for level in range(max_level + 1)]
    
    def is_manifold_aligned(self, variance: float, tolerance: float = 0.15) -> Tuple[bool, int]:
        """
        Check if variance is aligned with a manifold fold.
        
        Returns (is_aligned, fold_level) tuple.
        """
        for level in range(5):  # Check up to level 4 (192-fold)
            fold = self.get_fold(level)
            expected = 1.0 / fold
            deviation = abs(variance - expected) / expected
            
            if deviation < tolerance:
                return True, fold
        
        return False, 0
    
    def get_power_of_2_boundaries(self, n: int) -> List[int]:
        """
        Get manifold-aligned power-of-2 boundaries for data of length n.
        
        ET Derivation: Data compression and structure follow 2^n
        patterns because |P(D)| = 2^n (power set of descriptors).
        """
        boundaries = []
        power = 1
        while power < n:
            if power >= self.base_symmetry:  # Start from 12 or nearest power of 2
                boundaries.append(power)
            power *= 2
        return boundaries


# ==============================================================================
# GLOBAL MANIFOLD INSTANCE
# ==============================================================================
# Create and register the global manifold geometry instance

MANIFOLD = ManifoldGeometry()
set_manifold(MANIFOLD)
