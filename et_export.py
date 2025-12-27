#!/usr/bin/env python3
"""
ET EXPORT - Comprehensive Export Functionality
===============================================
Exception Theory Universal Astronomer v7.5

This module provides comprehensive export functionality for all ET analysis results.
In ET terms, this is the manifestation of internal configurations to external forms.

From Rules of Exception Law:
"Active Existence: E = T_engaged ° (P°D)_selected"
Exporting is the externalization of substantiated configurations.
"""

import os
from datetime import datetime
from typing import Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .et_core import LOGGER


class ETExporter:
    """
    Comprehensive export functionality for all ET analysis results.
    
    ET Derivation:
    Export is the process of binding internal configurations to external
    descriptor spaces (files). Each export format is a different D-space.
    """
    
    @staticmethod
    def export_logs(filepath: str, category: str = None, subcategory: str = None):
        """
        Export logs to text file.
        
        ET Derivation:
        Logs are the traversal history through the analysis manifold.
        Exporting them creates a permanent descriptor record.
        """
        logs = LOGGER.get_logs(category, subcategory)
        
        with open(filepath, 'w') as f:
            f.write(f"ET Universal Astronomer v7.5 - Log Export\n")
            f.write(f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*60}\n\n")
            f.write(logs)
        
        LOGGER.log_system(f"Exported logs to: {filepath}")
    
    @staticmethod
    def export_image(fig: Figure, filepath: str, dpi: int = 150):
        """
        Export matplotlib figure as image.
        
        ET Derivation:
        Visual representation is a 2D projection of the analysis manifold.
        The figure is a D-binding of computed results.
        """
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
        LOGGER.log_system(f"Exported image to: {filepath}")
    
    @staticmethod
    def export_data_csv(filepath: str, data_dict: Dict[str, np.ndarray]):
        """
        Export numerical data to CSV.
        
        ET Derivation:
        CSV is a tabular descriptor format - each column is a distinct
        D-type, each row a configuration sample.
        """
        max_len = max(len(v) for v in data_dict.values() if hasattr(v, '__len__'))
        
        with open(filepath, 'w') as f:
            # Header
            f.write(','.join(data_dict.keys()) + '\n')
            
            # Data rows
            for i in range(max_len):
                row = []
                for key, val in data_dict.items():
                    if hasattr(val, '__len__'):
                        if i < len(val):
                            row.append(str(val[i]))
                        else:
                            row.append('')
                    else:
                        row.append(str(val) if i == 0 else '')
                f.write(','.join(row) + '\n')
        
        LOGGER.log_system(f"Exported data to: {filepath}")
    
    @staticmethod
    def export_analysis_report(filepath: str, analysis_results: Dict[str, Any],
                                manifold_info: Dict[str, Any] = None):
        """
        Export comprehensive analysis report.
        
        ET Derivation:
        The analysis report is the complete substantiation record -
        all P-D-T bindings observed during the analysis traversal.
        """
        with open(filepath, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("ET UNIVERSAL ASTRONOMER v7.5 - ANALYSIS REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Manifold information
            if manifold_info:
                f.write("-" * 40 + "\n")
                f.write("MANIFOLD GEOMETRY\n")
                f.write("-" * 40 + "\n")
                for key, val in manifold_info.items():
                    f.write(f"  {key}: {val}\n")
                f.write("\n")
            
            # Analysis results
            f.write("-" * 40 + "\n")
            f.write("ANALYSIS RESULTS\n")
            f.write("-" * 40 + "\n")
            for key, val in analysis_results.items():
                if isinstance(val, np.ndarray):
                    f.write(f"  {key}: array shape={val.shape}, mean={np.nanmean(val):.6f}\n")
                elif isinstance(val, dict):
                    f.write(f"  {key}:\n")
                    for k2, v2 in val.items():
                        f.write(f"    {k2}: {v2}\n")
                else:
                    f.write(f"  {key}: {val}\n")
            
            f.write("\n")
            f.write("-" * 40 + "\n")
            f.write("LOGS\n")
            f.write("-" * 40 + "\n")
            f.write(LOGGER.get_logs())
        
        LOGGER.log_system(f"Exported report to: {filepath}")
    
    @staticmethod
    def export_2d_image(data: np.ndarray, filepath: str, colormap: str = 'viridis'):
        """
        Export 2D array as image file.
        
        ET Derivation:
        2D data represents a manifold slice. Different export formats
        capture different aspects of the descriptor structure.
        """
        ext = os.path.splitext(filepath)[1].lower()
        
        if ext in ['.png', '.jpg', '.jpeg', '.tiff']:
            # Normalize to 0-255 range
            data_norm = data - np.nanmin(data)
            data_max = np.nanmax(data_norm)
            if data_max > 0:
                data_norm = (data_norm / data_max * 255).astype(np.uint8)
            else:
                data_norm = np.zeros_like(data, dtype=np.uint8)
            
            # Apply colormap
            cmap = plt.get_cmap(colormap)
            colored = (cmap(data_norm / 255.0) * 255).astype(np.uint8)
            
            # Save using matplotlib
            plt.imsave(filepath, colored)
        
        elif ext == '.npy':
            np.save(filepath, data)
        
        elif ext == '.fits':
            try:
                from astropy.io import fits
                hdu = fits.PrimaryHDU(data)
                hdu.writeto(filepath, overwrite=True)
            except ImportError:
                # Fallback to numpy
                np.save(filepath.replace('.fits', '.npy'), data)
        
        LOGGER.log_system(f"Exported 2D image to: {filepath}")
    
    @staticmethod
    def export_spectrum_csv(filepath: str, wavelengths: np.ndarray, 
                            flux: np.ndarray, binding: np.ndarray = None):
        """
        Export spectrum data with optional binding column.
        
        ET Derivation:
        Spectrum export captures the wavelength-flux-binding relationship,
        the fundamental P-D-T triple for spectral analysis.
        """
        with open(filepath, 'w') as f:
            if binding is not None:
                f.write("wavelength,flux,binding\n")
                for i in range(len(wavelengths)):
                    w = wavelengths[i] if i < len(wavelengths) else ''
                    fl = flux[i] if i < len(flux) else ''
                    b = binding[i] if i < len(binding) else ''
                    f.write(f"{w},{fl},{b}\n")
            else:
                f.write("wavelength,flux\n")
                for i in range(len(wavelengths)):
                    w = wavelengths[i] if i < len(wavelengths) else ''
                    fl = flux[i] if i < len(flux) else ''
                    f.write(f"{w},{fl}\n")
        
        LOGGER.log_system(f"Exported spectrum to: {filepath}")
    
    @staticmethod
    def export_manifold_state(filepath: str, manifold):
        """
        Export current manifold geometry state.
        
        ET Derivation:
        The manifold state is the current configuration of the
        12-fold geometry - the detected fold level and variances.
        """
        import json
        
        state = {
            'detected_fold': manifold.detected_fold,
            'detected_variance': manifold.detected_variance,
            'base_symmetry': manifold.base_symmetry,
            'fold_sequence': manifold.get_fold_sequence(),
            'resonance_threshold': manifold.get_resonance_threshold(),
            'gaze_threshold': manifold.get_gaze_threshold(),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        LOGGER.log_system(f"Exported manifold state to: {filepath}")
