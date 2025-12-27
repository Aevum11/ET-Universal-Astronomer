#!/usr/bin/env python3
"""
ET APP - The Main GUI Application
==================================
Exception Theory Universal Astronomer v7.5

This module provides the complete GUI application.
In ET terms, this is the Binding (°) that unifies P, D, and T.

From Rules of Exception Law:
"S = (P°D°T)" - Something is the binding of all three primitives.
The application is the interface where the three primitives meet.

v7.5 Features:
- Comprehensive export for all tabs
- Zoom controls for dense visualizations
- Complete state reset on new file load
- Dynamic manifold detection and display
"""

import os
import math
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from .et_core import LOGGER
from .et_manifold import MANIFOLD
from .et_math import ETMath
from .et_scanner import ETConfig, ETManifoldScanner
from .et_ingestion import UniversalIngestor
from .et_export import ETExporter


class ETApp(tk.Tk):
    """
    ET Universal Astronomer v7.5 GUI Application.
    
    ET Derivation:
    The application is the binding interface (°) that allows:
    - P (data/manifold) to be substantiated
    - D (constraints/parameters) to be configured
    - T (scanner/analysis) to traverse and reveal structure
    
    v7.5 Features:
    - Comprehensive export for all tabs
    - Zoom controls for dense visualizations
    - Complete state reset on new file load
    - Dynamic manifold detection and display
    """
    
    def __init__(self):
        super().__init__()
        
        self.title("ET UNIVERSAL ASTRONOMER v7.5 - THE DYNAMIC MANIFOLD")
        self.geometry("1400x900")
        self.configure(bg="#121212")
        
        # Initialize state
        self.reset_state()
        
        # Build GUI
        self.build_gui()
        
        LOGGER.log_system("ETApp initialized")
    
    def reset_state(self):
        """Reset all application state - called when loading new file."""
        LOGGER.reset()
        UniversalIngestor.reset()
        
        # Data state
        self.data = np.array([])
        self.data_full = np.array([])
        self.wavelengths = None
        self.wavelengths_full = None
        self.binding_data = np.array([])
        self.filename = ""
        self.data_type = '1D'
        
        # 2D image state
        self.image_2d = None
        self.processed_2d = None
        self.d_field = None
        self.unc_map = None
        self.quality_mask = None
        
        # Processing
        self.scanner = ETManifoldScanner()
        self.monitoring = False
        
        # Discovered values
        self.discovered_lines = {}
        self.discovered_fold = 12
        
        LOGGER.log_system("Application state reset")
    
    def build_gui(self):
        """Build the complete GUI."""
        # Status bar
        status_frame = tk.Frame(self, bg="#1a1a1a")
        status_frame.pack(fill="x", side="top")
        
        self.lbl_status = tk.Label(status_frame, text="ET UNIVERSAL ASTRONOMER v7.5",
                                   bg="#1a1a1a", fg="#0f0", font=("Consolas", 10))
        self.lbl_status.pack(side="left", padx=10, pady=5)
        
        # Manifold indicator
        self.lbl_manifold = tk.Label(status_frame, text="Manifold: 12-fold",
                                     bg="#1a1a1a", fg="#ff0", font=("Consolas", 10))
        self.lbl_manifold.pack(side="right", padx=10, pady=5)
        
        # Toolbar
        toolbar = tk.Frame(self, bg="#1a1a1a")
        toolbar.pack(fill="x")
        
        tk.Button(toolbar, text="📂 LOAD FILE", command=self.load_file,
                 bg="#004400", fg="#fff", font=("Consolas", 10)).pack(side="left", padx=5, pady=5)
        
        tk.Button(toolbar, text="💾 EXPORT ALL", command=self.export_all,
                 bg="#440044", fg="#fff", font=("Consolas", 10)).pack(side="left", padx=5, pady=5)
        
        tk.Button(toolbar, text="📋 EXPORT LOGS", command=self.export_logs_dialog,
                 bg="#444400", fg="#fff", font=("Consolas", 10)).pack(side="left", padx=5, pady=5)
        
        tk.Button(toolbar, text="🔄 RESET", command=self.full_reset,
                 bg="#440000", fg="#fff", font=("Consolas", 10)).pack(side="left", padx=5, pady=5)
        
        # Notebook (tabs)
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create tabs
        self.tab_scanner = tk.Frame(self.notebook, bg="#121212")
        self.tab_observatory = tk.Frame(self.notebook, bg="#121212")
        self.tab_astro = tk.Frame(self.notebook, bg="#121212")
        self.tab_image = tk.Frame(self.notebook, bg="#121212")
        
        self.notebook.add(self.tab_scanner, text="📡 TRAVERSER SCANNER")
        self.notebook.add(self.tab_observatory, text="🔭 OBSERVATORY")
        self.notebook.add(self.tab_astro, text="🌟 CONCRETE ASTRONOMY")
        self.notebook.add(self.tab_image, text="🖼 IMAGE PROCESSING")
        
        # Build each tab
        self.build_scanner_tab()
        self.build_observatory_tab()
        self.build_astro_tab()
        self.build_image_tab()
    
    def build_scanner_tab(self):
        """Build the Traverser Scanner tab."""
        f = tk.Frame(self.tab_scanner, bg="#121212")
        f.pack(fill="both", expand=True)
        
        # Left panel - Scanner readouts
        left = tk.Frame(f, bg="#1a1a1a", width=350)
        left.pack(side="left", fill="y", padx=10, pady=10)
        left.pack_propagate(False)
        
        tk.Label(left, text="ET TRAVERSER SCANNER v7.5", bg="#1a1a1a", fg="#0f0",
                font=("Consolas", 14, "bold")).pack(pady=20)
        
        self.lbl_class = tk.Label(left, text="UNSUBSTANTIATED", bg="#1a1a1a", fg="#888",
                                  font=("Consolas", 18, "bold"))
        self.lbl_class.pack(pady=10)
        
        self.lbl_desc = tk.Label(left, text="Load data to begin analysis", bg="#1a1a1a", fg="#666",
                                 font=("Consolas", 10, "italic"))
        self.lbl_desc.pack(pady=5)
        
        # Scanner metrics
        metrics_frame = tk.Frame(left, bg="#1a1a1a")
        metrics_frame.pack(fill="x", padx=20, pady=20)
        
        self.scan_vars = {}
        metrics = [
            "BINDING (GRAVITY)",
            "JERK (AGENCY)",
            "VARIANCE (CHAOS)",
            "MANIFOLD ALIGN",
            "MANIFOLD FOLD",
            "GAZE RATIO",
            "T-TIME DILATION"
        ]
        
        for metric in metrics:
            row = tk.Frame(metrics_frame, bg="#1a1a1a")
            row.pack(fill="x", pady=3)
            tk.Label(row, text=f"{metric}:", bg="#1a1a1a", fg="#aaa",
                    font=("Consolas", 9), width=20, anchor="w").pack(side="left")
            var = tk.StringVar(value="---")
            self.scan_vars[metric] = var
            tk.Label(row, textvariable=var, bg="#1a1a1a", fg="#0f0",
                    font=("Consolas", 9, "bold"), width=15, anchor="e").pack(side="right")
        
        # Export button for this tab
        tk.Button(left, text="📋 EXPORT SCANNER", command=self.export_scanner,
                 bg="#333", fg="#fff", font=("Consolas", 10)).pack(pady=20, padx=20, fill="x")
        
        # Right panel - Figure
        self.fig_scanner = Figure(figsize=(8, 6), dpi=100, facecolor="#000")
        self.canvas_scanner = FigureCanvasTkAgg(self.fig_scanner, master=f)
        self.canvas_scanner.get_tk_widget().pack(side="right", fill="both", expand=True, padx=10, pady=10)
    
    def build_observatory_tab(self):
        """Build the Observatory tab with sub-tabs and zoom controls."""
        f = tk.Frame(self.tab_observatory, bg="#121212")
        f.pack(fill="both", expand=True)
        
        # Control panel at top
        control_frame = tk.Frame(f, bg="#1a1a1a")
        control_frame.pack(fill="x", padx=10, pady=5)
        
        # View buttons
        views_frame = tk.Frame(control_frame, bg="#1a1a1a")
        views_frame.pack(side="left")
        
        tk.Label(views_frame, text="VIEW:", bg="#1a1a1a", fg="#fff",
                font=("Consolas", 10, "bold")).pack(side="left", padx=5)
        
        self.observatory_views = [
            ("Decomposition", self.plot_decomposition),
            ("Phase Space", self.plot_phase),
            ("Variance Horizon", self.plot_horizon),
            ("Rotation Curve", self.plot_rotation),
            ("Spectrum", self.plot_spectrum),
            ("Fourier", self.plot_fourier)
        ]
        
        for name, cmd in self.observatory_views:
            tk.Button(views_frame, text=name, command=cmd,
                     bg="#333", fg="#fff", font=("Consolas", 9)).pack(side="left", padx=2)
        
        # Zoom controls
        zoom_frame = tk.Frame(control_frame, bg="#1a1a1a")
        zoom_frame.pack(side="right")
        
        tk.Label(zoom_frame, text="ZOOM:", bg="#1a1a1a", fg="#fff",
                font=("Consolas", 10, "bold")).pack(side="left", padx=5)
        
        self.zoom_level = tk.DoubleVar(value=1.0)
        
        tk.Button(zoom_frame, text="−", command=lambda: self.adjust_zoom(-0.2),
                 bg="#333", fg="#fff", font=("Consolas", 12), width=3).pack(side="left", padx=2)
        
        self.zoom_label = tk.Label(zoom_frame, text="100%", bg="#1a1a1a", fg="#0f0",
                                   font=("Consolas", 10), width=6)
        self.zoom_label.pack(side="left", padx=5)
        
        tk.Button(zoom_frame, text="+", command=lambda: self.adjust_zoom(0.2),
                 bg="#333", fg="#fff", font=("Consolas", 12), width=3).pack(side="left", padx=2)
        
        tk.Button(zoom_frame, text="Reset", command=lambda: self.set_zoom(1.0),
                 bg="#333", fg="#fff", font=("Consolas", 9)).pack(side="left", padx=5)
        
        # Export button
        tk.Button(control_frame, text="📷 EXPORT VIEW", command=self.export_observatory,
                 bg="#440044", fg="#fff", font=("Consolas", 9)).pack(side="right", padx=10)
        
        # Main figure area with scrollable canvas
        self.fig = Figure(figsize=(10, 7), dpi=100, facecolor="#000")
        self.canvas = FigureCanvasTkAgg(self.fig, master=f)
        
        # Add navigation toolbar
        toolbar_frame = tk.Frame(f, bg="#1a1a1a")
        toolbar_frame.pack(fill="x")
        self.nav_toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.nav_toolbar.update()
        
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
    
    def adjust_zoom(self, delta: float):
        """Adjust zoom level."""
        new_zoom = max(0.2, min(5.0, self.zoom_level.get() + delta))
        self.set_zoom(new_zoom)
    
    def set_zoom(self, level: float):
        """Set zoom level and update view."""
        self.zoom_level.set(level)
        self.zoom_label.config(text=f"{int(level * 100)}%")
        
        # Apply zoom to figure
        current_size = self.fig.get_size_inches()
        base_size = (10, 7)
        new_size = (base_size[0] * level, base_size[1] * level)
        self.fig.set_size_inches(new_size)
        self.canvas.draw()
        
        LOGGER.log_observatory('zoom', f"Zoom set to {int(level * 100)}%")
    
    def build_astro_tab(self):
        """Build the Concrete Astronomy tab."""
        f = tk.Frame(self.tab_astro, bg="#121212")
        f.pack(fill="both", expand=True)
        
        # Chemistry Analysis panel
        chem_frame = tk.LabelFrame(f, text="CHEMO-STRUCTURAL ANALYSIS", bg="#121212", fg="#0f0",
                                   font=("Consolas", 12))
        chem_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        
        chem_toolbar = tk.Frame(chem_frame, bg="#121212")
        chem_toolbar.pack(fill="x")
        
        tk.Button(chem_toolbar, text="📋 Export", command=lambda: self.export_text(self.chem_text, "chemistry"),
                 bg="#333", fg="#fff", font=("Consolas", 9)).pack(side="right", padx=5, pady=5)
        
        self.chem_text = scrolledtext.ScrolledText(chem_frame, bg="#000", fg="#0f0",
                                                    font=("Consolas", 10))
        self.chem_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Orbital Prediction panel
        orbit_frame = tk.LabelFrame(f, text="ORBITAL & COORDINATE PREDICTION", bg="#121212", fg="#0f0",
                                    font=("Consolas", 12))
        orbit_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        
        orbit_toolbar = tk.Frame(orbit_frame, bg="#121212")
        orbit_toolbar.pack(fill="x")
        
        tk.Button(orbit_toolbar, text="📋 Export", command=lambda: self.export_text(self.orbit_text, "orbital"),
                 bg="#333", fg="#fff", font=("Consolas", 9)).pack(side="right", padx=5, pady=5)
        
        self.orbit_text = scrolledtext.ScrolledText(orbit_frame, bg="#000", fg="#0f0",
                                                     font=("Consolas", 10))
        self.orbit_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Run button
        btn_run = tk.Button(f, text="RUN ASTRONOMICAL SOLVER", command=self.run_astro_solver,
                           bg="#004400", fg="#fff", font=("Consolas", 12, "bold"))
        btn_run.pack(side="bottom", fill="x", padx=10, pady=10)
    
    def build_image_tab(self):
        """Build the 2D Image Processing tab."""
        f = tk.Frame(self.tab_image, bg="#121212")
        f.pack(fill="both", expand=True)
        
        # Control panel
        side = tk.Frame(f, bg="#1a1a1a", width=280)
        side.pack(side="left", fill="y")
        side.pack_propagate(False)
        
        tk.Label(side, text="2D MANIFOLD SCANNER v7.5", bg="#1a1a1a", fg="#fff",
                font=("Consolas", 12, "bold")).pack(pady=10)
        
        # Options
        options_frame = tk.Frame(side, bg="#1a1a1a")
        options_frame.pack(fill="x", padx=10)
        
        self.var_destripe = tk.BooleanVar(value=False)
        tk.Checkbutton(options_frame, text="Destripe", variable=self.var_destripe,
                      bg="#1a1a1a", fg="#0f0", selectcolor="#333",
                      command=self.update_scanner_config).pack(anchor="w")
        
        self.var_fft = tk.BooleanVar(value=False)
        tk.Checkbutton(options_frame, text="FFT Filter", variable=self.var_fft,
                      bg="#1a1a1a", fg="#0f0", selectcolor="#333",
                      command=self.update_scanner_config).pack(anchor="w")
        
        self.var_fft_auto = tk.BooleanVar(value=True)
        tk.Checkbutton(options_frame, text="FFT Auto-tune", variable=self.var_fft_auto,
                      bg="#1a1a1a", fg="#0f0", selectcolor="#333",
                      command=self.update_scanner_config).pack(anchor="w")
        
        self.var_preserve = tk.BooleanVar(value=True)
        tk.Checkbutton(options_frame, text="Preservation Mode", variable=self.var_preserve,
                      bg="#1a1a1a", fg="#0f0", selectcolor="#333",
                      command=self.update_scanner_config).pack(anchor="w")
        
        tk.Label(side, text="FFT Threshold:", bg="#1a1a1a", fg="#aaa",
                font=("Consolas", 9)).pack(pady=(10, 0))
        self.var_fft_thresh = tk.DoubleVar(value=5.0)
        tk.Scale(side, from_=2.0, to=15.0, resolution=0.5, orient="horizontal",
                variable=self.var_fft_thresh, bg="#1a1a1a", fg="#0f0",
                command=lambda x: self.update_scanner_config()).pack(fill="x", padx=10)
        
        tk.Button(side, text="PROCESS 2D IMAGE", command=self.process_2d_image,
                 bg="#004400", fg="#fff", font=("Consolas", 10, "bold")).pack(pady=20, padx=10, fill="x")
        
        tk.Label(side, text="VIEW:", bg="#1a1a1a", fg="#fff",
                font=("Consolas", 10, "bold")).pack(pady=(20, 5))
        
        tk.Button(side, text="Processed", command=lambda: self.show_2d_view('processed'),
                 bg="#333", fg="#fff", width=18).pack(pady=2)
        tk.Button(side, text="D-Field", command=lambda: self.show_2d_view('d_field'),
                 bg="#333", fg="#fff", width=18).pack(pady=2)
        tk.Button(side, text="Uncertainty", command=lambda: self.show_2d_view('uncertainty'),
                 bg="#333", fg="#fff", width=18).pack(pady=2)
        tk.Button(side, text="Quality", command=lambda: self.show_2d_view('quality'),
                 bg="#333", fg="#fff", width=18).pack(pady=2)
        tk.Button(side, text="Original", command=lambda: self.show_2d_view('original'),
                 bg="#333", fg="#fff", width=18).pack(pady=2)
        
        # Export buttons
        tk.Label(side, text="EXPORT:", bg="#1a1a1a", fg="#fff",
                font=("Consolas", 10, "bold")).pack(pady=(20, 5))
        
        tk.Button(side, text="📷 Export Image", command=self.export_2d_image,
                 bg="#440044", fg="#fff", width=18).pack(pady=2)
        tk.Button(side, text="📋 Export Data", command=self.export_2d_data,
                 bg="#444400", fg="#fff", width=18).pack(pady=2)
        
        self.lbl_2d_stats = tk.Label(side, text="No 2D data", bg="#1a1a1a", fg="#666",
                                     font=("Consolas", 8), justify="left")
        self.lbl_2d_stats.pack(pady=20, padx=10)
        
        # Main figure
        self.fig_2d = Figure(figsize=(6, 6), dpi=100, facecolor="#000")
        self.canvas_2d = FigureCanvasTkAgg(self.fig_2d, master=f)
        self.canvas_2d.get_tk_widget().pack(side="right", fill="both", expand=True)
    
    def update_scanner_config(self):
        """Update scanner configuration from GUI controls."""
        self.scanner.cfg.destripe = self.var_destripe.get()
        self.scanner.cfg.fft_filter = self.var_fft.get()
        self.scanner.cfg.fft_auto_tune = self.var_fft_auto.get()
        self.scanner.cfg.fft_threshold = self.var_fft_thresh.get()
        self.scanner.cfg.preservation_mode = self.var_preserve.get()
    
    # =========================================================================
    # FILE LOADING AND RESET
    # =========================================================================
    
    def full_reset(self):
        """Complete reset of application state."""
        self.reset_state()
        
        # Clear displays
        self.lbl_status.config(text="ET UNIVERSAL ASTRONOMER v7.5")
        self.lbl_manifold.config(text="Manifold: 12-fold")
        self.lbl_class.config(text="UNSUBSTANTIATED", fg="#888")
        self.lbl_desc.config(text="Load data to begin analysis")
        
        for var in self.scan_vars.values():
            var.set("---")
        
        self.fig_scanner.clear()
        self.canvas_scanner.draw()
        
        self.fig.clear()
        self.canvas.draw()
        
        self.fig_2d.clear()
        self.canvas_2d.draw()
        
        self.chem_text.delete(1.0, tk.END)
        self.orbit_text.delete(1.0, tk.END)
        
        self.lbl_2d_stats.config(text="No 2D data")
        
        LOGGER.log_system("Full reset complete")
    
    def load_file(self):
        """Load a data file with complete state reset."""
        path = filedialog.askopenfilename(
            filetypes=[
                ("All Supported", "*.fits *.fit *.nc *.csv *.txt *.dat *.json"),
                ("FITS Files", "*.fits *.fit"),
                ("NetCDF Files", "*.nc"),
                ("Text Files", "*.csv *.txt *.dat"),
                ("JSON Files", "*.json"),
                ("All Files", "*.*")
            ]
        )
        if not path:
            return
        
        # COMPLETE RESET before loading new file
        self.full_reset()
        
        self.filename = os.path.basename(path)
        
        # Load data
        self.data_full = UniversalIngestor.load(path)
        self.wavelengths_full = UniversalIngestor._last_wavelengths
        self.data_type = UniversalIngestor._last_data_type or '1D'
        
        # Detect manifold fold from data
        MANIFOLD.detect_fold_from_data(self.data_full.flatten())
        self.discovered_fold = MANIFOLD.detected_fold
        self.lbl_manifold.config(text=f"Manifold: {self.discovered_fold}-fold")
        
        # Check for 2D image data
        is_2d_data = (
            len(self.data_full.shape) == 2 and
            min(self.data_full.shape) > 1 and
            self.data_type == '2D'
        )
        
        if is_2d_data:
            self.image_2d = self.data_full
            self.data = self.data_full.flatten()[:UniversalIngestor.MAX_DISPLAY_POINTS]
            self.wavelengths = None
            self.binding_data = np.zeros_like(self.data)
            
            LOGGER.log_system(f"Loaded 2D image: {self.image_2d.shape}")
            self.lbl_status.config(text=f"2D: {self.filename} [{self.image_2d.shape}]")
            
            self.show_2d_view('original')
            self.notebook.select(self.tab_image)
            return
        
        # Handle 1D spectral data
        if len(self.data_full) > UniversalIngestor.MAX_DISPLAY_POINTS:
            self.data, self.wavelengths = UniversalIngestor.downsample_for_display(
                self.data_full, self.wavelengths_full)
        else:
            self.data = self.data_full
            self.wavelengths = self.wavelengths_full
        
        # Calculate binding
        self.binding_data = ETMath.calculate_binding(self.data, self.wavelengths)
        
        # Discover absorption lines dynamically
        if self.wavelengths is not None:
            self.discovered_lines = ETMath.discover_absorption_lines(
                self.wavelengths, self.binding_data)
            LOGGER.log_system(f"Discovered {len(self.discovered_lines)} absorption features")
        
        # Data quality diagnostics
        valid_data = self.data[np.isfinite(self.data)]
        if len(valid_data) > 0:
            LOGGER.log_system(f"Data loaded: {len(valid_data)} valid points")
            LOGGER.log_system(f"Flux range: {np.min(valid_data):.4e} to {np.max(valid_data):.4e}")
            
            if self.wavelengths is not None:
                res_info = ETMath.classify_resolution(self.wavelengths)
                LOGGER.log_system(f"Resolution: {res_info['class']} ({res_info['step_angstroms']:.4f} Å/pixel)")
        
        valid_points = np.sum(np.isfinite(self.data))
        total_points = len(self.data_full) if hasattr(self, 'data_full') else valid_points
        self.lbl_status.config(text=f"SOURCE: {self.filename} [{valid_points:,} display / {total_points:,} total]")
        
        self.run_scanner_analysis()
    
    # =========================================================================
    # ANALYSIS FUNCTIONS
    # =========================================================================
    
    def run_scanner_analysis(self):
        """Run the Traverser Scanner analysis."""
        if len(self.binding_data) < 2:
            return
        
        clean_binding = self.binding_data[np.isfinite(self.binding_data)]
        clean_data = self.data[np.isfinite(self.data)]
        
        if len(clean_binding) < 10:
            self.lbl_class.config(text="INSUFFICIENT DATA", fg="#ff0")
            return
        
        t_sigs = ETMath.classify_t_signature(clean_binding)
        var = ETMath.variance_flow(clean_data)
        
        # Gaze ratio
        mean_val = np.nanmean(clean_data)
        gaze = np.nanmax(clean_data) / mean_val if mean_val != 0 else 0
        
        t_type = t_sigs["Type"]
        color = "#888"
        desc = "Unsubstantiated Field"
        
        if "GRAVITY" in t_type:
            color = "#0f0"
            desc = "Structured Binding (Stars/Mass)"
        elif "AGENCY" in t_type:
            color = "#f00"
            desc = "Navigational Intent (Life/Ship)"
        elif "RESONANT" in t_type:
            color = "#fff"
            desc = f"Manifold Resonance Lock ({t_sigs['Manifold_Fold']}-fold)"
        
        self.lbl_class.config(text=t_type, fg=color)
        self.lbl_desc.config(text=desc)
        
        self.scan_vars["BINDING (GRAVITY)"].set(f"{t_sigs['T_Gravity']*100:.1f}%")
        self.scan_vars["JERK (AGENCY)"].set(f"{t_sigs['T_Agency']:.4f}")
        self.scan_vars["VARIANCE (CHAOS)"].set(f"{var:.6f}")
        self.scan_vars["MANIFOLD ALIGN"].set("LOCKED" if t_sigs['Manifold_Align'] else "DRIFTING")
        self.scan_vars["MANIFOLD FOLD"].set(f"{t_sigs['Manifold_Fold']}")
        self.scan_vars["GAZE RATIO"].set(f"{gaze:.2f}")
        
        t_dilation = ETMath.et_time_dilation(var) if var < 1.0 else 1.0
        self.scan_vars["T-TIME DILATION"].set(f"{min(t_dilation, 999.9):.2f}x")
        
        # Log results
        LOGGER.log_scanner(f"Classification: {t_type}")
        LOGGER.log_scanner(f"T_Gravity: {t_sigs['T_Gravity']*100:.2f}%")
        LOGGER.log_scanner(f"T_Agency: {t_sigs['T_Agency']:.4f}")
        LOGGER.log_scanner(f"Variance: {var:.6f}")
        LOGGER.log_scanner(f"Manifold Fold: {t_sigs['Manifold_Fold']}")
        
        # Update scanner visualization
        self.update_scanner_plot()
    
    def update_scanner_plot(self):
        """Update the scanner visualization."""
        if len(self.data) < 2:
            return
        
        self.fig_scanner.clear()
        
        try:
            clean_data = np.where(np.isfinite(self.data), self.data, 0)
            clean_binding = np.where(np.isfinite(self.binding_data), self.binding_data, 0)
            
            ax = self.fig_scanner.add_subplot(111, facecolor="black")
            
            if self.wavelengths is not None and len(self.wavelengths) == len(clean_data):
                x = self.wavelengths
                ax.set_xlabel("Wavelength (Å)", color="white")
            else:
                x = np.arange(len(clean_data))
                ax.set_xlabel("Sample Index", color="white")
            
            ax.plot(x, clean_data, color="#333", linewidth=0.5, label="Flux (D)")
            
            ax2 = ax.twinx()
            ax2.plot(x, clean_binding, color="lime", alpha=0.7, linewidth=0.8, label="Binding (T)")
            
            # Mark manifold resonance on binding
            ax2.axhline(MANIFOLD.detected_variance, color="red", linestyle="--",
                       linewidth=1, alpha=0.5, label=f"1/{MANIFOLD.detected_fold} Resonance")
            
            ax.set_title("TRAVERSER SCANNER: Flux & Binding", color="white")
            ax.set_ylabel("Flux (D)", color="white")
            ax2.set_ylabel("Binding (T)", color="lime")
            ax.tick_params(colors="white")
            ax2.tick_params(colors="lime")
            
            self.canvas_scanner.draw()
            
        except Exception as e:
            LOGGER.log_scanner(f"Plot error: {e}")
    
    # =========================================================================
    # OBSERVATORY PLOTS
    # =========================================================================
    
    def plot_decomposition(self):
        """Plot signal decomposition."""
        if len(self.data) < 2:
            return
        self.fig.clear()
        
        try:
            clean_data = np.where(np.isfinite(self.data), self.data, 0)
            clean_binding = np.where(np.isfinite(self.binding_data), self.binding_data, 0)
            
            grav, agency, chaos = ETMath.decompose_signal(clean_data)
            
            LOGGER.log_observatory('decomposition', f"Gravity mean: {np.mean(grav):.4f}")
            LOGGER.log_observatory('decomposition', f"Agency std: {np.std(agency):.4f}")
            LOGGER.log_observatory('decomposition', f"Chaos std: {np.std(chaos):.4f}")
            
            ax1 = self.fig.add_subplot(311, facecolor="black")
            ax1.plot(clean_data, color="#333", label="Raw Flux (D)", linewidth=0.5)
            ax1b = ax1.twinx()
            ax1b.plot(clean_binding, color="lime", label="Binding (T)", alpha=0.5, linewidth=0.8)
            ax1.set_title("SUBSTANTIATION & BINDING (ET) / Flux & Absorption (Conv.)", color="white", fontsize=8)
            ax1.tick_params(colors="white")
            ax1b.tick_params(colors="lime")
            ax1.set_ylabel("Flux (D)", color="white", fontsize=7)
            ax1b.set_ylabel("Binding (T)", color="lime", fontsize=7)
            
            ax2 = self.fig.add_subplot(312, facecolor="black")
            ax2.plot(agency, color="red", linewidth=0.8)
            ax2.set_title("AGENCY (T_intent) / High-Jerk Residual", color="white", fontsize=8)
            ax2.tick_params(colors="white")
            ax2.set_ylabel("Amplitude", color="white", fontsize=7)
            
            ax3 = self.fig.add_subplot(313, facecolor="black")
            ax3.plot(chaos, color="cyan", alpha=0.5, linewidth=0.5)
            ax3.set_title("CHAOS (Entropy) / Noise Field", color="white", fontsize=8)
            ax3.tick_params(colors="white")
            ax3.set_ylabel("Amplitude", color="white", fontsize=7)
            
            self.fig.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            LOGGER.log_observatory('decomposition', f"Error: {e}")
    
    def plot_phase(self):
        """Plot phase space."""
        if len(self.data) < 2:
            return
        self.fig.clear()
        
        try:
            clean_data = self.data[np.isfinite(self.data)]
            if len(clean_data) < 3:
                return
            
            ax = self.fig.add_subplot(111, facecolor="black")
            d = clean_data[:-1]
            v = np.diff(clean_data)
            
            colors = np.arange(len(d))
            scatter = ax.scatter(d, v, c=colors, cmap="inferno", s=1, alpha=0.6)
            
            ax.set_title("PHASE SPACE: Position (D) vs Velocity (∂D/∂t)", color="white")
            ax.set_xlabel("Position Descriptor (D)", color="white")
            ax.set_ylabel("Velocity = ∂D/∂t", color="white")
            ax.tick_params(colors="white")
            
            self.fig.colorbar(scatter, ax=ax, label="Time Index")
            
            LOGGER.log_observatory('phase', f"Phase space points: {len(d)}")
            
            self.canvas.draw()
            
        except Exception as e:
            LOGGER.log_observatory('phase', f"Error: {e}")
    
    def plot_horizon(self):
        """Plot variance horizon."""
        if len(self.data) < 20:
            return
        self.fig.clear()
        
        try:
            clean_data = self.data[np.isfinite(self.data)]
            if len(clean_data) < 20:
                return
            
            ax = self.fig.add_subplot(111, facecolor="black")
            
            # Use manifold-aligned window
            window = max(5, min(50, len(clean_data) // MANIFOLD.base_symmetry))
            
            vars_list = []
            for i in range(len(clean_data) - window):
                segment = clean_data[i:i + window]
                if len(segment) >= 2:
                    vars_list.append(np.var(segment))
                else:
                    vars_list.append(0.0)
            
            if len(vars_list) == 0:
                return
            
            ax.plot(vars_list, color="yellow", linewidth=0.8, label="Running Variance")
            
            # Show detected manifold fold and higher harmonics
            for level in range(3):
                fold = MANIFOLD.get_fold(level)
                resonance = 1.0 / fold
                ax.axhline(resonance, color=["red", "orange", "yellow"][level],
                          linestyle="--", linewidth=1 if level == 0 else 0.5,
                          alpha=0.7 if level == 0 else 0.4,
                          label=f"1/{fold} ({fold}-fold)")
            
            ax.set_title(f"VARIANCE HORIZON (Detected: {MANIFOLD.detected_fold}-fold)", color="white")
            ax.set_xlabel("Position", color="white")
            ax.set_ylabel("Variance", color="white")
            ax.legend(loc="upper right", fontsize=8)
            ax.tick_params(colors="white")
            
            LOGGER.log_observatory('horizon', f"Variance range: {min(vars_list):.6f} - {max(vars_list):.6f}")
            
            self.canvas.draw()
            
        except Exception as e:
            LOGGER.log_observatory('horizon', f"Error: {e}")
    
    def plot_rotation(self):
        """Plot galactic rotation curve."""
        if len(self.data) < 2:
            return
        self.fig.clear()
        
        try:
            clean_data = self.data[np.isfinite(self.data)]
            ax = self.fig.add_subplot(111, facecolor="black")
            
            r = np.linspace(1, 50, max(50, len(clean_data)))
            
            m = np.cumsum(np.abs(clean_data[:len(r)])) if len(clean_data) >= len(r) else \
                np.cumsum(np.abs(np.interp(np.arange(len(r)), np.arange(len(clean_data)), clean_data)))
            
            if np.max(m) > 0:
                m = m / np.max(m) * 1e10
            
            v_newt, v_et = ETMath.solve_rotation_curve(r, m)
            
            ax.plot(r, v_newt, color="cyan", linestyle="--", linewidth=2,
                   label="Newtonian (BASE MATH)")
            ax.plot(r, v_et, color="lime", linewidth=2,
                   label=f"ET Corrected (1/{MANIFOLD.detected_fold} Variance)")
            
            v_obs = v_newt * (1 + 0.3 * np.log10(r + 1) / np.log10(51))
            ax.plot(r, v_obs, color="white", linestyle=":", linewidth=1, alpha=0.5,
                   label="Typical Observed (schematic)")
            
            ax.set_title("GALACTIC ROTATION CURVE: Newtonian vs ET", color="white")
            ax.set_xlabel("Radius (kpc)", color="white")
            ax.set_ylabel("Velocity (km/s)", color="white")
            ax.legend(loc="lower right", fontsize=8)
            ax.tick_params(colors="white")
            ax.set_xlim(0, 55)
            
            LOGGER.log_observatory('rotation', f"Max Newtonian v: {np.max(v_newt):.2f} km/s")
            LOGGER.log_observatory('rotation', f"Max ET v: {np.max(v_et):.2f} km/s")
            
            self.canvas.draw()
            
        except Exception as e:
            LOGGER.log_observatory('rotation', f"Error: {e}")
    
    def plot_spectrum(self):
        """Plot spectrum with binding overlay."""
        if len(self.data) < 2:
            return
        self.fig.clear()
        
        try:
            clean_data = self.data[np.isfinite(self.data)]
            clean_binding = self.binding_data[np.isfinite(self.binding_data)]
            
            if self.wavelengths is not None and len(self.wavelengths) == len(self.data):
                wv = self.wavelengths[np.isfinite(self.data)]
            else:
                wv = np.linspace(3800, 8000, len(clean_data))
            
            ax = self.fig.add_subplot(111, facecolor="black")
            
            ax.plot(wv, clean_data, color="white", linewidth=0.5, label="Flux")
            
            ax2 = ax.twinx()
            ax2.plot(wv, clean_binding[:len(wv)], color="lime", alpha=0.3, linewidth=0.5, label="Binding")
            
            # Mark discovered absorption lines
            wv_min, wv_max = np.min(wv), np.max(wv)
            for target_wv, (name, depth) in self.discovered_lines.items():
                if wv_min <= target_wv <= wv_max:
                    ax.axvline(target_wv, color="red", linestyle="--", alpha=0.5, linewidth=0.5)
                    ax.text(target_wv, ax.get_ylim()[1], name.split()[0],
                           color="red", fontsize=6, rotation=90, va="top")
            
            ax.set_title("SPECTRUM ANALYSIS: Flux & Binding", color="white")
            ax.set_xlabel("Wavelength (Å)", color="white")
            ax.set_ylabel("Flux (D)", color="white")
            ax2.set_ylabel("Binding (T)", color="lime")
            ax.tick_params(colors="white")
            ax2.tick_params(colors="lime")
            
            LOGGER.log_observatory('spectrum', f"Wavelength range: {wv_min:.1f} - {wv_max:.1f} Å")
            
            self.canvas.draw()
            
        except Exception as e:
            LOGGER.log_observatory('spectrum', f"Error: {e}")
    
    def plot_fourier(self):
        """Plot Fourier power spectrum."""
        if len(self.data) < 20:
            return
        self.fig.clear()
        
        try:
            clean_data = self.data[np.isfinite(self.data)]
            
            fft = np.fft.fft(clean_data - np.mean(clean_data))
            power = np.abs(fft) ** 2
            freq = np.fft.fftfreq(len(clean_data))
            
            pos_mask = freq > 0
            freq = freq[pos_mask]
            power = power[pos_mask]
            
            ax = self.fig.add_subplot(111, facecolor="black")
            ax.semilogy(freq, power, color="magenta", linewidth=0.5)
            
            threshold = np.percentile(power, 95)
            peaks = power > threshold
            ax.scatter(freq[peaks], power[peaks], color="yellow", s=10, zorder=5)
            
            ax.set_title("FOURIER POWER SPECTRUM", color="white")
            ax.set_xlabel("Frequency (1/sample)", color="white")
            ax.set_ylabel("Power (log scale)", color="white")
            ax.tick_params(colors="white")
            
            LOGGER.log_observatory('fourier', f"Peak power: {np.max(power):.2e}")
            
            self.canvas.draw()
            
        except Exception as e:
            LOGGER.log_observatory('fourier', f"Error: {e}")
    
    # =========================================================================
    # ASTRONOMY SOLVER
    # =========================================================================
    
    def run_astro_solver(self):
        """Run the complete astronomical analysis."""
        if len(self.binding_data) < 10:
            return
        
        valid_mask = np.isfinite(self.data) & np.isfinite(self.binding_data)
        clean_binding = self.binding_data[valid_mask]
        clean_data = self.data[valid_mask]
        
        if len(clean_binding) < 10:
            self.chem_text.delete(1.0, tk.END)
            self.chem_text.insert(tk.END, "[ERROR] Insufficient valid data points\n")
            return
        
        # Chemistry Analysis
        self.chem_text.delete(1.0, tk.END)
        self.chem_text.insert(tk.END, "=" * 60 + "\n")
        self.chem_text.insert(tk.END, "  CHEMO-STRUCTURAL RESONANCE ANALYSIS v7.5\n")
        self.chem_text.insert(tk.END, "  ET Binding vs Conventional Absorption Depth\n")
        self.chem_text.insert(tk.END, "=" * 60 + "\n\n")
        
        # Manifold info
        self.chem_text.insert(tk.END, f"[MANIFOLD] Detected fold: {MANIFOLD.detected_fold}\n")
        self.chem_text.insert(tk.END, f"[MANIFOLD] Base variance: {MANIFOLD.detected_variance:.6f}\n\n")
        
        if self.wavelengths is None or len(self.wavelengths) != len(self.data):
            self.chem_text.insert(tk.END, "[ERROR] NO WAVELENGTH DATA AVAILABLE!\n")
            self.chem_text.insert(tk.END, "Please install astropy: pip install astropy\n")
            LOGGER.log_astronomy('chemistry', "No wavelength calibration")
            return
        
        wv = self.wavelengths[valid_mask]
        res_info = ETMath.classify_resolution(wv)
        
        self.chem_text.insert(tk.END, f"[RESOLUTION]\n")
        self.chem_text.insert(tk.END, f"  Class: {res_info['class']}\n")
        self.chem_text.insert(tk.END, f"  Step: {res_info['step_angstroms']:.4f} Å/pixel\n")
        self.chem_text.insert(tk.END, f"  {res_info['description']}\n\n")
        
        # Dynamically discovered lines
        self.chem_text.insert(tk.END, f"[DYNAMICALLY DISCOVERED FEATURES]\n")
        self.chem_text.insert(tk.END, f"  Found: {len(self.discovered_lines)} absorption features\n\n")
        
        if self.discovered_lines:
            self.chem_text.insert(tk.END, f"{'Wavelength':<12} {'Element':<25} {'Binding':<10}\n")
            self.chem_text.insert(tk.END, "-" * 50 + "\n")
            
            for wv_found, (name, depth) in sorted(self.discovered_lines.items()):
                self.chem_text.insert(tk.END, f"{wv_found:>10.2f}Å  {name:<25} {depth*100:>7.2f}%\n")
                LOGGER.log_astronomy('chemistry', f"Found: {name} at {wv_found:.2f}Å ({depth*100:.2f}%)")
        
        self.chem_text.insert(tk.END, "\n")
        
        # Known lines analysis
        self.chem_text.insert(tk.END, "[KNOWN LINE COMPARISON]\n\n")
        
        flux_p95 = np.percentile(clean_data, 95)
        wv_min, wv_max = np.min(wv), np.max(wv)
        
        detection_threshold = res_info['detection_threshold']
        found_any = False
        
        for target_wv, (name, expected_depth) in ETMath.CHEMO_RESONANCE_MAP.items():
            if wv_min <= target_wv <= wv_max:
                idx = np.argmin(np.abs(wv - target_wv))
                
                context_radius = max(3, int(5 / max(res_info['step_angstroms'], 0.1)))
                start_idx = max(0, idx - context_radius)
                end_idx = min(len(clean_binding), idx + context_radius + 1)
                
                local_binding = clean_binding[start_idx:end_idx]
                observed_binding = np.max(local_binding) if len(local_binding) > 0 else 0
                
                status = "✗ (not detected)"
                if observed_binding >= detection_threshold:
                    found_any = True
                    ratio = observed_binding / max(expected_depth * res_info['resolution_factor'], 0.01)
                    if 0.5 <= ratio <= 2.0:
                        status = "✓ CONFIRMED"
                    elif ratio > 2.0:
                        status = "✓ ENHANCED"
                    else:
                        status = "⚠ WEAK"
                
                self.chem_text.insert(tk.END, f"  {name:<25} @ {target_wv:>7.1f}Å: {status}\n")
                self.chem_text.insert(tk.END, f"    Binding: {observed_binding*100:>6.2f}%\n")
        
        # Summary
        self.chem_text.insert(tk.END, "\n" + "-" * 40 + "\n")
        self.chem_text.insert(tk.END, f"[DATA SUMMARY]\n")
        self.chem_text.insert(tk.END, f"  Wavelength: {wv_min:.1f} - {wv_max:.1f} Å\n")
        self.chem_text.insert(tk.END, f"  Mean Flux: {np.nanmean(clean_data):.4e}\n")
        self.chem_text.insert(tk.END, f"  Manifold Fold: {MANIFOLD.detected_fold}\n")
        
        if found_any:
            self.chem_text.insert(tk.END, "\n[SUCCESS] Spectral fingerprints detected.\n")
        else:
            self.chem_text.insert(tk.END, "\n[INFO] No primary absorption lines found.\n")
        
        # Orbital Prediction
        self.orbit_text.delete(1.0, tk.END)
        self.orbit_text.insert(tk.END, "=" * 55 + "\n")
        self.orbit_text.insert(tk.END, "  ORBITAL TRAJECTORY PREDICTION v7.5\n")
        self.orbit_text.insert(tk.END, f"  Using {MANIFOLD.detected_fold}-fold manifold\n")
        self.orbit_text.insert(tk.END, "=" * 55 + "\n\n")
        
        mass = max(1.0, np.nanmean(np.abs(clean_data)) * 10)
        pos = np.array([1.0, 0.0, 0.0])
        vel = np.array([0.0, 0.5, 0.1])
        
        traj_newton = ETMath.predict_orbit_newtonian(pos, vel, mass)
        traj_et = ETMath.predict_orbit_et(pos, vel, mass)
        
        self.orbit_text.insert(tk.END, f"Initial Conditions:\n")
        self.orbit_text.insert(tk.END, f"  Position: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]\n")
        self.orbit_text.insert(tk.END, f"  Velocity: [{vel[0]:.2f}, {vel[1]:.2f}, {vel[2]:.2f}]\n")
        self.orbit_text.insert(tk.END, f"  Mass: {mass:.2f} M☉\n")
        self.orbit_text.insert(tk.END, f"  Manifold Variance: 1/{MANIFOLD.detected_fold}\n\n")
        
        self.orbit_text.insert(tk.END, f"{'STEP':<6} {'RA_N':<10} {'DEC_N':<10} {'DIST_N':<10} │ {'RA_ET':<10} {'DEC_ET':<10} {'DIST_ET':<10}\n")
        self.orbit_text.insert(tk.END, "-" * 75 + "\n")
        
        for i in range(0, 100, 20):
            p_n = traj_newton[i]
            p_e = traj_et[i]
            
            dist_n = np.linalg.norm(p_n)
            dist_e = np.linalg.norm(p_e)
            
            dec_n = math.degrees(math.asin(p_n[2] / dist_n)) if dist_n > 0 else 0
            ra_n = math.degrees(math.atan2(p_n[1], p_n[0])) % 360
            
            dec_e = math.degrees(math.asin(p_e[2] / dist_e)) if dist_e > 0 else 0
            ra_e = math.degrees(math.atan2(p_e[1], p_e[0])) % 360
            
            self.orbit_text.insert(tk.END,
                f"{i:<6} {ra_n:<10.2f} {dec_n:<10.2f} {dist_n:<10.4f} │ {ra_e:<10.2f} {dec_e:<10.2f} {dist_e:<10.4f}\n")
        
        final_diff = np.linalg.norm(traj_newton[-1] - traj_et[-1])
        self.orbit_text.insert(tk.END, f"\n[DIVERGENCE] Final: {final_diff:.6f} units\n")
        self.orbit_text.insert(tk.END, f"[ET] Variance field: 1/{MANIFOLD.detected_fold} = {MANIFOLD.detected_variance:.6f}\n")
        
        LOGGER.log_astronomy('orbital', f"Trajectory divergence: {final_diff:.6f}")
    
    # =========================================================================
    # 2D IMAGE PROCESSING
    # =========================================================================
    
    def process_2d_image(self):
        """Process 2D image with ET Manifold Scanner."""
        if self.image_2d is None:
            LOGGER.log_image('processing', "No 2D image loaded")
            return
        
        LOGGER.log_image('processing', f"Processing {self.image_2d.shape}...")
        header = UniversalIngestor._last_header
        
        processed, d_field, unc_map, quality = self.scanner._process_2d_slice(
            self.image_2d, header=header)
        
        self.processed_2d = processed
        self.d_field = d_field
        self.unc_map = unc_map
        self.quality_mask = quality
        
        # Update stats
        stats = (f"Shape: {self.image_2d.shape}\n"
                f"D-field mean: {np.mean(d_field):.3f}\n"
                f"Good pixels: {np.sum(quality==0)/quality.size*100:.1f}%\n"
                f"Manifold: {MANIFOLD.detected_fold}-fold")
        self.lbl_2d_stats.config(text=stats)
        
        LOGGER.log_image('stats', stats.replace('\n', ' | '))
        
        self.show_2d_view('processed')
    
    def show_2d_view(self, view_type: str):
        """Display 2D image view."""
        self.fig_2d.clear()
        ax = self.fig_2d.add_subplot(111, facecolor="black")
        
        if view_type == 'original' and self.image_2d is not None:
            im = ax.imshow(self.image_2d, cmap='viridis', origin='lower')
            ax.set_title("ORIGINAL", color="white")
            self.fig_2d.colorbar(im, ax=ax)
        elif view_type == 'processed' and self.processed_2d is not None:
            im = ax.imshow(self.processed_2d, cmap='viridis', origin='lower')
            ax.set_title("PROCESSED", color="white")
            self.fig_2d.colorbar(im, ax=ax)
        elif view_type == 'd_field' and self.d_field is not None:
            im = ax.imshow(self.d_field, cmap='inferno', origin='lower', vmin=0, vmax=1)
            ax.set_title("D-FIELD", color="white")
            self.fig_2d.colorbar(im, ax=ax)
        elif view_type == 'uncertainty' and self.unc_map is not None:
            im = ax.imshow(np.log10(self.unc_map + 1e-10), cmap='plasma', origin='lower')
            ax.set_title("UNCERTAINTY (log)", color="white")
            self.fig_2d.colorbar(im, ax=ax)
        elif view_type == 'quality' and self.quality_mask is not None:
            im = ax.imshow(self.quality_mask, cmap='RdYlGn_r', origin='lower', vmin=0, vmax=2)
            ax.set_title("QUALITY", color="white")
            self.fig_2d.colorbar(im, ax=ax)
        else:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha='center', color='white')
        
        ax.tick_params(colors="white")
        self.canvas_2d.draw()
    
    # =========================================================================
    # EXPORT FUNCTIONS
    # =========================================================================
    
    def export_all(self):
        """Export comprehensive analysis report."""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Export Complete Analysis"
        )
        if not filepath:
            return
        
        # Gather analysis results
        results = {
            'filename': self.filename,
            'data_type': self.data_type,
            'data_points': len(self.data),
            'manifold_fold': MANIFOLD.detected_fold,
            'manifold_variance': MANIFOLD.detected_variance,
        }
        
        if len(self.binding_data) > 0:
            t_sigs = ETMath.classify_t_signature(self.binding_data[np.isfinite(self.binding_data)])
            results['t_classification'] = t_sigs
        
        if self.discovered_lines:
            results['discovered_lines'] = {str(k): v for k, v in self.discovered_lines.items()}
        
        manifold_info = {
            'detected_fold': MANIFOLD.detected_fold,
            'detected_variance': MANIFOLD.detected_variance,
            'resonance_threshold': MANIFOLD.get_resonance_threshold(),
            'gaze_threshold': MANIFOLD.get_gaze_threshold(),
            'fold_sequence': MANIFOLD.get_fold_sequence()
        }
        
        ETExporter.export_analysis_report(filepath, results, manifold_info)
        messagebox.showinfo("Export Complete", f"Analysis exported to:\n{filepath}")
    
    def export_logs_dialog(self):
        """Export all logs to file."""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".log",
            filetypes=[("Log files", "*.log"), ("Text files", "*.txt"), ("All files", "*.*")],
            title="Export Logs"
        )
        if not filepath:
            return
        
        ETExporter.export_logs(filepath)
        messagebox.showinfo("Export Complete", f"Logs exported to:\n{filepath}")
    
    def export_scanner(self):
        """Export scanner tab data and visualization."""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("PDF", "*.pdf"), ("All files", "*.*")],
            title="Export Scanner View"
        )
        if not filepath:
            return
        
        ETExporter.export_image(self.fig_scanner, filepath)
        
        # Also export data as CSV
        csv_path = filepath.rsplit('.', 1)[0] + "_data.csv"
        data_dict = {
            'data': self.data,
            'binding': self.binding_data
        }
        if self.wavelengths is not None:
            data_dict['wavelength'] = self.wavelengths
        
        ETExporter.export_data_csv(csv_path, data_dict)
        messagebox.showinfo("Export Complete", f"Scanner exported to:\n{filepath}\n{csv_path}")
    
    def export_observatory(self):
        """Export current observatory view."""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("PDF", "*.pdf"), ("All files", "*.*")],
            title="Export Observatory View"
        )
        if not filepath:
            return
        
        ETExporter.export_image(self.fig, filepath, dpi=int(150 * self.zoom_level.get()))
        messagebox.showinfo("Export Complete", f"View exported to:\n{filepath}")
    
    def export_text(self, text_widget, name: str):
        """Export text widget content."""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title=f"Export {name.title()}"
        )
        if not filepath:
            return
        
        content = text_widget.get(1.0, tk.END)
        with open(filepath, 'w') as f:
            f.write(content)
        
        LOGGER.log_system(f"Exported {name} to: {filepath}")
        messagebox.showinfo("Export Complete", f"Exported to:\n{filepath}")
    
    def export_2d_image(self):
        """Export 2D processed image."""
        if self.processed_2d is None:
            messagebox.showwarning("No Data", "No processed 2D image to export")
            return
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG Image", "*.png"),
                ("FITS File", "*.fits"),
                ("NumPy Array", "*.npy"),
                ("All files", "*.*")
            ],
            title="Export 2D Image"
        )
        if not filepath:
            return
        
        ETExporter.export_2d_image(self.processed_2d, filepath)
        messagebox.showinfo("Export Complete", f"Image exported to:\n{filepath}")
    
    def export_2d_data(self):
        """Export all 2D processing results as CSV."""
        if self.processed_2d is None:
            messagebox.showwarning("No Data", "No processed 2D data to export")
            return
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".npz",
            filetypes=[
                ("NumPy Archive", "*.npz"),
                ("All files", "*.*")
            ],
            title="Export 2D Data"
        )
        if not filepath:
            return
        
        np.savez(filepath,
                 original=self.image_2d,
                 processed=self.processed_2d,
                 d_field=self.d_field,
                 uncertainty=self.unc_map,
                 quality=self.quality_mask)
        
        LOGGER.log_system(f"Exported 2D data to: {filepath}")
        messagebox.showinfo("Export Complete", f"Data exported to:\n{filepath}")
