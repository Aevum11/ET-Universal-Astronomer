"""
EXCEPTION THEORY UNIVERSAL ASTRONOMER
======================================
Modular Package Architecture v7.5

This package implements astronomical analysis tools built entirely on Exception Theory (ET)
mathematics. The modular structure directly mirrors the ET primitive hierarchy:

ARCHITECTURE MAPPING TO ET PRIMITIVES:
--------------------------------------

    ┌─────────────────────────────────────────────────────────────┐
    │                    THE EXCEPTION (E)                        │
    │                      et_core.py                             │
    │   "For every exception there is an exception, except the    │
    │    exception" - The grounding infrastructure all share      │
    └─────────────────────────────────────────────────────────────┘
                                │
            ┌───────────────────┼───────────────────┐
            │                   │                   │
            ▼                   ▼                   ▼
    ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
    │   P (Point)   │   │ D (Descriptor)│   │ T (Traverser) │
    │  et_manifold  │   │   et_math     │   │  et_scanner   │
    │   Infinite    │   │    Finite     │   │ Indeterminate │
    │   Substrate   │   │  Constraints  │   │    Agency     │
    └───────────────┘   └───────────────┘   └───────────────┘
            │                   │                   │
            └───────────────────┼───────────────────┘
                                │
                        ┌───────┴───────┐
                        │    ° (Bind)   │
                        │    et_app     │
                        │   P°D°T → S   │
                        │  (Something)  │
                        └───────────────┘
                                │
            ┌───────────────────┼───────────────────┐
            │                   │                   │
            ▼                   ▼                   ▼
    ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
    │  et_ingestion │   │   et_export   │   │    et_app     │
    │  Data Enters  │   │ Data Manifests│   │  Full Unity   │
    │  The Manifold │   │ From Manifold │   │    (GUI)      │
    └───────────────┘   └───────────────┘   └───────────────┘


MATHEMATICAL FOUNDATIONS:
-------------------------
All mathematics derive from the 12-fold manifold symmetry:
    - 3 Primitives × 4 Logic States = 12 Base Fold
    - Higher folds: 12 × 2^n (24, 48, 96, 192...)
    - Base Variance: 1/fold (dynamically detected)
    - Resonance Threshold: (fold + 1) / fold
    - Gaze Threshold: 1 + 2.4 / fold

KEY CONSTANTS (ET-Derived):
    - Koide Constant: 2/3
    - Dark Energy Ratio: 2/3 (~68%)
    - Dark Matter Ratio: 1/4 (~25%)
    - Baryonic Ratio: 1/12 (~8%)


MODULE DESCRIPTIONS:
--------------------
et_core.py      - Logging infrastructure, global state management
et_manifold.py  - Manifold geometry, fold calculations, symmetry operations
et_math.py      - All mathematical operations, physical constants, ET physics
et_scanner.py   - 2D image processing, kernel operations, D-field detection
et_ingestion.py - File loading (FITS, NetCDF, CSV, TXT, DAT, JSON, binary)
et_export.py    - Export functionality (logs, images, CSV, reports)
et_app.py       - Complete Tkinter GUI application


USAGE:
------
    # As a package
    from astrogazer_modular import ETApp
    app = ETApp()
    app.run()
    
    # Or run directly
    python -m astrogazer_modular
    
    # Or via main.py
    python main.py


Author: Exception Theory Development
Version: 7.5 Modular
License: ET Open Source
"""

__version__ = "7.5.0"
__author__ = "Exception Theory Development"

# ==============================================================================
# IMPORT ORDER MATTERS - Follow the ET Dependency Hierarchy
# ==============================================================================
# 1. The Exception (Core/Grounding) - must be first
from .et_core import (
    ETLogger,
    LOGGER,
    get_manifold,
    set_manifold
)

# 2. P (Point/Substrate) - Manifold geometry
from .et_manifold import (
    ManifoldGeometry,
    MANIFOLD
)

# 3. D (Descriptor/Constraints) - Mathematical operations  
from .et_math import ETMath

# 4. T (Traverser/Agency) - Scanner and processing
from .et_scanner import (
    ETConfig,
    ETManifoldScanner
)

# 5. Data flow - Ingestion (entry) and Export (manifestation)
from .et_ingestion import UniversalIngestor
from .et_export import ETExporter

# 6. The Binding (°) - Application unifies P°D°T
# GUI is optional - only import if tkinter is available
_ETApp = None  # Lazy-loaded

def _get_etapp():
    """Lazy-load ETApp to avoid tkinter dependency for non-GUI usage."""
    global _ETApp
    if _ETApp is None:
        try:
            from .et_app import ETApp
            _ETApp = ETApp
        except ImportError as e:
            raise ImportError(
                f"ETApp requires tkinter for GUI functionality: {e}\n"
                "Install tkinter or use non-GUI modules directly."
            )
    return _ETApp

# Try to import ETApp, but don't fail if tkinter unavailable
try:
    from .et_app import ETApp
except ImportError:
    ETApp = None  # Will raise helpful error if accessed


# ==============================================================================
# PACKAGE-LEVEL CONVENIENCE FUNCTIONS
# ==============================================================================

def create_app():
    """Create and return a new ETApp instance."""
    app_class = _get_etapp()
    return app_class()


def run():
    """Create and run the application."""
    app_class = _get_etapp()
    app = app_class()
    app.run()


def get_version():
    """Return package version string."""
    return __version__


def get_manifold_info():
    """Return current manifold state information."""
    m = get_manifold()
    if m is None:
        return "Manifold not initialized"
    return {
        'fold': m.fold,
        'base_variance': m.base_variance,
        'resonance_threshold': m.resonance_threshold,
        'gaze_threshold': m.gaze_threshold,
        'higher_folds': m.higher_folds[:5]  # First 5 higher folds
    }


# ==============================================================================
# EXPORTED NAMES
# ==============================================================================
__all__ = [
    # Version info
    '__version__',
    '__author__',
    
    # Core
    'ETLogger',
    'LOGGER',
    'get_manifold',
    'set_manifold',
    
    # Manifold
    'ManifoldGeometry',
    'MANIFOLD',
    
    # Math
    'ETMath',
    
    # Scanner
    'ETConfig',
    'ETManifoldScanner',
    
    # Data flow
    'UniversalIngestor',
    'ETExporter',
    
    # Application
    'ETApp',
    
    # Convenience functions
    'create_app',
    'run',
    'get_version',
    'get_manifold_info'
]
