#!/usr/bin/env python3
"""
EXCEPTION THEORY UNIVERSAL ASTRONOMER
======================================
Main Entry Point v7.5

This is the main entry point for running the modular astrogazer application.

USAGE:
------
    python main.py
    
    # Or as module
    python -m astrogazer_modular


ET ARCHITECTURE:
----------------
This entry point represents the "invocation" - the moment when the Exception
manifests into Something. It is the P°D°T binding that brings the application
into existence.

From Rules of Exception Law:
"The Exception binds to become Something, or it remains Nothing."
"""

import sys
import os

# Ensure we can import from the package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    """
    Main entry point for the Exception Theory Universal Astronomer.
    
    This function:
    1. Initializes the ET manifold (P)
    2. Loads mathematical operations (D)
    3. Prepares the scanner/traverser (T)
    4. Binds everything into the application (°)
    5. Runs the GUI
    """
    print("=" * 70)
    print("EXCEPTION THEORY UNIVERSAL ASTRONOMER v7.5")
    print("=" * 70)
    print()
    print("Initializing ET Modular Architecture...")
    print()
    
    try:
        # Import the application
        from astrogazer_modular import ETApp, get_manifold_info
        
        # Display manifold state
        print("Manifold State:")
        info = get_manifold_info()
        if isinstance(info, dict):
            print(f"  - Fold: {info['fold']}")
            print(f"  - Base Variance: {info['base_variance']:.6f}")
            print(f"  - Resonance Threshold: {info['resonance_threshold']:.6f}")
            print(f"  - Gaze Threshold: {info['gaze_threshold']:.6f}")
            print(f"  - Higher Folds: {info['higher_folds']}")
        else:
            print(f"  {info}")
        print()
        
        print("Starting GUI Application...")
        print("-" * 70)
        
        # Create and run the application
        app = ETApp()
        app.run()
        
    except ImportError as e:
        print(f"ERROR: Failed to import module: {e}")
        print()
        print("Attempting direct import fallback...")
        
        # Fallback: Try importing from current directory
        try:
            from et_core import LOGGER
            from et_manifold import MANIFOLD
            from et_math import ETMath
            from et_scanner import ETConfig, ETManifoldScanner
            from et_ingestion import UniversalIngestor
            from et_export import ETExporter
            from et_app import ETApp
            
            print("Direct import successful!")
            app = ETApp()
            app.run()
            
        except ImportError as e2:
            print(f"ERROR: Direct import also failed: {e2}")
            print()
            print("Please ensure all modules are in the correct location:")
            print("  - et_core.py")
            print("  - et_manifold.py")
            print("  - et_math.py")
            print("  - et_scanner.py")
            print("  - et_ingestion.py")
            print("  - et_export.py")
            print("  - et_app.py")
            sys.exit(1)
            
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def cli():
    """
    Command-line interface entry point.
    Supports additional arguments for batch processing.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Exception Theory Universal Astronomer v7.5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Launch GUI
  python main.py --version          # Show version
  python main.py --info             # Show manifold info
  python main.py --check            # Check module integrity
        """
    )
    
    parser.add_argument('--version', action='store_true',
                        help='Show version information')
    parser.add_argument('--info', action='store_true',
                        help='Show manifold state information')
    parser.add_argument('--check', action='store_true',
                        help='Check module integrity')
    
    args = parser.parse_args()
    
    if args.version:
        print("Exception Theory Universal Astronomer")
        print("Version: 7.5.0 Modular")
        print("Architecture: ET PDT (Point-Descriptor-Traverser)")
        return
    
    if args.info:
        try:
            from astrogazer_modular import get_manifold_info, MANIFOLD
            print("ET Manifold State")
            print("=" * 40)
            info = get_manifold_info()
            if isinstance(info, dict):
                for key, value in info.items():
                    print(f"  {key}: {value}")
            print()
            print("Full Manifold Constants:")
            print(f"  Koide: {MANIFOLD.koide_constant}")
            print(f"  Dark Energy: {MANIFOLD.dark_energy_ratio}")
            print(f"  Dark Matter: {MANIFOLD.dark_matter_ratio}")
            print(f"  Baryonic: {MANIFOLD.baryonic_ratio}")
        except ImportError as e:
            print(f"Error loading manifold: {e}")
        return
    
    if args.check:
        print("Checking Module Integrity...")
        print("=" * 40)
        modules = [
            ('et_core', ['ETLogger', 'LOGGER', 'get_manifold', 'set_manifold']),
            ('et_manifold', ['ManifoldGeometry', 'MANIFOLD']),
            ('et_math', ['ETMath']),
            ('et_scanner', ['ETConfig', 'ETManifoldScanner']),
            ('et_ingestion', ['UniversalIngestor']),
            ('et_export', ['ETExporter']),
            ('et_app', ['ETApp'])
        ]
        
        all_ok = True
        for mod_name, exports in modules:
            try:
                # Try package import
                try:
                    mod = __import__(f'astrogazer_modular.{mod_name}', fromlist=exports)
                except ImportError:
                    mod = __import__(mod_name)
                
                missing = []
                for exp in exports:
                    if not hasattr(mod, exp):
                        missing.append(exp)
                
                if missing:
                    print(f"  [WARN] {mod_name}: Missing {missing}")
                    all_ok = False
                else:
                    print(f"  [OK] {mod_name}: All exports present")
                    
            except ImportError as e:
                print(f"  [FAIL] {mod_name}: {e}")
                all_ok = False
        
        print()
        if all_ok:
            print("All modules OK!")
        else:
            print("Some issues detected. Check module files.")
        return
    
    # Default: run GUI
    main()


if __name__ == "__main__":
    cli()
