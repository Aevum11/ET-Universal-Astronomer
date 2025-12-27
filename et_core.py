#!/usr/bin/env python3
"""
ET CORE - The Grounding Infrastructure
=======================================
Exception Theory Universal Astronomer v7.5

This module provides the foundational logging system and global infrastructure.
In ET terms, this is The Exception - the grounding that cannot be otherwise.

From Rules of Exception Law:
"The Exception is the grounding of reality. All else is prosperity."

All other modules depend on this core infrastructure.
"""

from datetime import datetime
from typing import Dict, Optional


class ETLogger:
    """
    Comprehensive logging system for ET analysis.
    Captures all analysis output for export functionality.
    
    ET Derivation:
    The logger is the "witness" - it records the traversal of data
    through the manifold, capturing the binding events (P°D°T).
    """
    
    def __init__(self):
        self.logs = {}
        self.reset()
    
    def reset(self):
        """Reset all logs - called when loading new file."""
        self.logs = {
            'scanner': [],
            'observatory': {
                'decomposition': [],
                'phase': [],
                'horizon': [],
                'rotation': [],
                'spectrum': [],
                'fourier': []
            },
            'astronomy': {
                'chemistry': [],
                'orbital': []
            },
            'image': {
                'processing': [],
                'stats': []
            },
            'manifold': [],
            'system': []
        }
        self.log_system("ETLogger initialized/reset")
    
    def log(self, category: str, subcategory: str, message: str):
        """Log a message to specified category/subcategory."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        entry = f"[{timestamp}] {message}"
        
        if subcategory and category in self.logs and isinstance(self.logs[category], dict):
            if subcategory in self.logs[category]:
                self.logs[category][subcategory].append(entry)
        elif category in self.logs and isinstance(self.logs[category], list):
            self.logs[category].append(entry)
        
        # Also print to console
        print(f"[{category}{'/' + subcategory if subcategory else ''}] {message}")
    
    def log_scanner(self, message: str):
        self.log('scanner', '', message)
    
    def log_observatory(self, subcategory: str, message: str):
        self.log('observatory', subcategory, message)
    
    def log_astronomy(self, subcategory: str, message: str):
        self.log('astronomy', subcategory, message)
    
    def log_image(self, subcategory: str, message: str):
        self.log('image', subcategory, message)
    
    def log_manifold(self, message: str):
        self.log('manifold', '', message)
    
    def log_system(self, message: str):
        self.log('system', '', message)
    
    def get_logs(self, category: str = None, subcategory: str = None) -> str:
        """Get logs as formatted string."""
        if category is None:
            # Return all logs
            output = []
            for cat, content in self.logs.items():
                output.append(f"\n{'='*60}\n{cat.upper()} LOGS\n{'='*60}\n")
                if isinstance(content, dict):
                    for subcat, entries in content.items():
                        if entries:
                            output.append(f"\n--- {subcat.upper()} ---\n")
                            output.extend(entries)
                else:
                    output.extend(content)
            return '\n'.join(output)
        
        if category in self.logs:
            if subcategory and isinstance(self.logs[category], dict):
                if subcategory in self.logs[category]:
                    return '\n'.join(self.logs[category][subcategory])
            elif isinstance(self.logs[category], list):
                return '\n'.join(self.logs[category])
            elif isinstance(self.logs[category], dict):
                output = []
                for subcat, entries in self.logs[category].items():
                    if entries:
                        output.append(f"\n--- {subcat.upper()} ---\n")
                        output.extend(entries)
                return '\n'.join(output)
        return ""


# ==============================================================================
# GLOBAL INSTANCES (The Grounding)
# ==============================================================================
# These are the singular instances that all modules share.
# In ET terms, this is analogous to "The Exception" - one but many share it.

LOGGER = ETLogger()

# Placeholder for MANIFOLD - will be set by et_manifold module
# This allows circular import resolution while maintaining global access
_MANIFOLD = None


def get_manifold():
    """Get the global manifold instance."""
    global _MANIFOLD
    return _MANIFOLD


def set_manifold(manifold):
    """Set the global manifold instance."""
    global _MANIFOLD
    _MANIFOLD = manifold
