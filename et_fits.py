#!usrbinenv python3

ET FITS ENGINE - The FITS Traverser
====================================
Exception Theory Universal Astronomer v7.5

This module implements the 'fitsio' feature set using ET principles.
It handles the traversal (T) of FITS manifolds (P), creating and 
resolving Descriptors (D) for Images and Tables.

From Rules of Exception Law
T travels P by using D as (P°D) are bound.
File IO is the mechanism of T traversing external Point substrates.


import os
import numpy as np
from astropy.io import fits
from typing import Union, Tuple, Optional, Any, List

from .et_core import LOGGER
from .et_math import ETMath

class ETFitsTraverser
    
    The Traverser agent responsible for navigating FITS manifolds.
    
    ET Derivation
    - The File is the Point Substrate (P).
    - The Header is the Descriptor Set (D).
    - The Data is the Configuration Value.
    - This class is T (Agency) resolving the [00] indeterminacy of readingwriting.
    

    @staticmethod
    def substantiate_extension(
        filename str, 
        ext Union[int, str] = 0,
        mode str = 'readonly'
    ) - Tuple[np.ndarray, fits.Header]
        
        Read (Substantiate) data from a specific FITS extension.
        
        ET Logic
        T engages with the file (P) at a specific extension (Descriptor Index).
        It collapses the potential file state into a substantiated array.
        
        Args
            filename The Point substrate location.
            ext The Descriptor index (int) or Name (str) to navigate to.
            mode Traversal mode ('readonly', 'update', 'denywrite').
            
        Returns
            (manifold_slice, descriptor_set) The data array and the header.
        
        LOGGER.log_system(fT-Traverser initiating substantiation of {filename} [{ext}])
        
        try
            # T binds to the file (P) using astropy as the interface
            with fits.open(filename, mode=mode) as hdul
                # T navigates to the specific Descriptor (Extension)
                if isinstance(ext, int) and ext = len(hdul)
                    raise IndexError(fDescriptor index {ext} out of bounds for manifold {filename})
                
                hdu = hdul[ext]
                
                # Verify Descriptor Type
                is_image = isinstance(hdu, (fits.PrimaryHDU, fits.ImageHDU))
                is_table = isinstance(hdu, (fits.BinTableHDU, fits.TableHDU))
                
                if not (is_image or is_table)
                    LOGGER.log_system(fWarning Extension type {type(hdu)} is obscure.)

                # Substantiate the data (Load into memory)
                # In ET, this is resolving the Indeterminate - Determinate
                data = hdu.data
                header = hdu.header
                
                # Check for Zero descriptors (Absence)
                if data is None
                    LOGGER.log_system(Substantiated 'Zero' (Absence) at this extension.)
                    data = np.array([]) # Empty manifold slice

                LOGGER.log_system(fSubstantiation complete. D-Count {len(header)} cards.)
                
                # Return disconnected copies (Substantiated independent moments)
                # We copy to ensure the 'Moment' persists after file closure (T disengagement)
                return data.copy() if data.size  0 else data, header.copy()

        except Exception as e
            LOGGER.log_system(fIncoherence detected during substantiation {e})
            raise e

    @staticmethod
    def bind_extension(
        filename str, 
        data np.ndarray, 
        header Optional[fits.Header] = None,
        ext_type str = 'IMAGE',
        ext_name str = None,
        clobber bool = False,
        append bool = False
    )
        
        Write (Bind) data to a FITS extension.
        
        ET Logic
        T takes an internal configuration (data) and binds it to an external 
        Point substrate (file), creating a permanent Descriptor record.
        
        Args
            filename The Point substrate location.
            data The configuration to bind (numpy array).
            header The descriptor set describing the configuration.
            ext_type 'IMAGE', 'BINARY_TBL', or 'ASCII_TBL'.
            ext_name The name descriptor for the extension.
            clobber If True, overwrite existing Point substrate.
            append If True, add to existing manifold structure.
        
        LOGGER.log_system(fT-Traverser initiating binding to {filename} as {ext_type})

        # Construct the HDU (The Container for P ∘ D)
        hdu = None
        
        if header is None
            header = fits.Header()
            
        if ext_name
            header['EXTNAME'] = ext_name

        try
            # Select appropriate Descriptor Template based on ext_type
            if ext_type.upper() in ['IMAGE', 'IMG']
                hdu = fits.ImageHDU(data=data, header=header)
            elif ext_type.upper() in ['BINARY_TBL', 'BINTABLE']
                # For tables, we must ensure data is a record array or Table object
                if not isinstance(data, (np.ndarray, fits.FITS_rec))
                     # Attempt conversion if T provided raw list
                     pass 
                hdu = fits.BinTableHDU(data=data, header=header)
            elif ext_type.upper() in ['ASCII_TBL', 'TABLE']
                hdu = fits.TableHDU(data=data, header=header)
            else
                raise ValueError(fUnknown Descriptor Type {ext_type})

            # Binding Operation (IO)
            if append
                # Append requires navigating to end of existing manifold
                if not os.path.exists(filename)
                     # If P doesn't exist, Create Primary + Extension
                     primary = fits.PrimaryHDU()
                     hdul = fits.HDUList([primary, hdu])
                     hdul.writeto(filename)
                else
                    # Append to existing
                    with fits.open(filename, mode='append') as hdul
                        hdul.append(hdu)
                        # Flush handled by context manager usually, but 'append' mode writes on close
            else
                # NewOverwrite
                hdul = fits.HDUList([fits.PrimaryHDU(), hdu]) if not isinstance(hdu, fits.PrimaryHDU) else fits.HDUList([hdu])
                hdul.writeto(filename, overwrite=clobber)
                
            LOGGER.log_system(fBinding complete. Configuration externalized to {filename}.)

        except Exception as e
            LOGGER.log_system(fIncoherence during binding {e})
            raise e

    @staticmethod
    def create_table_column(
        name str, 
        format_type str, 
        array np.ndarray, 
        unit str = None
    ) - fits.Column
        
        Create a Column Descriptor for tables.
        
        ET Logic
        Defining a sub-descriptor (Column) for a larger Descriptor set (Table).
        
        return fits.Column(name=name, format=format_type, array=array, unit=unit)