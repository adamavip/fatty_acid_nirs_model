import numpy as np
import scipy
import scipy.signal
import scipy.io as io
import scipy.ndimage as nd

def msc(input_data, reference=None):
    ''' 
    Perform Multiplicative scatter correction

    :Args:
        input_data: numpy.ndarray
        reference: scalar
    
    :Returns:
        corrected spectra: numpy.ndarray 
    
    '''
    input_data = input_data.copy()
     # mean centre correction
    for i in range(input_data.shape[0]):
        input_data[i,:] -= input_data[i,:].mean()
 
    # Get the reference spectrum. If not given, estimate it from the mean    
    if reference is None:    
        ref = np.mean(input_data, axis=0)
    else:
        ref = reference
 
    # Define a new array and populate it with the corrected data    
    data_msc = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
        # Run regression
        fit = np.polyfit(ref, input_data[i,:], 1, full=True)
        # Apply correction
        data_msc[i,:] = (input_data[i,:] - fit[0][1]) / fit[0][0] 
 
    return data_msc

def snv(input_data):
    """
    Performs a Standard Normal Variate (SNV) correction.

    Args:
        input_data <numpy.ndarray>: NIR data matrix
        reference <float>: reference value to correct data 
    
    Returns:
        spectra <numpy.ndarray>: corrected spectra 
    
    """
    # Define a new array and populate it with the corrected data  
    output_data = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
 
        # Apply correction
        output_data[i,:] = (input_data[i,:] - np.mean(input_data[i,:])) / np.std(input_data[i,:])
 
    return output_data


def smooth(spectra, filter_win=11, window_type='flat', mode='reflect'):
    """ Smooths the spectra using convolution.
    Args:
        spectra <numpy.ndarray>: NIRS data matrix.
        filter_win <float>: length of the filter window in samples.
        window_type <str>: filtering window to use for convolution (see scipy.signal.windows)
        mode <str>: convolution mode
    Returns:
        spectra <numpy.ndarray>: Smoothed NIR spectra.
    """

    if window_type == 'flat':
        window = np.ones(filter_win)
    else:
        window = scipy.signal.windows.get_window(window_type, filter_win)
    window = window / np.sum(window)

    for row in range(spectra.shape[0]):
        spectra[row,:] = nd.convolve(spectra[row,:], window, mode=mode)

    return spectra


def savgol(spectra, filter_win=15, poly_order=2, deriv_order=0, delta=1.0):
    """ Perform Savitzky–Golay filtering on the data (also calculates derivatives). This function is a wrapper for
    scipy.signal.savgol_filter.

    Args:
        spectra <numpy.ndarray>: NIRS data matrix.
        filter_win <int>: Size of the filter window in samples (default 11).
        poly_order <int>: Order of the polynomial estimation (default 3).
        deriv_order <int>: Order of the derivation (default 0).

    Returns:
        spectra <numpy.ndarray>: NIRS data smoothed with Savitzky-Golay filtering
    """
    return scipy.signal.savgol_filter(spectra, filter_win, poly_order, deriv_order, delta=delta, axis=1)