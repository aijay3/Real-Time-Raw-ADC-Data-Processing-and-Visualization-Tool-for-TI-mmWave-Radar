"""
DSP module for radar signal processing.
"""

import numpy as np
from typing import Union, Tuple, List, Optional, Dict
import logging
from dataclasses import dataclass
from radar_parameters import RadarParameters
import scipy.signal as signal
import pyfftw
import matplotlib.pyplot as plt

# Configure pyfftw to use multiple threads for better performance
pyfftw.config.NUM_THREADS = 4
pyfftw.interfaces.cache.enable()

# Cache for FFTW plans
_FFTW_PLAN_CACHE = {}

# Configure logging for padding information only
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Pre-compute common windows for typical radar data sizes
_WINDOW_CACHE = {}

# Cache for matched filter coefficients
_MATCHED_FILTER_CACHE = {}

def apply_cfar(range_profile: np.ndarray, cfar_params: dict = None) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    """
    Apply Cell-Averaging Constant False Alarm Rate (CA-CFAR) detection with peak grouping.
    
    Args:
        range_profile: Range profile magnitude data [num_range_bins, num_rx]
        cfar_params: Dictionary containing CFAR parameters:
            - guard_cells: Number of guard cells on each side of CUT (default: 2)
            - training_cells: Number of training cells on each side of guard region (default: 4)
            - pfa: Probability of false alarm (default: 0.1)
            - group_peaks: Whether to enable peak grouping (default: True)
        
    Returns:
        Tuple of:
            - Binary detection mask same shape as input
            - List of (range_idx, magnitude) tuples for detected objects, with nearby peaks grouped
    """
    num_range_bins, num_rx = range_profile.shape
    
    # Set default parameters if not provided
    if cfar_params is None:
        cfar_params = {
            'guard_cells': 2,
            'training_cells': 4,
            'pfa': 0.1,
            'group_peaks': True
        }
    
    guard_cells = cfar_params.get('guard_cells', 2)
    training_cells = cfar_params.get('training_cells', 4)
    pfa = cfar_params.get('pfa', 0.1)
    group_peaks = cfar_params.get('group_peaks', True)
    group_threshold = 2.0  # Fixed threshold for grouping nearby peaks
    
    # Calculate threshold factor based on PFA and number of training cells
    num_training_total = 2 * training_cells
    alpha = (num_training_total * (pfa**(-1/num_training_total) - 1)) * 0.7
    
    # Initialize detection mask
    detection_mask = np.zeros_like(range_profile)
    detected_objects = []
    
    # Sum across all RX channels
    power_profile = np.sum(range_profile, axis=1)
    
    # Process the summed profile
    for cut_idx in range(num_range_bins):
            # Define training regions
            left_guard = max(0, cut_idx - guard_cells)
            left_train = max(0, cut_idx - guard_cells - training_cells)
            right_guard = min(num_range_bins, cut_idx + guard_cells + 1)
            right_train = min(num_range_bins, cut_idx + guard_cells + training_cells + 1)
            
            # Extract training cells (excluding guard cells and CUT)
            training_cells_left = power_profile[left_train:left_guard]
            training_cells_right = power_profile[right_guard:right_train]
            training_cells_data = np.concatenate([training_cells_left, training_cells_right])
            
            if len(training_cells_data) > 0:
                # Calculate threshold
                noise_level = np.mean(training_cells_data)
                threshold = alpha * noise_level
                
            # Compare CUT to threshold
            if power_profile[cut_idx] > threshold:
                detection_mask[cut_idx, :] = 1  # Mark detection across all channels
                # Store range index and magnitude for detected objects
                # Convert to dB using 20*log10 since we're working with magnitude
                detected_objects.append((cut_idx, 20 * np.log10(power_profile[cut_idx])))
    
    # Group nearby peaks if enabled
    if group_peaks:
        grouped_objects = []
        current_group = []
        
        # Sort detected objects by range index
        detected_objects.sort(key=lambda x: x[0])
        
        for obj in detected_objects:
            if not current_group:
                current_group = [obj]
            else:
                last_range = current_group[-1][0]
                if abs(obj[0] - last_range) <= group_threshold:
                    current_group.append(obj)
                else:
                    if current_group:
                        avg_range = round(sum(x[0] for x in current_group) / len(current_group))
                        # Convert back from dB, average, then back to dB
                        combined_mag = 20 * np.log10(sum(10**(x[1]/20) for x in current_group) / len(current_group))
                        grouped_objects.append((avg_range, combined_mag))
                    current_group = [obj]
        
        # Process last group
        if current_group:
            avg_range = round(sum(x[0] for x in current_group) / len(current_group))
            # Convert back from dB, average, then back to dB
            combined_mag = 20 * np.log10(sum(10**(x[1]/20) for x in current_group) / len(current_group))
            grouped_objects.append((avg_range, combined_mag))
        
        # Sort grouped objects by magnitude
        grouped_objects.sort(key=lambda x: x[1], reverse=True)
        return detection_mask, grouped_objects
    else:
        # Return ungrouped detections
        detected_objects.sort(key=lambda x: x[1], reverse=True)
        return detection_mask, detected_objects

def create_lfm_matched_filter(num_samples: int, freq_slope: float, sample_rate: float, 
                             window_type: str = 'blackmanharris') -> np.ndarray:
    """
    Create a matched filter for Linear Frequency Modulation (LFM) or Chirp pulse compression.
    
    This function creates a matched filter for LFM/Chirp signals to improve signal-to-noise ratio
    through pulse compression. The matched filter is the complex conjugate of the expected
    chirp signal.
    
    Args:
        num_samples: Number of samples in the chirp
        freq_slope: Frequency slope of the chirp in Hz/s
        sample_rate: Sampling rate in Hz
        window_type: Type of window to apply to the matched filter
        
    Returns:
        np.ndarray: Complex matched filter coefficients
    """
    # Check if we have this filter in cache
    cache_key = (num_samples, freq_slope, sample_rate, window_type)
    if cache_key in _MATCHED_FILTER_CACHE:
        return _MATCHED_FILTER_CACHE[cache_key]
    
    # Time vector
    t = np.arange(num_samples) / sample_rate
    
    # Create the chirp signal (complex exponential with quadratic phase)
    # For an FMCW radar, the beat signal phase is proportional to t²
    phase = np.pi * freq_slope * t**2
    chirp = np.exp(1j * phase)
    
    # Apply window to reduce sidelobes
    window = create_window(num_samples, window_type)
    windowed_chirp = chirp * window
    
    # Matched filter is the complex conjugate of the time-reversed chirp
    matched_filter = np.conj(windowed_chirp[::-1])
    
    # Normalize the filter
    matched_filter = matched_filter / np.sqrt(np.sum(np.abs(matched_filter)**2))
    
    # Cache the result
    _MATCHED_FILTER_CACHE[cache_key] = matched_filter
    
    return matched_filter

def apply_pulse_compression(radar_data: np.ndarray, radar_params: Optional[RadarParameters] = None) -> np.ndarray:
    """
    Apply pulse compression using matched filtering to improve SNR.
    
    Args:
        radar_data: Input radar data array [num_samples, num_rx, num_chirps]
        radar_params: Optional RadarParameters object containing radar configuration
        
    Returns:
        np.ndarray: Pulse-compressed radar data with improved SNR
    """
    if radar_params is not None:
        num_adc_samples = radar_params.config_params['adc_samples']
        sampling_rate = radar_params.config_params['sample_rate'] * 1e3  # Convert ksps to Hz
        freq_slope = radar_params.config_params['freq_slope'] * 1e12  # Convert MHz/μs to Hz/s
    else:
        num_adc_samples = radar_data.shape[0]
        sampling_rate = 10e6  # Default 10 MHz
        freq_slope = 60e12   # Default 60 MHz/μs = 60e12 Hz/s
    
    # Get the matched filter
    matched_filter = create_lfm_matched_filter(num_adc_samples, freq_slope, sampling_rate)
    
    # Get the shape of the input data
    num_samples, num_rx, num_chirps = radar_data.shape
    
    # Initialize output array
    compressed_data = np.zeros_like(radar_data)
    
    # Apply matched filter to each RX channel and chirp
    for rx in range(num_rx):
        for chirp in range(num_chirps):
            # Extract the signal for this RX and chirp
            signal = radar_data[:, rx, chirp]
            
            # Apply matched filter using convolution
            # Use 'same' mode to keep the same length as the input
            compressed = np.convolve(signal, matched_filter, mode='same')
            
            # Store the result
            compressed_data[:, rx, chirp] = compressed
    
    return compressed_data

def create_window(size: int, window_type: str = 'blackmanharris') -> np.ndarray:
    """
    Create or retrieve a cached window of specified size and type.
    
    Args:
        size: Length of the window
        window_type: Type of window ('blackmanharris', 'no window', 'hann', 'blackman', 'hamming')
    
    Returns:
        np.ndarray: Window function of specified size and type
    """
    cache_key = (size, window_type)
    if cache_key not in _WINDOW_CACHE:
        if window_type == 'no window':
            _WINDOW_CACHE[cache_key] = np.ones(size)
        elif window_type == 'hann':
            _WINDOW_CACHE[cache_key] = np.hanning(size)
        elif window_type == 'blackman':
            _WINDOW_CACHE[cache_key] = np.blackman(size)
        elif window_type == 'hamming':
            _WINDOW_CACHE[cache_key] = np.hamming(size)
        elif window_type == 'blackmanharris':
            # Standard 4-term Blackman-Harris window coefficients
            a0, a1, a2, a3 = 0.35875, 0.48829, 0.14128, 0.01168
            n = np.arange(size)
            _WINDOW_CACHE[cache_key] = (a0 
                                      - a1 * np.cos(2 * np.pi * n / (size - 1))
                                      + a2 * np.cos(4 * np.pi * n / (size - 1))
                                      - a3 * np.cos(6 * np.pi * n / (size - 1)))
        else:
            raise ValueError(f"Unsupported window type: {window_type}")
    return _WINDOW_CACHE[cache_key]

def suggest_padding_size(data_shape: tuple, radar_params: Optional[RadarParameters] = None,
                        range_padding: Optional[int] = None) -> List[int]:
    """
    Get FFT padding sizes based on input dimensions and user-specified padding.
    
    Args:
        data_shape: Shape of input data
        radar_params: Optional radar parameters
        range_padding: Optional user-specified range padding size
        
    Returns:
        List[int]: [range_padding, 0]
    """
    num_rx, _ = data_shape
    if radar_params is not None:
        num_chirps = radar_params.config_params['chirps_per_frame'] * radar_params.config_params['num_loops']
        num_adc_samples = radar_params.config_params['adc_samples']
    else:
        num_adc_samples = 256  # Default ADC samples
        num_chirps = data_shape[1] // num_adc_samples
    
    def log_padding_info(dimension: str, original: int, padded: int):
        if original != padded:
            logger.info(f"Padding applied to {dimension}: {original} -> {padded} samples")
        else:
            logger.info(f"No padding needed for {dimension}: already optimal at {original} samples")
    
    # Use user-specified padding if provided, otherwise calculate optimal padding
    if range_padding is None:
        next_pow2 = 2 ** np.ceil(np.log2(num_adc_samples)).astype(int)
        range_padding = next_pow2 * 2
    log_padding_info("range", num_adc_samples, range_padding)
    
    return [range_padding, 0]

def remove_static_clutter(range_data: np.ndarray, n_components: int = 2) -> np.ndarray:
    """
    Remove static clutter from range profile data using Principal Component Analysis (PCA).
    
    PCA is used to identify and remove the principal components that represent static clutter.
    The assumption is that static clutter will appear as strong, consistent patterns across
    multiple range bins and can be captured by the first few principal components.
    
    Args:
        range_data: Range profile data [num_range_bins, num_rx]
        n_components: Number of principal components to remove (default=2)
                     These components typically capture static clutter patterns
    
    Returns:
        Range profile data with static clutter removed using PCA
    """
    # Center the data by removing the mean
    data_mean = np.mean(range_data, axis=0, keepdims=True)
    centered_data = range_data - data_mean
    
    # Calculate covariance matrix
    covariance_matrix = np.cov(centered_data.T)
    
    # Calculate eigenvectors and eigenvalues
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    
    # Sort eigenvectors by eigenvalues in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, idx]
    
    # Select the principal components that represent clutter (first n_components)
    clutter_components = eigenvectors[:, :n_components]
    
    # Project data onto clutter subspace
    clutter = centered_data @ clutter_components @ clutter_components.T
    
    # Remove clutter by subtracting the projection
    range_data_clean = range_data - clutter
    
    return range_data_clean

def calculate_range_profile(radar_data: np.ndarray, radar_params: Optional[RadarParameters] = None, 
                          remove_clutter: bool = False, use_pulse_compression: bool = True,
                          window_type: str = 'blackmanharris', range_padding: Optional[int] = None) -> tuple:
    """
    Calculate range profile from radar data using FFT processing and averaging across chirps.
    
    This function implements proper sampling and FFT processing following Nyquist criteria:
    1. Ensures sampling rate (fs) is at least 2x the maximum beat frequency
    2. Uses zero-padding to the next power of 2 for better frequency resolution
    3. Applies windowing to reduce spectral leakage
    4. Only returns the positive frequency components (0 to fs/2) to avoid aliasing
    5. Optionally applies pulse compression to improve SNR
    
    Args:
        radar_data: Input radar data array [numRX, numChirps * numADCSamples]
        radar_params: Optional RadarParameters object containing radar configuration
        remove_clutter: Whether to apply clutter removal
        use_pulse_compression: Whether to apply LFM/Chirp pulse compression
        
    Returns:
        Tuple of (range_profile, range_axis, detected_points) where:
            range_profile: Range profile magnitude data
            range_axis: Corresponding range values in meters
            detected_points: List of (range, magnitude) tuples for detected objects,
                           with nearby peaks grouped together as single objects
    """
    # Validate input data and check Nyquist criteria
    validate_input(radar_data, radar_params)
    num_rx = radar_data.shape[0]
    if radar_params is not None:
        num_chirps = radar_params.config_params['chirps_per_frame'] * radar_params.config_params['num_loops']
        num_adc_samples = radar_params.config_params['adc_samples']
        sampling_rate = radar_params.config_params['sample_rate'] * 1e3  # Convert ksps to Hz
        freq_slope = radar_params.config_params['freq_slope'] * 1e12  # Convert MHz/μs to Hz/s (1e6 * 1e6 = 1e12)
        c = 3e8
        print(f"DEBUG: Using radar_params - num_chirps: {num_chirps}, num_adc_samples: {num_adc_samples}, sampling_rate: {sampling_rate}, freq_slope: {freq_slope}")
    else:
        num_adc_samples = 256
        num_chirps = radar_data.shape[1] // num_adc_samples
        sampling_rate = 10e6
        freq_slope = 60e6
        c = 3e8
        print(f"DEBUG: Using default values - num_chirps: {num_chirps}, num_adc_samples: {num_adc_samples}, sampling_rate: {sampling_rate}, freq_slope: {freq_slope}")
    
    reshaped_data = np.zeros((num_adc_samples, num_rx, num_chirps), dtype=np.complex64)
    for rx in range(num_rx):
        rx_data = radar_data[rx, :].reshape(-1, num_adc_samples).T
        reshaped_data[:, rx, :] = rx_data[:, :num_chirps]
    print(f"DEBUG: Reshaped data shape: {reshaped_data.shape}, sample values: {reshaped_data[0, 0, 0:3]}")
    
    # Apply pulse compression if requested
    if use_pulse_compression:
        logger.info("Applying LFM/Chirp pulse compression to improve SNR")
        reshaped_data = apply_pulse_compression(reshaped_data, radar_params)
        print(f"DEBUG: After pulse compression, data shape: {reshaped_data.shape}, sample values: {reshaped_data[0, 0, 0:3]}")
    
    window = create_window(num_adc_samples, window_type)
    windowed_data = reshaped_data * window[:, np.newaxis, np.newaxis]
    print(f"DEBUG: After windowing, data shape: {windowed_data.shape}, sample values: {windowed_data[0, 0, 0:3]}")
    
    # Get padding size with user-specified range padding
    padding_sizes = suggest_padding_size(radar_data.shape, radar_params, range_padding=range_padding)
    print(f"DEBUG: Padding sizes: {padding_sizes}")
    
    # Zero pad the data at the end
    padded_data = np.zeros((padding_sizes[0], num_rx, num_chirps), dtype=np.complex64)
    # Place data at the beginning of the padded array
    padded_data[:num_adc_samples] = windowed_data
    print(f"DEBUG: After padding, data shape: {padded_data.shape}, non-zero elements: {np.count_nonzero(padded_data)}")
    
    # Apply FFT on padded data using pyfftw
    # Check if we have a cached FFTW plan for this shape
    fft_shape = (padding_sizes[0], num_rx, num_chirps)
    fft_axis = 0
    cache_key = (fft_shape, fft_axis, 'fft')
    
    if cache_key in _FFTW_PLAN_CACHE:
        fft_obj = _FFTW_PLAN_CACHE[cache_key]
        # Copy data to input array
        np.copyto(fft_obj.input_array, padded_data)
        # Execute the plan
        fft_obj()
        # Get the result
        range_ffts = fft_obj.output_array
    else:
        # Create a new FFTW plan
        fft_input = pyfftw.empty_aligned(fft_shape, dtype='complex64')
        fft_output = pyfftw.empty_aligned(fft_shape, dtype='complex64')
        fft_obj = pyfftw.FFTW(fft_input, fft_output, axes=(fft_axis,), 
                             flags=('FFTW_MEASURE',), threads=pyfftw.config.NUM_THREADS)
        # Copy data to input array
        np.copyto(fft_input, padded_data)
        # Execute the plan
        fft_obj()
        # Get the result
        range_ffts = fft_output
        # Cache the plan
        _FFTW_PLAN_CACHE[cache_key] = fft_obj
    print(f"DEBUG: After FFT, data shape: {range_ffts.shape}, sample values: {range_ffts[0, 0, 0:3]}")
    # Calculate range profile based on chirp averaging
    range_profile = np.max(np.abs(range_ffts), axis=2)
    print(f"DEBUG: Range profile shape after max across chirps: {range_profile.shape}, sample values: {range_profile[0:3, 0]}")
    
    # Calculate sum across all antennas if needed
    if range_profile.shape[1] > 1:  # Only if we have multiple antennas
        # Add sum as first column, keeping original channels after
        range_profile = np.column_stack([np.sum(range_profile, axis=1, keepdims=True), range_profile])
        print(f"DEBUG: Range profile shape after antenna summation: {range_profile.shape}, sample values: {range_profile[0:3, 0]}")
    
    if remove_clutter:
        range_profile = remove_static_clutter(range_profile)
        print(f"DEBUG: Range profile shape after clutter removal: {range_profile.shape}, sample values: {range_profile[0:3, 0]}")
    
    # For FMCW radar, range calculation with proper scaling and FFT bin mapping
    if radar_params is not None:
        max_range = radar_params.max_range
        range_resolution = radar_params.range_resolution
        num_range_bins = padding_sizes[0] // 2  # Only positive frequencies
        
        # Apply range offset calibration based on padding size
        if radar_params.auto_adjust_offset:
            # Automatically adjust offset based on padding size
            if padding_sizes[0] == 256:
                range_offset = -1.0
            elif padding_sizes[0] == 512:
                range_offset = -3.0
            else:
                # For other padding sizes, scale proportionally
                padding_scale_factor = padding_sizes[0] / radar_params.base_range_padding
                range_offset = radar_params.range_offset * padding_scale_factor
        else:
            # Use the user-specified offset from the GUI
            range_offset = radar_params.range_offset
        
        # Apply range offset calibration
        range_axis = np.arange(num_range_bins) * range_resolution + range_offset
        print(f"DEBUG: Range axis created from radar_params - shape: {range_axis.shape}, range_resolution: {range_resolution}, max_range: {max_range}, base_offset: {radar_params.range_offset}, applied_offset: {range_offset}, padding: {padding_sizes[0]}")
    else:
        # Fallback calculation when radar_params not available
        # Calculate range resolution and use it to create range axis
        bandwidth = freq_slope * (num_adc_samples / sampling_rate)  # Hz
        range_resolution = c / (2 * bandwidth)
        range_axis = np.arange(padding_sizes[0] // 2) * range_resolution
        print(f"DEBUG: Range axis created from calculated values - shape: {range_axis.shape}, bandwidth: {bandwidth}, range_resolution: {range_resolution}")
    print(f"DEBUG: Range axis initial values - shape: {range_axis.shape}, range resolution: {range_resolution}, sample values: {range_axis[0:3]}, min: {np.min(range_axis)}, max: {np.max(range_axis)}")
    
    # Get positive frequencies/ranges (first half of FFT output)
    range_profile = range_profile[:padding_sizes[0]//2]
    range_axis = range_axis[:padding_sizes[0]//2]
    print(f"DEBUG: After selecting positive frequencies - range profile shape: {range_profile.shape}, range axis shape: {range_axis.shape}, range axis min: {np.min(range_axis)}, max: {np.max(range_axis)}, sample values: {range_axis[0:3]}")
    
    # Apply CFAR detection with processor parameters
    detection_mask, detected_objects = apply_cfar(range_profile, cfar_params=radar_params.cfar_params if hasattr(radar_params, 'cfar_params') else None)
    print(f"DEBUG: After CFAR, detection mask shape: {detection_mask.shape}, number of detected objects: {len(detected_objects)}")
    
    # Convert detected object indices to actual ranges and format for display
    detected_points = []
    print(f"DEBUG: Converting detected objects to range values using range_axis of length {len(range_axis)}")
    for idx, mag in detected_objects:
        # Ensure index is within bounds
        if 0 <= idx < len(range_axis):
            range_value = range_axis[int(idx)]
            detected_points.append((range_value, mag))
            print(f"DEBUG: Converted index {idx} to range value {range_value} meters with magnitude {mag} dB")
        else:
            print(f"DEBUG: Skipped index {idx} as it's out of range_axis bounds (0-{len(range_axis)-1})")
    print(f"DEBUG: Number of detected points after conversion to range values: {len(detected_points)}")
    
    print(f"DEBUG: Final range profile shape: {range_profile.shape}, range axis shape: {range_axis.shape}, range axis min: {np.min(range_axis)}, max: {np.max(range_axis)}")
    print(f"DEBUG: Returning {len(detected_points)} detected points: {detected_points}")
    return range_profile, range_axis, detected_points

def validate_input(radar_data: np.ndarray, radar_params: Optional[RadarParameters] = None) -> bool:
    """
    Validate input data shape, type and check Nyquist criteria.
    
    Args:
        radar_data: Input radar data array
        radar_params: Optional RadarParameters object containing radar configuration
    
    Returns:
        bool: True if validation passes
        
    Raises:
        TypeError: If input type is invalid
        ValueError: If input shape is invalid or Nyquist criteria not met
    """
    if not isinstance(radar_data, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if len(radar_data.shape) != 2:
        raise ValueError("Input must be 2D array [numRX, numChirps * numADCSamples]")
