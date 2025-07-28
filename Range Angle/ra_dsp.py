"""
DSP module for radar signal processing.
"""

import numpy as np
from numba import njit
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

# Numba optimized CFAR core computation
@njit
def _cfar_core(power_profile, num_range_bins, num_second_dim_bins, 
               range_guard, second_dim_guard, range_training, second_dim_training, alpha):
    """
    Numba-accelerated core CFAR computation.
    Supports both range-doppler and range-angle processing.
    """
    detection_mask = np.zeros_like(power_profile)
    detected_objects = []
    
    for range_idx in range(num_range_bins):
        for second_dim_idx in range(num_second_dim_bins):
            # Define 2D training regions
            range_start = max(0, range_idx - range_training - range_guard)
            range_end = min(num_range_bins, range_idx + range_training + range_guard + 1)
            second_dim_start = max(0, second_dim_idx - second_dim_training - second_dim_guard)
            second_dim_end = min(num_second_dim_bins, second_dim_idx + second_dim_training + second_dim_guard + 1)
            
            # Define guard regions
            guard_range_start = max(0, range_idx - range_guard)
            guard_range_end = min(num_range_bins, range_idx + range_guard + 1)
            guard_second_dim_start = max(0, second_dim_idx - second_dim_guard)
            guard_second_dim_end = min(num_second_dim_bins, second_dim_idx + second_dim_guard + 1)
            
            # Extract training region
            training_region = power_profile[range_start:range_end, second_dim_start:second_dim_end]
            
            # Create guard region mask
            guard_mask = np.ones((range_end - range_start, second_dim_end - second_dim_start), dtype=np.bool_)
            local_guard_range_start = guard_range_start - range_start
            local_guard_range_end = guard_range_end - range_start
            local_guard_second_dim_start = guard_second_dim_start - second_dim_start
            local_guard_second_dim_end = guard_second_dim_end - second_dim_start
            
            # Set guard region in mask
            if (local_guard_range_start >= 0 and local_guard_range_end <= training_region.shape[0] and
                local_guard_second_dim_start >= 0 and local_guard_second_dim_end <= training_region.shape[1]):
                guard_mask[local_guard_range_start:local_guard_range_end,
                         local_guard_second_dim_start:local_guard_second_dim_end] = False
            
            # Calculate noise level from training cells (excluding guard region)
            training_sum = 0.0
            num_training_cells = 0
            for i in range(training_region.shape[0]):
                for j in range(training_region.shape[1]):
                    if guard_mask[i, j]:
                        training_sum += training_region[i, j]
                        num_training_cells += 1
            
            if num_training_cells > 0:
                # Calculate threshold
                noise_level = training_sum / num_training_cells
                threshold = alpha * noise_level
                
                # Compare CUT to threshold
                if power_profile[range_idx, second_dim_idx] > threshold:
                    detection_mask[range_idx, second_dim_idx] = 1
                    # Store detection
                    detected_objects.append((range_idx, second_dim_idx, 
                                          10 * np.log10(power_profile[range_idx, second_dim_idx])))
    
    return detection_mask, detected_objects

def apply_cfar(data: np.ndarray, cfar_params: dict = None, mode: str = 'range_doppler') -> Tuple[np.ndarray, List[Tuple[float, float, float]]]:
    """
    Apply Cell-Averaging Constant False Alarm Rate (CA-CFAR) detection with peak grouping.
    Supports both range-doppler and range-angle processing modes.
    Uses Numba-accelerated core computation for improved performance.
    
    Args:
        data: Input magnitude data, either:
            - Range-Doppler map [num_range_bins, num_doppler_bins] for mode='range_doppler'
            - Range-Angle map [num_range_bins, num_angle_bins] for mode='range_angle'
        cfar_params: Dictionary containing CFAR parameters:
            - range_guard: Number of guard cells in range dimension (default: 2)
            - doppler_guard/angle_guard: Guard cells in second dimension (default: 2)
            - range_training: Number of training cells in range dimension (default: 8)
            - doppler_training/angle_training: Training cells in second dimension (default: 4)
            - pfa: Probability of false alarm (default: 0.01)
            - group_peaks: Whether to enable peak grouping (default: True)
        mode: Processing mode, either 'range_doppler' or 'range_angle'
        
    Returns:
        Tuple of:
            - Binary detection mask same shape as input
            - List of (range_idx, second_dim_idx, magnitude) tuples for detected objects
    """
    num_range_bins, num_second_dim_bins = data.shape
    
    # Validate mode
    if mode not in ['range_doppler', 'range_angle']:
        raise ValueError("Mode must be either 'range_doppler' or 'range_angle'")
    
    # Set default parameters if not provided
    if cfar_params is None:
        cfar_params = {
            'range_guard': 2,
            'doppler_guard' if mode == 'range_doppler' else 'angle_guard': 2,
            'range_training': 8,
            'doppler_training' if mode == 'range_doppler' else 'angle_training': 4,
            'pfa': 0.01,
            'group_peaks': True
        }
    
    range_guard = cfar_params.get('range_guard', 2)
    second_dim_guard = cfar_params.get('doppler_guard' if mode == 'range_doppler' else 'angle_guard', 2)
    range_training = cfar_params.get('range_training', 8)
    second_dim_training = cfar_params.get('doppler_training' if mode == 'range_doppler' else 'angle_training', 4)
    pfa = cfar_params.get('pfa', 0.01)
    group_peaks = cfar_params.get('group_peaks', True)
    group_threshold = 2.0  # Fixed threshold for grouping nearby peaks
    
    # Calculate total number of training cells
    num_training_total = ((2 * range_training + 2 * range_guard + 1) * 
                         (2 * second_dim_training + 2 * second_dim_guard + 1) - 
                         (2 * range_guard + 1) * (2 * second_dim_guard + 1))
    
    # Calculate threshold factor based on PFA and number of training cells
    alpha = num_training_total * (pfa**(-1/num_training_total) - 1)
    
    # Process each cell using Numba-accelerated function
    power_profile = np.abs(data)**2
    detection_mask, detected_objects = _cfar_core(
        power_profile, num_range_bins, num_second_dim_bins,
        range_guard, second_dim_guard, range_training, second_dim_training, alpha
    )
    
    # Group nearby peaks if enabled
    if group_peaks and detected_objects:
        grouped_objects = []
        current_group = []
        
        # Sort detected objects by range index
        detected_objects.sort(key=lambda x: (x[0], x[1]))
        
        for obj in detected_objects:
            if not current_group:
                current_group = [obj]
            else:
                last_obj = current_group[-1]
                # Check if current object is close to last object in both dimensions
                if (abs(obj[0] - last_obj[0]) <= group_threshold and 
                    abs(obj[1] - last_obj[1]) <= group_threshold):
                    current_group.append(obj)
                else:
                    if current_group:
                        # Average position and combine magnitudes
                        avg_range = round(sum(x[0] for x in current_group) / len(current_group))
                        avg_doppler = round(sum(x[1] for x in current_group) / len(current_group))
                        combined_mag = 10 * np.log10(sum(10**(x[2]/10) for x in current_group))
                        grouped_objects.append((avg_range, avg_doppler, combined_mag))
                    current_group = [obj]
        
        # Process last group
        if current_group:
            avg_range = round(sum(x[0] for x in current_group) / len(current_group))
            avg_doppler = round(sum(x[1] for x in current_group) / len(current_group))
            combined_mag = 10 * np.log10(sum(10**(x[2]/10) for x in current_group))
            grouped_objects.append((avg_range, avg_doppler, combined_mag))
        
        # Sort grouped objects by magnitude
        grouped_objects.sort(key=lambda x: x[2], reverse=True)
        return detection_mask, grouped_objects
    else:
        # Return ungrouped detections sorted by magnitude
        detected_objects.sort(key=lambda x: x[2], reverse=True)
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
                        range_padding: Optional[int] = None, doppler_padding: Optional[int] = None,
                        angle_padding: Optional[int] = None) -> List[int]:
    """
    Get FFT padding sizes based on input dimensions and user-specified padding.
    
    Args:
        data_shape: Shape of input data
        radar_params: Optional radar parameters
        range_padding: Optional user-specified range padding size
        doppler_padding: Optional user-specified doppler padding size
        angle_padding: Optional user-specified angle padding size
        
    Returns:
        List[int]: [range_padding, 0, doppler_padding, angle_padding]
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
        range_padding = next_pow2
    log_padding_info("range", num_adc_samples, range_padding)
    
    if doppler_padding is None:
        next_pow2 = 2 ** np.ceil(np.log2(num_chirps)).astype(int)
        doppler_padding = next_pow2
    log_padding_info("doppler", num_chirps, doppler_padding)
    
    # Calculate angle padding if not provided
    if angle_padding is None:
        # Default angle padding for 181 azimuth angles (-90° to 90° in 1° steps)
        angle_padding = 181
    log_padding_info("angle", 181, angle_padding)
    
    return [range_padding, 0, doppler_padding, angle_padding]

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

@njit
def _beamform_range_bin(range_bin_data_avg: np.ndarray, angles: np.ndarray, 
                       virtual_array_positions: np.ndarray, wavelength: float, d: float, 
                       is_azimuth: bool = True) -> np.ndarray:
    """
    Numba-accelerated beamforming calculation for a single range bin.
    
    Args:
        range_bin_data_avg: Complex range bin data averaged across chirps
        angles: Array of angles (azimuth or elevation) in degrees
        virtual_array_positions: Virtual array element positions
        wavelength: Radar wavelength
        d: Element spacing
        is_azimuth: True for azimuth beamforming, False for elevation
        
    Returns:
        np.ndarray: Beamformed output for each angle
    """
    num_angles = len(angles)
    beamformed_output = np.zeros(num_angles, dtype=np.float32)
    
    for a_idx in range(num_angles):
        # Convert angle to radians
        angle_rad = np.deg2rad(angles[a_idx])
        
        # Calculate direction vector components
        if is_azimuth:
            k_x = np.sin(angle_rad)
            k_z = 0  # No elevation component
            k_y = np.cos(angle_rad)
        else:
            k_x = 0  # No azimuth component
            k_z = np.sin(angle_rad)
            k_y = np.cos(angle_rad)
        
        # Calculate phase shifts
        phase_shifts = 2 * np.pi * (virtual_array_positions[:, 0] * k_x + 
                                  virtual_array_positions[:, 1] * k_y + 
                                  virtual_array_positions[:, 2] * k_z) / wavelength * d
        
        # Create steering vector with explicit dtype to match input data
        steering_vector = np.exp(1j * phase_shifts).astype(np.complex64)
        
        # Apply beamforming (ensuring both arrays are complex64)
        beamformed_signal = np.abs(np.dot(range_bin_data_avg, np.conj(steering_vector).astype(np.complex64)))
        beamformed_output[a_idx] = beamformed_signal
    
    return beamformed_output

def calculate_range_angle(radar_data: np.ndarray, radar_params: Optional[RadarParameters] = None,
                         remove_clutter: bool = False, use_pulse_compression: bool = True,
                         window_type: str = 'blackmanharris', range_padding: Optional[int] = None,
                         doppler_padding: Optional[int] = None, angle_padding: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                                                     List[Tuple[float, float, float]], List[Tuple[float, float, float]]]:
    """
    Calculate Range-Angle maps using beamforming with steering vectors for azimuth and elevation.
    
    Args:
        radar_data: Input radar data array [numRX, numChirps * numADCSamples]
        radar_params: Optional RadarParameters object containing radar configuration
        remove_clutter: Whether to apply clutter removal
        use_pulse_compression: Whether to apply LFM/Chirp pulse compression
        window_type: Type of window to apply
        range_padding: Optional user-specified range padding size
        doppler_padding: Optional user-specified doppler padding size
        angle_padding: Optional user-specified angle padding size
        
    Returns:
        Tuple of (range_azimuth, range_elevation, range_axis, azimuth_angles, elevation_angles, azimuth_points, elevation_points) where:
            range_azimuth: Range-Azimuth map
            range_elevation: Range-Elevation map
            range_axis: Corresponding range values in meters
            azimuth_angles: Corresponding azimuth angles in degrees
            elevation_angles: Corresponding elevation angles in degrees
            azimuth_points: List of (range, angle, magnitude) tuples for detected objects in azimuth
            elevation_points: List of (range, angle, magnitude) tuples for detected objects in elevation
    """
    num_rx, _ = radar_data.shape
    if radar_params is not None:
        num_chirps = radar_params.config_params['chirps_per_frame'] * radar_params.config_params['num_loops']
        num_adc_samples = radar_params.config_params['adc_samples']
        sampling_rate = radar_params.config_params['sample_rate'] * 1e3
        freq_slope = radar_params.config_params['freq_slope'] * 1e6
        c = 3e8
        carrier_freq = radar_params.config_params['start_freq'] * 1e9
        wavelength = c / carrier_freq
    else:
        num_adc_samples = 256
        num_chirps = radar_data.shape[1] // num_adc_samples
        sampling_rate = 10e6
        freq_slope = 60e6
        c = 3e8
        carrier_freq = 77e9  # Default 77 GHz
        wavelength = c / carrier_freq
    
    # Reshape data efficiently using vectorized operations
    reshaped_data = radar_data.reshape(num_rx, num_chirps, num_adc_samples).transpose(2, 0, 1)
    
    # Apply pulse compression if requested
    if use_pulse_compression:
        logger.info("Applying LFM/Chirp pulse compression to improve SNR in range-angle processing")
        reshaped_data = apply_pulse_compression(reshaped_data, radar_params)
    
    # Get window
    range_window = create_window(num_adc_samples, window_type)
    
    # Get padding sizes
    padding_sizes = suggest_padding_size(radar_data.shape, radar_params, 
                                       range_padding=range_padding,
                                       doppler_padding=doppler_padding,
                                       angle_padding=angle_padding)
    
    # 1. Process Range Dimension
    # Apply window to range dimension
    range_window = create_window(num_adc_samples, window_type)
    windowed_data = reshaped_data * range_window[:, np.newaxis, np.newaxis]
    
    # Zero pad range dimension
    padded_data = np.zeros((padding_sizes[0], num_rx, num_chirps), dtype=np.complex64)
    padded_data[:num_adc_samples] = windowed_data
    
    # Apply Range FFT
    range_fft_shape = (padding_sizes[0], num_rx, num_chirps)
    range_fft_axis = 0
    range_cache_key = (range_fft_shape, range_fft_axis, 'fft')
    
    if range_cache_key in _FFTW_PLAN_CACHE:
        range_fft_obj = _FFTW_PLAN_CACHE[range_cache_key]
        np.copyto(range_fft_obj.input_array, padded_data)
        range_fft_obj()
        range_ffts = range_fft_obj.output_array
    else:
        range_fft_input = pyfftw.empty_aligned(range_fft_shape, dtype='complex64')
        range_fft_output = pyfftw.empty_aligned(range_fft_shape, dtype='complex64')
        range_fft_obj = pyfftw.FFTW(range_fft_input, range_fft_output, axes=(range_fft_axis,),
                                   flags=('FFTW_MEASURE',), threads=pyfftw.config.NUM_THREADS)
        np.copyto(range_fft_input, padded_data)
        range_fft_obj()
        range_ffts = range_fft_output
        _FFTW_PLAN_CACHE[range_cache_key] = range_fft_obj
    
    # Get positive frequencies/ranges
    range_ffts = range_ffts[:padding_sizes[0]//2]
    
    # 2. Process Doppler Dimension
    # Apply window to doppler dimension
    doppler_window = create_window(num_chirps, window_type)
    windowed_doppler = range_ffts * doppler_window[np.newaxis, np.newaxis, :]
    
    # Zero pad doppler dimension
    padded_doppler = np.zeros((range_ffts.shape[0], num_rx, padding_sizes[2]), dtype=np.complex64)
    padded_doppler[:, :, :num_chirps] = windowed_doppler
    
    # Apply Doppler FFT
    doppler_fft_shape = padded_doppler.shape
    doppler_fft_axis = 2
    doppler_cache_key = (doppler_fft_shape, doppler_fft_axis, 'fft')
    
    if doppler_cache_key in _FFTW_PLAN_CACHE:
        doppler_fft_obj = _FFTW_PLAN_CACHE[doppler_cache_key]
        np.copyto(doppler_fft_obj.input_array, padded_doppler)
        doppler_fft_obj()
        doppler_ffts = doppler_fft_obj.output_array
    else:
        doppler_fft_input = pyfftw.empty_aligned(doppler_fft_shape, dtype='complex64')
        doppler_fft_output = pyfftw.empty_aligned(doppler_fft_shape, dtype='complex64')
        doppler_fft_obj = pyfftw.FFTW(doppler_fft_input, doppler_fft_output, axes=(doppler_fft_axis,),
                                     flags=('FFTW_MEASURE',), threads=pyfftw.config.NUM_THREADS)
        np.copyto(doppler_fft_input, padded_doppler)
        doppler_fft_obj()
        doppler_ffts = doppler_fft_output
        _FFTW_PLAN_CACHE[doppler_cache_key] = doppler_fft_obj
    
    # 3. Process Angle Dimension
    # Apply window to angle dimension
    angle_window = create_window(num_rx, window_type)
    windowed_angle = doppler_ffts * angle_window[np.newaxis, :, np.newaxis]
    
    # Zero pad angle dimension
    padded_angle = np.zeros((doppler_ffts.shape[0], padding_sizes[3], doppler_ffts.shape[2]), dtype=np.complex64)
    padded_angle[:, :num_rx, :] = windowed_angle
    
    # Apply Angle FFT
    angle_fft_shape = padded_angle.shape
    angle_fft_axis = 1
    angle_cache_key = (angle_fft_shape, angle_fft_axis, 'fft')
    
    if angle_cache_key in _FFTW_PLAN_CACHE:
        angle_fft_obj = _FFTW_PLAN_CACHE[angle_cache_key]
        np.copyto(angle_fft_obj.input_array, padded_angle)
        angle_fft_obj()
        angle_ffts = angle_fft_obj.output_array
    else:
        angle_fft_input = pyfftw.empty_aligned(angle_fft_shape, dtype='complex64')
        angle_fft_output = pyfftw.empty_aligned(angle_fft_shape, dtype='complex64')
        angle_fft_obj = pyfftw.FFTW(angle_fft_input, angle_fft_output, axes=(angle_fft_axis,),
                                   flags=('FFTW_MEASURE',), threads=pyfftw.config.NUM_THREADS)
        np.copyto(angle_fft_input, padded_angle)
        angle_fft_obj()
        angle_ffts = angle_fft_output
        _FFTW_PLAN_CACHE[angle_cache_key] = angle_fft_obj
    
    # Apply clutter removal if requested
    if remove_clutter:
        for chirp in range(num_chirps):
            angle_ffts[:, :, chirp] = remove_static_clutter(angle_ffts[:, :, chirp])
    
    # Calculate range axis with proper scaling and FFT bin mapping
    if radar_params is not None:
        max_range = radar_params.max_range
        range_resolution = radar_params.range_resolution
        num_range_bins = padding_sizes[0] // 2  # Only positive frequencies
        range_axis = np.arange(num_range_bins) * range_resolution
    else:
        bin_indices = np.arange(padding_sizes[0] // 2)  # Only positive frequencies
        range_axis = (bin_indices * c * sampling_rate) / (2 * freq_slope * padding_sizes[0])
    
    # Define antenna array geometry
    # For a typical MIMO radar with 4 RX and 3 TX antennas in a grid pattern
    # Assuming λ/2 spacing between elements
    d = wavelength / 2  # Element spacing (λ/2)
    
    # Define virtual array positions for both azimuth and elevation configurations
    # Based on Figures 2-5 and 2-6
    
    # Range-Azimuth configuration (Figure 2-6)
    virtual_array_positions_azimuth = np.array([
        # Top row (circled in figure)
        [0, 0, 0],  # RX0 × TX0
        [1, 0, 0],  # RX1 × TX0
        [2, 0, 0],  # RX2 × TX0
        [3, 0, 0],  # RX3 × TX0
        # Second row
        [0, 1, 0],  # RX0 × TX1
        [1, 1, 0],  # RX1 × TX1
        [2, 1, 0],  # RX2 × TX1
        [3, 1, 0],  # RX3 × TX1
        # Third row
        [0, 2, 0],  # RX0 × TX2
        [1, 2, 0],  # RX1 × TX2
        [2, 2, 0],  # RX2 × TX2
        [3, 2, 0],  # RX3 × TX2
    ])
    
    # Range-Elevation configuration (Figure 2-5)
    virtual_array_positions_elevation = np.array([
        # First column (circled in figure)
        [0, 0, 0],  # RX0 × TX0
        [0, 1, 0],  # RX0 × TX1
        [0, 2, 0],  # RX0 × TX2
        # Second column
        [1, 0, 0],  # RX1 × TX0
        [1, 1, 0],  # RX1 × TX1
        [1, 2, 0],  # RX1 × TX2
        # Third column
        [2, 0, 0],  # RX2 × TX0
        [2, 1, 0],  # RX2 × TX1
        [2, 2, 0],  # RX2 × TX2
        # Fourth column
        [3, 0, 0],  # RX3 × TX0
        [3, 1, 0],  # RX3 × TX1
        [3, 2, 0],  # RX3 × TX2
    ])
    
    # Define angle ranges for beamforming
    azimuth_angles = np.linspace(-90, 90, 181 if angle_padding is None else angle_padding)  # -90° to 90° in 1° steps
    elevation_angles = np.linspace(-45, 45, 91 if angle_padding is None else angle_padding)  # -45° to 45° in 1° steps
    
    # Initialize Range-Angle maps
    num_range_bins = len(range_axis)
    num_azimuth_angles = len(azimuth_angles)
    num_elevation_angles = len(elevation_angles)
    
    range_azimuth = np.zeros((num_range_bins, num_azimuth_angles), dtype=np.float32)
    range_elevation = np.zeros((num_range_bins, num_elevation_angles), dtype=np.float32)
    
    # Perform beamforming for each range bin using Numba-optimized function
    for r in range(num_range_bins):
        # Extract range bin data for all virtual channels and chirps
        range_bin_data = range_ffts[r, :, :]
        
        # Average across chirps to improve SNR
        range_bin_data_avg = np.mean(range_bin_data, axis=1)
        
        # Beamforming for azimuth using optimized function
        range_azimuth[r, :] = _beamform_range_bin(
            range_bin_data_avg, 
            azimuth_angles,
            virtual_array_positions_azimuth,
            wavelength,
            d,
            True
        )
        
        # Beamforming for elevation using optimized function
        range_elevation[r, :] = _beamform_range_bin(
            range_bin_data_avg,
            elevation_angles,
            virtual_array_positions_elevation,
            wavelength,
            d,
            False
        )
    
    # Normalize the maps
    range_azimuth = range_azimuth / np.max(range_azimuth)
    range_elevation = range_elevation / np.max(range_elevation)
    
    # Apply CFAR to range-azimuth map
    azimuth_detection_mask, azimuth_detected_points = apply_cfar(
        range_azimuth, 
        cfar_params=radar_params.cfar_params if hasattr(radar_params, 'cfar_params') else None,
        mode='range_angle'
    )

    # Apply CFAR to range-elevation map
    elevation_detection_mask, elevation_detected_points = apply_cfar(
        range_elevation,
        cfar_params=radar_params.cfar_params if hasattr(radar_params, 'cfar_params') else None,
        mode='range_angle'
    )

    # Convert azimuth detections to (range, angle, magnitude)
    azimuth_points = [
        (range_axis[int(r)], azimuth_angles[int(a)], mag)
        for r, a, mag in azimuth_detected_points
        if 0 <= int(r) < len(range_axis) and 0 <= int(a) < len(azimuth_angles)
    ]

    # Convert elevation detections to (range, angle, magnitude)
    elevation_points = [
        (range_axis[int(r)], elevation_angles[int(e)], mag)
        for r, e, mag in elevation_detected_points
        if 0 <= int(r) < len(range_axis) and 0 <= int(e) < len(elevation_angles)
    ]

    return range_azimuth, range_elevation, range_axis, azimuth_angles, elevation_angles, azimuth_points, elevation_points

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

# Pre-compile the Numba functions to avoid JIT compilation delay during runtime
_cfar_core(np.zeros((10, 10), dtype=np.float32), 10, 10, 2, 2, 4, 4, 1.0)
_beamform_range_bin(
    np.zeros(10, dtype=np.complex64),  # range_bin_data_avg as complex64
    np.linspace(-90, 90, 10, dtype=np.float32),  # angles as float32
    np.zeros((10, 3), dtype=np.float32),  # virtual_array_positions as float32
    np.float32(0.00386),  # wavelength for 77 GHz as float32
    np.float32(0.00193),  # d = wavelength/2 as float32
    True  # is_azimuth
)
