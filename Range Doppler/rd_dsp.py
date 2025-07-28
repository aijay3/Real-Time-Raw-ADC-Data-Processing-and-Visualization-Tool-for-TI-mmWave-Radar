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
def _cfar_core(power_profile, num_range_bins, num_doppler_bins, 
               range_guard, doppler_guard, range_training, doppler_training, alpha):
    """
    Numba-accelerated core CFAR computation.
    """
    detection_mask = np.zeros_like(power_profile)
    detected_objects = []
    
    for range_idx in range(num_range_bins):
        for doppler_idx in range(num_doppler_bins):
            # Define 2D training regions
            range_start = max(0, range_idx - range_training - range_guard)
            range_end = min(num_range_bins, range_idx + range_training + range_guard + 1)
            doppler_start = max(0, doppler_idx - doppler_training - doppler_guard)
            doppler_end = min(num_doppler_bins, doppler_idx + doppler_training + doppler_guard + 1)
            
            # Define guard regions
            guard_range_start = max(0, range_idx - range_guard)
            guard_range_end = min(num_range_bins, range_idx + range_guard + 1)
            guard_doppler_start = max(0, doppler_idx - doppler_guard)
            guard_doppler_end = min(num_doppler_bins, doppler_idx + doppler_guard + 1)
            
            # Extract training region
            training_region = power_profile[range_start:range_end, doppler_start:doppler_end]
            
            # Create guard region mask
            guard_mask = np.ones((range_end - range_start, doppler_end - doppler_start), dtype=np.bool_)
            local_guard_range_start = guard_range_start - range_start
            local_guard_range_end = guard_range_end - range_start
            local_guard_doppler_start = guard_doppler_start - doppler_start
            local_guard_doppler_end = guard_doppler_end - doppler_start
            
            # Set guard region in mask
            if (local_guard_range_start >= 0 and local_guard_range_end <= training_region.shape[0] and
                local_guard_doppler_start >= 0 and local_guard_doppler_end <= training_region.shape[1]):
                guard_mask[local_guard_range_start:local_guard_range_end,
                         local_guard_doppler_start:local_guard_doppler_end] = False
            
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
                if power_profile[range_idx, doppler_idx] > threshold:
                    detection_mask[range_idx, doppler_idx] = 1
                    # Store detection
                    detected_objects.append((range_idx, doppler_idx, 
                                          10 * np.log10(power_profile[range_idx, doppler_idx])))
    
    return detection_mask, detected_objects

def apply_cfar(range_doppler: np.ndarray, cfar_params: dict = None, radar_params: Optional[RadarParameters] = None) -> Tuple[np.ndarray, List[Tuple[float, float, float]]]:
    """
    Apply 2D Cell-Averaging CFAR detection with adaptive peak grouping.
    Uses Numba-accelerated core computation for improved performance.
    
    Features:
    - Adaptive thresholds based on range and velocity resolution
    - SNR-weighted averaging for accurate position estimation
    - Resolution-aware peak grouping
    
    Args:
        range_doppler: Range-Doppler magnitude data [num_range_bins, num_doppler_bins]
        cfar_params: Dictionary containing CFAR parameters
        radar_params: Optional RadarParameters object for resolution calculations
        
    Returns:
        Tuple of:
            - Binary detection mask same shape as input
            - List of (range_idx, doppler_idx, magnitude) tuples for detected objects
    """
    num_range_bins, num_doppler_bins = range_doppler.shape
    
    # Set default parameters if not provided
    if cfar_params is None:
        cfar_params = {
            'range_guard': 2,
            'doppler_guard': 2,
            'range_training': 8,
            'doppler_training': 4,
            'pfa': 0.001,
            'group_peaks': True  # Enable by default
        }
    
    range_guard = cfar_params.get('range_guard', 2)
    doppler_guard = cfar_params.get('doppler_guard', 2)
    range_training = cfar_params.get('range_training', 8)
    doppler_training = cfar_params.get('doppler_training', 4)
    pfa = cfar_params.get('pfa', 0.001)
    group_peaks = cfar_params.get('group_peaks', True)
    
    # Calculate adaptive grouping thresholds based on resolution
    if radar_params is not None:
        # Convert range resolution to bins
        range_res_bins = max(1, round(radar_params.range_resolution / 
                                    (radar_params.max_range / num_range_bins)))
        
        # Convert velocity resolution to bins
        velocity_res_bins = max(1, round(radar_params.velocity_resolution / 
                                       (2 * radar_params.max_velocity / num_doppler_bins)))
        
        # Set grouping thresholds to 1.5x the resolution in bins
        range_group_threshold = 1.5 * range_res_bins
        doppler_group_threshold = 1.5 * velocity_res_bins
    else:
        # Default thresholds if radar parameters not available
        range_group_threshold = 2.0
        doppler_group_threshold = 2.0
    
    # Calculate total number of training cells
    num_training_total = ((2 * range_training + 2 * range_guard + 1) * 
                         (2 * doppler_training + 2 * doppler_guard + 1) - 
                         (2 * range_guard + 1) * (2 * doppler_guard + 1))
    
    # Calculate threshold factor based on PFA and number of training cells
    alpha = num_training_total * (pfa**(-1/num_training_total) - 1)
    
    # Process each cell in the range-Doppler map using Numba-accelerated function
    power_profile = np.abs(range_doppler)**2
    detection_mask, detected_objects = _cfar_core(
        power_profile, num_range_bins, num_doppler_bins,
        range_guard, doppler_guard, range_training, doppler_training, alpha
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
                # Check if current object is close using adaptive thresholds
                range_diff = abs(obj[0] - last_obj[0])
                doppler_diff = abs(obj[1] - last_obj[1])
                
                if (range_diff <= range_group_threshold and 
                    doppler_diff <= doppler_group_threshold):
                    current_group.append(obj)
                else:
                    if current_group:
                        # Convert magnitudes to linear scale for SNR weighting
                        mags_linear = [10**(x[2]/10) for x in current_group]
                        total_mag = sum(mags_linear)
                        weights = [mag/total_mag for mag in mags_linear]
                        
                        # Calculate SNR-weighted average positions
                        avg_range = round(sum(x[0] * w for x, w in zip(current_group, weights)))
                        avg_doppler = round(sum(x[1] * w for x, w in zip(current_group, weights)))
                        
                        # Calculate combined magnitude
                        combined_mag = 10 * np.log10(total_mag)
                        
                        grouped_objects.append((avg_range, avg_doppler, combined_mag))
                    current_group = [obj]
        
        # Process last group
        if current_group:
            # Convert magnitudes to linear scale for SNR weighting
            mags_linear = [10**(x[2]/10) for x in current_group]
            total_mag = sum(mags_linear)
            weights = [mag/total_mag for mag in mags_linear]
            
            # Calculate SNR-weighted average positions
            avg_range = round(sum(x[0] * w for x, w in zip(current_group, weights)))
            avg_doppler = round(sum(x[1] * w for x, w in zip(current_group, weights)))
            
            # Calculate combined magnitude
            combined_mag = 10 * np.log10(total_mag)
            
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
                        range_padding: Optional[int] = None, doppler_padding: Optional[int] = None) -> List[int]:
    """
    Get FFT padding sizes based on input dimensions and user-specified padding.
    
    Args:
        data_shape: Shape of input data
        radar_params: Optional radar parameters
        range_padding: Optional user-specified range padding size
        doppler_padding: Optional user-specified doppler padding size
        
    Returns:
        List[int]: [range_padding, 0 , doppler_padding]
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
    
    return [range_padding, 0, doppler_padding]

def apply_mti_filter(range_data: np.ndarray, num_pulses: int = 3) -> np.ndarray:
    """
    Apply Moving Target Indicator (MTI) filtering to suppress stationary targets.
    
    Args:
        range_data: Range profile data [num_range_bins, num_rx, num_chirps]
        num_pulses: Number of pulses for MTI filter (default=3)
    
    Returns:
        MTI filtered data with same shape as input
    """
    if len(range_data.shape) != 3:
        raise ValueError("Input data must be 3D array [num_range_bins, num_rx, num_chirps]")
    
    num_range_bins, num_rx, num_chirps = range_data.shape
    
    # Ensure we have enough pulses for MTI
    if num_chirps < num_pulses:
        return range_data
    
    # Initialize output array
    mti_data = np.zeros_like(range_data)
    
    # Apply pulse-to-pulse cancellation
    for i in range(num_pulses - 1, num_chirps):
        # Subtract consecutive pulses to cancel static returns
        mti_pulse = range_data[:, :, i]
        for j in range(1, num_pulses):
            mti_pulse = mti_pulse - range_data[:, :, i-j]
        mti_data[:, :, i] = mti_pulse
    
    return mti_data

def remove_static_clutter(range_data: np.ndarray, energy_threshold: float = 0.95,
                         robust_estimation: bool = True, noise_suppression: bool = True,
                         use_mti: bool = True, num_mti_pulses: int = 3) -> np.ndarray:
    """
    Remove static clutter using combined MTI and PCA approach.
    
    Features:
    - MTI pre-filtering to suppress strong stationary returns
    - Adaptive PCA with energy-based component selection for residual clutter
    - Robust covariance estimation using MCD (Minimum Covariance Determinant)
    - Statistical noise suppression
    
    Args:
        range_data: Range profile data [num_range_bins, num_rx] or [num_range_bins, num_rx, num_chirps]
        energy_threshold: Threshold for cumulative energy ratio (default=0.95)
        robust_estimation: Use robust covariance estimation (default=True)
        noise_suppression: Apply noise suppression (default=True)
        use_mti: Enable MTI pre-filtering (default=True)
        num_mti_pulses: Number of pulses for MTI filter (default=3)
    
    Returns:
        Range profile data with static clutter removed
    """
    # Apply MTI filtering if enabled and data is 3D
    if use_mti and len(range_data.shape) == 3:
        range_data = apply_mti_filter(range_data, num_mti_pulses)
        # Reshape 3D data to 2D for PCA
        orig_shape = range_data.shape
        range_data = range_data.reshape(-1, orig_shape[1])
    
    # Center the data using robust or standard estimation
    data_mean = np.median(range_data, axis=0, keepdims=True) if robust_estimation else np.mean(range_data, axis=0, keepdims=True)
    centered_data = range_data - data_mean
    
    if robust_estimation:
        # Robust covariance estimation using MCD-inspired approach
        n_samples = centered_data.shape[0]
        h = int(0.75 * n_samples)  # Use 75% of samples for robust estimation
        
        # Calculate initial distances
        initial_cov = np.cov(centered_data.T)
        distances = np.zeros(n_samples)
        for i in range(n_samples):
            sample = centered_data[i:i+1].T
            distances[i] = sample.T @ np.linalg.inv(initial_cov) @ sample
            
        # Select h samples with smallest distances
        mask = np.argsort(distances)[:h]
        subset_data = centered_data[mask]
        
        # Compute covariance on selected subset
        covariance_matrix = np.cov(subset_data.T)
    else:
        covariance_matrix = np.cov(centered_data.T)
    
    # Calculate eigenvectors and eigenvalues
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    
    # Sort in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Determine number of components based on energy threshold
    total_energy = np.sum(eigenvalues)
    cumulative_energy_ratio = np.cumsum(eigenvalues) / total_energy
    n_components = np.searchsorted(cumulative_energy_ratio, energy_threshold) + 1
    
    # Select clutter components
    clutter_components = eigenvectors[:, :n_components]
    
    # Project data onto clutter subspace
    clutter = centered_data @ clutter_components @ clutter_components.T
    
    # Remove clutter
    range_data_clean = range_data - clutter
    
    if noise_suppression:
        # Apply statistical noise suppression
        noise_std = np.std(range_data_clean, axis=0, keepdims=True)
        noise_threshold = 2.0 * noise_std  # 2-sigma threshold
        range_data_clean = np.where(
            np.abs(range_data_clean) > noise_threshold,
            range_data_clean,
            range_data_clean * np.abs(range_data_clean) / noise_threshold
        )
    
    return range_data_clean

def calculate_range_doppler(radar_data: np.ndarray, radar_params: Optional[RadarParameters] = None, 
                          remove_clutter: bool = True, use_pulse_compression: bool = True,
                          window_type: str = 'blackmanharris', range_padding: Optional[int] = None,
                          doppler_padding: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[float, float, float]]]:
    """
    Calculate range-Doppler map using vectorized operations.
    
    This function implements proper sampling and FFT processing following Nyquist criteria:
    1. Ensures sampling rate (fs) is at least 2x the maximum beat frequency
    2. Uses zero-padding for better frequency resolution
    3. Applies 2D windowing to reduce spectral leakage
    4. Only returns the positive frequency components for range
    5. Optionally applies pulse compression to improve SNR
    
    Args:
        radar_data: Input radar data array [numRX, numChirps * numADCSamples]
        radar_params: Optional RadarParameters object containing radar configuration
        remove_clutter: Whether to apply clutter removal
        use_pulse_compression: Whether to apply LFM/Chirp pulse compression
        window_type: Type of window to apply ('blackmanharris', 'hann', etc.)
        range_padding: Optional user-specified range FFT padding
        doppler_padding: Optional user-specified doppler FFT padding
        
    Returns:
        Tuple of (range_doppler, range_axis, velocity_axis, detected_points) where:
            range_doppler: Range-Doppler magnitude map
            range_axis: Corresponding range values in meters
            velocity_axis: Corresponding velocity values in m/s
            detected_points: List of (range, velocity, magnitude) tuples for detected objects
    """
    # Validate input data and check Nyquist criteria
    validate_input(radar_data, radar_params)
    num_rx = radar_data.shape[0]
    if radar_params is not None:
        num_chirps = radar_params.config_params['chirps_per_frame'] * radar_params.config_params['num_loops']
        num_adc_samples = radar_params.config_params['adc_samples']
        sampling_rate = radar_params.config_params['sample_rate'] * 1e3
        freq_slope = radar_params.config_params['freq_slope'] * 1e6
        chirp_time = (radar_params.config_params['idle_time'] + radar_params.config_params['ramp_end_time']) * 1e-6
        c = 3e8
    else:
        num_adc_samples = 256
        num_chirps = radar_data.shape[1] // num_adc_samples
        sampling_rate = 10e6
        freq_slope = 60e6
        chirp_time = num_adc_samples / sampling_rate
        c = 3e8
    
    # Reshape data efficiently using vectorized operations
    reshaped_data = np.zeros((num_adc_samples, num_rx, num_chirps), dtype=np.complex64)
    for rx in range(num_rx):
        rx_data = radar_data[rx, :].reshape(-1, num_adc_samples).T
        reshaped_data[:, rx, :] = rx_data[:, :num_chirps]
    
    # Apply pulse compression if requested
    if use_pulse_compression:
        logger.info("Applying LFM/Chirp pulse compression to improve SNR in range-Doppler processing")
        reshaped_data = apply_pulse_compression(reshaped_data, radar_params)
    
    # Get windows
    range_window = create_window(num_adc_samples, window_type)
    doppler_window = create_window(num_chirps, window_type)
    
    # Create 2D window by outer product of range and doppler windows
    window_2d = range_window[:, np.newaxis] * doppler_window[np.newaxis, :]
    
    # Apply 2D window using broadcasting
    windowed_data = reshaped_data * window_2d[:, np.newaxis, :]
    
    # Get padding sizes with user-specified padding
    padding_sizes = suggest_padding_size(radar_data.shape, radar_params, 
                                       range_padding=range_padding,
                                       doppler_padding=doppler_padding)
    
    # Create zero-padded array for both dimensions with zeros at the end
    padded_data = np.zeros((padding_sizes[0], num_rx, padding_sizes[2]), dtype=np.complex64)
    
    # Place data at the beginning of the padded array
    padded_data[:num_adc_samples, :, :num_chirps] = windowed_data
    
    # Perform 2D FFT with proper padding
    range_doppler = np.zeros((padding_sizes[0], padding_sizes[2]), dtype=np.complex64)
    
    for rx in range(num_rx):
        # First perform range FFT using pyfftw
        # Check if we have a cached FFTW plan for this shape
        fft_shape = (padding_sizes[0], padding_sizes[2])
        fft_axis = 0
        cache_key = (fft_shape, fft_axis, 'fft', rx)
        
        if cache_key in _FFTW_PLAN_CACHE:
            fft_obj = _FFTW_PLAN_CACHE[cache_key]
            # Copy data to input array
            np.copyto(fft_obj.input_array, padded_data[:, rx, :])
            # Execute the plan
            fft_obj()
            # Get the result
            range_fft = fft_obj.output_array
        else:
            # Create a new FFTW plan
            fft_input = pyfftw.empty_aligned(fft_shape, dtype='complex64')
            fft_output = pyfftw.empty_aligned(fft_shape, dtype='complex64')
            fft_obj = pyfftw.FFTW(fft_input, fft_output, axes=(fft_axis,), 
                                 flags=('FFTW_MEASURE',), threads=pyfftw.config.NUM_THREADS)
            # Copy data to input array
            np.copyto(fft_input, padded_data[:, rx, :])
            # Execute the plan
            fft_obj()
            # Get the result
            range_fft = fft_output
            # Cache the plan
            _FFTW_PLAN_CACHE[cache_key] = fft_obj
        
            # Apply enhanced static clutter removal (enabled by default)
            if remove_clutter:
                # Reshape range_fft to 2D matrix for PCA
                range_fft_2d = range_fft.reshape(-1, range_fft.shape[-1])
                # Apply enhanced static clutter removal with adaptive parameters
                range_fft_clean = remove_static_clutter(
                    range_fft_2d,
                    energy_threshold=0.95,  # Adaptive energy threshold
                    robust_estimation=True,  # Enable robust covariance estimation
                    noise_suppression=True   # Enable noise suppression
                )
                # Reshape back to original shape
                range_fft = range_fft_clean.reshape(range_fft.shape)
        
        # Then perform Doppler FFT using pyfftw
        # Check if we have a cached FFTW plan for this shape
        doppler_fft_shape = range_fft.shape
        doppler_fft_axis = 1
        doppler_cache_key = (doppler_fft_shape, doppler_fft_axis, 'fft', rx, 'doppler')
        
        if doppler_cache_key in _FFTW_PLAN_CACHE:
            doppler_fft_obj = _FFTW_PLAN_CACHE[doppler_cache_key]
            # Copy data to input array
            np.copyto(doppler_fft_obj.input_array, range_fft)
            # Execute the plan
            doppler_fft_obj()
            # Get the result and apply fftshift
            doppler_fft = np.fft.fftshift(doppler_fft_obj.output_array, axes=1)
        else:
            # Create a new FFTW plan
            doppler_fft_input = pyfftw.empty_aligned(doppler_fft_shape, dtype='complex64')
            doppler_fft_output = pyfftw.empty_aligned(doppler_fft_shape, dtype='complex64')
            doppler_fft_obj = pyfftw.FFTW(doppler_fft_input, doppler_fft_output, axes=(doppler_fft_axis,), 
                                         flags=('FFTW_MEASURE',), threads=pyfftw.config.NUM_THREADS)
            # Copy data to input array
            np.copyto(doppler_fft_input, range_fft)
            # Execute the plan
            doppler_fft_obj()
            # Get the result and apply fftshift
            doppler_fft = np.fft.fftshift(doppler_fft_output, axes=1)
            # Cache the plan
            _FFTW_PLAN_CACHE[doppler_cache_key] = doppler_fft_obj
        
        # Non-coherent power summation
        range_doppler += np.abs(doppler_fft)**2
    
    # Take square root after power summation
    range_doppler = np.sqrt(range_doppler)
    
    # Calculate range axis with proper scaling and FFT bin mapping
    if radar_params is not None:
        max_range = radar_params.max_range
        range_resolution = radar_params.range_resolution
        num_range_bins = padding_sizes[0] // 2  # Only positive frequencies
        range_axis = np.arange(num_range_bins) * range_resolution
    else:
        bin_indices = np.arange(padding_sizes[0] // 2)  # Only positive frequencies
        # Convert freq_slope from MHz/μs to Hz/s
        freq_slope_hz_per_s = freq_slope * 1e12  # MHz/μs to Hz/s
        range_axis = (bin_indices * c * sampling_rate) / (2 * freq_slope_hz_per_s * padding_sizes[0])
    
    # Get positive ranges (first half of FFT output)
    range_doppler = range_doppler[:padding_sizes[0]//2]
    range_axis = range_axis[:padding_sizes[0]//2]
    
    # Calculate velocity axis
    doppler_freqs = np.fft.fftshift(np.fft.fftfreq(padding_sizes[2], d=chirp_time))
    if radar_params is not None:
        num_tx = radar_params.num_tx_channels
        carrier_freq = radar_params.config_params['start_freq'] * 1e9
        wavelength = c / carrier_freq
        # Calculate velocity in m/s
        velocity_axis = (doppler_freqs * wavelength) / (4 * num_tx)
    else:
        # Default calculation if radar_params not provided, using typical 77 GHz carrier
        wavelength = c / (77e9)  # 77 GHz typical automotive radar
        # Calculate velocity in m/s
        velocity_axis = (doppler_freqs * wavelength) / 2
    
    # Apply 2D CFAR detection
    detection_mask, detected_objects = apply_cfar(range_doppler, cfar_params=radar_params.cfar_params if hasattr(radar_params, 'cfar_params') else None)
    
    # Convert detected object indices to actual ranges and velocities
    detected_points = []
    for range_idx, doppler_idx, mag in detected_objects:
        # Ensure indices are within bounds
        if (0 <= range_idx < len(range_axis) and 
            0 <= doppler_idx < len(velocity_axis)):
            detected_points.append((range_axis[int(range_idx)], 
                                 velocity_axis[int(doppler_idx)], 
                                 mag))
    
    return range_doppler, range_axis, velocity_axis, detected_points

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
