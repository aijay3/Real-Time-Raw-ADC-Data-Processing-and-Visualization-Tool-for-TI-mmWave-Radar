"""
Real-time processing module for radar data.
Handles UDP data reception and processing for radar visualization.
"""

import threading as th
import numpy as np
import socket
import logging
import time
import psutil
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import List, Tuple, Optional, Union, Dict
from queue import Queue, Empty
from dataclasses import dataclass
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
HEADER_SIZE = 10  # Size of UDP packet header in bytes
SOCKET_TIMEOUT = 0.1  # seconds
QUEUE_TIMEOUT = 0.5  # seconds
LOG_INTERVAL = 100  # frames

# Import dsp module for radar signal processing
import rd_dsp

class UdpListener(th.Thread):
    """Thread class for receiving and processing UDP data streams from radar."""
    
    def __init__(self, name: str, binary_data_queue: Queue, frame_length: int, 
                 data_address: Tuple[str, int], buffer_size: int):
        """
        Initialize UDP listener thread.
        
        Args:
            name: Thread name
            binary_data_queue: Queue to store ADC data from UDP stream
            frame_length: Length of a single frame
            data_address: Tuple of (host IP address, port)
            buffer_size: Socket buffer size
        """
        self._stop_event = th.Event()
        super().__init__(name=name)
        self.binary_data_queue = binary_data_queue
        self.frame_length = frame_length
        self.data_address = data_address
        self.buffer_size = buffer_size
        self.data_socket = None

    def stop(self):
        """Signal the thread to stop and cleanup resources."""
        self._stop_event.set()
        if self.data_socket:
            try:
                self.data_socket.close()
            except Exception as e:
                logger.error(f"Error closing socket: {e}")

    def run(self):
        """Main thread execution loop for receiving UDP data."""
        try:
            # Configure data type for binary conversion
            data_type = np.dtype(np.int16).newbyteorder('<')
            frame_buffer = []
            frame_count = 0
            
            # Initialize socket with error handling
            try:
                self.data_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.data_socket.bind(self.data_address)
                self.data_socket.settimeout(SOCKET_TIMEOUT)
                logger.info("UDP socket created successfully")
                logger.info("Starting data stream reception")
            except socket.error as e:
                logger.error(f"Socket initialization failed: {e}")
                return
            
            # Main reception loop
            while not self._stop_event.is_set():
                try:
                    packet_data, _ = self.data_socket.recvfrom(self.buffer_size)
                    
                    # Validate packet size
                    if len(packet_data) <= HEADER_SIZE:
                        logger.warning(f"Received undersized packet: {len(packet_data)} bytes")
                        continue
                    
                    # Process packet and convert binary data to int16
                    frame_data = packet_data[HEADER_SIZE:]  # Remove header
                    int16_samples = np.frombuffer(frame_data, dtype=data_type)
                    frame_buffer.extend(int16_samples)
                    
                    # Store the raw packet data for display
                    # We'll pass this along with the processed data
                    raw_packet_data = packet_data[:50]  # Store first 50 bytes of raw packet
                    binary_data = frame_data[:50]  # Store first 50 bytes of binary data
                    
                    # Process complete frames
                    while len(frame_buffer) >= self.frame_length:
                        frame_count += 1
                        current_frame = frame_buffer[:self.frame_length]
                        frame_buffer = frame_buffer[self.frame_length:]
                        
                        try:
                            # Create a tuple with the frame data and raw packet data
                            frame_data_with_raw = {
                                'frame_data': current_frame,
                                'raw_packet': raw_packet_data,
                                'binary_data': binary_data
                            }
                            self.binary_data_queue.put(frame_data_with_raw, timeout=QUEUE_TIMEOUT)
                            if frame_count % LOG_INTERVAL == 0:
                                logger.debug(f"Processed frame {frame_count}")
                        except Queue.Full:
                            logger.warning("Queue full, dropping frame")
                            
                except socket.timeout:
                    continue
                except Exception as e:
                    logger.error(f"Error processing UDP data: {e}")
                    
        except Exception as e:
            logger.error(f"Fatal error in UdpListener: {e}")
        finally:
            if self.data_socket:
                try:
                    self.data_socket.close()
                    logger.info("UDP socket closed")
                except Exception as e:
                    logger.error(f"Error closing socket: {e}")

class DataProcessor(th.Thread):
    """Thread class for processing radar data and generating visualizations."""
    
    def __init__(self, name: str, config: List[int], 
                 binary_queue: Queue, range_doppler_queue: Queue, 
                 range_profile_queue: Queue, range_angle_queue: Queue = None,
                 selected_channel: int = 0, radar_params=None):
        """Initialize with additional processing resources."""
        super().__init__(name=name)
        self._stop_event = th.Event()
        self.remove_clutter = True  # Static clutter removal enabled by default
        self.window_type = 'blackmanharris'  # Default window type (Blackman-Harris)
        self.range_padding = 512  # Default range padding
        self.doppler_padding = 64  # Default doppler padding
        
        # Enhanced CFAR parameters with adaptive thresholds
        self.cfar_params = {
            'range_guard': 2,      # Default guard cells in range
            'doppler_guard': 2,    # Default guard cells in doppler
            'range_training': 8,   # Default training cells in range
            'doppler_training': 4, # Default training cells in doppler
            'pfa': 0.001,         # Default probability of false alarm
            'group_peaks': True    # Peak grouping enabled by default
        }
        
        # Configuration parameters
        self.num_adc_samples = config[0]
        self.num_chirps = config[1]
        self.num_tx_channels = config[2]
        self.num_rx_channels = config[3]
        
        # Data queues
        self.binary_queue = binary_queue
        self.range_profile_queue = range_profile_queue
        self.range_doppler_queue = range_doppler_queue
        self.selected_channel = selected_channel
        self.radar_params = radar_params
        
        # Store raw data samples for display
        self.last_raw_packet = None  # Raw packet from UDP socket
        self.last_binary_data = None  # Binary data
        
        # Initialize thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Cache for intermediate results
        self.cache = {}
        
        # Frame stats metrics
        self.frame_count = 0
        
        logger.info(f"Initialized DataProcessor with config: "
                   f"samples={self.num_adc_samples}, "
                   f"chirps={self.num_chirps}, "
                   f"tx={self.num_tx_channels}, "
                   f"rx={self.num_rx_channels}")

    def set_clutter_removal(self, enabled: bool):
        """Set the static clutter removal state."""
        self.remove_clutter = enabled
        logger.info(f"Static clutter removal {'enabled' if enabled else 'disabled'}")
        
    def set_channel(self, selected_channel: int):
        """Set the selected channel for processing."""
        self.selected_channel = selected_channel
        logger.info(f"Selected channel set to {selected_channel}")
        
    def set_cfar_params(self, guard_cells: int = None, training_cells: int = None, 
                       pfa: float = None, group_peaks: bool = None):
        """Update CFAR processing parameters."""
        if guard_cells is not None:
            self.cfar_params['guard_cells'] = guard_cells
        if training_cells is not None:
            self.cfar_params['training_cells'] = training_cells
        if pfa is not None:
            self.cfar_params['pfa'] = pfa
        if group_peaks is not None:
            self.cfar_params['group_peaks'] = group_peaks
        logger.info(f"Updated CFAR parameters: {self.cfar_params}")

    def set_window_type(self, window_type: str):
        """Set the window type for processing."""
        # Convert GUI window names to internal names
        window_map = {
            "Blackman-Harris": "blackmanharris",
            "No Window": "no window",
            "Hamming": "hamming",
            "Hann": "hann",
            "Blackman": "blackman"
        }
        self.window_type = window_map.get(window_type, "blackmanharris")
        logger.info(f"Window type set to {window_type} (internal: {self.window_type})")
        
    def set_range_padding(self, padding: int):
        """Set the range dimension padding size."""
        self.range_padding = padding
        logger.info(f"Range padding set to {padding}")

    def set_doppler_padding(self, padding: int):
        """Set the doppler dimension padding size."""
        self.doppler_padding = padding
        logger.info(f"Doppler padding set to {padding}")
        

    def _process_frame(self, int16_samples: Union[List, np.ndarray]) -> np.ndarray:
        """
        Process a single frame of radar data in 2I2Q interleaved format (I1, I2, Q1, Q2).
        
        Args:
            int16_samples: Int16 ADC samples in format [I1, I2, Q1, Q2, ...].
            
        Returns:
            Processed data in format [numRX, numChirps * numADCSamples].
        """
        try:
            # Convert list to numpy array while preserving int16 dtype from UDPListener
            if isinstance(int16_samples, list):
                int16_samples = np.array(int16_samples)
            
            # Calculate num_chirps from radar parameters as num_loops x chirps_per_frame
            if self.radar_params:
                num_chirps = self.radar_params.config_params['num_loops'] * self.radar_params.config_params['chirps_per_frame']
            else:
                num_chirps = len(int16_samples) // (2 * self.num_adc_samples * self.num_rx_channels)

            # Reshape data to separate I and Q samples
            data = int16_samples.reshape(-1, 4)  # Reshape into groups of 4 samples
            
            # Combine I and Q for complex samples
            complex_samples = data[:, 0::2] + 1j * data[:, 2::2]  # Combine I and Q samples

            # Reshape the data: each column represents data from a chirp
            complex_samples = complex_samples.reshape((self.num_adc_samples * self.num_rx_channels, num_chirps), order="F")
            complex_samples = complex_samples.T  # Transpose so that rows represent chirps
            
            # Organize data for the 3x4 antenna array with λ/2 spacing
            rx_chirp_samples = np.zeros((12, num_chirps * self.num_adc_samples), dtype=np.complex64)
            
            # Map physical RX channels to virtual array positions
            # The array is arranged in a 3x4 grid:
            # [0  1  2  3 ]  <- Top row (RX 0-3)
            # [4  5  6  7 ]  <- Middle row (RX 4-7)
            # [8  9  10 11]  <- Bottom row (RX 8-11)
            
            for rx in range(self.num_rx_channels):
                for chirp in range(num_chirps):
                    # The chirp sequence follows TX0->TX1->TX2 pattern for TDM-MIMO
                    # Each TX antenna creates a set of 4 virtual channels
                    virtual_rx = rx + (chirp % self.num_tx_channels) * self.num_rx_channels
                    rx_chirp_samples[virtual_rx, (chirp // self.num_tx_channels) * self.num_adc_samples:
                                              ((chirp // self.num_tx_channels) + 1) * self.num_adc_samples] = \
                        complex_samples[chirp, rx * self.num_adc_samples:(rx + 1) * self.num_adc_samples]

            return rx_chirp_samples

        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            raise

    def stop(self):
        """Signal the thread to stop and cleanup resources."""
        self._stop_event.set()
        # Clean up thread pool
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=False)
        # Clear caches
        if hasattr(self, 'cache'):
            self.cache.clear()

    def run(self):
        """Main processing loop."""
        self.frame_count = 0
        
        while not self._stop_event.is_set():
            try:
                # Get data with timeout
                frame_data_with_raw = self.binary_queue.get(timeout=SOCKET_TIMEOUT)
                
                # Extract frame data and raw packet data
                if isinstance(frame_data_with_raw, dict):
                    raw_frame = frame_data_with_raw.get('frame_data')
                    self.last_raw_packet = frame_data_with_raw.get('raw_packet')
                    self.last_binary_data = frame_data_with_raw.get('binary_data')
                else:
                    # For backward compatibility
                    raw_frame = frame_data_with_raw
                
                # Process frame
                try:
                    # Process chirp
                    processed_data = self._process_frame(raw_frame)
                    
                    self.frame_count += 1
                    
                    # Generate visualizations with user-selected padding
                    # Calculate range doppler with range and doppler axis using selected window
                    # Add CFAR parameters to radar_params for DSP module
                    if hasattr(self, 'cfar_params'):
                        if not hasattr(self.radar_params, 'cfar_params'):
                            self.radar_params.cfar_params = {}
                        self.radar_params.cfar_params.update(self.cfar_params)
                    
                    range_doppler, range_axis, velocity_axis, detected_points = rd_dsp.calculate_range_doppler(
                        processed_data,
                        self.radar_params,
                        remove_clutter=self.remove_clutter,
                        use_pulse_compression=True,  # Enable LFM/Chirp pulse compression
                        window_type=self.window_type,
                        range_padding=self.range_padding,
                        doppler_padding=self.doppler_padding
                    )
                    
                    # Store processed data in cache with unique key
                    # Use a string representation of the first few values as a simple hash
                    # This avoids the "unhashable type: numpy.ndarray" error
                    data_key = str(processed_data[:20].tobytes())
                    self.cache[data_key] = processed_data
                    
                    # Update queues with latest data, dropping old data if queue is full
                    try:
                        if not self.range_doppler_queue.empty():
                            self.range_doppler_queue.get_nowait()
                    except Empty:
                        pass
                    
                    # Queue results with proper backpressure
                    try:
                        if all(x is not None for x in [range_doppler, range_axis, velocity_axis, detected_points]):
                            self.range_doppler_queue.put((range_doppler, range_axis, velocity_axis, detected_points), timeout=1.0)
                    except Exception as e:
                        logger.error(f"Error queueing results: {e}")
                        continue
                    
                    if self.frame_count % LOG_INTERVAL == 0:
                        logger.debug(f"Processed {self.frame_count} frames")
                        
                except Exception as e:
                    logger.error(f"Error in frame processing: {e}")
                    continue
                    
            except Empty:
                continue  # Queue timeout, check stop condition
            except Exception as e:
                logger.error(f"Error in main processing loop: {e}")
                continue
                
        logger.info(f"DataProcessor stopped after processing {self.frame_count} frames")
