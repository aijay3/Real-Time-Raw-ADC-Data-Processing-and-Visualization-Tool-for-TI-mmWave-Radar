"""
Main application module for real-time radar visualization.
"""

from matplotlib import pyplot as plt
from rp_real_time_process import UdpListener, DataProcessor
from radar_config import SerialConfig
from radar_parameters import RadarParameters
from queue import Queue, Empty
import pyqtgraph as pg
# Enable OpenGL acceleration for better performance
pg.setConfigOptions(useOpenGL=True, antialias=True)
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import numpy as np  
import threading
import time
import sys
import socket
import logging
from typing import Optional, List, Tuple
from rp_app_layout import Ui_MainWindow
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import os
import serial.tools.list_ports

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_COM_PORT = "COM5"
SOCKET_TIMEOUT = 0.1  # seconds
BUFFER_SIZE = 2097152  # bytes
FRAME_QUEUE_TIMEOUT = 0.1  # seconds
PACKET_HEADER = (0xA55A).to_bytes(2, byteorder='little', signed=False)
PACKET_FOOTER = (0xEEAA).to_bytes(2, byteorder='little', signed=False)
PACKET_SIZE_ZERO = (0x00).to_bytes(2, byteorder='little', signed=False)
PACKET_SIZE_SIX = (0x06).to_bytes(2, byteorder='little', signed=False)

# FPGA Command Codes
CMD_INIT = (0x01).to_bytes(2, byteorder='little', signed=False)
CMD_CONFIG = (0x02).to_bytes(2, byteorder='little', signed=False)
CMD_SET_FPGA = (0x03).to_bytes(2, byteorder='little', signed=False)
CMD_GET_STATUS = (0x04).to_bytes(2, byteorder='little', signed=False)
CMD_START = (0x05).to_bytes(2, byteorder='little', signed=False)
CMD_STOP = (0x06).to_bytes(2, byteorder='little', signed=False)
CMD_RESET = (0x07).to_bytes(2, byteorder='little', signed=False)
CMD_DEBUG = (0x08).to_bytes(2, byteorder='little', signed=False)
CMD_CONNECT = (0x09).to_bytes(2, byteorder='little', signed=False)
CMD_SET_MODE = (0x0A).to_bytes(2, byteorder='little', signed=False)
CMD_SET_PACKET = (0x0B).to_bytes(2, byteorder='little', signed=False)
CMD_GET_CONFIG = (0x0C).to_bytes(2, byteorder='little', signed=False)
CMD_SET_PARAMS = (0x0D).to_bytes(2, byteorder='little', signed=False)
CMD_GET_VERSION = (0x0E).to_bytes(2, byteorder='little', signed=False)

# FPGA Configuration Data
FPGA_CONFIG_DATA = (0x01020102031e).to_bytes(6, byteorder='big', signed=False)
PACKET_CONFIG_DATA = (0xc005350c0000).to_bytes(6, byteorder='big', signed=False)

# Network Configuration
RADAR_HOST = '192.168.33.30'
RADAR_DATA_PORT = 4098
RADAR_CONFIG_PORT = 4096
FPGA_CONFIG_HOST = '192.168.33.180'
FPGA_CONFIG_PORT = 4096

# Data Processing Queues
binary_data_queue = Queue()
range_profile_queue = Queue()

# Default Radar Configuration (updated when config file is loaded)
num_adc_samples = 256  # Default values
num_chirps = 16
num_tx_channels = 3
num_rx_channels = 4  
radar_config = [num_adc_samples, num_chirps, num_tx_channels, num_rx_channels]
frame_length = num_adc_samples * num_chirps * num_tx_channels * num_rx_channels * 2

# Network addresses
data_address = (RADAR_HOST, RADAR_DATA_PORT)
config_address = (RADAR_HOST, RADAR_CONFIG_PORT)
fpga_address = (FPGA_CONFIG_HOST, FPGA_CONFIG_PORT)

# Global instances
config = None
radar_params = None
plot_widget = None
ui = None
collector = None
processor = None
radar_ctrl = None
fpga_socket = None
main_window = None
plot_rpl = None  # Range profile plot

def get_available_com_ports() -> List[str]:
    """Get a list of available COM ports."""
    try:
        ports = [port.device for port in serial.tools.list_ports.comports()]
        logger.debug(f"Found COM ports: {ports}")
        return ports
    except Exception as e:
        logger.error(f"Error getting COM ports: {e}")
        return []

def browse_config():
    """Browse and load radar configuration file."""
    global ui, radar_params
    filename, _ = QFileDialog.getOpenFileName(
        None,
        "Select Configuration File",
        os.path.dirname(os.path.abspath(__file__)),
        "Config Files (*.cfg);;All Files (*.*)"
    )
    if filename:
        try:
            radar_params = RadarParameters(filename)
            ui.config_path.setText(os.path.basename(filename))
            ui.config_path.setProperty("fullPath", filename)
        except Exception as e:
            logger.error(f"Failed to process config file: {e}")

def create_command_packet(command_code: str) -> bytes:
    """Create a command packet for FPGA communication."""
    if command_code == '9':
        return PACKET_HEADER + CMD_CONNECT + PACKET_SIZE_ZERO + PACKET_FOOTER
    elif command_code == 'E':
        return PACKET_HEADER + CMD_GET_VERSION + PACKET_SIZE_ZERO + PACKET_FOOTER
    elif command_code == '3':
        return PACKET_HEADER + CMD_SET_FPGA + PACKET_SIZE_SIX + FPGA_CONFIG_DATA + PACKET_FOOTER
    elif command_code == 'B':
        return PACKET_HEADER + CMD_SET_PACKET + PACKET_SIZE_SIX + PACKET_CONFIG_DATA + PACKET_FOOTER
    elif command_code == '5':
        return PACKET_HEADER + CMD_START + PACKET_SIZE_ZERO + PACKET_FOOTER
    elif command_code == '6':
        return PACKET_HEADER + CMD_STOP + PACKET_SIZE_ZERO + PACKET_FOOTER
    else:
        return b'NULL'

def update_figure() -> None:
    """Update range profile plot with new data."""
    global plot_rpl, update_time, plot_widget, ui, radar_params, processor
    
    # Calculate time since last update
    now = time.time()
    dt = now - update_time
    
    # Track frame timing for smoothness metrics
    frame_times = getattr(update_figure, 'frame_times', [])
    if len(frame_times) > 100:  # Keep only the last 100 frame times
        frame_times.pop(0)
    frame_times.append(dt)
    update_figure.frame_times = frame_times
    
    # Calculate frame time statistics for adaptive timing
    if len(frame_times) > 10:
        avg_frame_time = sum(frame_times[-10:]) / 10
        frame_time_std = np.std(frame_times[-10:])
        jitter = frame_time_std / avg_frame_time if avg_frame_time > 0 else 0
        # Log frame timing stats occasionally
        if getattr(update_figure, 'frame_count', 0) % 100 == 0:
            logger.debug(f"Frame timing: avg={avg_frame_time*1000:.1f}ms, jitter={jitter*100:.1f}%")
    else:
        jitter = 0
    
    # Increment frame counter
    update_figure.frame_count = getattr(update_figure, 'frame_count', 0) + 1
    
    # Use non-blocking queue gets with short timeout to prevent UI freezing
    try:
        # Try to get range profile data with a short timeout
        try:
            range_profile_data, range_axis, detected_points = range_profile_queue.get(timeout=0.01)
            update_figure.last_range_profile = (range_profile_data, range_axis, detected_points)
        except Empty:
            # Use the last known data if available
            if hasattr(update_figure, 'last_range_profile'):
                range_profile_data, range_axis, detected_points = update_figure.last_range_profile
            else:
                # Skip this update if no data is available yet
                logger.debug("No range profile data available yet")
                schedule_next_update(jitter)
                return
            
        # Update range profile with error handling
        try:
            # Get selected channel index
            channel_idx = ui.channel_radio_group.checkedId()
            
            # Always clear the plot widget
            plot_widget.clear()
            
            # Define colors for each channel
            channel_colors = ['y', 'g', 'b', 'r']
            
            # Initialize variables
            channel_color = 'y'  # Default to yellow
            profile_data_db = None
            plot_name = "Sum" if channel_idx == 0 else f"CH{channel_idx}"
            
            if channel_idx == 0:  # Sum mode
                profile_data_selected = range_profile_data[:, 0]  # First column is the sum
                profile_data_db = 20 * np.log10(profile_data_selected)
            elif channel_idx == 5:  # ALL channels mode
                # Add legend
                #plot_widget.addLegend()
                
                # Plot all channels (skip the sum column)
                for ch in range(1, min(5, range_profile_data.shape[1])):  # Start from 1 to skip sum, limit to 4 channels
                    profile_data_selected = range_profile_data[:, ch]
                    profile_data_db = 20 * np.log10(profile_data_selected)
                    
                    # Get color for this channel
                    channel_color = channel_colors[ch-1]  # Adjust index since we skip sum column
                    
                    # Create a new plot with legend entry
                    plot_widget.plot(
                        range_axis, 
                        profile_data_db, 
                        pen=pg.mkPen(channel_color, width=2),
                        name=f"Channel {ch}",  # More descriptive name for legend
                        antialias=True,
                        useOpenGL=True
                    )
                
                # Set profile_data_db to None to skip the final plot since we've already plotted all channels
                profile_data_db = None
            elif channel_idx > 0 and channel_idx < range_profile_data.shape[1]:
                # Single channel mode (channels 1-4 are at indices 1-4 since 0 is sum)
                profile_data_selected = range_profile_data[:, channel_idx]
                # Convert magnitude directly to dB
                profile_data_db = 20 * np.log10(profile_data_selected)
                
                # Get the appropriate color for the selected channel (adjust index since we skip sum)
                channel_color = channel_colors[channel_idx-1] if channel_idx-1 < len(channel_colors) else 'y'
                
            # Only create the plot if we have valid data
            if profile_data_db is not None:
                plot_rpl = plot_widget.plot(
                    range_axis, 
                    profile_data_db, 
                    pen=pg.mkPen(channel_color, width=2),
                    name=plot_name,
                    antialias=True,
                    useOpenGL=True
                )
            
            # Update data display table with detected objects
            ui.data_display_table.setRowCount(0)  # Clear existing rows
            if detected_points:
                # Sort detected points by range
                detected_points.sort(key=lambda x: x[0])
                
                # Add rows to table
                for range_val, magnitude in detected_points:
                    row_position = ui.data_display_table.rowCount()
                    ui.data_display_table.insertRow(row_position)
                    
                    # Create and set range item
                    range_item = QtWidgets.QTableWidgetItem(f"{range_val:.2f}")
                    range_item.setTextAlignment(QtCore.Qt.AlignCenter)
                    ui.data_display_table.setItem(row_position, 0, range_item)
                    
                    # Create and set magnitude item
                    magnitude_item = QtWidgets.QTableWidgetItem(f"{magnitude:.1f}")
                    magnitude_item.setTextAlignment(QtCore.Qt.AlignCenter)
                    ui.data_display_table.setItem(row_position, 1, magnitude_item)
        except Exception as e:
            logger.error(f"Error updating range profile: {e}")
            
    except Exception as e:
        logger.error(f"Error in update_figure: {e}")
        schedule_next_update(jitter)
        return
        
    # Schedule next update with adaptive timing
    schedule_next_update(jitter)
    update_time = now

def schedule_next_update(jitter: float = 0):
    """Schedule the next UI update with fixed timing."""
    global update_time
    
    # Use a fixed update interval of 100ms (10 FPS)
    fixed_interval = 100  # milliseconds
    
    # Schedule the next update
    QtCore.QTimer.singleShot(fixed_interval, update_figure)

def initialize_radar() -> None:
    """Initialize and start radar data collection."""
    global radar_ctrl, config, ui, radar_config, frame_length, radar_params
    
    # Get configuration
    com_port = ui.com_select.currentText()
    config_path = ui.config_path.property("fullPath")
    
    # Validate inputs
    if not com_port:
        logger.error("Please select a COM port")
        return
    if not config_path or not os.path.exists(config_path):
        logger.error("Please select a valid configuration file")
        return
    
    # Initialize radar parameters
    if not radar_params:
        radar_params = RadarParameters(config_path)
    
    # Display parameters
    params = radar_params.get_all_parameters()
    logger.info("\n=== Calculated Radar Parameters ===")
    for key, value in params.items():
        if isinstance(value, float):
            logger.info(f"{key}: {value:.2f}")
        else:
            logger.info(f"{key}: {value}")
    logger.info("================================")
    
    # Update configuration
    num_adc_samples = radar_params.config_params['adc_samples']
    num_chirps = radar_params.config_params['chirps_per_frame']
    num_tx_channels = radar_params.num_tx_channels
    num_rx_channels = radar_params.num_rx_channels
    
    radar_config = [num_adc_samples, num_chirps, num_tx_channels, num_rx_channels]
    frame_length = num_adc_samples * num_chirps * num_tx_channels * num_rx_channels * 2
    
    # Initialize radar
    radar_ctrl = SerialConfig(name='ConnectRadar', CLIPort=com_port, BaudRate=115200)
    radar_ctrl.StopRadar()
    radar_ctrl.SendConfig(config_path)
    
    # Start radar
    time.sleep(1)
    radar_ctrl.StartRadar()
    logger.info("Radar started and streaming data...")
    
    update_figure()

def cleanup() -> None:
    """Cleanup all resources and stop data collection."""
    global collector, processor, fpga_socket, radar_ctrl
    logger.info("Starting cleanup...")
    
    # Stop radar
    if hasattr(radar_ctrl, 'StopRadar'):
        try:
            radar_ctrl.StopRadar()
            logger.info("Radar stopped")
        except Exception as e:
            logger.error(f"Error stopping radar: {e}")
    
    # Stop threads with proper cleanup
    for thread, name in [(collector, "Collector"), (processor, "Processor")]:
        if thread and thread.is_alive():
            try:
                thread.stop()
                thread.join(timeout=1)
                if thread.is_alive():
                    logger.warning(f"{name} thread taking longer to stop, waiting...")
                    thread.join(timeout=3)
                    if thread.is_alive():
                        logger.warning(f"{name} thread did not stop gracefully")
                    else:
                        logger.info(f"{name} thread stopped successfully")
            except AttributeError as e:
                logger.error(f"Error stopping {name} thread: Missing stop method")
            except Exception as e:
                logger.error(f"Error stopping {name} thread: {e}")
    
    # Close socket
    if fpga_socket:
        try:
            fpga_socket.sendto(create_command_packet('6'), fpga_address)
            fpga_socket.close()
            logger.info("Config socket closed")
        except Exception as e:
            logger.error(f"Error closing config socket: {e}")
    
    # Clear queues
    for queue in [binary_data_queue, range_profile_queue]:
        try:
            while not queue.empty():
                queue.get_nowait()
        except Exception as e:
            logger.debug(f"Error clearing queue: {e}")
            
    logger.info("Cleanup completed")

def initialize_gui() -> None:
    """Initialize and run the main application window."""
    global plot_rpl, update_time, collector, processor, fpga_socket
    global radar_ctrl, plot_widget, ui, main_window
    
    app = QtWidgets.QApplication(sys.argv)
    main_window = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(main_window)
    
    # Configure close event
    def handle_close(event):
        cleanup()
        app.quit()
        event.accept()
        sys.exit()
    
    main_window.closeEvent = handle_close
    main_window.show()
    
    # Initialize buttons
    start_button = ui.start_button
    exit_button = ui.exit_button
    
    # Setup COM ports
    com_ports = get_available_com_ports()
    ui.com_select.addItems(com_ports)
    com5_index = ui.com_select.findText(DEFAULT_COM_PORT)
    if com5_index >= 0:
        ui.com_select.setCurrentIndex(com5_index)
    
    # Setup range profile plot
    view_rpl = ui.range_profile_view.addViewBox()
    view_rpl.setAspectLocked(False)
    plot_widget = pg.PlotItem(viewBox=view_rpl)
    
    # Configure range profile plot
    ui.range_profile_view.setCentralItem(plot_widget)
    plot_widget.setLabel('left', 'Relative Power', units='dB')
    plot_widget.setLabel('bottom', 'Range', units='m')
    plot_widget.showGrid(x=True, y=True)
    # Set grid color to dark gray for better visibility on black background
    plot_widget.getAxis('left').setPen(pg.mkPen(color='w', width=1))
    plot_widget.getAxis('bottom').setPen(pg.mkPen(color='w', width=1))
    plot_widget.getAxis('left').setTextPen(pg.mkPen(color='w', width=1))
    plot_widget.getAxis('bottom').setTextPen(pg.mkPen(color='w', width=1))
    # Get maximum range from radar parameters
    max_range = radar_params.max_range if radar_params else 10
    
    # Set range profile X-axis range
    plot_widget.setXRange(0, max_range)
    # Set Y-axis range to 40-120 dB
    plot_widget.setYRange(40, 120)
    # Create range profile plot with OpenGL acceleration
    plot_rpl = plot_widget.plot(pen='y', antialias=True, useOpenGL=True)
    
    update_time = time.time()
    
    # Try to load default config and update UI
    if load_default_config() and ui:
        ui.config_path.setText(os.path.basename(default_config))
        ui.config_path.setProperty("fullPath", default_config)
    
    # Connect buttons
    ui.browse_button.clicked.connect(browse_config)
    start_button.clicked.connect(initialize_radar)
    exit_button.clicked.connect(lambda: (cleanup(), app.quit(), sys.exit()))
    
    # Connect radio buttons for channel selection
    def on_radio_button_changed(id):
        # Update processor directly
        if processor:
            processor.set_channel(id)
            logger.info(f"Selected channel changed to {id}")
    
    ui.channel_radio_group.buttonClicked[int].connect(on_radio_button_changed)
    
    # Connect window and padding selection dropdowns
    def on_window_changed(window_type):
        if processor:
            processor.set_window_type(window_type)
            logger.info(f"Window type changed to {window_type}")
    
    def on_range_padding_changed(padding_size):
        if processor:
            processor.set_range_padding(int(padding_size))
            logger.info(f"Range padding changed to {padding_size}")
    
    def on_range_offset_changed(offset_value):
        if radar_params:
            radar_params.range_offset = offset_value
            # Disable auto-adjust when user manually changes the offset
            radar_params.auto_adjust_offset = False
            logger.info(f"Range offset calibration changed to {offset_value} meters (auto-adjust disabled)")
    
    ui.window_select.currentTextChanged.connect(on_window_changed)
    ui.range_pad_select.currentTextChanged.connect(on_range_padding_changed)
    ui.range_offset_spin.valueChanged.connect(on_range_offset_changed)
    
    # Connect static clutter removal button
    def on_clutter_removal_changed(checked):
        if processor:
            processor.set_clutter_removal(checked)
    
    ui.remove_clutter_button.toggled.connect(on_clutter_removal_changed)
    
    # Connect CFAR parameter controls
    def on_guard_cells_changed(value):
        if processor:
            processor.set_cfar_params(guard_cells=value)
            
    def on_training_cells_changed(value):
        if processor:
            processor.set_cfar_params(training_cells=value)
            
    def on_false_alarm_changed(value):
        if processor:
            processor.set_cfar_params(pfa=value)
            
    def on_group_peaks_changed(checked):
        if processor:
            processor.set_cfar_params(group_peaks=checked)
    
    # Connect CFAR spinboxes
    ui.guard_cells_spin.valueChanged.connect(on_guard_cells_changed)
    ui.training_cells_spin.valueChanged.connect(on_training_cells_changed)
    ui.false_alarm_spin.valueChanged.connect(on_false_alarm_changed)
    ui.group_peaks_button.toggled.connect(on_group_peaks_changed)
    
    app.instance().exec_()

def initialize_fpga() -> socket.socket:
    """Initialize FPGA with retry mechanism and proper error handling."""
    command_sequence = ['9', 'E', '3', 'B', '5', '6']
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            sock.bind(config_address)
        except OSError as e:
            logger.error(f"Failed to bind socket: {e}")
            try:
                import psutil
                current_pid = os.getpid()
                for proc in psutil.process_iter(['pid', 'name', 'connections']):
                    if proc.info['name'] == 'python.exe' and proc.pid != current_pid:
                        try:
                            connections = proc.connections()
                            for conn in connections:
                                if conn.laddr.port == config_address[1]:
                                    logger.info(f"Terminating process {proc.pid}")
                                    proc.terminate()
                                    proc.wait(timeout=3)
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            continue
                
                time.sleep(1)
                sock.bind(config_address)
            except Exception as bind_error:
                logger.error(f"Could not bind socket: {bind_error}")
                raise
        
        for command in command_sequence[:5]:
            sock.sendto(create_command_packet(command), fpga_address)
            try:
                msg, server = sock.recvfrom(2048)
            except socket.timeout:
                logger.warning(f"Timeout waiting for FPGA response on command {command}")
            time.sleep(0.1)
        
        return sock
    
    except Exception as e:
        logger.error(f"Failed to initialize FPGA: {e}")
        if 'sock' in locals():
            sock.close()
        raise

# Initialize system
fpga_socket = initialize_fpga()

# Define default config path
default_config = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             '..', 'config', 'AWR1843_cfg.cfg')

def load_default_config():
    """Load default radar configuration file."""
    global radar_params
    if os.path.exists(default_config):
        radar_params = RadarParameters(default_config)
        logger.info("Loaded default radar parameters")
        return True
    else:
        logger.warning("Default config file not found")
        return False

# Start data collection
if not radar_params and not load_default_config():
    logger.warning("No radar parameters available. Processing will use defaults.")

collector = UdpListener('Listener', binary_data_queue, frame_length, data_address, BUFFER_SIZE)
processor = DataProcessor('Processor', radar_config, binary_data_queue,
                        None, range_profile_queue, None,
                        selected_channel=0, radar_params=radar_params)
# Ensure CH1 is selected by default
collector.start()
processor.start()

if __name__ == '__main__':
    try:
        initialize_gui()
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        cleanup()
        logger.info("Program closed")
        sys.exit()
