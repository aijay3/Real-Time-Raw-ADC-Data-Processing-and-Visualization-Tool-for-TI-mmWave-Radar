# Real-Time Raw ADC Data Processing and Visualization Tool for TI mmWave Radar

## Overview

This project provides a real-time radar data capture and processing pipeline for **TI mmWave radar sensors** using the **DCA1000EVM** and a **Python-based software stack** — without relying on TI’s mmWave Studio. It streams and processes **raw ADC data** directly from compatible radar devices and presents the output in intuitive live visualizations.

Traditional workflows with mmWave Studio are limited to file-based offline processing and require multiple dependencies, specific software versions, and complex configuration steps. This tool bypasses those limitations by enabling **live UDP data capture**, **multi-threaded signal processing**, and **interactive GUI-based visualization**.

### 🧠 Key Features

- **Real-time radar visualization modes**:
  - 📈 **Range Profile** – object distance via FFT
  - 🔁 **Range Doppler** – distance and velocity mapping
  - 🧭 **Range Angle** – angular position estimation using beamforming
- **Direct raw ADC capture** via DCA1000 (no mmWave Studio)
- **CFAR detection**, **clutter removal**, and **windowed FFT processing**
- Interactive GUI with:
  - Parameter tuning (FFT size, padding, channel selection)
  - Live plot updates (~10 FPS)
- Modular Python implementation with PyQt5 and PyQtGraph

<div align="center">
  <img src="images/launcher_interface.png" width="600"/>
  <p><i>Launcher with selectable visualization modes</i></p>
</div>

### 🛠 Expected Hardware Compatibility

This tool is designed and tested for the **AWR1843AOP** mmWave radar sensor and DCA1000EVM, but it is expected to work with other TI AWR and IWR series sensors supported by the DCA1000 interface.

Examples include:
- AWR1243, AWR1443, AWR1642, AWR2243, AWR2944, AWR6843, AWR6843AOP
- IWR1443, IWR1642, IWR1843, IWR2944, IWR6843, IWR6843AOP

> ⚠️ While not all devices are explicitly tested, compatibility is expected based on shared interface and data capture protocols. Ensure your radar is supported by the DCA1000EVM and outputs raw ADC via LVDS.

<div align="center">
  <img src="images/range_profile.png" width="600"/>
  <p><i>Range Profile view: signal power vs. distance</i></p>
</div>

### 🚧 Motivation

Setting up real-time raw ADC capture with mmWave Studio often involves:
- Complex DIP switch configurations and unclear hardware instructions
- Platform-specific dependencies (Windows-only, MATLAB runtime)
- Fragmented documentation across toolboxes, user guides, and forums
- No support for live processing — only offline `.bin` analysis

This project addresses those limitations by providing:
- A **streamlined setup** (Python + Ethernet + config file)
- A **user-friendly interface** with parameter control and diagnostics
- A fully **extensible signal processing pipeline** for education, prototyping, and research

<div align="center">
  <img src="images/range_doppler.png" width="600"/>
  <p><i>Range Doppler view: distance vs. relative speed</i></p>
</div>

<div align="center">
  <img src="images/range_angle.png" width="600"/>
  <p><i>Range Angle view: spatial mapping via beamforming</i></p>
</div>

```mermaid
graph TD
  %% UI and Launcher
  A[Launcher GUI (launcher.py)] --> B[Range Profile App (rp_main.py)]
  A --> C[Range Doppler App (rd_main.py)]
  A --> D[Range Angle App (ra_main.py)]

  %% Configuration
  E[Radar Config File (.cfg)] --> F[SerialConfig Module]
  F --> G[mmWave Radar Sensor]

  %% Data Acquisition
  G --> H[FPGA (DCA1000)]
  H --> I[UDP Listener]
  I --> J[Binary Data Queue]

  %% Range Profile Pipeline
  J --> K[RP Data Processor]
  K --> L[1D FFT]
  L --> M[CFAR Detection]
  M --> N[Range Profile Queue]
  N --> O[RP GUI]

  %% Range Doppler Pipeline
  J --> P[RD Data Processor]
  P --> Q[2D FFT]
  Q --> R[2D CFAR Detection]
  R --> S[Range Doppler Queue]
  S --> T[RD GUI]

  %% Range Angle Pipeline
  J --> U[RA Data Processor]
  U --> V[3D FFT]
  V --> W[2D CFAR Detection]
  W --> X[Range Angle Queue]
  X --> Y[RA GUI]

```


This is a real-time ADC sample capture and processing tool to obtain and analyze raw data from TI mmWave radar ***AWR1843AOP EVM*** cascading with ***DCA1000 EVM*** using Python. The tool enables real-time processing to generate Range Profile, Range-Doppler, and Range-Angle images under 1 Transmitter and 4 Receiver (in this version) setting without using mmWave studio.

![System Overview](images/system_overview.png)
*System architecture showing the complete radar processing pipeline*

## Table of Contents
- [Overview](#overview)
- [Hardware Requirements](#hardware-requirements)
- [Software Requirements](#software-requirements)
- [Installation Guide](#installation-guide)
- [Hardware Setup](#hardware-setup)
- [Network Configuration](#network-configuration)
- [Running the Software](#running-the-software)
- [Using the Applications](#using-the-applications)
- [Signal Processing Features](#signal-processing-features)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)
- [Performance Specifications](#performance-specifications)
- [Future Enhancements](#future-enhancements)
- [Contact](#contact)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)

## Overview

This project addresses the significant challenges in accessing and processing raw ADC data from Texas Instruments mmWave radar sensors. Traditional methods involve complex proprietary tools, multiple software installations, and intricate setup procedures that often result in compatibility issues and operational inconsistencies.

Our solution provides:
- **Streamlined Workflow**: Reduces complex multi-step processes to a single application launch
- **Real-time Processing**: Immediate visualization and analysis without file-based delays
- **User-friendly Interface**: Intuitive GUI that abstracts radar complexity
- **Modular Architecture**: Extensible design for future enhancements
- **Direct Hardware Communication**: Bypasses proprietary tools like mmWave Studio

![Application Interface](images/application_interface.png)
*Main application interface showing real-time radar visualization*

## Hardware Requirements

### Essential Hardware Components

#### Primary Radar System
- **Texas Instruments AWR1843AOP EVM** (Evaluation Module)
  - Operating frequency: 76-81 GHz (W-Band FMCW)
  - Bandwidth: Up to 4 GHz available
  - TX Effective Isotropic Radiated Power (EIRP): 16 dBm
  - RX Effective Isotropic Noise Figure: 10 dB
  - Integrated 3TX/4RX Antenna-on-Package (AoP) configuration
  - Field of View: 140° Azimuth × 140° Elevation
  - On-board C67x DSP core and ARM Cortex-R4F processor

![AWR1843AOP EVM](images/awr1843_evm.png)
*AWR1843AOP EVM with integrated antenna array*

#### Data Capture Hardware
- **DCA1000 EVM** (Data Capture Card)
  - Real-time LVDS data capture and streaming
  - 1-Gbps Ethernet connection for data transfer
  - Supports up to 600 Mbps LVDS data rates
  - 60-pin Samtec connector for radar interface
  - FPGA-based processing for data handling

![DCA1000 EVM](images/dca1000_evm.png)
*DCA1000 EVM data capture card*

#### Host Computer Requirements
- **Operating System**: Windows 10 or higher (64-bit recommended)
- **Processor**: Intel Core i5 2.5 GHz or equivalent (minimum)
- **Memory**: 8 GB RAM (minimum), 16 GB recommended
- **Storage**: 500 GB available space for data capture
- **Network**: Gigabit Ethernet port (dedicated for radar communication)
- **USB Ports**: Multiple USB 2.0/3.0 ports for device connections

#### Connectivity Hardware
- **Ethernet Cable**: CAT5e or CAT6 (for DCA1000 to PC connection)
- **USB Cables**: 
  - USB-A to Micro-USB (for radar configuration)
  - USB-A to Micro-USB (for DCA1000 control)
- **Power Supplies**:
  - 5V DC, 2.5A minimum for AWR1843AOP EVM
  - 5V DC adapter for DCA1000 EVM (or powered via radar EVM)

![Hardware Setup](images/hardware_setup.png)
*Complete hardware setup showing all connections*

### Optional Hardware
- **Tripod or Mounting System**: For stable radar positioning
- **RF Absorbers**: For controlled testing environments
- **Calibration Targets**: For system validation and testing

## Software Requirements

### Core Dependencies

The project requires Python 3.9 or higher with the following key packages:

#### Essential Python Libraries
```
colorama==0.4.6
contourpy==1.2.0
cycler==0.12.1
fonttools==4.56.0
importlib_resources==6.5.2
kiwisolver==1.4.4
llvmlite==0.39.1
matplotlib==3.7.1
numba==0.56.4
numpy==1.23.5
opencv-python==4.8.1.78
packaging==24.2
pillow==11.1.0
psutil==5.9.5
pyFFTW==0.13.1
pyparsing==3.2.1
PyQt5==5.15.9
PyQt5-Qt5==5.15.2
PyQt5-sip==12.12.2
pyqtgraph==0.13.3
pyserial==3.5
python-dateutil==2.9.0.post0
scipy==1.11.1
six==1.17.0
tqdm==4.67.1
zipp==3.23.0
```

### Conda Environment
The `environment.yml` file is also provided for creating a conda environment.
```yml
name: real-time-radar
channels:
  - conda-forge
  - defaults
dependencies:
  - bzip2=1.0.8
  - ca-certificates=2025.2.25
  - expat=2.7.1
  - libffi=3.4.4
  - openssl=3.0.16
  - pip=25.1
  - python=3.9.23
  - setuptools=78.1.1
  - sqlite=3.50.2
  - tk=8.6.14
  - tzdata=2025b
  - ucrt=10.0.22621.0
  - vc=14.3
  - vc14_runtime=14.44.35208
  - vs2015_runtime=14.44.35208
  - wheel=0.45.1
  - xz=5.6.4
  - zlib=1.2.13
  - pip:
      - colorama==0.4.6
      - contourpy==1.2.0
      - cycler==0.12.1
      - fonttools==4.56.0
      - importlib-resources==6.5.2
      - kiwisolver==1.4.4
      - llvmlite==0.39.1
      - matplotlib==3.7.1
      - numba==0.56.4
      - numpy==1.23.5
      - opencv-python==4.8.1.78
      - packaging==24.2
      - pillow==11.1.0
      - psutil==5.9.5
      - pyfftw==0.13.1
      - pyparsing==3.2.1
      - pyqt5==5.15.9
      - pyqt5-qt5==5.15.2
      - pyqt5-sip==12.12.2
      - pyqtgraph==0.13.3
      - pyserial==3.5
      - python-dateutil==2.9.0.post0
      - scipy==1.11.1
      - six==1.17.0
      - tqdm==4.67.1
      - zipp==3.23.0
```

![Software Architecture](images/software_architecture.png)
*Software architecture diagram showing component relationships*

### Development Environment
- **IDE**: Visual Studio Code, PyCharm, or similar
- **Version Control**: Git (for development)
- **Package Manager**: pip or conda

## Installation Guide

### Step 1: Clone or Download the Repository

```bash
git clone https://github.com/aijay3/Live-Radar-Demonstrator-using-Raw-ADC-Data-in-Python.git
cd Live-Radar-Demonstrator-using-Raw-ADC-Data-in-Python
```

### Step 2: Set Up Python Environment

#### Option 1: Using Conda (Recommended)

```bash
# Create and activate the conda environment
conda env create -f environment.yml
conda activate real-time-radar
```

#### Option 2: Using pip and venv

```bash
# Create a virtual environment
python -m venv radar_env

# Activate the virtual environment
# On Windows:
radar_env\Scripts\activate
# On macOS/Linux:
source radar_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Verify Installation

```bash
# Test the installation
python -c "import numpy, PyQt5, pyqtgraph, serial; print('All dependencies installed successfully')"
```

![Installation Process](images/installation_process.png)
*Step-by-step installation verification*

## Hardware Setup

### Physical Connections

#### Step 1: Prepare the Hardware
1. Ensure all components are powered off
2. Set AWR1843AOP EVM switches to DCA1000(SDK) mode
3. Configure DCA1000 EVM switches for RAW ADC capture

![Switch Configuration](images/switch_configuration.png)
*Correct switch positions for both EVMs*

#### Step 2: Connect AWR1843AOP EVM to DCA1000 EVM
1. Mount the AWR1843AOP EVM to DCA1000 EVM using the J11 connector
2. Ensure secure connection with proper alignment

#### Step 3: Connect to Computer
1. **Power Connections**:
   - Connect 5V DC power supply to AWR1843AOP EVM
   - Connect 5V DC power supply to DCA1000 EVM (or use power from radar EVM)

2. **Data Connections**:
   - USB-A to Micro-USB: AWR1843AOP UART port to PC
   - USB-A to Micro-USB: DCA1000 FTDI port to PC
   - Ethernet cable: DCA1000 to PC Ethernet port

![Connection Diagram](images/connection_diagram.png)
*Detailed connection diagram showing all interfaces*

### Hardware Validation

After connections, verify:
- Power LEDs are illuminated on both EVMs
- Windows Device Manager shows COM ports for both devices
- Network adapter recognizes Ethernet connection

## Network Configuration

### Ethernet Adapter Setup

1. **Configure Static IP**:
   - Open Network and Sharing Center
   - Change adapter settings
   - Right-click Ethernet adapter → Properties
   - Select IPv4 → Properties
   - Set manual configuration:
     ```
     IP Address: 192.168.33.30
     Subnet Mask: 255.255.255.0
     Default Gateway: (leave empty)
     DNS: (leave empty)
     ```

![Network Configuration](images/network_config.png)
*Network adapter configuration screen*

2. **Firewall Configuration**:
   - Open Windows Defender Firewall
   - Create inbound/outbound rules for UDP ports 4096 and 4098
   - Or temporarily disable firewall for testing

3. **Verify Network Connection**:
   ```bash
   ping 192.168.33.180
   ```

## Running the Software

### Using the Launcher (Recommended)

The launcher provides a unified interface to access all radar applications:

```bash
python launcher.py
```

![Launcher Interface](images/launcher_interface.png)
*Main launcher showing application selection*

### Running Individual Applications

#### Range Profile Application
```bash
python "Range Profile/rp_main.py"
```
- **Purpose**: 1D radar processing for object distance detection
- **Output**: Signal strength vs. distance plot
- **Use Cases**: Basic presence detection, distance measurement

![Range Profile](images/range_profile.png)
*Range Profile visualization showing detected objects*

#### Range Doppler Application
```bash
python "Range Doppler/rd_main.py"
```
- **Purpose**: 2D radar processing for distance and velocity detection
- **Output**: Range-velocity heatmap
- **Use Cases**: Motion tracking, velocity estimation

![Range Doppler](images/range_doppler.png)
*Range Doppler heatmap showing moving targets*

#### Range Angle Application
```bash
python "Range Angle/ra_main.py"
```
- **Purpose**: 2D radar processing for distance and angle detection
- **Output**: Range-angle spatial map
- **Use Cases**: Spatial localization, multi-target tracking

![Range Angle](images/range_angle.png)
*Range Angle visualization showing spatial target distribution*

## Using the Applications

### Common Interface Elements

All applications share a consistent interface design:

#### 1. Configuration Panel
- **COM Port Selection**: Choose the correct serial port for radar communication
- **Configuration File**: Load TI-standard .cfg files
- **Connection Status**: Real-time connection indicators

#### 2. Processing Controls
- **Window Functions**: Blackman-Harris, Hamming, Hann, Rectangular
- **FFT Parameters**: Size, zero-padding options
- **Channel Selection**: Individual RX channels or combined processing
- **Clutter Removal**: Static background suppression

#### 3. CFAR Detection Settings
- **Guard Cells**: Number of cells around target (typical: 2-8)
- **Training Cells**: Cells for noise estimation (typical: 8-16)
- **False Alarm Rate**: Detection threshold (typical: 10^-3 to 10^-6)
- **Peak Grouping**: Combine nearby detections

#### 4. Real-time Display
- **Live Visualization**: Updates at ~10 FPS
- **Interactive Controls**: Zoom, pan, color scaling
- **Data Tables**: Numerical values for detected objects
- **Status Information**: Frame rate, processing health

![Interface Elements](images/interface_elements.png)
*Detailed view of common interface components*

### Starting Data Capture

1. **Hardware Connection**:
   - Verify all hardware connections
   - Check power indicators and COM port recognition

2. **Software Configuration**:
   - Select correct COM port from dropdown
   - Load appropriate configuration file (config/AWR1843_cfg.cfg)
   - Verify network settings

3. **Initiate Processing**:
   - Click "Send Radar Config" to configure the radar
   - Data visualization begins automatically
   - Monitor status indicators for system health

4. **Parameter Optimization**:
   - Adjust processing parameters in real-time
   - Observe immediate effects in visualization
   - Fine-tune for specific application requirements

## Signal Processing Features

### Advanced Processing Capabilities

#### Fast Fourier Transform (FFT) Processing
- **Range FFT**: Converts time-domain to frequency-domain for distance estimation
- **Doppler FFT**: Analyzes phase changes across chirps for velocity detection
- **Angle FFT**: Processes spatial antenna array for direction finding

#### Window Functions
- **Blackman-Harris**: Excellent sidelobe suppression (-92 dB)
- **Hamming**: Good balance of resolution and sidelobe performance (-43 dB)
- **Hann**: Standard choice for most applications (-31 dB)
- **Rectangular**: Maximum resolution, poor sidelobe performance (-13 dB)

![Window Functions](images/window_functions.png)
*Comparison of different window functions and their effects*

#### CFAR Detection
- **Cell Averaging (CA-CFAR)**: Standard implementation
- **Adaptive Thresholding**: Maintains constant false alarm rate
- **Configurable Parameters**: Guard cells, training cells, threshold scaling

#### Beamforming and Angle Estimation
- **Virtual Array Processing**: MIMO radar creates 3×4 virtual antenna array
- **Steering Vector Calculation**: Precise angle-of-arrival estimation
- **Spatial Filtering**: Enhanced angular resolution and interference rejection

### Performance Optimization

#### Multi-threading Architecture
- **Data Acquisition Thread**: High-speed UDP packet reception
- **Processing Thread**: Signal processing pipeline
- **Visualization Thread**: GUI updates and user interaction
- **Main Thread**: Application control and coordination

#### Memory Management
- **Buffer Reuse**: Preallocated arrays for efficient processing
- **Queue Management**: Thread-safe data transfer
- **Garbage Collection**: Automatic memory cleanup

![Processing Pipeline](images/processing_pipeline.png)
*Signal processing pipeline showing data flow*

## Troubleshooting

### Common Issues and Solutions

#### Hardware Connection Problems

**Issue**: COM ports not detected
- **Solution**: 
  - Verify USB cable connections
  - Check Device Manager for FTDI devices
  - Reinstall FTDI drivers if necessary
  - Try different USB ports

**Issue**: Network communication failure
- **Solution**:
  - Verify Ethernet cable connection
  - Check IP address configuration (192.168.33.30)
  - Disable firewall temporarily
  - Restart network adapter

![Troubleshooting Guide](images/troubleshooting.png)
*Visual troubleshooting flowchart*

#### Software Issues

**Issue**: Application crashes during startup
- **Solution**:
  - Verify all dependencies are installed
  - Check Python version compatibility (3.8+)
  - Run in virtual environment
  - Check console output for error messages

**Issue**: No data visualization
- **Solution**:
  - Verify radar configuration is sent successfully
  - Check network connectivity to DCA1000
  - Ensure correct switch positions on hardware
  - Monitor status indicators for errors

#### Performance Issues

**Issue**: Low frame rate or delayed visualization
- **Solution**:
  - Reduce FFT size or zero-padding
  - Disable unnecessary processing features
  - Close other applications to free resources
  - Check CPU and memory usage

**Issue**: High memory consumption
- **Solution**:
  - Restart application periodically
  - Reduce buffer sizes in configuration
  - Monitor memory usage with Task Manager
  - Consider system upgrade if persistent

### Hardware Debugging

#### DCA1000 EVM Status LEDs
- **Power LED**: Solid green when powered correctly
- **Ethernet LED**: Flashing during data transfer
- **FPGA LED**: Solid when FPGA is configured
- **Error LEDs**: Indicate specific hardware issues

#### AWR1843AOP EVM Indicators
- **Power LED**: Confirms proper power supply
- **Status LEDs**: Show radar operational state
- **Button Response**: Test hardware responsiveness

## Project Structure

The project follows a modular architecture with clear separation of concerns:

```
Real Time Radar/
├── launcher.py                 # Main application launcher
├── README.md                   # This documentation
├── requirements.txt            # Python dependencies
├── environment.yml            # Conda environment specification
├── .gitignore                 # Git ignore rules
│
├── config/                    # Configuration files
│   └── AWR1843_cfg.cfg       # Default radar configuration
│
├── Range Profile/             # 1D range processing
│   ├── rp_main.py            # Main application
│   ├── rp_dsp.py             # Signal processing
│   ├── rp_app_layout.py      # GUI layout
│   ├── rp_real_time_process.py # Real-time processing
│   ├── radar_config.py       # Radar configuration
│   ├── radar_config_params.py # Configuration parameters
│   └── radar_parameters.py   # Operational parameters
│
├── Range Doppler/             # 2D range-velocity processing
│   ├── rd_main.py            # Main application
│   ├── rd_dsp.py             # Signal processing
│   ├── rd_app_layout.py      # GUI layout
│   ├── rd_real_time_process.py # Real-time processing
│   ├── radar_config.py       # Radar configuration
│   ├── radar_config_params.py # Configuration parameters
│   └── radar_parameters.py   # Operational parameters
│
├── Range Angle/               # 2D range-angle processing
│   ├── ra_main.py            # Main application
│   ├── ra_dsp.py             # Signal processing
│   ├── ra_app_layout.py      # GUI layout
│   ├── ra_real_time_process.py # Real-time processing
│   ├── coordinate_transforms.py # Coordinate utilities
│   ├── radar_config.py       # Radar configuration
│   ├── radar_config_params.py # Configuration parameters
│   └── radar_parameters.py   # Operational parameters
│
└── images/                    # Documentation images
    ├── system_overview.png
    ├── hardware_setup.png
    ├── software_architecture.png
    └── ...
```

![Project Structure](images/project_structure.png)
*Visual representation of project organization*

### Module Descriptions

#### Core Components
- **launcher.py**: Unified entry point with GUI for application selection
- **radar_config.py**: Serial communication interface for radar configuration
- **radar_parameters.py**: Calculation and management of radar operational parameters

#### Signal Processing Modules
- **rp_dsp.py, rd_dsp.py, ra_dsp.py**: Specialized DSP implementations for each mode
- **coordinate_transforms.py**: Spatial coordinate conversion utilities

#### User Interface
- **app_layout.py**: PyQt5-based GUI layouts for each application
- **real_time_process.py**: Threading and real-time data handling

## Performance Specifications

### System Performance Metrics

#### Processing Performance
- **Visualization Update Rate**: ~10 FPS (typical)
- **Processing Latency**: <100ms from data acquisition to display
- **Memory Usage**: 200-500 MB (depending on configuration)
- **CPU Usage**: 20-40% on Intel Core i5 (typical)

#### Radar Performance
- **Maximum Detection Range**: Up to 10 meters (configuration dependent)
- **Range Resolution**: ~4 cm (with 1.8 GHz bandwidth)
- **Velocity Resolution**: ~0.13 m/s (typical configuration)
- **Angular Resolution**: ~30° (3TX × 4RX configuration)
- **Angular Field of View**: 140° azimuth × 140° elevation

![Performance Metrics](images/performance_metrics.png)
*Performance benchmarks across different system configurations*

#### Data Throughput
- **ADC Sampling Rate**: Up to 5 MSPS per channel
- **Data Rate**: ~600 Mbps LVDS (maximum)
- **Ethernet Throughput**: Up to 1 Gbps
- **Frame Rate**: Configurable (1-100 Hz typical)

### Optimization Guidelines

#### For Maximum Range Resolution
- Increase chirp bandwidth
- Use appropriate window functions
- Optimize FFT zero-padding

#### For Maximum Velocity Resolution
- Increase number of chirps per frame
- Reduce chirp repetition interval
- Use coherent processing

#### For Maximum Angular Resolution
- Utilize all available TX/RX channels
- Implement advanced beamforming
- Calibrate antenna array response

## Future Enhancements

### Planned Features

#### Machine Learning Integration
- **Object Classification**: Distinguish between different target types
- **Activity Recognition**: Human gesture and movement analysis
- **Anomaly Detection**: Identify unusual patterns or behaviors
- **Adaptive Processing**: ML-optimized parameter selection

![ML Integration](images/ml_integration.png)
*Conceptual diagram of machine learning integration*

#### Advanced Signal Processing
- **3D Visualization**: Elevation-azimuth mapping
- **Micro-Doppler Analysis**: Fine-grained motion signatures
- **Multi-target Tracking**: Kalman filtering and track association
- **Interference Mitigation**: Advanced filtering techniques

#### System Enhancements
- **Multi-radar Fusion**: Synchronized operation of multiple units
- **Cloud Integration**: Remote processing and data storage
- **Mobile Support**: Android/iOS companion applications
- **Web Interface**: Browser-based control and monitoring

#### Performance Improvements
- **GPU Acceleration**: CUDA-based signal processing
- **Real-time Optimization**: Sub-millisecond processing latency
- **Memory Efficiency**: Reduced memory footprint
- **Parallel Processing**: Multi-core optimization

### Research Directions

#### Academic Applications
- **Educational Platform**: Teaching radar principles and signal processing
- **Research Tool**: Platform for algorithm development and testing
- **Benchmarking**: Standardized performance evaluation

#### Industrial Applications
- **Automotive Radar**: ADAS and autonomous vehicle development
- **Industrial Automation**: Robotics and process monitoring
- **Security Systems**: Perimeter monitoring and intrusion detection
- **Healthcare**: Vital sign monitoring and patient tracking

![Future Applications](images/future_applications.png)
*Potential application domains for the radar system*

## Contact

For questions, support, or collaboration opportunities:

- **Primary Developer**: Jayanth Balaji Uppu
- **Institution**: Ostbayerische Technische Hochschule (OTH) Regensburg
- **Department**: Sensorik-Applikations Zentrum (SappZ)
- **Supervisor**: Prof. Dr.-Ing. Matthias Ehrnsperger

- **GitHub Repository**: [Live-Radar-Demonstrator-using-Raw-ADC-Data-in-Python](https://github.com/aijay3/Live-Radar-Demonstrator-using-Raw-ADC-Data-in-Python)
- **Issues and Bug Reports**: Please use GitHub Issues for technical problems
- **Feature Requests**: Submit via GitHub Issues with enhancement label

## Acknowledgement

This work was conducted at the Sensorik-ApplikationsZentrum (SappZ), Faculty of Applied Natural and Cultural Sciences, Ostbayerische Technische Hochschule (OTH) Regensburg.

Special thanks to:
- **Prof. Dr.-Ing. Matthias Ehrnsperger** for supervision and guidance
- **SappZ Team** for providing research infrastructure and technical support
- **Texas Instruments** for hardware support and documentation
- **Open Source Community** for the excellent Python libraries that made this project possible

![Acknowledgements](images/acknowledgements.png)
*Institutional logos and collaborators*

## Citation

If you use this software in your research or projects, please cite:

```bibtex
@mastersthesis{uppu2025radar,
  title={Design and Construction of a W-Band FMCW Radar Demonstrator for Object Detection: Real-time data capture and processing system in Python for Texas Instruments mmWave radar sensors (AWR1843AOP EVM) integrated with DCA1000 EVM},
  author={Uppu, Jayanth Balaji},
  year={2025},
  school={Ostbayerische Technische Hochschule (OTH) Regensburg},
  type={Master's Thesis},
  address={Regensburg, Germany}
}
```

### Related Publications

For theoretical background and detailed technical analysis, refer to:
- Master's Thesis: "Design and Construction of a W-Band FMCW Radar Demonstrator for Object Detection" (April 2025)
- Technical documentation available in the repository's `docs/` folder

---

**License**: This project is open source. Please check the LICENSE file for specific terms and conditions.

**Version**: 1.0.0 (April 2025)

**Last Updated**: April 2025

![Footer](images/footer.png)
*Project logo and version information*
