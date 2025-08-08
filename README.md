# Real-Time Raw ADC Data Processing and Visualization Tool for TI mmWave Radar

## Overview

This project provides a real-time radar data capture and processing pipeline for TI mmWave radar sensors using the DCA1000EVM and a Python-based software stack — without relying on TI's mmWave Studio. It streams and processes raw ADC data directly from compatible radar devices and presents the output in intuitive live visualizations.

Traditional workflows with mmWave Studio are limited to file-based offline processing and require multiple dependencies, specific software versions, and complex configuration steps. This tool bypasses those limitations by enabling live UDP data capture, multi-threaded signal processing, and interactive GUI-based visualization.

## 🧠 Key Features

- **Real-time radar visualization modes:**
  - 📈 **Range Profile** – object distance via FFT
  - 🔁 **Range Doppler** – distance and velocity mapping
  - 🧭 **Range Angle** – angular position estimation using beamforming
- **Direct raw ADC capture via DCA1000 (no mmWave Studio)**
- **CFAR detection, clutter removal, and windowed FFT processing**
- **Interactive GUI with:**
  - Parameter tuning (FFT size, padding, channel selection)
  - Live plot updates
- **Modular Python implementation with PyQt5 and PyQtGraph**

<p align="center">
<img width="439" height="481" alt="image" src="https://github.com/user-attachments/assets/2c6f995e-a21a-4bd3-afcd-7a9498e2cb59" />
<img width="945" height="389" alt="image" src="https://github.com/user-attachments/assets/923b5ef4-98da-49f8-ac35-691a834037b4" />
<img width="945" height="327" alt="image" src="https://github.com/user-attachments/assets/e9825a6d-4cbd-4cdb-b713-87bcc49202f1" />
<img width="897" height="298" alt="image" src="https://github.com/user-attachments/assets/1959800e-8df3-4cbf-89b2-d3964cfb17b8" />
</p>

## 🛠 Expected Hardware Compatibility

This tool is designed and tested for the **AWR1843AOP mmWave radar sensor** and **DCA1000EVM**, but it is expected to work with other TI AWR and IWR series sensors supported by the DCA1000 interface.

**Examples include:**
- AWR1243, AWR1443, AWR1642, AWR2243, AWR2944, AWR6843, AWR6843AOP
- IWR1443, IWR1642, IWR1843, IWR2944, IWR6843, IWR6843AOP

⚠️ **While not all devices are explicitly tested, compatibility is expected based on shared interface and data capture protocols. Ensure your radar is supported by the DCA1000EVM and outputs raw ADC via LVDS.**

## 🚧 Motivation

Setting up real-time raw ADC capture with mmWave Studio often involves:
- Complex DIP switch configurations and unclear hardware instructions
- Platform-specific dependencies (Windows-only, MATLAB runtime)
- Fragmented documentation across toolboxes, user guides, and forums
- No support for live processing — only offline .bin analysis

This project addresses those limitations by providing:
- A streamlined setup (Python + Ethernet + config file)
- A user-friendly interface with parameter control and diagnostics
- A fully extensible signal processing pipeline for education, prototyping, and research

---

## Hardware Guide

### Required Hardware

#### 📡 1. TI Radar Sensor Module
**Texas Instruments AWR1843AOP EVM**
- 76–81 GHz mmWave radar sensor with integrated antenna
- 3 Transmitters, 4 Receivers (3TX/4RX AoP configuration)
- Onboard C67x DSP and Cortex-R4F MCU
- Field of View: ~140° (Azimuth & Elevation)

<p align="center">
<img width="509" height="286" alt="image" src="https://github.com/user-attachments/assets/6128ab2d-add0-4f37-8c2a-a6b6c864e652" />
</p>


💡 **Other TI AWR/IWR series radar EVMs may also work if supported by DCA1000 and output raw ADC data over LVDS.**

#### 🔌 2. Data Capture Module
**DCA1000EVM**
- Streams raw ADC data from radar over Ethernet
- 1 Gbps Ethernet support
- Connects to radar via 60-pin Samtec connector

<p align="center">
<img width="582" height="328" alt="image" src="https://github.com/user-attachments/assets/8160ffb5-c97c-4045-b4da-8697094ceb0f" />
</p>

#### 🖥️ 3. Host Computer
**Recommended system:**
- **Operating System:** Windows 10 or 11 (64-bit)
- **CPU:** Intel Core i5 or equivalent (2.5 GHz+)
- **Memory:** 8 GB RAM minimum (16 GB recommended)
- **Storage:** At least 500 GB free (for ADC capture files)
- **Ports:**
  - 2x USB (Micro USB for radar + DCA1000 control)
  - 1x Gigabit Ethernet (dedicated for radar streaming)

#### 🧩 4. Connectivity and Accessories
- **Ethernet Cable:** CAT5e or CAT6
- **USB Cables:**
  - USB-A to Micro-USB (for radar UART)
  - USB-A to Micro-USB (for DCA1000 FTDI control)
- **Power Supplies:**
  - 5V DC, 2.5–3A for AWR1843AOP EVM
  - Optional: separate 5V DC for DCA1000 (or use EVM passthrough)
- **Optional:**
  - Tripod or mounting system
  - RF absorbers for test lab
  - Corner reflectors or calibration targets

### Physical Hardware Setup

#### Prepare the Hardware:

**AWR1843AOP EVM**
- Ensure all components are powered off before starting
- Mount the radar module on a stable surface or tripod
- Make sure the switches on AWR1843AOP are in position of DCA1000(SDK) mode

<p align="center">
<img width="685" height="551" alt="image" src="https://github.com/user-attachments/assets/c7c07d45-cdb3-4b33-be86-e42f21239e30" />
</p>

<p align="center">  
<img width="509" height="161" alt="image" src="https://github.com/user-attachments/assets/39890aba-0778-48ba-9d0d-55f015805d13" />
</p>


**DCA1000**
- Make sure the switches on DCA1000 are in position for RAW ADC Capture from AWR1843AOP EVM

<p align="center">
<img width="509" height="257" alt="image" src="https://github.com/user-attachments/assets/356cc4d8-a46f-4037-87d2-fe3e796c5179" />
</p>

**Connecting Setup - AWR1843AOP EVM to DCA1000 EVM:**
- Mount the AWR1843AOP EVM to DCA1000 EVM using the J11 connector

<p align="center">
<img width="648" height="384" alt="image" src="https://github.com/user-attachments/assets/106c1ab5-11e1-4fe9-a750-fbfaf0ed2a59" />
</p>

**Connecting AWR1843AOP EVM + DCA1000 EVM to Computer:**

1. Connect the 12V DC power supply to the DCA1000's and AWR1843AOP's power input
2. Connect the 5V power supply for the AWR1843AOP module
3. Use a USB-A to Micro-USB cable to connect the UART USB port on the AWR1843AOP EVM to the USB port on your computer
4. Use a USB-A to Micro-USB cable to connect the FTDI USB port on the DCA1000 EVM to the USB port on your computer

<p align="center">
<img width="729" height="362" alt="image" src="https://github.com/user-attachments/assets/fe93839d-08d3-456d-85d0-3450234b9e9f" />
</p>

### Drivers Installation

#### Installation of FTDI and XDS driver

#### FTDI Drivers for DCA1000

FTDI (Future Technology Devices International) drivers are software components that allow an operating system to communicate with FTDI chips, which are commonly used for USB-to-serial (UART) and other USB bridge connections.

**Types of FTDI Drivers:**
- **VCP (Virtual COM Port):** Emulates a standard serial port on the PC, allowing legacy software to communicate with devices as if they are connected by traditional RS232/COM ports
- **D2XX:** Provides direct access to the USB device without creating a COM port. Used by custom applications needing more control or performance
- **D3XX:** Used for higher-speed USB3.0 chips (not common with DCA1000)

• USB-to-Serial Interface: The DCA1000EVM integrates an FTDI chip (such as the FT4232) to implement its USB-to-serial (UART) interface. This lets the host computer connect to the DCA1000EVM using a standard USB cable.

• Driver Requirement: For the PC to recognize and communicate with the DCA1000EVM's UART interface, the correct FTDI driver (most commonly the VCP driver) must be installed. When installed, the DCA1000 will appear as a COM port in the Device Manager.

Open Windows Device Manager and expand the "Ports (COM & LPT)" list for confirming hardware detection. When the DCA1000 + AWR1843AOPEVM is connected to the PC and switched ON for the first time, Windows may not be able to recognize the device and it may show up under "Other devices" in Device Manager. The FTDI device ports of the DCA1000 board will appear with a yellow label when the driver is not installed.

<p align="center">
<img width="468" height="236" alt="image" src="https://github.com/user-attachments/assets/2a3deb0c-ebcd-4640-8a1c-95d91fc5957b" />
</p>

**Installation Steps:**

1. **Download the latest FTDI CDM driver** from the [FTDI website](https://ftdichip.com/drivers/d2xx-drivers/) and unzip it into a location on your PC

2. **Run the FTDI driver exe** with administrator privileges and follow the installation wizard

3. **Verify Installation:** Open Windows Device Manager and expand the "Ports (COM & LPT)" list. When properly installed, you should see 4 new COM ports without yellow warning labels.

**Troubleshooting FTDI Installation:**
- If devices still appear with yellow labels, try downloading and installing mmWave Studio from [TI.com](https://www.ti.com/tool/MMWAVE-STUDIO). The required FTDI drivers will be installed automatically at the end of mmWave Studio installation.

<p align="center">
<img width="785" height="454" alt="image" src="https://github.com/user-attachments/assets/a63f16af-e8c6-45d0-b29e-6e2c08fd7fa1" />
</p>

- If the automatic FTDI installation fails, in Windows Device Manager right-click on these devices and update the drivers by pointing to the location of the FTDI driver.

<p align="center">
<img width="696" height="480" alt="image" src="https://github.com/user-attachments/assets/ce981dba-9b72-4554-80ca-c1b3edec6680" />
<p align="center">

- This must be done for all four COM ports. If after updating the FTDI driver, Device Manager still does not show 4 new COM Ports, update the FTDI driver once again.

<p align="center">
<img width="289" height="225" alt="image" src="https://github.com/user-attachments/assets/75e618ef-6b1a-47aa-a418-c3f6a6a7d1e6" />
<p align="center">
  
-    When all four COM ports are installed, Device Manager recognizes these devices and indicates the COM port numbers.

<p align="center">
<img width="688" height="551" alt="image" src="https://github.com/user-attachments/assets/bab132d4-ddc6-4f10-899a-e1600817ec9a" />
<p align="center">

#### Silicon Labs USB-to-UART Driver for AWR1843AOP

The CP210x series from Silicon Labs consists of USB-to-UART bridge chips that play a crucial role in connecting TI mmWave radar sensors to host computers.

Here's how they are related:

- TI mmWave radar sensors use UART (Universal Asynchronous Receiver-Transmitter) interfaces for configuration and data transfer. Many TI radar evaluation modules have built-in USB-to-UART bridges so they can easily connect to a PC using a USB cable.
- CP210x chips are the USB-to-UART bridge used on many TI mmWave radar sensor boards (such as IWR and AWR series EVMs). The CP210x device converts USB from the host PC into the UART protocol used by the radar hardware. This enables:
  - Flashing firmware to the radar sensor.
  - Sending configuration commands.
  - Receiving radar data as a serial stream.
- Drive Requirements: For the serial (COM) ports of TI mmWave radar sensors to function correctly on a computer, the Silicon Labs CP210x drivers must be installed. When properly installed, the system will detect one or more serial ports (COM ports) corresponding to the radar sensor’s UART channels—these are recognized as "Silicon Labs CP210x USB to UART Bridge" ports in the device manager.

**Installation Steps:**

1. **Download the CP210x Universal Windows Drivers** from [Silicon Labs](https://www.silabs.com/software-and-tools/usb-to-uart-bridge-vcp-drivers?tab=downloads)

2. **Right-click the folder** and unzip the installation files

3. **Right-click on the silabser.inf file** and select Install

4. **Follow the installation wizard**, click the "Next" button, and agree with the terms of use to complete the installation process and the CP210x USB drivers will be installed successfully.

<p align="center">
<img width="657" height="509" alt="image" src="https://github.com/user-attachments/assets/322109d4-79e6-4f12-865d-d7f3935e99cb" />
<p align="center">

**Verification:**  In the Windows Device Manager, the COM ports should appear as this when both the FTDI drivers and Silicon Labs USB-to-UART Drivers are installed.

<p align="center">
<img width="739" height="200" alt="image" src="https://github.com/user-attachments/assets/fa8f259c-ac2e-432d-b65f-3c798ed1c2fd" />
<p align="center">

### Network Configuration

#### 1. Configure the Network Adapter:
- Open Windows Control Panel > Network and Internet > Network Connections
- Identify and right-click the Ethernet adapter connected to the DCA1000 and select Properties
- Select Internet Protocol Version 4 (TCP/IPv4), then click Properties
- Set the following IP configuration:
  - **IP Address:** 192.168.33.30
  - **Subnet Mask:** 255.255.255.0
  - **Default Gateway:** (leave empty)
- Click OK to save the settings

<p align="center">
<img width="651" height="517" alt="image" src="https://github.com/user-attachments/assets/3ba200eb-9283-44ec-973d-1130cdfd7c78" />
<p align="center">

#### 2. Configure Windows Firewall:
- Open Windows Defender Firewall with Advanced Security
- Create inbound and outbound rules for UDP ports 4096 and 4098 or disable the firewall

### Hardware Validation

After connections, verify:
- Power LEDs are illuminated on both EVMs
- Windows Device Manager shows COM ports for both devices
- Network adapter recognizes Ethernet connection

---

## Software Guide

### Core Dependencies

The project requires **Python 3.8 or higher** with the following key packages:

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

#### Conda Environment

The `environment.yml` file is also provided for creating a conda environment:

```yaml
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

#### Development Environment
- **IDE:** Visual Studio Code, PyCharm, or similar
- **Version Control:** Git (for development)
- **Package Manager:** pip or conda

### Installation Guide

#### Step 1: Clone or Download the Repository
```bash
git clone https://github.com/aijay3/Live-Radar-Demonstrator-using-Raw-ADC-Data-in-Python.git
cd Live-Radar-Demonstrator-using-Raw-ADC-Data-in-Python
```

#### Step 2: Set Up Python Environment

**Option 1: Using Conda (Recommended)**
```bash
# Create and activate the conda environment
conda env create -f environment.yml
conda activate real-time-radar
```

**Option 2: Using pip and venv**
```bash
# Create a virtual environment
python -m venv radar_env

# Activate the virtual environment
radar_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### Step 3: Verify Installation
```bash
# Test the installation
python -c "import numpy, PyQt5, pyqtgraph, serial; print('All dependencies installed successfully')"
```

#### Step 4: Open the project in your code editor

### Running the Software

#### Using the Launcher (Recommended)
The launcher provides a unified interface to access all radar applications:
```bash
python launcher.py
```

#### Running Individual Applications

##### Range Profile Application
```bash
python "Range Profile/rp_main.py"
```

##### Range-Doppler Application
```bash
python "Range Doppler/rd_main.py"
```

##### Range-Angle Application
```bash
python "Range Angle/ra_main.py"
```

### Starting Data Capture

#### 1. Hardware Connection:
- Verify all hardware connections
- Check power indicators and COM port recognition

#### 2. Software Configuration:
- Select correct COM port from dropdown
- Load appropriate configuration file (`config/AWR1843_cfg.cfg`)

#### Steps to Obtain a Config File

1. **Connect the mmWave Device and Launch the Visualizer**
   - Connect your TI mmWave sensor to the PC using a USB cable
   - Open the mmWave Demo Visualizer application in your browser or as a standalone app
   - Ensure the correct serial (COM) ports are selected for communication with the device

2. **Select the Appropriate Platform and SDK Version**
   - In the CONFIGURE tab, choose the correct platform (e.g., xwr68xx, xwr16xx) that matches your hardware
   - Select the SDK version that matches the firmware running on your mmWave device

3. **Set Up Configuration Parameters**
   - Use the "Setup Details" and "Scene Selection" sections to specify radar parameters such as range, resolution, and detection settings
   - Adjust other options as needed, including chirp profile, sampling rate, and frame rate

4. **Generate and Save the Config File**
   - Once all settings are finalized, click the SAVE CONFIG TO PC button (or similarly named option) in the CONFIGURE tab
   - The Visualizer generates a .cfg file containing the sequence of CLI commands that configure the radar sensor
   - Save this file to your desired location on your computer

5. **Use or Edit the Config File**
   - The saved .cfg file can be used to reconfigure the device later by loading it in the Visualizer or sending it via command line tools
   - You can also edit the file manually if needed, as it is a text file with documented commands

- Verify network settings

#### 3. Initiate Processing:
- Click "Send Radar Config" to configure the radar
- Data visualization begins automatically

#### 4. Parameter Optimization:
- Adjust processing parameters in real-time
- Observe immediate effects in visualization
- Fine-tune for specific application requirements

### Using the Applications

#### Common Interface Elements
All applications share a consistent interface design with real-time display and interactive controls including zoom, pan, and data tables.

#### 1. Configuration Panel
- **COM Port Selection:** Choose the correct serial port for radar communication
- **Configuration File:** Load TI-standard .cfg files
- **Connection Status:** Real-time connection indicators

#### 2. Processing Controls
- **Window Functions:** Blackman-Harris, Hamming, Hann, Rectangular
- **FFT Parameters:** Size, zero-padding options
- **Channel Selection:** Individual RX channels or combined processing
- **Clutter Removal:** Static background suppression

#### 3. CFAR Detection Settings
- **Guard Cells:** Number of cells around target (typical: 2-8)
- **Training Cells:** Cells for noise estimation (typical: 8-16)
- **False Alarm Rate:** Detection threshold (typical: 10^-3 to 10^-6)
- **Peak Grouping:** Combine nearby detections

---

## System Architecture

The system is built around a set of core components that are shared across all three processing modes. These components handle the initial setup, data acquisition, and basic configuration of the radar system.

### 1. Launcher (`launcher.py`)
The `launcher.py` script serves as the main entry point for the entire application. It provides a graphical user interface (GUI) built with PyQt5 that allows the user to select and launch one of the three radar processing applications. This centralized launcher simplifies the user experience and provides a consistent starting point for all operations.

### 2. Radar Configuration
The radar configuration is handled by a set of files and modules that work together to initialize the radar hardware with the desired parameters.

- **Configuration File (`config/AWR1843_cfg.cfg`):** This is a plain text file that contains a series of commands for configuring the AWR1843AOP EVM. These commands set up the radar's chirp parameters, frame structure, ADC settings, and data output format. The file is parsed by the `radar_parameters.py` module to extract the configuration values.

- **`radar_config.py`:** This module contains the `SerialConfig` class, which is responsible for communicating with the radar EVM over a serial (COM) port. It reads the commands from the .cfg file and sends them to the radar one by one. It also provides methods for starting and stopping the radar sensor.

- **`radar_parameters.py`:** This module defines the `RadarParameters` class, which parses the .cfg file and calculates a set of derived parameters that are essential for the signal processing pipeline. These parameters include:
  - Range resolution
  - Maximum range
  - Velocity resolution
  - Maximum velocity
  - Dimensions of the radar data cube

### 3. Data Acquisition
The data acquisition process is managed by a dedicated thread that receives raw ADC data from the radar and makes it available to the processing pipeline.

- **UdpListener** (in `rp_real_time_process.py`, `rd_real_time_process.py`, `ra_real_time_process.py`): This class, which runs in its own thread, is responsible for listening for UDP packets from the DCA1000 EVM on a specific network port. It receives the raw ADC data, removes the packet headers, and places the binary data into a shared queue for the DataProcessor to consume.

- **Binary Data Queue:** A thread-safe queue that is used to transfer the raw ADC data from the UdpListener to the DataProcessor. This decouples the data acquisition and data processing tasks, allowing them to run in parallel and preventing data loss.

### Processing Pipelines

The system features three distinct processing pipelines, one for each of the supported radar modes. Each pipeline is implemented in its own set of scripts and is responsible for processing the raw ADC data and generating the corresponding visualization.

#### 1. Range Profile
The Range Profile mode provides a 1D visualization of the signal power as a function of distance from the radar.

- **`rp_main.py`:** The main script for the Range Profile application. It initializes the GUI, starts the UdpListener and DataProcessor threads, and handles user interactions.

- **`rp_real_time_process.py`:** Contains the DataProcessor class for this mode. It retrieves raw data from the binary queue, reshapes it, and passes it to the `rp_dsp` module for processing.

- **`rp_dsp.py`:** This module contains the core signal processing functions for the Range Profile mode:
  - **1D FFT:** A Fast Fourier Transform is applied to the ADC data to transform it from the time domain to the frequency domain, which corresponds to the range domain
  - **Pulse Compression:** A matched filter is applied to the data to improve the signal-to-noise ratio (SNR) through pulse compression
  - **CFAR Detection:** A Constant False Alarm Rate (CFAR) algorithm is used to detect peaks in the range profile, which correspond to detected objects
  - **Clutter Removal:** A static clutter removal algorithm based on Principal Component Analysis (PCA) can be applied to remove stationary background objects

- **`rp_app_layout.py`:** Defines the PyQt5 GUI for the Range Profile application. It includes a 1D plot for the range profile, a table to display the range and magnitude of detected objects, and controls for configuring the processing parameters.

#### 2. Range-Doppler
The Range-Doppler mode provides a 2D visualization of the signal power as a function of both range and velocity.

- **`rd_main.py`:** The main script for the Range-Doppler application.

- **`rd_real_time_process.py`:** Contains the DataProcessor for this mode, which calls the `rd_dsp` module for processing.

- **`rd_dsp.py`:** The signal processing module for the Range-Doppler mode:
  - **2D FFT:** A 2D FFT is performed on the radar data cube to generate the Range-Doppler map. The first FFT is along the range dimension, and the second is along the Doppler (chirp) dimension
  - **2D CFAR Detection:** A 2D version of the CFAR algorithm is used to detect objects in the Range-Doppler map
  - **MTI Filtering:** A Moving Target Indicator (MTI) filter can be applied to suppress stationary targets and enhance the detection of moving objects

- **`rd_app_layout.py`:** The GUI for the Range-Doppler application. It features a 2D heatmap for the Range-Doppler map and a data table that displays the range, speed, and direction of detected objects.

#### 3. Range-Angle
The Range-Angle mode provides a 2D visualization of the signal power as a function of range and angle, allowing for the spatial localization of objects.

- **`ra_main.py`:** The main script for the Range-Angle application.

- **`ra_real_time_process.py`:** Contains the DataProcessor for this mode, which calls the `ra_dsp` module for processing.

- **`ra_dsp.py`:** The signal processing module for the Range-Angle mode:
  - **3D FFT:** A 3D FFT is performed on the radar data cube. The first two dimensions compute the Range-Doppler map, and the third dimension (across the virtual antenna array) is used to estimate the angle of arrival
  - **Beamforming:** A beamforming algorithm with steering vectors is used to generate the Range-Angle map for both azimuth and elevation
  - **2D CFAR Detection:** The CFAR algorithm is applied to the Range-Angle map to detect objects

- **`ra_app_layout.py`:** The GUI for the Range-Angle application. It includes a 2D heatmap for the Range-Angle map and a data table that displays the range and angle of detected objects.

- **`coordinate_transforms.py`:** A utility module that provides functions for converting polar coordinates (range and angle) to Cartesian coordinates (x and z) for visualization.

---

## Troubleshooting

### Common Issues and Solutions

#### Hardware Connection Problems

**Issue: COM ports not detected**
- **Solution:**
  - Verify USB cable connections
  - Check Device Manager for FTDI devices
  - Reinstall FTDI drivers if necessary
  - Try different USB ports

**Issue: Network communication failure**
- **Solution:**
  - Verify Ethernet cable connection
  - Check IP address configuration (192.168.33.30)
  - Disable firewall temporarily
  - Restart network adapter

#### Software Issues

**Issue: Application crashes during startup**
- **Solution:**
  - Verify all dependencies are installed
  - Check Python version compatibility (3.8+)
  - Run in virtual environment
  - Check console output for error messages

**Issue: No data visualization**
- **Solution:**
  - Verify radar configuration is sent successfully
  - Check network connectivity to DCA1000
  - Ensure correct switch positions on hardware
  - Monitor status indicators for errors

#### Performance Issues

**Issue: Low frame rate or delayed visualization**
- **Solution:**
  - Reduce FFT size or zero-padding
  - Disable unnecessary processing features
  - Close other applications to free resources
  - Check CPU and memory usage

**Issue: High memory consumption**
- **Solution:**
  - Restart application periodically
  - Reduce buffer sizes in configuration
  - Monitor memory usage with Task Manager
  - Consider system upgrade if persistent

### Visual Troubleshooting Flowchart

For any other problems, go through the documents folder which contains the user guides and other documents related to troubleshooting.

---


