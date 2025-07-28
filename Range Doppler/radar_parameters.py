import math
import re
import logging

class RadarParameters:
    # Physical constants
    SPEED_OF_LIGHT = 3e8  # Speed of light in m/s

    def __init__(self, config_file_path):
        self.config_file_path = config_file_path
        self.config_params = self._parse_config_file()
        
    def _parse_config_file(self):
        """Parse the mmWave config file to extract parameters."""
        config_params = {}
        with open(self.config_file_path, 'r') as config_file:
            config_lines = config_file.readlines()
            
        for config_line in config_lines:
            config_line = config_line.strip()
            if config_line.startswith('%') or not config_line:  # Skip comments and empty lines
                continue
                
            if config_line.startswith('dfeDataOutputMode'):
                # dfeDataOutputMode <modeType>
                param_values = config_line.split()
                config_params['dfe_mode'] = int(param_values[1])  # 1=frame based, 2=continuous, 3=advanced
                
            elif config_line.startswith('channelCfg'):
                # channelCfg <rxChannelEn> <txChannelEn> <cascading>
                param_values = config_line.split()
                config_params['rx_mask'] = int(param_values[1])  # e.g., 15 for 4 RX antennas (0x1111b)
                config_params['tx_mask'] = int(param_values[2])  # TX antenna mask
                
            elif config_line.startswith('adcCfg'):
                # adcCfg <numADCBits> <adcOutputFmt>
                param_values = config_line.split()
                adc_bits_map = {0: 12, 1: 14, 2: 16}
                config_params['adc_bits'] = adc_bits_map.get(int(param_values[1]), 12)
                config_params['adc_format'] = int(param_values[2])  # 0=real, 1=complex 1x, 2=complex 2x
                
            elif config_line.startswith('adcbufCfg'):
                # adcbufCfg <subFrameIdx> <adcOutputFmt> <SampleSwap> <ChanInterleave> <ChirpThreshold>
                param_values = config_line.split()
                config_params['adc_buf_fmt'] = int(param_values[2])  # 0=complex, 1=real
                config_params['sample_swap'] = int(param_values[3])  # 1=Q in LSB, I in MSB
                config_params['chan_interleave'] = int(param_values[4])  # 1=non-interleaved
                
            elif config_line.startswith('profileCfg'):
                # profileCfg <profileId> <startFreq> <idleTime> <adcStartTime> <rampEndTime> 
                # <txOutPower> <txPhaseShifter> <freqSlopeConst> <txStartTime> <numAdcSamples> 
                # <digOutSampleRate> <hpfCornerFreq1> <hpfCornerFreq2> <rxGain>
                param_values = config_line.split()
                config_params['profile_id'] = int(param_values[1])
                config_params['start_freq'] = float(param_values[2])  # GHz
                config_params['idle_time'] = float(param_values[3])  # μs
                config_params['adc_start_time'] = float(param_values[4])  # μs
                config_params['ramp_end_time'] = float(param_values[5])  # μs
                config_params['tx_power'] = int(param_values[6])
                config_params['tx_phase'] = int(param_values[7])
                # Ensure frequency slope is properly parsed as float
                freq_slope = float(param_values[8])
                if freq_slope == 0:
                    logging.error("Invalid frequency slope value: 0")
                    raise ValueError("Frequency slope cannot be 0")
                config_params['freq_slope'] = freq_slope  # MHz/μs
                logging.info(f"Parsed frequency slope: {freq_slope} MHz/μs")
                config_params['tx_start_time'] = float(param_values[9])  # μs
                config_params['adc_samples'] = int(param_values[10])
                config_params['sample_rate'] = float(param_values[11])  # ksps
                config_params['hpf1_corner_freq'] = int(param_values[12])
                config_params['hpf2_corner_freq'] = int(param_values[13])
                config_params['rx_gain'] = int(param_values[14])
                
            elif config_line.startswith('chirpCfg'):
                # chirpCfg <startIdx> <endIdx> <profileId> <startFreqVar> <freqSlopeVar> 
                # <idleTimeVar> <adcStartTimeVar> <txEnable>
                param_values = config_line.split()
                if 'chirp_configs' not in config_params:
                    config_params['chirp_configs'] = []
                config_params['chirp_configs'].append({
                    'start_idx': int(param_values[1]),
                    'end_idx': int(param_values[2]),
                    'profile_id': int(param_values[3]),
                    'start_freq_var': float(param_values[4]),  # Hz
                    'freq_slope_var': float(param_values[5]),  # kHz/μs
                    'idle_time_var': float(param_values[6]),  # μs
                    'adc_start_time_var': float(param_values[7]),  # μs
                    'tx_enable': int(param_values[8])  # TX antenna enable mask
                })
                
            elif config_line.startswith('frameCfg'):
                # frameCfg <StartIdx> <chirpEndIdx> <numLoops> <numFrames> 
                # <framePeriodicity> <triggerSelect> <frameTriggerDelay>
                param_values = config_line.split()
                config_params['chirp_start_idx'] = int(param_values[1])
                config_params['chirp_end_idx'] = int(param_values[2])
                config_params['num_loops'] = int(param_values[3])
                config_params['num_frames'] = int(param_values[4])
                config_params['frame_periodicity'] = float(param_values[5])  # ms
                config_params['trigger_select'] = int(param_values[6])
                if len(param_values) > 7:
                    config_params['frame_trigger_delay'] = float(param_values[7])  # ms
                
            elif config_line.startswith('lowPower'):
                # lowPower <don't_care> <adcMode>
                param_values = config_line.split()
                config_params['adc_mode'] = int(param_values[2])  # 0=regular ADC mode
                
            elif config_line.startswith('guiMonitor'):
                # guiMonitor parameters for data export configuration
                param_values = config_line.split()
                config_params['detected_objects'] = int(param_values[2])
                config_params['log_mag_range'] = int(param_values[3])
                config_params['noise_profile'] = int(param_values[4])
                config_params['range_azimuth_map'] = int(param_values[5])
                config_params['range_doppler_map'] = int(param_values[6])
                config_params['stats_info'] = int(param_values[7])
                
        # Calculate derived parameters
        config_params['chirps_per_frame'] = config_params['chirp_end_idx'] - config_params['chirp_start_idx'] + 1
        
        # Log key parameters
        logging.info("Radar Configuration Parameters:")
        logging.info(f"  Start Frequency: {config_params['start_freq']:.4f} GHz")
        logging.info(f"  Frequency Slope: {config_params['freq_slope']:.4f} MHz/μs")
        logging.info(f"  Ramp End Time: {config_params['ramp_end_time']:.4f} μs")
        logging.info(f"  Sample Rate: {config_params['sample_rate']:.4f} ksps")
        logging.info(f"  ADC Samples: {config_params['adc_samples']}")
        logging.info(f"  Number of Loops: {config_params['num_loops']}")
        
        return config_params

    @property
    def num_tx_channels(self):
        """Number of TX Channels = count of enabled TX channels"""
        return bin(self.config_params['tx_mask']).count('1') if 'tx_mask' in self.config_params else 0

    @property
    def num_rx_channels(self):
        """Number of RX Channels = count of enabled RX channels"""
        return bin(self.config_params['rx_mask']).count('1') if 'rx_mask' in self.config_params else 0

    @property
    def is_complex_output(self):
        """Check if ADC output is complex"""
        return self.config_params.get('adc_buf_fmt', 0) == 0  # 0=complex, 1=real

    @property
    def adc_sampling_time(self):
        """ADC Sampling Time = (NADCSamples / (fadc * KHz2Hz)) * sec2usec"""
        if 'adc_samples' not in self.config_params or 'sample_rate' not in self.config_params:
            return 0
        KHz2Hz = 1e3
        sec2usec = 1e6
        return (self.config_params['adc_samples'] / (self.config_params['sample_rate'] * KHz2Hz)) * sec2usec  # μs

    @property
    def inter_chirp_time(self):
        """Inter-Chirp Time = Tid + (Tramp end - Tadc)"""
        if not all(key in self.config_params for key in ['idle_time', 'ramp_end_time']):
            return 0
        return self.config_params['idle_time'] + (self.config_params['ramp_end_time'] - self.adc_sampling_time)  # μs

    @property
    def active_frame_time(self):
        """Active Frame Time = Nchirps × (Tid + Tramp end) × Nloops × usec to msec"""
        if not all(key in self.config_params for key in ['chirps_per_frame', 'idle_time', 'ramp_end_time', 'num_loops']):
            return 0
        return (self.config_params['chirps_per_frame'] * 
                (self.config_params['idle_time'] + self.config_params['ramp_end_time']) * 
                self.config_params['num_loops']) / 1000  # ms

    @property
    def frame_time(self):
        """Frame Time = Tframe period"""
        return self.config_params.get('frame_periodicity', 0)  # ms

    @property
    def duty_cycle(self):
        """Duty Cycle (%) = (Tactive / Tframe) × 100"""
        if self.frame_time == 0:
            return 0
        return (self.active_frame_time / self.frame_time) * 100

    @property
    def total_bandwidth(self):
        """Total Bandwidth = Tramp end × Sslope (in MHz)"""
        if not all(key in self.config_params for key in ['ramp_end_time', 'freq_slope']):
            return 0
        return self.config_params['freq_slope'] * self.config_params['ramp_end_time']  # MHz

    @property
    def valid_bandwidth(self):
        """Valid Bandwidth = Tadc × Sslope (in MHz)"""
        if not all(key in self.config_params for key in ['freq_slope']):
            return 0
        return self.config_params['freq_slope'] * self.adc_sampling_time  # MHz

    @property
    def range_resolution(self):
        """Range Resolution = c / (2 × B), where B is valid bandwidth in Hz"""
        if self.valid_bandwidth == 0:
            return 0
        MHz2Hz = 1e6
        bandwidth_hz = self.valid_bandwidth * MHz2Hz
        resolution = self.SPEED_OF_LIGHT / (2 * bandwidth_hz)  # meters
        #logging.info(f"Range Resolution: {resolution:.4f} meters")
        return resolution

    @property
    def max_range(self):
        """Maximum Range = (IFmax × c) / (2 × S), where IFmax = 0.8 × fs for complex 1x"""
        if not all(key in self.config_params for key in ['sample_rate', 'freq_slope']):
            return 0
        
        # Convert sampling rate from ksps to Hz
        fs_hz = self.config_params['sample_rate'] * 1e3
        
        # Convert frequency slope from MHz/μs to Hz/s
        # MHz/μs -> Hz/s: × 1e6 (MHz to Hz) × 1e6 (μs to s)
        slope_hz_per_sec = self.config_params['freq_slope'] * 1e12
        
        # IFmax = 0.8 × fs for complex 1x mode
        IFmax = 0.8 * fs_hz
        
        # Calculate maximum range
        max_range = (IFmax * self.SPEED_OF_LIGHT) / (2 * slope_hz_per_sec)  # meters
        #logging.info(f"Maximum Range: {max_range:.4f} meters")
        return max_range

    @property
    def wavelength(self):
        """Wavelength = c / fcarrier"""
        if 'start_freq' not in self.config_params:
            return 0
        wavelength = self.SPEED_OF_LIGHT / (self.config_params['start_freq'] * 1e9)  # meters
        logging.info(f"Wavelength: {wavelength:.4f} meters")
        return wavelength

    @property
    def velocity_resolution(self):
        """Velocity Resolution = λ / (2 × Nloops × numTX × Tchirp)"""
        if not all(key in self.config_params for key in ['num_loops', 'idle_time', 'ramp_end_time']):
            return 0
        sec2usec = 1e6
        GHz2Hz = 1e9
        # Calculate carrier frequency (using start frequency)
        carrier_freq = self.config_params['start_freq'] * GHz2Hz
        wavelength = self.SPEED_OF_LIGHT / carrier_freq
        # Calculate chirp time in seconds
        chirp_time = (self.config_params['idle_time'] + self.config_params['ramp_end_time']) / sec2usec
        resolution = wavelength / (2 * self.config_params['num_loops'] * self.num_tx_channels * chirp_time)  # m/s
        logging.info(f"Velocity Resolution: {resolution:.4f} m/s")
        return resolution

    @property
    def max_velocity(self):
        """Maximum Velocity = λ / (4 × numTX × Tchirp)"""
        if not all(key in self.config_params for key in ['idle_time', 'ramp_end_time']):
            return 0
        sec2usec = 1e6
        GHz2Hz = 1e9
        # Calculate carrier frequency (using start frequency)
        carrier_freq = self.config_params['start_freq'] * GHz2Hz
        wavelength = self.SPEED_OF_LIGHT / carrier_freq
        # Calculate chirp time in seconds
        chirp_time = (self.config_params['idle_time'] + self.config_params['ramp_end_time']) / sec2usec
        max_vel = wavelength / (4 * self.num_tx_channels * chirp_time)  # m/s
        logging.info(f"Maximum Velocity: {max_vel:.4f} m/s")
        return max_vel

    @property
    def num_doppler_bins(self):
        """Number of Doppler Bins = 2^⌈log2(Nloops)⌉"""
        if 'num_loops' not in self.config_params:
            return 0
        return 2 ** math.ceil(math.log2(self.config_params['num_loops']))

    @property
    def num_range_bins(self):
        """Number of Range Bins = 2^⌈log2(NADCSamples)⌉"""
        if 'adc_samples' not in self.config_params:
            return 0
        return 2 ** math.ceil(math.log2(self.config_params['adc_samples']))

    @property
    def radar_cube_size(self):
        """Radar Cube Size (KB) = (bytesPerSample × NADCSamples × Nchirps × Nloops × NRX) / 1024"""
        if not all(key in self.config_params for key in ['adc_samples', 'chirps_per_frame', 'num_loops']):
            return 0
        bytes_per_sample = 4  # Complex data uses 4 bytes per sample (2 bytes each for I and Q)
        return (bytes_per_sample * self.config_params['adc_samples'] * self.config_params['chirps_per_frame'] * 
                self.config_params['num_loops'] * self.num_rx_channels) / 1024  # KB

    def get_all_parameters(self):
        """Return all calculated parameters as a dictionary."""
        return {
            'num_tx_channels': self.num_tx_channels,
            'num_rx_channels': self.num_rx_channels,
            'adc_sampling_time': self.adc_sampling_time,
            'inter_chirp_time': self.inter_chirp_time,
            'active_frame_time': self.active_frame_time,
            'frame_time': self.frame_time,
            'duty_cycle': self.duty_cycle,
            'total_bandwidth': self.total_bandwidth,
            'valid_bandwidth': self.valid_bandwidth,
            'range_resolution': self.range_resolution,
            'max_range': self.max_range,
            'velocity_resolution': self.velocity_resolution,
            'max_velocity': self.max_velocity,
            'num_doppler_bins': self.num_doppler_bins,
            'num_range_bins': self.num_range_bins,
            'radar_cube_size': self.radar_cube_size
        }
