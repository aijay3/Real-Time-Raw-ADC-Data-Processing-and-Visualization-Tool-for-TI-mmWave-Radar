"""
Radar configuration parameters and their descriptions.
"""

RADAR_PARAMS = {
    "dfeDataOutputMode": {
        "description": "Data output mode configuration",
        "parameters": {
            "modeType": {
                "description": "Output mode type",
                "values": {
                    1: "Frame based chirps - Fixed number of chirps per frame",
                    2: "Continuous chirping - Continuous stream of chirps", 
                    3: "Advanced frame config - Multiple subframes with different configurations"
                }
            }
        }
    },
    
    "channelCfg": {
        "description": "Channel configuration",
        "parameters": {
            "rxChannelEn": {
                "description": "Receive antenna mask e.g for 4 antennas, it is 0x1111b = 15"
            },
            "txChannelEn": {
                "description": "Transmit antenna mask"
            },
            "cascading": {
                "description": "SoC cascading, not applicable, set to 0"
            }
        }
    },
    
    "adcCfg": {
        "description": "ADC configuration",
        "parameters": {
            "numADCBits": {
                "description": "Number of ADC bits",
                "values": {
                    0: "12-bits",
                    1: "14-bits",
                    2: "16-bits"
                }
            },
            "adcOutputFmt": {
                "description": "Output format",
                "values": {
                    0: "real",
                    1: "complex 1x (image band filtered output)",
                    2: "complex 2x (image band visible)"
                }
            }
        }
    },
    
    "adcbufCfg": {
        "description": "ADC buffer configuration",
        "parameters": {
            "subFrameIdx": {
                "description": "Subframe Index (-1 for legacy mode, or specific subframe number for advanced frame mode)"
            },
            "adcOutputFmt": {
                "description": "ADCBUF output format",
                "values": {
                    0: "Complex (only supported mode)",
                    1: "Real (not supported)"
                }
            },
            "SampleSwap": {
                "description": "ADCBUF IQ swap selection",
                "values": {
                    0: "I in LSB, Q in MSB",
                    1: "Q in LSB, I in MSB (only supported option)"
                }
            },
            "ChanInterleave": {
                "description": "ADCBUF channel interleave configuration",
                "values": {
                    0: "Interleaved (only supported for XWR14xx)",
                    1: "Non-interleaved (only supported option)"
                }
            },
            "ChirpThreshold": {
                "description": "Set to 1 when LVDS streaming is enabled"
            }
        }
    },
    
    "profileCfg": {
        "description": "Chirp profile configuration",
        "parameters": {
            "profileId": {"description": "Profile Identifier"},
            "startFreq": {"description": "Start frequency in GHz"},
            "idleTime": {"description": "Idle time in microseconds"},
            "adcStartTime": {"description": "ADC valid start time in microseconds"},
            "rampEndTime": {"description": "Ramp end time in microseconds"},
            "txOutPower": {"description": "Tx output power back-off code"},
            "txPhaseShifter": {"description": "Tx phase shifter for Tx antennas"},
            "freqSlopeConst": {"description": "Frequency slope in MHz/usec"},
            "txStartTime": {"description": "Tx start time in microseconds"},
            "numAdcSamples": {"description": "Number of ADC samples collected"},
            "digOutSampleRate": {"description": "ADC sampling frequency in ksps"},
            "hpfCornerFreq1": {"description": "HPF1 corner frequency"},
            "hpfCornerFreq2": {"description": "HPF2 corner frequency"},
            "rxGain": {"description": "RX gain (OR'ed value of RX gain in dB and RF gain target)"}
        }
    },
    
    "chirpCfg": {
        "description": "Chirp configuration",
        "parameters": {
            "startIdx": {"description": "Chirp start index"},
            "endIdx": {"description": "Chirp end index"},
            "profileId": {"description": "Profile identifier"},
            "startFreqVar": {"description": "Start frequency variation (Hz)"},
            "freqSlopeVar": {"description": "Frequency slope variation (kHz/us)"},
            "idleTimeVar": {"description": "Idle time variation (µs)"},
            "adcStartTimeVar": {"description": "ADC start time variation (µs)"},
            "txEnable": {"description": "Tx antenna enable mask"}
        }
    },
    
    "lowPower": {
        "description": "Low power mode configuration",
        "parameters": {
            "dont_care": {"description": "Set to 0 (mandatory value)"},
            "adcMode": {"description": "ADC mode configuration (0x00: Regular ADC mode)"}
        }
    },
    
    "frameCfg": {
        "description": "Frame configuration",
        "parameters": {
            "chirpStartIdx": {"description": "Chirp start index (0-511)"},
            "chirpEndIdx": {"description": "Chirp end index (chirpStartIdx-511)"},
            "numLoops": {"description": "Number of loops (1-255)"},
            "numFrames": {"description": "Number of frames (0 for infinite, 1-65535)"},
            "framePeriodicity": {"description": "Frame periodicity in milliseconds"},
            "triggerSelect": {"description": "Frame trigger selection (1: Software trigger)"},
            "frameTriggerDelay": {"description": "Frame trigger delay in milliseconds"}
        }
    },
    
    "advFrameCfg": {
        "description": "Advanced frame configuration",
        "parameters": {
            "numOfSubFrames": {"description": "Number of subframes enabled in the frame"},
            "forceProfile": {"description": "Force profile (only 0 is supported)"},
            "numFrames": {"description": "Number of frames to transmit (1 frame = all enabled subframes)"},
            "triggerSelect": {"description": "Trigger selection (1: Software trigger)"},
            "frameTrigDelay": {"description": "Frame trigger delay in milliseconds (float values allowed)"}
        }
    },
    
    "subFrameCfg": {
        "description": "Subframe configuration",
        "parameters": {
            "subFrameNum": {"description": "Subframe number (0 to RL_MAX_SUBFRAMES-1)"},
            "forceProfileIdx": {"description": "Force profile index (ignored as forceProfile in advFrameCfg should be set to 0)"},
            "chirpStartIdx": {"description": "Start index of chirp (corresponding chirpCfg should be defined)"},
            "numOfChirps": {"description": "Number of unique chirps per burst, including the start index"},
            "numLoops": {"description": "Number of loops through unique chirps (≥4; must be multiple of 4 for DSP Doppler DPU)"},
            "burstPeriodicity": {"description": "Burst periodicity in milliseconds (float values allowed; must meet timing constraints)"},
            "chirpStartIdxOffset": {"description": "Chirp start index increment for next burst (set to 0 as only one burst per subframe is supported)"},
            "numOfBurst": {"description": "Number of bursts in the subframe (set to 1 as only one burst per subframe is supported)"},
            "numOfBurstLoops": {"description": "Number of times to loop over bursts (set to 1 as only one burst per subframe is supported)"},
            "subFramePeriodicity": {"description": "Subframe periodicity in milliseconds (set to same as burstPeriodicity)"}
        }
    },
    
    "guiMonitor": {
        "description": "GUI monitoring configuration",
        "parameters": {
            "subFrameIdx": {
                "description": "Subframe Index (-1 for legacy mode, or specific subframe number for advanced frame mode)"
            },
            "detectedObjects": {
                "description": "Export of point cloud",
                "values": {
                    0: "Disable",
                    1: "Enable export with side info (SNR, noise)",
                    2: "Enable export without side info"
                }
            },
            "logMagRange": {
                "description": "Export of log magnitude range profile at zero Doppler",
                "values": {
                    0: "Disable",
                    1: "Enable"
                }
            },
            "noiseProfile": {
                "description": "Export of log magnitude noise profile",
                "values": {
                    0: "Disable",
                    1: "Enable"
                }
            },
            "rangeAzimuthHeatMap": {
                "description": "Range-Azimuth heat map (AoA DPU required)",
                "values": {
                    0: "Disable",
                    1: "Enable export of zero Doppler radar cube matrix"
                }
            },
            "rangeDopplerHeatMap": {
                "description": "Range-Doppler heat map",
                "values": {
                    0: "Disable",
                    1: "Enable export of detection matrix"
                }
            },
            "statsInfo": {
                "description": "Export of system statistics (CPU load, margins, temperature)",
                "values": {
                    0: "Disable",
                    1: "Enable"
                }
            }
        }
    },
    
    "cfarCfg": {
        "description": "CFAR detection configuration",
        "parameters": {
            "subFrameIdx": {"description": "Subframe Index (-1 for legacy mode or to apply to all subframes)"},
            "procDirection": {
                "description": "Processing direction",
                "values": {
                    0: "CFAR detection in range direction",
                    1: "CFAR detection in Doppler direction"
                }
            },
            "mode": {
                "description": "CFAR averaging mode",
                "values": {
                    0: "CFAR_CA (Cell Averaging)",
                    1: "CFAR_CAGO (Cell Averaging Greatest Of)",
                    2: "CFAR_CASO (Cell Averaging Smallest Of)"
                }
            },
            "noiseWin": {"description": "Noise averaging window length (in samples)"},
            "guardLen": {"description": "One-sided guard length (in samples)"},
            "divShift": {"description": "Cumulative noise sum divisor (expressed as a shift)"},
            "cyclicMode": {
                "description": "Cyclic mode (wrapped around mode)",
                "values": {
                    0: "Disabled",
                    1: "Enabled"
                }
            },
            "thresholdScale": {"description": "Detection threshold scale in dB (float value, max 100 dB)"},
            "peakGrouping": {
                "description": "Peak grouping",
                "values": {
                    0: "Disabled",
                    1: "Enabled"
                }
            }
        }
    },
    
    "multiObjBeamForming": {
        "description": "Multi-Object Beamforming configuration",
        "parameters": {
            "subFrameIdx": {"description": "Subframe Index (-1 for legacy mode or to apply to all subframes)"},
            "featureEnabled": {
                "description": "Multi-Object Beamforming feature",
                "values": {
                    0: "Disabled",
                    1: "Enabled"
                }
            },
            "threshold": {"description": "Detection threshold for second peak in Azimuth FFT output (0 to 1)"}
        }
    },
    
    "calibDcRangeSig": {
        "description": "DC range calibration configuration",
        "parameters": {
            "subFrameIdx": {"description": "Subframe Index (-1 for legacy mode or to apply to all subframes)"},
            "enabled": {
                "description": "Enable DC removal using first few chirps",
                "values": {
                    0: "Disabled",
                    1: "Enabled"
                }
            },
            "negativeBinIdx": {"description": "Maximum negative range FFT index for compensation"},
            "positiveBinIdx": {"description": "Maximum positive range FFT index for compensation"},
            "numAvg": {"description": "Number of chirps to average for DC signature (power of 2, > num Doppler bins)"}
        }
    },
    
    "clutterRemoval": {
        "description": "Static clutter removal configuration",
        "parameters": {
            "subFrameIdx": {"description": "Subframe Index (-1 for legacy mode or to apply to all subframes)"},
            "enabled": {
                "description": "Enable static clutter removal",
                "values": {
                    0: "Disabled",
                    1: "Enabled"
                }
            }
        }
    },
    
    "aoaFovCfg": {
        "description": "Angle of Arrival Field of View configuration",
        "parameters": {
            "subFrameIdx": {"description": "Subframe Index (-1 for legacy mode or to apply to all subframes)"},
            "minAzimuthDeg": {"description": "Minimum azimuth angle (degrees) for field of view"},
            "maxAzimuthDeg": {"description": "Maximum azimuth angle (degrees) for field of view"},
            "minElevationDeg": {"description": "Minimum elevation angle (degrees) for field of view"},
            "maxElevationDeg": {"description": "Maximum elevation angle (degrees) for field of view"}
        }
    },
    
    "cfarFovCfg": {
        "description": "CFAR Field of View configuration",
        "parameters": {
            "subFrameIdx": {"description": "Subframe Index (-1 for legacy mode or to apply to all subframes)"},
            "procDirection": {
                "description": "Processing direction",
                "values": {
                    0: "Point filtering in range direction (meters)",
                    1: "Point filtering in Doppler direction (meters/sec)"
                }
            },
            "min": {"description": "Minimum limit (meters or m/s) - points below are filtered"},
            "max": {"description": "Maximum limit (meters or m/s) - points above are filtered"}
        }
    },
    
    "compRangeBiasAndRxChanPhase": {
        "description": "Range bias and RX channel phase compensation",
        "parameters": {
            "rangeBias": {"description": "Compensation for range estimation bias (meters)"},
            "phaseBias": {"description": "Complex values (Re,Im pairs) for Rx channel phase compensation"}
        }
    },
    
    "measureRangeBiasAndRxChanPhase": {
        "description": "Range bias and RX channel phase measurement",
        "parameters": {
            "enabled": {
                "description": "Enable measurement",
                "values": {
                    0: "Disabled (default for non-calibration profiles)",
                    1: "Enabled (use with profile_calibration.cfg)"
                }
            },
            "targetDistance": {"description": "Distance to strong reflector/test object (meters)"},
            "searchWin": {"description": "Search window around targetDistance (meters)"}
        }
    },
    
    "extendedMaxVelocity": {
        "description": "Extended maximum velocity configuration",
        "parameters": {
            "subFrameIdx": {"description": "Subframe Index (-1 for legacy mode or to apply to all subframes)"},
            "enabled": {
                "description": "Enable velocity disambiguation",
                "values": {
                    0: "Disabled",
                    1: "Enabled"
                }
            }
        }
    },
    
    "CQRxSatMonitor": {
        "description": "RX Saturation Monitoring configuration",
        "parameters": {
            "profile": {"description": "Valid profile ID for monitoring configuration"},
            "satMonSel": {"description": "RX Saturation monitoring mode"},
            "priSliceDuration": {"description": "Duration of each slice (1LSB = 0.16us)"},
            "numSlices": {"description": "Number of primary and secondary slices (1-127, max primary 64)"},
            "rxChanMask": {"description": "RX channel mask (1 - Mask, 0 - Unmask)"}
        }
    },
    
    "analogMonitor": {
        "description": "Analog monitoring configuration",
        "parameters": {
            "rxSaturation": {
                "description": "CQRxSatMonitor enable/disable",
                "values": {
                    0: "Disable",
                    1: "Enable"
                }
            },
            "sigImgBand": {
                "description": "CQSigImgMonitor enable/disable",
                "values": {
                    0: "Disable",
                    1: "Enable"
                }
            }
        }
    },
    
    "lvdsStreamCfg": {
        "description": "LVDS streaming configuration",
        "parameters": {
            "subFrameIdx": {"description": "Subframe Index (-1 for legacy mode or to apply to all subframes)"},
            "enableHeader": {
                "description": "Enable/disable HSI header for all active streams",
                "values": {
                    0: "Disable HSI header",
                    1: "Enable HSI header"
                }
            },
            "dataFmt": {
                "description": "Data format for HW streaming",
                "values": {
                    0: "HW Streaming Disabled",
                    1: "ADC",
                    4: "CP_ADC_CQ"
                }
            },
            "enableSW": {
                "description": "Enable/disable user data (SW session)",
                "values": {
                    0: "Disable user data",
                    1: "Enable user data"
                }
            }
        }
    },
    
    "bpmCfg": {
        "description": "Beamforming MIMO configuration",
        "parameters": {
            "subFrameIdx": {"description": "Subframe Index (-1 for legacy mode or to apply to all subframes)"},
            "enabled": {
                "description": "BPM enable/disable",
                "values": {
                    0: "Disabled",
                    1: "Enabled"
                }
            },
            "chirp0Idx": {"description": "Chirp index for first BPM chirp (phase 0 on TXA/TXB)"},
            "chirp1Idx": {"description": "Chirp index for second BPM chirp (phase 0 on TXA, 180 on TXB)"}
        }
    },
    
    "calibdata": {
        "description": "Calibration data configuration",
        "parameters": {
            "saveEnable": {
                "description": "Save enable/disable",
                "values": {
                    0: "Save enabled (boot-time calibration and save to FLASH)",
                    1: "Save disabled"
                }
            },
            "restoreEnable": {
                "description": "Restore enable/disable",
                "values": {
                    0: "Restore enabled (restore from FLASH, skip boot-time calibration)",
                    1: "Restore disabled"
                }
            },
            "flashOffset": {"description": "FLASH address offset for calibration data"}
        }
    },
    
    "compressCfg": {
        "description": "Data compression configuration",
        "parameters": {
            "subFrameIdx": {"description": "Subframe Index (-1 for legacy mode or to apply to all subframes)"},
            "ratio": {
                "description": "Compression ratio for radar data cube",
                "values": {
                    0.25: "25% compression",
                    0.5: "50% compression",
                    0.75: "75% compression"
                }
            },
            "numRangeBins": {"description": "Number of range bins per compressed block (1-32)"}
        }
    }
}

def get_param_description(command, param_name=None):
    """Get description for a command or specific parameter."""
    if command not in RADAR_PARAMS:
        return "Unknown command"
        
    if param_name is None:
        return RADAR_PARAMS[command]["description"]
        
    params = RADAR_PARAMS[command]["parameters"]
    if param_name not in params:
        return "Unknown parameter"
        
    param_info = params[param_name]
    if isinstance(param_info, dict):
        if "values" in param_info:
            desc = param_info["description"] + "\nValues:\n"
            for val, val_desc in param_info["values"].items():
                desc += f"  {val}: {val_desc}\n"
            return desc
        return param_info["description"]
    return param_info
