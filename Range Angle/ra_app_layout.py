# -*- coding: utf-8 -*-

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QButtonGroup, QRadioButton
import pyqtgraph as pg
from pyqtgraph import GraphicsLayoutWidget

# Enable OpenGL acceleration for better performance
pg.setConfigOptions(useOpenGL=True, antialias=True)

# Set default theme for pyqtgraph to dark background
pg.setConfigOption('background', 'k')  # 'k' is black
pg.setConfigOption('foreground', 'w')  # 'w' is white

class Ui_MainWindow(object):

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        # Set a reasonable default size while still allowing resizing
        MainWindow.resize(1400, 300)  # Increased width to accommodate data display
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        MainWindow.setFont(font)
        MainWindow.setContextMenuPolicy(QtCore.Qt.NoContextMenu)
        
        # Set black background for the main window
        MainWindow.setStyleSheet("background-color: black; color: white;")
        
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.centralwidget.setStyleSheet("background-color: black; color: white;")

        # Main horizontal layout (left side for plots, right side for controls)
        self.main_layout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.main_layout.setSpacing(10)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Left side layout (for plots and data display)
        self.left_side_layout = QtWidgets.QVBoxLayout()
        self.left_side_layout.setSpacing(10)
        
        # Right side layout (for controls)
        self.right_side_layout = QtWidgets.QVBoxLayout()
        self.right_side_layout.setSpacing(10)
        
        # Configure fonts
        button_font = QtGui.QFont()
        button_font.setFamily("Arial")
        button_font.setBold(True)
        button_font.setWeight(75)
        button_font.setPointSize(10)  # Standard size for buttons
        
        title_font = QtGui.QFont()
        title_font.setFamily("Arial")
        title_font.setPointSize(12)  # Standardized title size
        title_font.setBold(True)
        title_font.setWeight(75)
        
        # Top controls layout (will be added to left side)
        self.top_controls = QtWidgets.QHBoxLayout()
        self.top_controls.setSpacing(20)

        # COM Port section
        com_layout = QtWidgets.QHBoxLayout()
        self.com_port_label = QtWidgets.QLabel(self.centralwidget)
        self.com_port_label.setFont(button_font)
        self.com_port_label.setObjectName("com_port_label")
        self.com_port_label.setStyleSheet("color: white;")
        
        self.com_select = QtWidgets.QComboBox(self.centralwidget)
        self.com_select.setFont(button_font)
        self.com_select.setObjectName("com_select")
        self.com_select.setStyleSheet("background-color: #333333; color: white; selection-background-color: #555555; border-radius: 4px;")
        self.com_select.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.com_select.setMinimumWidth(120)

        com_layout.addWidget(self.com_port_label)
        com_layout.addWidget(self.com_select)
        self.top_controls.addLayout(com_layout)
        
        # Config file section
        config_layout = QtWidgets.QHBoxLayout()
        self.config_label = QtWidgets.QLabel(self.centralwidget)
        self.config_label.setFont(button_font)
        self.config_label.setObjectName("config_label")
        self.config_label.setStyleSheet("color: white;")
        
        self.config_path = QtWidgets.QLineEdit(self.centralwidget)
        self.config_path.setFont(button_font)
        self.config_path.setObjectName("config_path")
        self.config_path.setReadOnly(True)
        self.config_path.setStyleSheet("background-color: #333333; color: white; border: 1px solid #555555; border-radius: 4px;")
        self.config_path.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        config_layout.addWidget(self.config_label)
        config_layout.addWidget(self.config_path)
        self.top_controls.addLayout(config_layout)
        
        # Buttons layout
        buttons_layout = QtWidgets.QHBoxLayout()
        self.browse_button = QtWidgets.QPushButton(self.centralwidget)
        self.browse_button.setFont(button_font)
        self.browse_button.setObjectName("browse_button")
        self.browse_button.setStyleSheet("background-color: #444444; color: white; border: 1px solid #666666; border-radius: 4px; padding: 4px;")
        self.browse_button.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.browse_button.setMinimumWidth(120)
        self.browse_button.setMinimumHeight(30)
        
        self.start_button = QtWidgets.QPushButton(self.centralwidget)
        self.start_button.setFont(button_font)
        self.start_button.setObjectName("start_button")
        self.start_button.setStyleSheet("background-color: #008800; color: white; border-radius: 4px; padding: 4px;")
        self.start_button.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.start_button.setMinimumWidth(180)
        self.start_button.setMinimumHeight(30)

        buttons_layout.addWidget(self.browse_button)
        buttons_layout.addWidget(self.start_button)
        self.top_controls.addLayout(buttons_layout)
        
        # Add top controls to left side layout
        self.left_side_layout.addLayout(self.top_controls)
        
        # Create exit button for top right
        self.exit_button = QtWidgets.QPushButton(self.centralwidget)
        self.exit_button.setFont(button_font)
        self.exit_button.setObjectName("exit_button")
        self.exit_button.setStyleSheet("background-color: #cc0000; color: white; border-radius: 4px; padding: 4px;")
        self.exit_button.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.exit_button.setMinimumWidth(90)
        self.exit_button.setMinimumHeight(30)
        
        # Add exit button to right side layout
        self.right_side_layout.addWidget(self.exit_button, 0, QtCore.Qt.AlignRight)
        
        # Top plots layout (will be added to left side)
        top_plots_layout = QtWidgets.QHBoxLayout()
        
        # Left plot (Range Angle)
        left_plot_layout = QtWidgets.QVBoxLayout()
        self.range_angle_label = QtWidgets.QLabel(self.centralwidget)
        self.range_angle_label.setFont(title_font)
        self.range_angle_label.setAlignment(QtCore.Qt.AlignCenter)
        self.range_angle_label.setObjectName("range_angle_label")
        self.range_angle_label.setStyleSheet("color: white;")
        left_plot_layout.addWidget(self.range_angle_label)
        
        # Range angle view
        self.range_angle_view = GraphicsLayoutWidget(self.centralwidget)
        self.range_angle_view.setObjectName("range_angle_view")
        self.range_angle_view.setBackground('black')
        self.range_angle_view.setStyleSheet("border: 2px solid white;")
        self.range_angle_view.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.range_angle_view.setMinimumSize(400, 300)
        left_plot_layout.addWidget(self.range_angle_view)
        
        top_plots_layout.addLayout(left_plot_layout)

        # Data Display section
        data_display_layout = QtWidgets.QVBoxLayout()
        self.data_display_label = QtWidgets.QLabel(self.centralwidget)
        self.data_display_label.setFont(title_font)
        self.data_display_label.setAlignment(QtCore.Qt.AlignCenter)
        self.data_display_label.setText("Detected Objects")
        self.data_display_label.setStyleSheet("color: white;")
        data_display_layout.addWidget(self.data_display_label)
        
        # Data Display table
        self.data_display_table = QtWidgets.QTableWidget(self.centralwidget)
        self.data_display_table.setFont(button_font)
        self.data_display_table.setStyleSheet("""
            QTableWidget {
                background-color: #333333;
                color: #00ff00;
                border: 2px solid white;
                border-radius: 4px;
                gridline-color: #555555;
            }
            QHeaderView::section {
                background-color: #444444;
                color: white;
                padding: 5px;
                border: 1px solid #555555;
            }
            QTableWidget::item {
                padding: 5px;
            }
        """)
        self.data_display_table.setMinimumWidth(200)
        self.data_display_table.setMinimumHeight(300)
        
        # Set up table columns for range-angle data
        self.data_display_table.setColumnCount(2)
        self.data_display_table.setHorizontalHeaderLabels(['Range (m)', 'Angle (°)'])
        
        # Adjust column widths
        header = self.data_display_table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        
        data_display_layout.addWidget(self.data_display_table)
        
        top_plots_layout.addLayout(data_display_layout)
        self.left_side_layout.addLayout(top_plots_layout)
        
        # Add CFAR controls in a horizontal layout below plots
        cfar_controls_container = QtWidgets.QHBoxLayout()
        
        # Add CFAR Parameters label
        self.cfar_params_label = QtWidgets.QLabel("CFAR Parameters:", self.centralwidget)
        self.cfar_params_label.setFont(title_font)
        self.cfar_params_label.setStyleSheet("color: white;")
        cfar_controls_container.addWidget(self.cfar_params_label)
        
        # Add spacing to move controls to the right
        cfar_controls_container.addSpacing(20)
        
        # Create layout for CFAR controls
        cfar_controls_layout = QtWidgets.QHBoxLayout()
        cfar_controls_layout.setSpacing(20)
        
        # Common style for spinboxes
        spinbox_style = """
            QSpinBox, QDoubleSpinBox {
                background-color: #333333;
                color: white;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 5px;
                min-width: 80px;
            }
            QSpinBox::up-button, QSpinBox::down-button,
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
                background-color: #444444;
                border: none;
                border-radius: 2px;
            }
            QSpinBox:hover, QDoubleSpinBox:hover {
                background-color: #3c3c3c;
            }
        """
        
        # Guard cells control
        guard_cells_container = QtWidgets.QHBoxLayout()
        self.guard_cells_label = QtWidgets.QLabel("Guard Cells:", self.centralwidget)
        self.guard_cells_label.setFont(button_font)
        self.guard_cells_label.setStyleSheet("color: white;")
        self.guard_cells_spin = QtWidgets.QSpinBox(self.centralwidget)
        self.guard_cells_spin.setStyleSheet(spinbox_style)
        self.guard_cells_spin.setRange(1, 20)
        self.guard_cells_spin.setValue(2)
        guard_cells_container.addWidget(self.guard_cells_label)
        guard_cells_container.addWidget(self.guard_cells_spin)
        cfar_controls_layout.addLayout(guard_cells_container)
        
        # Training cells control
        training_cells_container = QtWidgets.QHBoxLayout()
        self.training_cells_label = QtWidgets.QLabel("Training Cells:", self.centralwidget)
        self.training_cells_label.setFont(button_font)
        self.training_cells_label.setStyleSheet("color: white;")
        self.training_cells_spin = QtWidgets.QSpinBox(self.centralwidget)
        self.training_cells_spin.setStyleSheet(spinbox_style)
        self.training_cells_spin.setRange(2, 50)
        self.training_cells_spin.setValue(8)
        training_cells_container.addWidget(self.training_cells_label)
        training_cells_container.addWidget(self.training_cells_spin)
        cfar_controls_layout.addLayout(training_cells_container)
        
        # False alarm rate control
        false_alarm_container = QtWidgets.QHBoxLayout()
        self.false_alarm_label = QtWidgets.QLabel("False Alarm Rate:", self.centralwidget)
        self.false_alarm_label.setFont(button_font)
        self.false_alarm_label.setStyleSheet("color: white;")
        self.false_alarm_spin = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.false_alarm_spin.setStyleSheet(spinbox_style)
        self.false_alarm_spin.setRange(0.01, 1.0)
        self.false_alarm_spin.setSingleStep(0.01)
        self.false_alarm_spin.setValue(0.01)
        false_alarm_container.addWidget(self.false_alarm_label)
        false_alarm_container.addWidget(self.false_alarm_spin)
        cfar_controls_layout.addLayout(false_alarm_container)

        # Group Peaks button
        self.group_peaks_button = QtWidgets.QPushButton(self.centralwidget)
        self.group_peaks_button.setFont(button_font)
        self.group_peaks_button.setObjectName("group_peaks_button")
        self.group_peaks_button.setText("Group Peaks")
        self.group_peaks_button.setCheckable(True)
        self.group_peaks_button.setStyleSheet("""
            QPushButton {
                background-color: #FFA500;
                color: black;
                border-radius: 4px;
                padding: 5px;
                min-width: 100px;
            }
            QPushButton:checked {
                background-color: #FF8C00;
                border: 1px solid #FF4500;
            }
        """)
        cfar_controls_layout.addWidget(self.group_peaks_button)
        
        # Add spacing to center the controls
        cfar_controls_layout.addStretch()
        
        # Add CFAR controls layout to container
        cfar_controls_container.addLayout(cfar_controls_layout)
        
        # Add CFAR controls container to left side layout
        self.left_side_layout.addLayout(cfar_controls_container)
        
        # ---- RIGHT SIDE CONTROLS ----
        
        # Window and padding controls (Processing Controls)
        controls_group_box = QtWidgets.QGroupBox("Processing Controls", self.centralwidget)
        controls_group_box.setStyleSheet("QGroupBox { color: white; border: 1px solid #555555; border-radius: 5px; padding-top: 15px; margin-top: 5px; } QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center; padding: 0 10px; }")
        controls_layout = QtWidgets.QVBoxLayout(controls_group_box)
        controls_layout.setSpacing(10)
        
        # Common fonts
        label_font = QtGui.QFont()
        label_font.setFamily("Arial")
        label_font.setBold(True)
        label_font.setPointSize(11)

        window_font = QtGui.QFont()
        window_font.setFamily("Arial")
        window_font.setBold(True)
        window_font.setPointSize(11)
        
        # Common dropdown style
        dropdown_style = """
            QComboBox {
                background-color: #2c2c2c;
                color: white;
                border: 1px solid #555555;
                border-radius: 5px;
                padding: 5px;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid white;
                margin-right: 5px;
            }
            QComboBox:hover {
                background-color: #3c3c3c;
            }
            QComboBox QAbstractItemView {
                background-color: #2c2c2c;
                color: white;
                selection-background-color: #3c3c3c;
                border: 1px solid #555555;
            }
        """

        # Window selection
        window_group = QtWidgets.QHBoxLayout()
        self.window_label = QtWidgets.QLabel(self.centralwidget)
        self.window_label.setFont(label_font)
        self.window_label.setText("Window: ")
        self.window_label.setStyleSheet("color: white;")
        
        self.window_select = QtWidgets.QComboBox(self.centralwidget)
        self.window_select.setFont(window_font)
        self.window_select.setObjectName("window_select")
        self.window_select.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.window_select.setStyleSheet(dropdown_style)

        window_group.addWidget(self.window_label)
        window_group.addWidget(self.window_select)
        controls_layout.addLayout(window_group)
        
        # Range padding
        range_group = QtWidgets.QHBoxLayout()
        self.range_pad_label = QtWidgets.QLabel(self.centralwidget)
        self.range_pad_label.setFont(label_font)
        self.range_pad_label.setText("Range Padding: ")
        self.range_pad_label.setStyleSheet("color: white;")
        
        self.range_pad_select = QtWidgets.QComboBox(self.centralwidget)
        self.range_pad_select.setFont(window_font)
        self.range_pad_select.setObjectName("range_pad_select")
        self.range_pad_select.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.range_pad_select.setStyleSheet(dropdown_style)
        
        range_group.addWidget(self.range_pad_label)
        range_group.addWidget(self.range_pad_select)
        controls_layout.addLayout(range_group)

        # Add Doppler padding
        doppler_group = QtWidgets.QHBoxLayout()
        self.doppler_pad_label = QtWidgets.QLabel(self.centralwidget)
        self.doppler_pad_label.setFont(label_font)
        self.doppler_pad_label.setText("Doppler Padding: ")
        self.doppler_pad_label.setStyleSheet("color: white;")
        
        self.doppler_pad_select = QtWidgets.QComboBox(self.centralwidget)
        self.doppler_pad_select.setFont(window_font)
        self.doppler_pad_select.setObjectName("doppler_pad_select")
        self.doppler_pad_select.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.doppler_pad_select.setStyleSheet(dropdown_style)
        
        doppler_group.addWidget(self.doppler_pad_label)
        doppler_group.addWidget(self.doppler_pad_select)
        controls_layout.addLayout(doppler_group)

        # Angle padding
        angle_group = QtWidgets.QHBoxLayout()
        self.angle_pad_label = QtWidgets.QLabel(self.centralwidget)
        self.angle_pad_label.setFont(label_font)
        self.angle_pad_label.setText("Angle Padding: ")
        self.angle_pad_label.setStyleSheet("color: white;")
        
        self.angle_pad_select = QtWidgets.QComboBox(self.centralwidget)
        self.angle_pad_select.setFont(window_font)
        self.angle_pad_select.setObjectName("angle_pad_select")
        self.angle_pad_select.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.angle_pad_select.setStyleSheet(dropdown_style)

        angle_group.addWidget(self.angle_pad_label)
        angle_group.addWidget(self.angle_pad_select)
        controls_layout.addLayout(angle_group)

        # Add controls group box to right side
        self.right_side_layout.addWidget(controls_group_box)
        
        # Remove Static Clutter button
        self.remove_clutter_button = QtWidgets.QPushButton(self.centralwidget)
        self.remove_clutter_button.setFont(button_font)
        self.remove_clutter_button.setObjectName("remove_clutter_button")
        self.remove_clutter_button.setCheckable(True)
        self.remove_clutter_button.setStyleSheet("""
            QPushButton {
                background-color: #FFFFE0;
                color: black;
                border-radius: 4px;
                padding: 5px;
            }
            QPushButton:checked {
                background-color: #FFD700;
                border: 1px solid #DAA520;
            }
        """)
        self.remove_clutter_button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.remove_clutter_button.setMinimumHeight(30)
        
        # Add Remove Clutter button to right side
        self.right_side_layout.addWidget(self.remove_clutter_button)
        
        # Add window options
        self.window_select.addItem("Blackman-Harris")  # Default
        self.window_select.addItem("Hamming")
        self.window_select.addItem("Hann")
        self.window_select.addItem("Blackman")
        self.window_select.addItem("No Window")
        
        # Add padding options (32 to 1024 in powers of 2)
        padding_options = [str(2**i) for i in range(3, 11)]  # 32, 64, 128, 256, 512, 1024
        for option in padding_options:
            self.range_pad_select.addItem(option)
            self.doppler_pad_select.addItem(option)  # Add options to doppler padding
            self.angle_pad_select.addItem(option)
        
        # Set default padding values
        self.range_pad_select.setCurrentText("512")
        self.doppler_pad_select.setCurrentText("64")  # Set default doppler padding
        self.angle_pad_select.setCurrentText("16")
        
        # Add right side layout to main layout with fixed width
        right_side_container = QtWidgets.QWidget()
        right_side_container.setLayout(self.right_side_layout)
        right_side_container.setFixedWidth(250)  # Control the width of the right side
        
        # Add left and right side layouts to main layout
        self.main_layout.addLayout(self.left_side_layout)
        self.main_layout.addWidget(right_side_container)
        
        # Status bar
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        self.statusbar.setStyleSheet("background-color: black; color: white;")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        # Set tooltips for dropdowns
        self.window_select.setToolTip(_translate("MainWindow", "Select window function for signal processing"))
        self.range_pad_select.setToolTip(_translate("MainWindow", "Select range dimension padding"))
        self.angle_pad_select.setToolTip(_translate("MainWindow", "Select angle dimension padding"))
        MainWindow.setWindowTitle(_translate("MainWindow", "Real-Time Radar"))
        self.range_angle_label.setText(_translate("MainWindow", "Range Angle"))
        self.com_port_label.setText(_translate("MainWindow", "COM Port:"))
        self.config_label.setText(_translate("MainWindow", "Config File:"))
        self.browse_button.setText(_translate("MainWindow", "Browse"))
        self.start_button.setText(_translate("MainWindow", "Send Radar Config"))
        self.remove_clutter_button.setText(_translate("MainWindow", "Remove Static Clutter"))
        self.exit_button.setText(_translate("MainWindow", "Exit"))
