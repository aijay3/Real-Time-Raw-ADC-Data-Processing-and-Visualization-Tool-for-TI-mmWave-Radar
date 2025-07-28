"""
Launcher application for mmWave Radar Processing Tools.
This script provides a GUI to launch the different radar processing applications.
"""

import os
import sys
import logging
import subprocess
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import QPalette, QColor
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LauncherWindow(QtWidgets.QMainWindow):
    """Main launcher window for the radar processing applications."""
    
    # Track the currently running application process
    current_app_process = None
    current_app_name = None
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("mmWave Radar Processing Tools")
        self.setMinimumSize(600, 550)  # Further increased height to accommodate longer descriptions
        
        # Set dark theme
        self.set_dark_theme()
        
        # Center the window on the screen
        self.center_on_screen()
        
        # Create the central widget and layout
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        
        # Create main layout
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        
        # Add title and description
        title_label = QtWidgets.QLabel("mmWave Radar Processing Tools")
        title_label.setAlignment(QtCore.Qt.AlignCenter)
        title_font = QtGui.QFont("Segoe UI", 18)  # Modern font
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #3498db; margin-bottom: 5px;")  # Modern blue color
        
        description_label = QtWidgets.QLabel(
            "Select an application to launch:"
        )
        description_label.setAlignment(QtCore.Qt.AlignCenter)
        desc_font = QtGui.QFont("Segoe UI", 10)  # Modern font
        description_label.setFont(desc_font)
        description_label.setStyleSheet("color: #ecf0f1; margin-bottom: 10px;")  # Light color
        
        main_layout.addWidget(title_label)
        main_layout.addWidget(description_label)
        main_layout.addSpacing(20)
        
        # Create buttons layout
        buttons_layout = QtWidgets.QVBoxLayout()
        buttons_layout.setSpacing(20)  # Increased spacing between buttons
        
        # Range Profile button
        rp_button = self.create_app_button(
            "Range Profile",
            "1D radar processing to detect object distance. Displays signal power vs. range, "
            "allowing detection of objects at different distances. Useful for basic presence "
            "detection and simple distance measurements.",
            self.launch_range_profile
        )
        buttons_layout.addWidget(rp_button)
        
        # Range Doppler button (moved up before Range Angle as per user request)
        rd_button = self.create_app_button(
            "Range Doppler",
            "2D radar processing to detect object distance and velocity. Visualizes both "
            "the range and relative speed of objects. Useful for tracking moving targets "
            "and distinguishing between stationary and moving objects.",
            self.launch_range_doppler
        )
        buttons_layout.addWidget(rd_button)
        
        # Range Angle button
        ra_button = self.create_app_button(
            "Range Angle",
            "2D radar processing to detect object distance and angle. Creates a 2D heatmap "
            "showing object positions in range-azimuth space. Useful for locating multiple "
            "objects and understanding their spatial distribution.",
            self.launch_range_angle
        )
        buttons_layout.addWidget(ra_button)
        
        # Add buttons layout to main layout
        main_layout.addLayout(buttons_layout)
        main_layout.addStretch(1)
        
        # Add status bar
        self.status_bar = QtWidgets.QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        self.status_bar.setStyleSheet("""
            QStatusBar {
                color: #ecf0f1;
                background-color: #000000;
                border-top: 1px solid #333333;
                padding: 3px;
            }
        """)
        
        # Add exit button
        exit_button = QtWidgets.QPushButton("Exit")
        exit_button.clicked.connect(self.close)
        exit_button.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 10pt;
                font-family: 'Segoe UI';
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
            QPushButton:pressed {
                background-color: #a93226;
            }
        """)
        exit_button.setMaximumWidth(150)
        exit_button_layout = QtWidgets.QHBoxLayout()
        exit_button_layout.addStretch()
        exit_button_layout.addWidget(exit_button)
        exit_button_layout.addStretch()
        main_layout.addLayout(exit_button_layout)
    
    def set_dark_theme(self):
        """Set dark theme for the application."""
        dark_palette = QPalette()
        
        # Set color roles from dark to light - using black background as requested
        dark_palette.setColor(QPalette.Window, QColor(0, 0, 0))  # Pure black
        dark_palette.setColor(QPalette.WindowText, QColor(236, 240, 241))  # Light gray
        dark_palette.setColor(QPalette.Base, QColor(13, 13, 13))  # Very dark gray
        dark_palette.setColor(QPalette.AlternateBase, QColor(25, 25, 25))  # Dark gray
        dark_palette.setColor(QPalette.ToolTipBase, QColor(236, 240, 241))
        dark_palette.setColor(QPalette.ToolTipText, QColor(236, 240, 241))
        dark_palette.setColor(QPalette.Text, QColor(236, 240, 241))
        dark_palette.setColor(QPalette.Button, QColor(25, 25, 25))  # Dark gray
        dark_palette.setColor(QPalette.ButtonText, QColor(236, 240, 241))
        dark_palette.setColor(QPalette.BrightText, QColor(52, 152, 219))  # Bright blue
        dark_palette.setColor(QPalette.Link, QColor(52, 152, 219))
        dark_palette.setColor(QPalette.Highlight, QColor(52, 152, 219))
        dark_palette.setColor(QPalette.HighlightedText, QColor(236, 240, 241))
        
        # Apply the palette
        self.setPalette(dark_palette)
    
    def create_app_button(self, title, description, callback):
        """Create a styled button for an application."""
        button = QtWidgets.QPushButton()
        button.setMinimumHeight(140)  # Further increased height for longer descriptions
        
        # Create button with black background as requested
        button.setStyleSheet("""
            QPushButton {
                background-color: #1a1a1a;
                color: white;
                border: 1px solid #333333;
                border-radius: 10px;
                text-align: left;
            }
            QPushButton:hover {
                background-color: #808080;  /* Gray hover color */
                border: 1px solid #666666;
            }
            QPushButton:pressed {
                background-color: #0a0a0a;
            }
        """)
        
        # Create layout for button content
        button_layout = QtWidgets.QHBoxLayout(button)
        button_layout.setContentsMargins(15, 15, 15, 15)
        
        # Add icon
        icon_label = QtWidgets.QLabel()
        icon_label.setFixedSize(50, 50)
        
        # Create a colored icon based on the button type
        if title == "Range Profile":
            color = QColor("#3498db")  # Blue
        elif title == "Range Angle":
            color = QColor("#2ecc71")  # Green
        elif title == "Range Doppler":
            color = QColor("#e74c3c")  # Red
        else:
            color = QColor("#9b59b6")  # Purple
        
        # Create a pixmap for the icon
        pixmap = QtGui.QPixmap(50, 50)
        pixmap.fill(QtCore.Qt.transparent)
        
        # Draw the icon
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        
        # Draw a filled circle
        painter.setBrush(QtGui.QBrush(color))
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawEllipse(5, 5, 40, 40)
        
        # Draw an icon or text based on the button type
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255)))
        font = QtGui.QFont("Arial", 20, QtGui.QFont.Bold)
        painter.setFont(font)
        
        # Draw different icons based on the button type
        if title == "Range Profile":
            # Draw a line chart icon
            painter.drawLine(15, 35, 20, 25)
            painter.drawLine(20, 25, 25, 30)
            painter.drawLine(25, 30, 30, 15)
            painter.drawLine(30, 15, 35, 20)
        elif title == "Range Angle":
            # Draw a radar-like icon
            painter.drawEllipse(20, 20, 15, 15)
            painter.drawLine(25, 25, 35, 15)
            painter.drawLine(25, 25, 15, 15)
        elif title == "Range Doppler":
            # Draw a speedometer-like icon
            painter.drawArc(15, 15, 25, 25, 30 * 16, 120 * 16)
            painter.drawLine(25, 25, 35, 20)
        else:
            # Draw the first letter of the title
            painter.drawText(pixmap.rect(), QtCore.Qt.AlignCenter, title[0])
        
        painter.end()
        
        # Set the pixmap to the label
        icon_label.setPixmap(pixmap)
        button_layout.addWidget(icon_label)
        
        # Create text layout with more space for description
        text_layout = QtWidgets.QVBoxLayout()
        text_layout.setSpacing(10)  # Increased spacing between title and description
        
        # Add title and description to button
        title_label = QtWidgets.QLabel(title)
        title_font = QtGui.QFont("Segoe UI", 12)  # Modern font
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setStyleSheet(f"color: {color.name()}; border: none;")
        
        desc_label = QtWidgets.QLabel(description)
        desc_font = QtGui.QFont("Segoe UI", 9)  # Modern font
        desc_label.setFont(desc_font)
        desc_label.setWordWrap(True)  # Enable word wrapping for longer descriptions
        desc_label.setStyleSheet("color: #ecf0f1; border: none;")  # Light color for description
        
        # Set maximum width for description to ensure proper wrapping
        desc_label.setMaximumWidth(450)
        
        text_layout.addWidget(title_label)
        text_layout.addWidget(desc_label)
        button_layout.addLayout(text_layout)
        
        # Connect button click to callback
        button.clicked.connect(callback)
        
        return button
    
    def center_on_screen(self):
        """Center the window on the screen."""
        screen_geometry = QtWidgets.QDesktopWidget().availableGeometry()
        window_geometry = self.geometry()
        
        x = (screen_geometry.width() - window_geometry.width()) // 2
        y = (screen_geometry.height() - window_geometry.height()) // 2
        
        self.move(x, y)
    
    def launch_application(self, app_path, app_name):
        """Launch an application and handle errors."""
        try:
            # Check if an application is already running
            if self.current_app_process is not None:
                # Terminate the currently running application
                try:
                    self.status_bar.showMessage(f"Closing {self.current_app_name}...")
                    self.current_app_process.terminate()
                    # Give it a moment to terminate
                    self.current_app_process.wait(timeout=2)
                    self.status_bar.showMessage(f"{self.current_app_name} closed successfully")
                except Exception as e:
                    logger.error(f"Error closing {self.current_app_name}: {str(e)}")
                    # Continue anyway to launch the new application
            
            self.status_bar.showMessage(f"Launching {app_name}...")
            
            # Get the absolute path to the application
            script_dir = os.path.dirname(os.path.abspath(__file__))
            app_full_path = os.path.join(script_dir, app_path)
            
            # Check if the file exists
            if not os.path.exists(app_full_path):
                raise FileNotFoundError(f"Application file not found: {app_full_path}")
            
            # Launch the application
            if getattr(sys, 'frozen', False):
                # If running as a bundled executable
                python_executable = sys.executable
                self.current_app_process = subprocess.Popen([python_executable, app_full_path])
            else:
                # If running as a script
                self.current_app_process = subprocess.Popen([sys.executable, app_full_path])
            
            # Update the current app name
            self.current_app_name = app_name
            
            self.status_bar.showMessage(f"{app_name} launched successfully")
            
        except Exception as e:
            error_message = f"Error launching {app_name}: {str(e)}"
            logger.error(error_message)
            self.status_bar.showMessage(error_message)
            
            # Show error dialog
            error_dialog = QtWidgets.QMessageBox()
            error_dialog.setIcon(QtWidgets.QMessageBox.Critical)
            error_dialog.setWindowTitle("Launch Error")
            error_dialog.setText(error_message)
            error_dialog.setStandardButtons(QtWidgets.QMessageBox.Ok)
            error_dialog.exec_()
    
    def launch_range_profile(self):
        """Launch the Range Profile application."""
        self.launch_application("Range Profile/rp_main.py", "Range Profile")
    
    def launch_range_angle(self):
        """Launch the Range Angle application."""
        self.launch_application("Range Angle/ra_main.py", "Range Angle")
    
    def launch_range_doppler(self):
        """Launch the Range Doppler application."""
        self.launch_application("Range Doppler/rd_main.py", "Range Doppler")
    
    def closeEvent(self, event):
        """Handle window close event."""
        # First check if our tracked process is still running
        if self.current_app_process is not None:
            try:
                if self.current_app_process.poll() is None:  # None means it's still running
                    # Add it to our children list to be terminated
                    children = [self.current_app_process]
                else:
                    children = []
            except Exception as e:
                logger.error(f"Error checking current app process: {e}")
                children = []
        else:
            children = []
        
        # Also check for any other child processes that might be running
        # This is a backup in case our tracking mechanism missed something
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    # Check if this is a Python process that might be one of our applications
                    if proc.info['name'] == 'python.exe' or proc.info['name'] == 'pythonw.exe':
                        cmdline = proc.info['cmdline']
                        if cmdline and any(app in ' '.join(cmdline) for app in 
                                          ['rp_main.py', 'ra_main.py', 'rd_main.py']):
                            # Check if this is not our already tracked process
                            if self.current_app_process is None or proc.pid != self.current_app_process.pid:
                                children.append(proc)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception as e:
            logger.error(f"Error checking for child processes: {e}")
        
        # If child processes are running, ask for confirmation
        if children:
            confirm_dialog = QtWidgets.QMessageBox()
            confirm_dialog.setIcon(QtWidgets.QMessageBox.Question)
            confirm_dialog.setWindowTitle("Confirm Exit")
            confirm_dialog.setText("One or more radar applications are still running.")
            confirm_dialog.setInformativeText("Do you want to close them and exit?")
            confirm_dialog.setStandardButtons(
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
            )
            confirm_dialog.setDefaultButton(QtWidgets.QMessageBox.No)
            
            if confirm_dialog.exec_() == QtWidgets.QMessageBox.Yes:
                # Terminate child processes
                for proc in children:
                    try:
                        proc.terminate()
                        logger.info(f"Terminated process {proc.pid}")
                    except Exception as e:
                        logger.error(f"Error terminating process {proc.pid}: {e}")
                
                # Accept the close event
                event.accept()
            else:
                # Reject the close event
                event.ignore()
        else:
            # No child processes, accept the close event
            event.accept()


def main():
    """Main entry point for the launcher application."""
    # Create the application
    app = QtWidgets.QApplication(sys.argv)
    
    # Set application style
    app.setStyle("Fusion")
    
    # Create and show the main window
    main_window = LauncherWindow()
    main_window.show()
    
    # Run the application
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
