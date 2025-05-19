import cv2
import numpy as np
import json
from pathlib import Path
from typing import List, Tuple, Dict
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                            QMessageBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen
import sys

class ColorCorrectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.source_img = None
        self.target_img = None
        self.source_rois = []
        self.target_rois = []
        self.current_roi = []
        self.drawing = False
        self.current_image = None
        self.current_window = None
        self.color_pairs = []
        self.temp_pixmap = None
        # For mapping mouse to image coordinates
        self.source_display_info = None  # (scale, x_offset, y_offset)
        self.target_display_info = None
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('Color Correction Tool')
        self.setGeometry(100, 100, 1200, 600)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create image display area
        image_layout = QHBoxLayout()
        
        # Source image display
        self.source_label = QLabel()
        self.source_label.setMinimumSize(500, 400)
        self.source_label.setAlignment(Qt.AlignCenter)
        self.source_label.setText("Source Image\n(Click to load)")
        self.source_label.setStyleSheet("border: 2px dashed gray;")
        self.source_label.mousePressEvent = self.source_image_clicked
        
        # Target image display
        self.target_label = QLabel()
        self.target_label.setMinimumSize(500, 400)
        self.target_label.setAlignment(Qt.AlignCenter)
        self.target_label.setText("Target Image\n(Click to load)")
        self.target_label.setStyleSheet("border: 2px dashed gray;")
        self.target_label.mousePressEvent = self.target_image_clicked
        
        image_layout.addWidget(self.source_label)
        image_layout.addWidget(self.target_label)
        layout.addLayout(image_layout)
        
        # Create buttons
        button_layout = QHBoxLayout()
        
        self.process_button = QPushButton('Process Images')
        self.process_button.clicked.connect(self.process_images)
        self.process_button.setEnabled(False)
        
        self.next_button = QPushButton('Next Pair')
        self.next_button.clicked.connect(self.next_pair)
        self.next_button.setEnabled(False)
        
        self.save_matrix_button = QPushButton('Save Matrix')
        self.save_matrix_button.clicked.connect(self.save_matrix_dialog)
        self.save_matrix_button.setEnabled(False)

        self.load_matrix_button = QPushButton('Load Matrix')
        self.load_matrix_button.clicked.connect(self.load_matrix_dialog)
        self.load_matrix_button.setEnabled(True)

        self.apply_matrix_button = QPushButton('Apply Correction')
        self.apply_matrix_button.clicked.connect(self.apply_matrix_dialog)
        self.apply_matrix_button.setEnabled(True)

        button_layout.addWidget(self.process_button)
        button_layout.addWidget(self.next_button)
        button_layout.addWidget(self.save_matrix_button)
        button_layout.addWidget(self.load_matrix_button)
        button_layout.addWidget(self.apply_matrix_button)
        layout.addLayout(button_layout)
        
        # Status label
        self.status_label = QLabel()
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        # Set up timer for ROI drawing
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_roi_drawing)
        self.timer.start(30)  # 30ms refresh rate
        
    def source_image_clicked(self, event):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Source Image", "", 
            "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_name:
            self.source_img = cv2.imread(file_name)
            if self.source_img is not None:
                self.display_image(self.source_img, self.source_label)
                self.check_images_loaded()
            else:
                QMessageBox.warning(self, "Error", "Failed to load source image")
    
    def target_image_clicked(self, event):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Target Image", "", 
            "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_name:
            self.target_img = cv2.imread(file_name)
            if self.target_img is not None:
                self.display_image(self.target_img, self.target_label)
                self.check_images_loaded()
            else:
                QMessageBox.warning(self, "Error", "Failed to load target image")
    
    def display_image(self, img, label):
        # Convert OpenCV image to QImage
        height, width, channel = img.shape
        bytes_per_line = 3 * width
        q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

        label_width = label.width()
        label_height = label.height()
        scale = min(label_width / width, label_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        x_offset = (label_width - new_width) // 2
        y_offset = (label_height - new_height) // 2

        # Create a blank pixmap and fill with a background color
        pixmap = QPixmap(label_width, label_height)
        pixmap.fill(Qt.darkGray)  # or Qt.black, or any color you like

        # Draw the scaled image onto the blank pixmap
        painter = QPainter(pixmap)
        scaled_img = QPixmap.fromImage(q_img).scaled(
            new_width, new_height, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        painter.drawPixmap(x_offset, y_offset, scaled_img)
        painter.end()

        label.setPixmap(pixmap)

        # Store display info for coordinate mapping
        if label == self.source_label:
            self.source_display_info = (scale, x_offset, y_offset, width, height)
        elif label == self.target_label:
            self.target_display_info = (scale, x_offset, y_offset, width, height)
    
    def check_images_loaded(self):
        if self.source_img is not None and self.target_img is not None:
            self.process_button.setEnabled(True)
            self.status_label.setText("Images loaded. Click 'Process Images' to start selecting ROIs.")
    
    def process_images(self):
        self.process_button.setEnabled(False)
        self.next_button.setEnabled(True)
        self.status_label.setText("Draw ROIs on both images by clicking and dragging")
        # Enable ROI drawing with correct label context
        self.source_label.mousePressEvent = lambda event: self.start_roi(event, self.source_label)
        self.source_label.mouseMoveEvent = lambda event: self.update_roi(event, self.source_label)
        self.source_label.mouseReleaseEvent = lambda event: self.end_roi(event, self.source_label)
        self.target_label.mousePressEvent = lambda event: self.start_roi(event, self.target_label)
        self.target_label.mouseMoveEvent = lambda event: self.update_roi(event, self.target_label)
        self.target_label.mouseReleaseEvent = lambda event: self.end_roi(event, self.target_label)
    
    def start_roi(self, event, label):
        self.drawing = True
        x, y = self.map_to_image_coords(label, event.x(), event.y())
        self.current_roi = [(x, y)]
        self.current_image = label
    
    def update_roi(self, event, label):
        if self.drawing:
            x, y = self.map_to_image_coords(label, event.x(), event.y())
            if len(self.current_roi) == 1:
                self.current_roi.append((x, y))
            else:
                self.current_roi[-1] = (x, y)
    
    def end_roi(self, event, label):
        if self.drawing:
            self.drawing = False
            x, y = self.map_to_image_coords(label, event.x(), event.y())
            if len(self.current_roi) == 1:
                self.current_roi.append((x, y))
            else:
                self.current_roi[-1] = (x, y)
            if len(self.current_roi) >= 2:
                if self.current_image == self.source_label:
                    self.source_rois.append(self.current_roi.copy())
                    self.show_roi_stats(self.source_img, self.current_roi, 'Source')
                else:
                    self.target_rois.append(self.current_roi.copy())
                    self.show_roi_stats(self.target_img, self.current_roi, 'Target')
            self.current_roi = []
    
    def show_roi_stats(self, img, roi, window_name):
        x1, y1 = roi[0]
        x2, y2 = roi[1]
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        roi_img = img[y1:y2, x1:x2]
        avg_color = np.mean(roi_img, axis=(0, 1))
        self.status_label.setText(f"{window_name} ROI {len(self.source_rois)} - Average RGB: {avg_color}")
    
    def next_pair(self):
        if len(self.source_rois) == len(self.target_rois) and len(self.source_rois) > 0:
            matrix = self.calculate_color_correction_matrix()
            self.save_correction_matrix(matrix, 'color_correction_matrix.npy')
            self.save_roi_data('roi_data.json')
            self.current_matrix = matrix
            self.save_matrix_button.setEnabled(True)
            self.status_label.setText('Matrix and ROI data saved. You can now save or apply the matrix.')
            reply = QMessageBox.question(
                self, 'Success', 
                'Matrix and ROI data saved. Process another pair?',
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.reset_for_next_pair()
            else:
                self.close()
        else:
            QMessageBox.warning(
                self, 'Warning',
                'Please draw the same number of ROIs on both images'
            )
    
    def reset_for_next_pair(self):
        self.source_img = None
        self.target_img = None
        self.source_rois = []
        self.target_rois = []
        self.source_label.clear()
        self.target_label.clear()
        self.source_label.setText("Source Image\n(Click to load)")
        self.target_label.setText("Target Image\n(Click to load)")
        self.process_button.setEnabled(False)
        self.next_button.setEnabled(False)
        self.status_label.setText("Load new image pair to continue")
    
    def calculate_color_correction_matrix(self) -> np.ndarray:
        if len(self.source_rois) != len(self.target_rois):
            raise ValueError("Number of ROIs must match between source and target images")
        
        source_colors = []
        target_colors = []
        
        for source_roi, target_roi in zip(self.source_rois, self.target_rois):
            # Calculate average colors for source ROI
            x1, y1 = source_roi[0]
            x2, y2 = source_roi[1]
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            source_avg = np.mean(self.source_img[y1:y2, x1:x2], axis=(0, 1))
            
            # Calculate average colors for target ROI
            x1, y1 = target_roi[0]
            x2, y2 = target_roi[1]
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            target_avg = np.mean(self.target_img[y1:y2, x1:x2], axis=(0, 1))
            
            source_colors.append(source_avg)
            target_colors.append(target_avg)
        
        # Convert to numpy arrays
        source_colors = np.array(source_colors)
        target_colors = np.array(target_colors)
        
        # Solve least squares problem: target_colors * matrix = source_colors
        matrix, residuals, rank, s = np.linalg.lstsq(target_colors, source_colors, rcond=None)
        
        return matrix.T  # Transpose to get 3x3 matrix
    
    def save_correction_matrix(self, matrix: np.ndarray, output_path: str):
        np.save(output_path, matrix)
    
    def save_roi_data(self, output_path: str):
        data = {
            'source_rois': self.source_rois,
            'target_rois': self.target_rois,
            'source_colors': [np.mean(self.source_img[min(y1,y2):max(y1,y2), min(x1,x2):max(x1,x2)], axis=(0,1)).tolist() 
                            for (x1,y1), (x2,y2) in self.source_rois],
            'target_colors': [np.mean(self.target_img[min(y1,y2):max(y1,y2), min(x1,x2):max(x1,x2)], axis=(0,1)).tolist() 
                            for (x1,y1), (x2,y2) in self.target_rois]
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

    def map_to_image_coords(self, label, x, y):
        if label == self.source_label:
            info = self.source_display_info
        else:
            info = self.target_display_info
        if info is None:
            return 0, 0
        scale, x_offset, y_offset, img_w, img_h = info
        # Remove offset, then scale
        x_img = int((x - x_offset) / scale)
        y_img = int((y - y_offset) / scale)
        # Clamp to image bounds
        x_img = max(0, min(img_w - 1, x_img))
        y_img = max(0, min(img_h - 1, y_img))
        return x_img, y_img

    def update_roi_drawing(self):
        # Redraw image with current ROI rectangle if drawing
        for label, img, rois, display_info in [
            (self.source_label, self.source_img, self.source_rois, self.source_display_info),
            (self.target_label, self.target_img, self.target_rois, self.target_display_info)
        ]:
            if img is not None:
                # Start with base image
                img_disp = img.copy()
                # Draw all existing ROIs
                for roi in rois:
                    x1, y1 = roi[0]
                    x2, y2 = roi[1]
                    cv2.rectangle(img_disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Draw current ROI if drawing on this label
                if self.drawing and self.current_image == label and len(self.current_roi) == 2:
                    x1, y1 = self.current_roi[0]
                    x2, y2 = self.current_roi[1]
                    cv2.rectangle(img_disp, (x1, y1), (x2, y2), (0, 0, 255), 2)
                self.display_image(img_disp, label)

    def save_matrix_dialog(self):
        if not hasattr(self, 'current_matrix') or self.current_matrix is None:
            QMessageBox.warning(self, 'Warning', 'No matrix to save!')
            return
        file_name, _ = QFileDialog.getSaveFileName(self, 'Save Matrix', '', 'NumPy Matrix (*.npy)')
        if file_name:
            np.save(file_name, self.current_matrix)
            QMessageBox.information(self, 'Saved', f'Matrix saved to {file_name}')

    def load_matrix_dialog(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Load Matrix', '', 'NumPy Matrix (*.npy)')
        if file_name:
            try:
                matrix = np.load(file_name)
                if matrix.shape == (3, 3):
                    self.current_matrix = matrix
                    self.save_matrix_button.setEnabled(True)
                    QMessageBox.information(self, 'Loaded', f'Matrix loaded from {file_name}')
                else:
                    QMessageBox.warning(self, 'Error', 'Selected file is not a valid 3x3 matrix!')
            except Exception as e:
                QMessageBox.warning(self, 'Error', f'Failed to load matrix: {e}')

    def apply_matrix_dialog(self):
        if not hasattr(self, 'current_matrix') or self.current_matrix is None:
            QMessageBox.warning(self, 'Warning', 'No matrix loaded!')
            return
        file_name, _ = QFileDialog.getOpenFileName(self, 'Select Image to Correct', '', 'Image Files (*.png *.jpg *.jpeg *.bmp)')
        if file_name:
            img = cv2.imread(file_name)
            if img is None:
                QMessageBox.warning(self, 'Error', 'Failed to load image!')
                return
            img_corr = self.apply_color_correction(img, self.current_matrix)
            # Show result in a new window
            self.show_corrected_image(img_corr)
            # Ask to save
            reply = QMessageBox.question(self, 'Save Corrected Image?', 'Save the corrected image?', QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                save_name, _ = QFileDialog.getSaveFileName(self, 'Save Corrected Image', '', 'Image Files (*.png *.jpg *.jpeg *.bmp)')
                if save_name:
                    cv2.imwrite(save_name, img_corr)
                    QMessageBox.information(self, 'Saved', f'Corrected image saved to {save_name}')

    def apply_color_correction(self, img, matrix):
        img_flat = img.reshape(-1, 3).astype(np.float32)
        img_corr = np.dot(img_flat, matrix.T)
        img_corr = np.clip(img_corr, 0, 255).astype(np.uint8)
        return img_corr.reshape(img.shape)

    def show_corrected_image(self, img):
        # Show corrected image in a new window
        height, width, channel = img.shape
        bytes_per_line = 3 * width
        q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        win = QMainWindow(self)
        win.setWindowTitle('Corrected Image Preview')
        label = QLabel()
        label.setPixmap(QPixmap.fromImage(q_img).scaled(600, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        win.setCentralWidget(label)
        win.resize(620, 440)
        win.show()

def main():
    app = QApplication(sys.argv)
    window = ColorCorrectionApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 