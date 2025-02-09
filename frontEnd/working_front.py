import sys
import cv2
import numpy as np
import mediapipe as mp
import joblib
from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer
import design  # Import the generated design.py file

class ISLConverter(QMainWindow, design.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)  # Load UI

        # Load the trained RandomForest model
        self.model = joblib.load("random_forest_model.pkl")

        # Class index-to-label mapping
        self.label_map = {
            0: '1', 1: '2', 2: '3', 3: '4', 4: '5', 5: '6', 6: '7', 7: '8', 8: '9',
            9: 'A', 10: 'B', 11: 'C', 12: 'D', 13: 'E', 14: 'F', 15: 'G', 16: 'H', 17: 'I',
            18: 'J', 19: 'K', 20: 'L', 21: 'M', 22: 'N', 23: 'O', 24: 'P', 25: 'Q', 26: 'R',
            27: 'S', 28: 'T', 29: 'U', 30: 'V', 31: 'W', 32: 'X', 33: 'Y', 34: 'Z'
        }

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

        # Set up camera
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.btn_start.clicked.connect(self.toggle_camera)
        self.camera_active = False

    def toggle_camera(self):
        if self.camera_active:
            self.timer.stop()
            if self.cap:
                self.cap.release()
            self.label_camera.clear()  # Clear QLabel
            self.btn_start.setText("Start Camera")
        else:
            self.cap = cv2.VideoCapture(0)  # Open webcam
            self.timer.start(20)  # Start timer to update the frame every 30ms
            self.btn_start.setText("Stop Camera")
        self.camera_active = not self.camera_active

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Flip the frame horizontally
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect hands using MediaPipe
            results = self.hands.process(frame_rgb)

            hand_landmarks_list = []

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])
                    hand_landmarks_list.append(landmarks)

            # Handle cases with 0, 1, or 2 hands detected
            if len(hand_landmarks_list) == 0:
                # No hands detected, use zeros as placeholders
                hand_input = np.zeros((1, 126))
            elif len(hand_landmarks_list) == 1:
                # One hand detected, pad with zeros
                hand_input = np.array(hand_landmarks_list[0] + [0] * 63).reshape(1, -1)
            else:
                # Two hands detected, concatenate features
                hand_input = np.array(hand_landmarks_list[0] + hand_landmarks_list[1]).reshape(1, -1)

            # Ensure the input size is correct before prediction
            if hand_input.shape[1] == 126:
                prediction = self.model.predict(hand_input)
                predicted_label = self.label_map.get(prediction[0], "Unknown")
                self.textEdit_result.setPlainText(f"Predicted Sign: {predicted_label}")

            # Convert to RGB for display in QLabel
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            
            # Display in QLabel
            self.label_camera.setPixmap(QPixmap.fromImage(qt_image))

    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ISLConverter()
    window.show()
    sys.exit(app.exec())
