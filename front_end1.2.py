import sys
import cv2
import numpy as np
import mediapipe as mp
from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer
import design  # Import the generated design.py file
from xgboost import XGBClassifier

class ISLConverter(QMainWindow, design.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)  # Load UI

        # ✅ Load XGBoost Model
        self.model = XGBClassifier()
        self.model.load_model("ISLClassifier.json")  # ✅ Ensure correct model loading

        # ✅ Define Class Mapping (0-25 → A-Z)
        self.label_map = {
            0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
            9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R',
            18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
        }

        # ✅ Initialize MediaPipe Hands (Allow 2 Hands)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

        # ✅ Set up camera
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.btn_start.clicked.connect(self.toggle_camera)
        self.camera_active = False

    def toggle_camera(self):
        """Starts or stops the webcam feed"""
        if self.camera_active:
            self.timer.stop()
            if self.cap:
                self.cap.release()
            self.label_camera.clear()  # Clear QLabel
            self.btn_start.setText("Start Camera")
        else:
            self.cap = cv2.VideoCapture(0)  # ✅ Open webcam safely
            if not self.cap.isOpened():
                self.textEdit_result.setPlainText("Error: Cannot access webcam")
                return
            self.timer.start(20)  # Update every 20ms
            self.btn_start.setText("Stop Camera")
        self.camera_active = not self.camera_active

    def update_frame(self):
        """Processes the webcam frame and predicts the sign"""
        ret, frame = self.cap.read()
        if not ret:
            return  # Skip if no frame

        # ✅ Flip frame horizontally for better UX
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ✅ Detect hands using MediaPipe
        results = self.hands.process(frame_rgb)

        hand_landmarks_list = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                hand_landmarks_list.append(landmarks)

        # ✅ Handle cases: 0, 1, or 2 hands detected
        if len(hand_landmarks_list) == 0:
            hand_input = np.zeros((1, 127))  # No hands, use zero-filled placeholder
        elif len(hand_landmarks_list) == 1:
            hand_input = np.array(hand_landmarks_list[0] + [0] * 63).reshape(1, -1)  # Pad with zeros
        else:
            hand_input = np.array(hand_landmarks_list[0] + hand_landmarks_list[1]).reshape(1, -1)

        # ✅ Ensure Feature Shape Matches Model (127 features)
        if hand_input.shape[1] == 126:
            hand_input = np.append(hand_input, 0).reshape(1, -1)  # Append a dummy feature

        # ✅ Perform Prediction
        if hand_input.shape[1] == 127:
            prediction = self.model.predict(hand_input)
            predicted_label = self.label_map.get(int(prediction[0]), "Unknown")
            self.textEdit_result.setPlainText(f"Predicted Sign: {predicted_label}")

        # ✅ Convert to RGB for PyQt display
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        # ✅ Display in QLabel
        self.label_camera.setPixmap(QPixmap.fromImage(qt_image))

    def closeEvent(self, event):
        """Ensure webcam is properly released on app close"""
        if self.cap:
            self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ISLConverter()
    window.show()
    sys.exit(app.exec())
