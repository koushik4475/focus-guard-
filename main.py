import cv2
import numpy as np
import time
import os
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List
import multiprocessing as mp_core
from queue import Empty
import gtts
import pygame
import mediapipe as mp
from scipy.spatial import distance as dist
import shutil # For cleanup
import sys # <<<--- NEW: Required for PyQt Application
import random # <<<--- NEW: Required for PyQt Animation
import math # <<<--- NEW: Required for PyQt Animation (and good practice)

# --- Enhanced Configuration ---
EAR_THRESHOLD = 0.22
EAR_CONSEC_FRAMES = 8
NOSE_OFFSET_THRESH = 0.04
LOOK_CONSEC_FRAMES = 5

# Advanced Phone Detection
PHONE_DETECTION_METHODS = 4  # Use multiple detection methods
PROXIMITY_PIX = 60
GRIP_THRESH = 85
PHONE_WARNING_THRESHOLD = 2
SAVE_COOLDOWN = 10.0

# Alert System
ALERT_COOLDOWN_CONTINUOUS = 2.0
DETECTION_CONSEC_FRAMES = 3
DEBUG = True
WINDOW_NAME = 'FOCUS GUARD BY VISIONEXUS' # Define a constant for the window name

# Animation Settings
ALERT_ANIMATION_DURATION = 1.5
PULSE_SPEED = 0.1

# --- MediaPipe Setup & Constants ---
mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [263, 387, 385, 362, 380, 373]
NOSE_IDX = 1

# Hand landmarks for phone detection
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20
WRIST = 0

# =================================================================
# 🎨 MODERN LOADING SCREEN COMPONENTS (PyQt6)
# =================================================================
# NOTE: These components will only be used for the initial loading screen

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QLabel, QProgressBar, QGraphicsDropShadowEffect)
from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, pyqtProperty, QPointF, QRectF
from PyQt6.QtGui import (QPainter, QColor, QLinearGradient, QRadialGradient)

# <<<--- THIS CLASS CONTAINS THE PARTICLE ANIMATION --->>>
class AnimatedBackground(QWidget):
    """Animated background with moving, glowing particles."""
    def __init__(self):
        super().__init__()
        self.particles = []
        self.time = 0
        self.initParticles()
        
    def initParticles(self):
        """Initializes 120 particles with random properties."""
        self.particles = []
        for _ in range(120):
            particle = {
                'x': random.uniform(0, 1), 
                'y': random.uniform(0, 1), 
                'size': random.uniform(2, 5), 
                'speed': random.uniform(0.2, 1.0), 
                # Theme-matched blue/cyan colors for the particles
                'color': QColor( 
                    random.randint(80, 150),
                    random.randint(100, 180),
                    random.randint(200, 255),
                    random.randint(80, 150)
                ),
                'angle': random.uniform(0, 2 * math.pi) 
            }
            self.particles.append(particle)
    
    def updateParticles(self):
        """Updates the position and size of each particle for animation."""
        self.time += 0.016
        for p in self.particles:
            p['angle'] += 0.02 * p['speed']
            p['x'] = 0.5 + 0.4 * math.cos(p['angle'] + self.time * p['speed'])
            p['y'] = 0.5 + 0.4 * math.sin(p['angle'] + self.time * p['speed'])
            p['size'] = 2 + 2 * math.sin(self.time * 2 + p['angle'])
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # --- Pitch Black Background (Ensures full coverage) ---
        painter.fillRect(self.rect(), QColor(0, 0, 0)) # Solid black background
        
        # Draw particles
        for p in self.particles:
            x = p['x'] * self.width() 
            y = p['y'] * self.height() 
            size = p['size']
            
            # Draw glow effect 
            radial = QRadialGradient(x, y, size * 3)
            radial.setColorAt(0, p['color'])
            radial.setColorAt(1, QColor(0, 0, 0, 0))
            painter.setBrush(radial)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(x, y), size * 3, size * 3)
            
            # Draw main particle
            painter.setBrush(p['color'])
            painter.drawEllipse(QPointF(x, y), size, size)

class FloatingOrb(QWidget):
    """Central, pulsating, and rotating orb."""
    def __init__(self):
        super().__init__()
        self._pulse = 0
        self._rotation = 0
        self.setFixedSize(200, 200)
        
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(50)
        # Theme-matched blue glow
        shadow.setColor(QColor(100, 150, 255, 150)) 
        shadow.setOffset(0, 0)
        self.setGraphicsEffect(shadow)
        
        self.pulse_animation = QPropertyAnimation(self, b"pulse")
        self.pulse_animation.setDuration(2000)
        self.pulse_animation.setLoopCount(-1)
        self.pulse_animation.setStartValue(0)
        self.pulse_animation.setEndValue(1)
        self.pulse_animation.setEasingCurve(QEasingCurve.Type.InOutSine)
        self.pulse_animation.start()
        
        self.rotation_animation = QPropertyAnimation(self, b"rotation")
        self.rotation_animation.setDuration(8000)
        self.rotation_animation.setLoopCount(-1)
        self.rotation_animation.setStartValue(0)
        self.rotation_animation.setEndValue(360)
        self.rotation_animation.start()
    
    @pyqtProperty(float)
    def pulse(self):
        return self._pulse
    
    @pulse.setter
    def pulse(self, value):
        self._pulse = value
        self.update() 
    
    @pyqtProperty(float)
    def rotation(self):
        return self._rotation
    
    @rotation.setter
    def rotation(self, value):
        self._rotation = value
        self.update() 
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        center = QPointF(self.width() / 2, self.height() / 2)
        
        # Outer glow
        radial = QRadialGradient(center, 100)
        radial.setColorAt(0, QColor(100, 150, 255, 100))
        radial.setColorAt(1, QColor(0, 0, 0, 0))
        painter.setBrush(radial)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(center, 100, 100)
      
        # Main orb with pulse effect
        pulse_size = 50 + 15 * math.sin(self._pulse * 2 * math.pi)
        radial = QRadialGradient(center, pulse_size)
        radial.setColorAt(0, QColor(255, 255, 255, 200))
        radial.setColorAt(0.5, QColor(150, 200, 255, 150))
        radial.setColorAt(1, QColor(100, 150, 255, 0))
        
        painter.setBrush(radial)
        painter.drawEllipse(center, pulse_size, pulse_size)
        
        # Rotating ring of dots
        painter.save()
        painter.translate(center)
        painter.rotate(self._rotation)
        
        ring_radius = 70
        for i in range(8):
            angle = i * 45
            rad_angle = math.radians(angle)
            x = ring_radius * math.cos(rad_angle)
            y = ring_radius * math.sin(rad_angle)
            
            dot_size = 3 + 2 * math.sin(self._pulse * 2 * math.pi + i * 0.5)
            painter.setBrush(QColor(200, 220, 255))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(x, y), dot_size, dot_size)
        
        painter.restore()

class ModernProgressBar(QProgressBar):
    """Sleek progress bar with gradient fill and animated glow."""
    def __init__(self):
        super().__init__()
        self.setFixedHeight(10)
        self.setTextVisible(False)
        self._glow_position = 0
        
        self.glow_animation = QPropertyAnimation(self, b"glow_position")
        self.glow_animation.setDuration(1500)
        self.glow_animation.setLoopCount(-1)
        self.glow_animation.setStartValue(0)
        self.glow_animation.setEndValue(1)
        self.glow_animation.start()
    
    @pyqtProperty(float)
    def glow_position(self):
        return self._glow_position
    
    @glow_position.setter
    def glow_position(self, value):
        self._glow_position = value
        self.update() 
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Background
        painter.setBrush(QColor(40, 45, 60))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(self.rect(), 5, 5)
        
        # Progress fill with gradient
        if self.value() > 0:
            progress_width = (self.value() / self.maximum()) * self.width()
            progress_rect = QRectF(0, 0, progress_width, self.height())
            
            gradient = QLinearGradient(0, 0, progress_width, 0)
            # Theme-matched blue gradient
            gradient.setColorAt(0, QColor(100, 150, 255))
            gradient.setColorAt(0.5, QColor(150, 200, 255))
            gradient.setColorAt(1, QColor(100, 150, 255))
            
            painter.setBrush(gradient)
            painter.drawRoundedRect(progress_rect, 5, 5)
            
            # Animated glow
            glow_width = 30
            glow_pos = self._glow_position * (progress_width + glow_width) - glow_width
            if glow_pos < progress_width:
                glow_rect = QRectF(glow_pos, 0, glow_width, self.height())
                glow_gradient = QLinearGradient(glow_rect.left(), 0, glow_rect.right(), 0)
                glow_gradient.setColorAt(0, QColor(255, 255, 255, 0))
                glow_gradient.setColorAt(0.5, QColor(255, 255, 255, 100))
                glow_gradient.setColorAt(1, QColor(255, 255, 255, 0))
                
                painter.setBrush(glow_gradient)
                painter.drawRoundedRect(glow_rect, 5, 5)
        
        # Border
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(self.rect(), 5, 5)

class ModernLoadingWindow(QMainWindow):
    """The main frameless window containing the loading screen elements."""
    def __init__(self, main_app_ref, capture_ready_event):
        super().__init__()
        # Event to check if the background worker (camera) is ready
        self.capture_ready_event = capture_ready_event 
        
        self.setWindowTitle("Focus Guard - Loading")
        self.setFixedSize(1000, 700)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        
        # SOLID BLACK BACKGROUND FIX: Ensures content behind is hidden
        self.setStyleSheet("background-color: black;")

        self.setupUI()
        self.setupAnimations()
        
    def setupUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        self.background = AnimatedBackground()
        layout.addWidget(self.background)
       
        overlay = QWidget()
        overlay.setStyleSheet("background: transparent;")
        overlay_layout = QVBoxLayout(overlay)
        overlay_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        overlay_layout.setSpacing(40)
        
        # Title
        title = QLabel("FOCUS GUARD")
        title.setStyleSheet("QLabel {color: #ffffff; font-size: 48px; font-weight: bold; font-family: 'Segoe UI', Arial, sans-serif; background: transparent; letter-spacing: 2px;}")
        title_shadow = QGraphicsDropShadowEffect()
        title_shadow.setBlurRadius(20)
        title_shadow.setColor(QColor(100, 150, 255, 150))
        title.setGraphicsEffect(title_shadow)
        
        subtitle = QLabel("Advanced Driver Monitoring System")
        subtitle.setStyleSheet("QLabel {color: #a0b0ff; font-size: 18px; font-weight: normal; font-family: 'Segoe UI', Arial, sans-serif; background: transparent; letter-spacing: 1px;}")
        
        self.orb = FloatingOrb()
        
        self.progress = ModernProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setFixedWidth(400)
        
        self.status_label = QLabel("Initializing...")
        self.status_label.setStyleSheet("QLabel {color: #c0d0ff; font-size: 14px; font-family: 'Segoe UI', Arial, sans-serif; background: transparent;}")
        
        # Custom footer with Koushik Hy credit
        footer_text = "Developed by Koushik Hy | Secure Driving Solutions"
        footer_style = "QLabel {color: #7080a0; font-size: 12px; font-family: 'Segoe UI', Arial, sans-serif; background: transparent; margin-top: 20px;}"
        footer = QLabel(footer_text)
        footer.setStyleSheet(footer_style)
        
        overlay_layout.addWidget(title, 0, Qt.AlignmentFlag.AlignCenter)
        overlay_layout.addWidget(subtitle, 0, Qt.AlignmentFlag.AlignCenter)
        overlay_layout.addWidget(self.orb, 0, Qt.AlignmentFlag.AlignCenter)
        overlay_layout.addWidget(self.progress, 0, Qt.AlignmentFlag.AlignCenter)
        overlay_layout.addWidget(self.status_label, 0, Qt.AlignmentFlag.AlignCenter)
        overlay_layout.addWidget(footer, 0, Qt.AlignmentFlag.AlignCenter)
        
        layout.addWidget(overlay)
        
    def setupAnimations(self):
        # Timer to run the particle animation (self.background.updateParticles() and self.background.update())
        self.bg_timer = QTimer()
        self.bg_timer.timeout.connect(self.animateBackground)
        self.bg_timer.start(16)
        
        self.loading_timer = QTimer()
        self.loading_timer.timeout.connect(self.updateLoading)
        self.loading_timer.start(80) 
        
        # Stages are adjusted to include a clear wait for the camera
        self.loading_stages = [
            ("Initializing PyQt UI", 10),
            ("Loading AI Models", 25), 
            ("Waiting for Camera Worker", 50), # Key stage that pauses
            ("Starting Main Loop", 75),
            ("Finalizing System Check", 95),
            ("Complete", 100)
        ]
        self.current_stage = 0
        self.progress_value = 0.0
        
    def animateBackground(self):
        # <<<--- THIS IS WHAT MAKES THE PARTICLES MOVE --->>>
        self.background.updateParticles()
        self.background.update()
        
    def updateLoading(self):
        if self.current_stage < len(self.loading_stages):
            stage_name, target_progress = self.loading_stages[self.current_stage]
            
            # Special check for camera readiness: Block progress until the event is set
            if stage_name == "Waiting for Camera Worker" and not self.capture_ready_event.is_set():
                dots = "." * (int(time.time() * 2) % 4)
                self.status_label.setText(f"{stage_name}{dots}")
                return # Pause progress
            
            # If camera is ready, or not the camera stage, proceed
            if self.progress_value < target_progress:
                self.progress_value += 0.5
                self.progress.setValue(int(self.progress_value))
                
                dots = "." * (int(time.time() * 2) % 4)
                self.status_label.setText(f"{stage_name}{dots}")
            else:
                self.current_stage += 1
        else:
            self.loading_timer.stop()
            self.status_label.setText("Ready! Starting application...")
            # Use QTimer to ensure the quit is processed correctly by the event loop
            QTimer.singleShot(1000, self.startMainApplication)
    
    def startMainApplication(self):
        # 1. Close the loading window
        self.close() 
        # 2. Quit the PyQt application entirely
        QApplication.quit() 
        
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
    
    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self.drag_position)
            event.accept()

def run_loading_screen(capture_ready_event: mp_core.Event):
    """
    Runs the PyQt loading screen as a temporary, blocking application
    and quits it when the loading process is complete.
    """
    try:
        # Get existing instance or create new one
        app = QApplication.instance()
        if not app:
            app = QApplication(sys.argv)
        
        splash = ModernLoadingWindow(None, capture_ready_event)
        splash.show()
        
        # Blocks until QApplication.quit() is called inside ModernLoadingWindow
        app.exec() 
        
    except Exception as e:
        print(f"[Loading Screen Error] PyQt execution failed: {e}")

# =================================================================
# 💻 ORIGINAL HELPER FUNCTIONS (mostly UNCHANGED)
# =================================================================

def eye_aspect_ratio(eye: np.ndarray) -> float:
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C) if C != 0 else 0.0

def landmarks_to_np(landmarks, w: int, h: int) -> np.ndarray:
    return np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.float32)

def calculate_hand_rectangle(hand_landmarks, w: int, h: int) -> Tuple[float, float, float, float]:
    """Calculate bounding rectangle for hand"""
    points = landmarks_to_np(hand_landmarks, w, h)
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    width = x_max - x_min
    height = y_max - y_min
    return x_min, y_min, width, height

def detect_phone_grip(hand_landmarks, w: int, h: int) -> float:
    """Detect if hand is in phone gripping position"""
    points = landmarks_to_np(hand_landmarks, w, h)
    
    grip_score = 0.0
    
    # Method 1: Thumb and index finger proximity (typical phone grip)
    thumb_index_dist = np.linalg.norm(points[THUMB_TIP] - points[INDEX_TIP])
    if thumb_index_dist < GRIP_THRESH:
        grip_score += 0.3
    
    # Method 2: Multiple fingers close together (holding object)
    thumb_middle_dist = np.linalg.norm(points[THUMB_TIP] - points[MIDDLE_TIP])
    thumb_ring_dist = np.linalg.norm(points[THUMB_TIP] - points[RING_TIP])
    
    if thumb_middle_dist < GRIP_THRESH * 1.2:
        grip_score += 0.2
    if thumb_ring_dist < GRIP_THRESH * 1.5:
        grip_score += 0.2
    
    # Method 3: Palm curvature (cupped hand for holding)
    palm_points = points[[0, 5, 9, 13, 17]]  # Wrist and finger bases
    palm_area = cv2.contourArea(palm_points.reshape(-1, 1, 2))
    if palm_area > 1000:  # Reasonable palm area indicates proper hand detection
        grip_score += 0.1
    
    # Method 4: Finger alignment (fingers curled for holding)
    finger_tips = points[[8, 12, 16, 20]]  # All finger tips
    finger_bases = points[[5, 9, 13, 17]]  # All finger bases
    tip_to_base_dists = [np.linalg.norm(tip - base) for tip, base in zip(finger_tips, finger_bases)]
    
    # If fingers are curled (tips close to bases), it's more likely to be holding something
    curled_fingers = sum(1 for dist in tip_to_base_dists if dist < 80)
    grip_score += curled_fingers * 0.05
    
    return min(grip_score, 1.0)

def detect_hand_orientation(hand_landmarks, w: int, h: int) -> str:
    """Detect hand orientation for phone usage patterns"""
    points = landmarks_to_np(hand_landmarks, w, h)
    
    # Vector from wrist to middle finger MCP
    wrist_to_middle = points[9] - points[0]
    angle = np.degrees(np.arctan2(wrist_to_middle[1], wrist_to_middle[0]))
    
    if -45 <= angle <= 45:
        return "horizontal"
    elif 45 < angle <= 135:
        return "vertical_down"
    elif -135 <= angle < -45:
        return "vertical_up"
    else:
        return "horizontal"  # Default

def advanced_phone_detection(hand_landmarks_list, face_center, face_radius, w: int, h: int) -> Tuple[bool, float, List[str]]:
    """Advanced phone detection using multiple methods with debugging info"""
    if not hand_landmarks_list or face_center is None:
        return False, 0.0, []
    
    detection_methods = []
    total_confidence = 0.0
    
    for hand_landmarks in hand_landmarks_list:
        points = landmarks_to_np(hand_landmarks.landmark, w, h)
        palm_center = np.mean(points[[0, 5, 9, 13, 17]], axis=0)
        
        # Method 1: Proximity to face
        face_distance = np.linalg.norm(palm_center - face_center)
        proximity_threshold = PROXIMITY_PIX + (face_radius if face_radius else 100)
        
        if face_distance < proximity_threshold:
            proximity_confidence = 1.0 - (face_distance / proximity_threshold)
            total_confidence += proximity_confidence * 0.3
            detection_methods.append(f"Proximity: {proximity_confidence:.2f}")
        
        # Method 2: Grip detection
        grip_confidence = detect_phone_grip(hand_landmarks.landmark, w, h)
        total_confidence += grip_confidence * 0.4
        if grip_confidence > 0.3:
            detection_methods.append(f"Grip: {grip_confidence:.2f}")
        
        # Method 3: Hand orientation
        orientation = detect_hand_orientation(hand_landmarks.landmark, w, h)
        orientation_confidence = 0.0
        if orientation in ["horizontal", "vertical_down"]:  # Common phone holding orientations
            orientation_confidence = 0.7
            total_confidence += orientation_confidence * 0.2
            detection_methods.append(f"Orientation: {orientation}")
        
        # Method 4: Hand position relative to face (below face is common for phone)
        if palm_center[1] > face_center[1]:  # Hand below face
            position_confidence = 0.6
            total_confidence += position_confidence * 0.1
            detection_methods.append("Position: Below face")
        
        # Method 5: Hand size and shape (phone holding often shows specific patterns)
        hand_width = np.linalg.norm(points[0] - points[9])  # Wrist to middle base
        if hand_width > 50:  # Reasonable hand size
            shape_confidence = 0.5
            total_confidence += shape_confidence * 0.1
            detection_methods.append("Shape: Valid")
    
    # Normalize confidence
    total_confidence = min(total_confidence, 1.0)
    is_phone = total_confidence > 0.5
    
    return is_phone, total_confidence, detection_methods

def draw_phone_detection_debug(frame: np.ndarray, hand_landmarks_list, face_center, 
                             phone_confidence: float, detection_methods: List[str]):
    """Draw debug information for phone detection"""
    h, w = frame.shape[:2]
    
    # Draw face center
    if face_center is not None:
        center = tuple(map(int, face_center)) 
        # Theme-matched color: Cyan/Yellow (0, 255, 255)
        cv2.circle(frame, center, 8, (0, 255, 255), -1)
        cv2.circle(frame, center, PROXIMITY_PIX, (0, 255, 255), 2)
    
    # Draw hand landmarks and connections for phone detection
    if hand_landmarks_list:
        for hand_landmarks in hand_landmarks_list:
            if isinstance(hand_landmarks, list):
                points = np.array(hand_landmarks, dtype=np.float32)
            else:
                points = landmarks_to_np(hand_landmarks.landmark, w, h)
            
            # Draw hand skeleton
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
                (0, 5), (5, 6), (6, 7), (7, 8),  # Index
                (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
                (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
                (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
            ]
            
            for start, end in connections:
                # Theme-matched color: Light Blue/Cyan (255, 255, 0 in BGR is Yellow/Cyan)
                cv2.line(frame, 
                         (int(points[start][0]), int(points[start][1])),
                         (int(points[end][0]), int(points[end][1])),
                         (255, 255, 0), 2) 
            
            # Draw key points for phone detection
            key_points = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
            for idx in key_points:
                color = (0, 255, 0) if idx in [THUMB_TIP, INDEX_TIP] else (255, 0, 0)
                cv2.circle(frame, (int(points[idx][0]), int(points[idx][1])), 6, color, -1)
    
    # Draw confidence bar
    if phone_confidence > 0:
        bar_width = 200
        bar_height = 20
        bar_x = 10
        bar_y = h - 100
        
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        fill_width = int(phone_confidence * bar_width)
        color = (0, 0, 255) if phone_confidence > 0.5 else (0, 165, 255) # Red/Orange
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), color, -1)
        
        cv2.putText(frame, f'Phone Confidence: {phone_confidence:.2f}', 
                    (bar_x, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Draw detection methods
    if detection_methods:
        method_y = h - 70
        for i, method in enumerate(detection_methods[:4]):
            cv2.putText(frame, method, (10, method_y + i * 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 100), 1)

def draw_animated_alert(frame: np.ndarray, alert_type: str, intensity: float):
    """Draw animated alert overlays"""
    h, w = frame.shape[:2]
    current_time = time.time()
    pulse = (np.sin(current_time * PULSE_SPEED * 2 * np.pi) + 1) / 2
    
    if alert_type == "drowsy":
        color = (0, 0, 255)  # Red
        text = "DROWSINESS DETECTED!"
    elif alert_type == "phone":
        color = (0, 165, 255)  # Orange
        text = "PHONE USAGE DETECTED!"
    elif alert_type == "lookaway":
        color = (0, 255, 255)  # Yellow
        text = "LOOKING AWAY!"
    else:
        color = (0, 128, 255)  # Darker Orange/Red
        text = "FACE NOT DETECTED!"
    
    # Pulsating overlay
    overlay = frame.copy()
    alpha = 0.3 + 0.2 * pulse
    cv2.rectangle(overlay, (0, 0), (w, h), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    # Animated text
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 2, 3)[0]
    text_x = (w - text_size[0]) // 2
    text_y = (h + text_size[1]) // 2
    
    # Text shadow effect
    shadow_color = tuple(max(0, c - 100) for c in color)
    cv2.putText(frame, text, (text_x + 3, text_y + 3), 
                cv2.FONT_HERSHEY_DUPLEX, 2, shadow_color, 3, cv2.LINE_AA)
    cv2.putText(frame, text, (text_x, text_y), 
                cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)
    
    # Flashing border
    border_thickness = 3 + int(5 * pulse)
    cv2.rectangle(frame, (0, 0), (w, h), color, border_thickness)

def draw_modern_ui(frame: np.ndarray, status_data: Dict[str, Any], fps: float, tts_active: bool):
    """
    Modern, animated UI with Cyber-Blue/Dark theme, matching the loading screen.
    """
    h, w = frame.shape[:2]
    current_time = time.time()
    pulse = (np.sin(current_time * PULSE_SPEED * 2 * np.pi) + 1) / 2
    
    # --- THEME COLORS (BGR) ---
    CYBER_BLUE = (255, 150, 100)  # Light Blue/Cyan for highlights
    DARK_BG = (15, 15, 15)       # Almost black background
    DARK_PANEL = (30, 30, 30)    # Slightly lighter dark for panels
    ALERT_RED = (0, 0, 255)      # Red for critical alerts
    ALERT_ORANGE = (0, 165, 255) # Orange for phone alerts
    ALERT_YELLOW = (0, 255, 255) # Yellow for lookaway
    ATTENTIVE_GREEN = (0, 180, 0) # Green for good status
    WHITE = (255, 255, 255)
    LIGHT_GREY = (200, 200, 200)

    # Extract status data
    is_drowsy = status_data.get('is_drowsy', False)
    is_looking_away = status_data.get('is_looking_away', False)
    is_using_phone = status_data.get('is_using_phone', False)
    face_found = status_data.get('face_found', False)
    ear_value = status_data.get('ear_value', 0.0)
    phone_warning_count = status_data.get('phone_warning_count', 0)
    phone_confidence = status_data.get('phone_confidence', 0.0)
    
    is_critical = is_drowsy or is_using_phone or is_looking_away or not face_found
    
    # --- Main Background Overlay (Darken the camera feed) ---
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), DARK_BG, -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame) # 40% dark overlay

    # --- Status Bar (Top Header) ---
    status_bar_h = 50
    
    # Draw solid DARK_PANEL background for status bar
    cv2.rectangle(frame, (0, 0), (w, status_bar_h), DARK_PANEL, -1)
    
    # Determine alert line color
    status_color_alert = ALERT_RED if is_critical else CYBER_BLUE
    
    # Add a thin colored line/glow effect at the bottom
    cv2.line(frame, (0, status_bar_h - 1), (w, status_bar_h - 1), status_color_alert, 3)
    
    # Status Text with Animation
    status_text = "ALERT! CHECK DRIVER" if is_critical else "DRIVER ATTENTIVE"
    text_color = WHITE
    text_scale = 1.0 + 0.1 * pulse if is_critical else 0.9
    
    # Animated glow for critical status
    if is_critical:
        shadow_color = status_color_alert
        cv2.putText(frame, status_text, (20 + 2, 35 + 2), cv2.FONT_HERSHEY_DUPLEX, 
                    text_scale, shadow_color, 4, cv2.LINE_AA) # Shadow/Glow
        
    cv2.putText(frame, status_text, (20, 35), cv2.FONT_HERSHEY_DUPLEX, 
                text_scale, text_color, 2, cv2.LINE_AA)
    
    # --- Metrics Panel (Modern Dark Themed) ---
    panel_width = 320
    panel_height = 200
    panel_x = w - panel_width - 10
    panel_y = status_bar_h + 10 
    
    # Panel background color (solid dark panel)
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x+panel_width, panel_y+panel_height), 
                  DARK_PANEL, -1)
    
    # Panel border with CYBER_BLUE highlight
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x+panel_width, panel_y+panel_height), 
                  CYBER_BLUE, 2)
    
    # Metrics content
    metrics_start_y = panel_y + 30
    line_height = 28
    text_h_offset = 10
    
    # Title: Metrics
    cv2.putText(frame, 'SYSTEM METRICS', (panel_x + text_h_offset, metrics_start_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, CYBER_BLUE, 2, cv2.LINE_AA)
    
    # Separator Line
    cv2.line(frame, (panel_x + text_h_offset, metrics_start_y + 5), 
             (panel_x + panel_width - text_h_offset, metrics_start_y + 5), (50, 50, 50), 1)

    # FPS
    cv2.putText(frame, f'FPS: {int(fps)}', (panel_x + text_h_offset, metrics_start_y + line_height),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, LIGHT_GREY, 1, cv2.LINE_AA)
    
    # Face Status
    face_status = "DETECTED" if face_found else "NOT FOUND"
    face_color = ATTENTIVE_GREEN if face_found else ALERT_RED
    cv2.putText(frame, f'Face: {face_status}', (panel_x + text_h_offset, metrics_start_y + line_height*2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, face_color, 2, cv2.LINE_AA)
    
    # EAR Value with animated bar
    ear_color = ATTENTIVE_GREEN if ear_value > EAR_THRESHOLD else ALERT_RED
    cv2.putText(frame, f'EAR: {ear_value:.3f}', (panel_x + text_h_offset, metrics_start_y + line_height*3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, ear_color, 2, cv2.LINE_AA)
    
    # EAR Progress Bar (The line below the text)
    bar_width = 150
    bar_x = panel_x + panel_width - bar_width - text_h_offset
    bar_y = metrics_start_y + line_height*3 + 5
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 8), (60, 60, 60), -1)
    ear_fill = int(min(ear_value / 0.4, 1.0) * bar_width)
    
    # Animated fill for low EAR
    draw_fill_color = ear_color
    if ear_value < EAR_THRESHOLD:
        animated_width = int(ear_fill * (0.8 + 0.2 * pulse))
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + animated_width, bar_y + 8), draw_fill_color, -1)
    else:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + ear_fill, bar_y + 8), draw_fill_color, -1)
    
    # Alert Status
    alert_status = 'ACTIVE' if is_critical else 'NONE'
    tts_status = ' (SPEAKING)' if tts_active else ''
    alert_color = ALERT_RED if is_critical else ATTENTIVE_GREEN
    cv2.putText(frame, f'Alert: {alert_status}{tts_status}',
               (panel_x + text_h_offset, metrics_start_y + line_height*4), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, alert_color, 2, cv2.LINE_AA)
    
    # --- Footer ---
    footer_h = 40
    footer_y = h - footer_h
    cv2.rectangle(frame, (0, footer_y), (w, h), DARK_PANEL, -1) # Solid DARK_PANEL footer
    
    # Bottom line CYBER_BLUE highlight
    cv2.line(frame, (0, footer_y), (w, footer_y), CYBER_BLUE, 2)
    
    # Main Info Text
    info_text = 'ADVANCED DRIVER MONITORING v6.0 | Enhanced Phone Detection | Press ESC or Q to Exit'
    cv2.putText(frame, info_text, (10, h - 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, LIGHT_GREY, 1, cv2.LINE_AA)
               
    # Developer Credit 
    dev_text = 'Developed by Koushik Hy'
    dev_text_size = cv2.getTextSize(dev_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    dev_text_x = w - dev_text_size[0] - 10
    cv2.putText(frame, dev_text, (dev_text_x, h - 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, CYBER_BLUE, 1, cv2.LINE_AA)
    
    # Draw individual alerts if critical (Full screen flashing)
    if is_critical:
        if is_drowsy:
            draw_animated_alert(frame, "drowsy", pulse)
        elif is_using_phone:
            draw_animated_alert(frame, "phone", pulse)
        elif is_looking_away:
            draw_animated_alert(frame, "lookaway", pulse)
        elif not face_found:
            draw_animated_alert(frame, "noface", pulse)

def save_alert_image(frame: np.ndarray, alert_type: str, phone_warning_count: int):
    """Save alert image when warnings exceed threshold"""
    if phone_warning_count >= PHONE_WARNING_THRESHOLD:
        os.makedirs('detected_images', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"detected_images/{alert_type}_alert_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        return True
    return False

# --- TTS Worker (Unchanged) ---
def _gtts_alert_worker(alert_queue: mp_core.Queue, tts_active_flag: mp_core.Value):
    if os.path.exists('temp_alerts'):
        shutil.rmtree('temp_alerts')
    os.makedirs('temp_alerts', exist_ok=True)
    print("[TTS Worker] Started. Initializing pygame mixer...")
    try:
        pygame.mixer.init(frequency=24000) 
    except pygame.error as e:
        print(f"[PYGAME ERROR] Could not initialize mixer: {e}")
        return

    while True:
        try:
            alert_data = alert_queue.get(timeout=1.0)
            if alert_data is None:
                break
            text, alert_type = alert_data
            tts = gtts.gTTS(text, lang='en', slow=False)
            filename = f'temp_alerts/{alert_type}.mp3'
            tts.save(filename)

            tts_active_flag.value = True
            sound = pygame.mixer.Sound(filename)
            channel = sound.play()
            while channel.get_busy():
                time.sleep(0.1)
            tts_active_flag.value = False
        except Empty:
            continue
        except Exception as e:
            tts_active_flag.value = False
    
    pygame.mixer.quit()
    if os.path.exists('temp_alerts'):
        shutil.rmtree('temp_alerts')
    print("[TTS Worker] Shutting down.")

def speak_continuous(text: str, alert_type: str, alert_history: dict, tts_active_flag: mp_core.Value, tts_queue: mp_core.Queue):
    current_time = time.time()
    if (current_time - alert_history.get(alert_type, 0.0) > ALERT_COOLDOWN_CONTINUOUS and not tts_active_flag.value):
        try:
            if tts_queue.qsize() > 0:
                tts_queue.get_nowait()
            tts_queue.put_nowait((text, alert_type))
            alert_history[alert_type] = current_time
        except Exception:
            pass

# --- Video Capture Worker (Unchanged) ---
def video_capture_worker(frame_queue: mp_core.Queue, stop_event: mp_core.Event, capture_ready_event: mp_core.Event):
    print("[Capture Worker] Starting camera...")
    cap = cv2.VideoCapture(0) 
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print('[Capture Worker] Cannot open camera! Trying index 1...')
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print('[Capture Worker] Cannot open camera on index 1 either. Exiting.')
            return

    capture_ready_event.set()
    print("[Capture Worker] Camera is ready.")
    
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        frame = cv2.flip(frame, 1)
        try:
            if frame_queue.qsize() > 0:
                frame_queue.get_nowait()
            frame_queue.put_nowait(frame)
        except Exception:
            continue
    cap.release()
    print("[Capture Worker] Shutting down.")

# --- Enhanced Processing Worker (Unchanged) ---
def processing_worker(frame_queue: mp_core.Queue, result_queue: mp_core.Queue, stop_event: mp_core.Event):
    print("[Processing Worker] Initializing MediaPipe...")
    eye_counter = 0
    look_persist = 0
    phone_persist = 0
    phone_warning_count = 0
    last_face_center: Optional[np.ndarray] = None
    last_face_radius: Optional[float] = None
    last_save_time = 0.0

    with mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                          min_detection_confidence=0.7, min_tracking_confidence=0.7) as face_mesh, \
          mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                        min_detection_confidence=0.6, min_tracking_confidence=0.6) as hands:

        while not stop_event.is_set():
            try:
                frame = frame_queue.get(timeout=0.01)
            except Empty:
                continue
                
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            is_drowsy = False
            is_looking_away = False
            is_using_phone = False
            face_found = False
            ear_value = 0.0
            phone_confidence = 0.0
            detection_methods = []
            face_landmarks = None
            hand_landmarks = None

            results = face_mesh.process(rgb)
            hands_res = hands.process(rgb)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                pts = landmarks_to_np(landmarks, w, h)
                face_found = True
                face_landmarks = pts.tolist()
                
                face_center_np = pts.mean(axis=0)
                face_radius_val = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)) * 0.25
                last_face_center = face_center_np 
                last_face_radius = face_radius_val 
                
                ear_left = eye_aspect_ratio(pts[LEFT_EYE_IDX])
                ear_right = eye_aspect_ratio(pts[RIGHT_EYE_IDX])
                ear_value = (ear_left + ear_right) / 2.0
                
                if ear_value < EAR_THRESHOLD:
                    eye_counter += 1
                    if eye_counter >= EAR_CONSEC_FRAMES:
                        is_drowsy = True
                else:
                    eye_counter = 0
                
                nose = pts[NOSE_IDX]
                offset_x = (nose[0] - face_center_np[0]) / w 
                if abs(offset_x) > NOSE_OFFSET_THRESH:
                    look_persist += 1
                    if look_persist >= LOOK_CONSEC_FRAMES:
                        is_looking_away = True
                else:
                    look_persist = 0
            else:
                eye_counter = 0
                look_persist = 0

            phone_detected_this_frame = False
            if hands_res and hands_res.multi_hand_landmarks and last_face_center is not None:
                hand_landmarks_list_for_result = []
                for hl in hands_res.multi_hand_landmarks:
                    hpts = landmarks_to_np(hl.landmark, w, h)
                    hand_landmarks_list_for_result.append(hpts.tolist())
                hand_landmarks = hand_landmarks_list_for_result

                phone_detected_this_frame, phone_confidence, detection_methods = advanced_phone_detection(
                    hands_res.multi_hand_landmarks, 
                    last_face_center, 
                    last_face_radius, 
                    w, h
                )
            
            if phone_detected_this_frame:
                phone_persist += 1
                if phone_persist >= DETECTION_CONSEC_FRAMES:
                    is_using_phone = True
                    if phone_persist == DETECTION_CONSEC_FRAMES:
                        phone_warning_count += 1
            else:
                phone_persist = 0

            results_data: Dict[str, Any] = {
                'is_drowsy': is_drowsy,
                'is_looking_away': is_looking_away,
                'is_using_phone': is_using_phone,
                'face_found': face_found,
                'ear_value': ear_value,
                'phone_warning_count': phone_warning_count, 
                'phone_confidence': phone_confidence,
                'face_landmarks': face_landmarks,
                'hand_landmarks': hand_landmarks, 
                'detection_methods': detection_methods,
                'face_center': last_face_center.tolist() if last_face_center is not None else None,
                'face_radius': last_face_radius
            }

            try:
                if result_queue.qsize() > 0:
                    result_queue.get_nowait()
                result_queue.put_nowait(results_data)
            except Exception:
                continue
    print("[Processing Worker] Shutting down.")

# --- Enhanced Main UI Worker ---
def main_ui_worker(frame_queue, result_queue, stop_event, tts_queue, alert_history, tts_active_flag):
    last_save_time = 0.0
    fps_time = time.time()
    fps_counter = 0
    fps_value = 0.0
    os.makedirs('detected_images', exist_ok=True)
    
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    
    results_data: Dict[str, Any] = {
        'is_drowsy': False, 'is_looking_away': False, 'is_using_phone': False,
        'face_found': False, 'ear_value': 0.0, 'phone_warning_count': 0,
        'phone_confidence': 0.0, 'face_landmarks': None, 'hand_landmarks': None,
        'detection_methods': [], 'face_center': None, 'face_radius': None
    }
    
    while not stop_event.is_set():
        current_time = time.time()
        
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            stop_event.set()
            break
            
        try:
            frame = frame_queue.get(timeout=0.001)
        except Empty:
            time.sleep(0.001)
            continue

        h, w = frame.shape[:2]

        try:
            latest_results = result_queue.get_nowait()
            results_data.update(latest_results)
        except Empty:
            pass

        fps_counter += 1
        if current_time - fps_time >= 1.0:
            fps_value = fps_counter / (current_time - fps_time)
            fps_counter = 0
            fps_time = current_time

        face_center_np = np.array(results_data['face_center']) if results_data['face_center'] is not None else None
        
        # ALERT MANAGEMENT
        is_drowsy = results_data['is_drowsy']
        is_looking_away = results_data['is_looking_away']
        is_using_phone = results_data['is_using_phone']
        face_found = results_data['face_found']
        phone_warning_count = results_data['phone_warning_count']

        if is_drowsy:
            speak_continuous('Warning! Drowsiness detected. Please take a break immediately.',
                             'drowsy', alert_history, tts_active_flag, tts_queue)
            if current_time - last_save_time > SAVE_COOLDOWN:
                if save_alert_image(frame, "drowsiness", phone_warning_count):
                    last_save_time = current_time
                    
        elif is_using_phone:
            speak_continuous('Warning! Put the phone down immediately. Focus on driving.',
                             'phone', alert_history, tts_active_flag, tts_queue)
            if (phone_warning_count >= PHONE_WARNING_THRESHOLD and 
                current_time - last_save_time > SAVE_COOLDOWN):
                if save_alert_image(frame, "phone", phone_warning_count):
                    last_save_time = current_time
        
        elif is_looking_away:
            if results_data.get('face_landmarks') and face_center_np is not None:
                pts = np.array(results_data['face_landmarks'])
                offset_x = (pts[NOSE_IDX][0] - w / 2) / w
                look_side = 'RIGHT' if offset_x > 0 else 'LEFT'
                speak_continuous(f'Warning! Looking {look_side}. Keep eyes on the road.',
                                 'lookaway', alert_history, tts_active_flag, tts_queue)
        
        elif not face_found:
            speak_continuous('Warning! Driver face not detected.', 'no_face', alert_history, tts_active_flag, tts_queue)
        else:
            for alert_type in ['drowsy', 'phone', 'lookaway', 'no_face']:
                alert_history.pop(alert_type, None)

        # Draw enhanced modern UI (Theme-matched function)
        draw_modern_ui(frame, results_data, fps_value, tts_active_flag.value)

        # Draw DEBUG info *after* UI to overlay everything
        if DEBUG and results_data['hand_landmarks']:
            draw_phone_detection_debug(
                frame, 
                results_data['hand_landmarks'], 
                face_center_np, 
                results_data['phone_confidence'], 
                results_data['detection_methods']
            )

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27: # ESC key is 27
            stop_event.set()
            break
            
    cv2.destroyAllWindows()
    print("[Main UI Worker] Shutting down.")

# --- Run Multi-Process System (Unchanged) ---
def run_multi_process_system():
    def final_cleanup_and_exit():
        cv2.destroyAllWindows()
        if os.path.exists('temp_alerts'):
             shutil.rmtree('temp_alerts')
        print("="*60)
        print("ADVANCED DRIVER MONITORING SYSTEM EXITED SUCCESSFULLY")
        print("All processes stopped. Cleanup complete.")
        print("="*60)
    
    manager = mp_core.Manager()
    alert_history = manager.dict()
    tts_active_flag = mp_core.Value('b', False)
    capture_ready_event = mp_core.Event() 
    
    tts_queue = mp_core.Queue(maxsize=1)
    frame_queue = mp_core.Queue(maxsize=1)
    result_queue = mp_core.Queue(maxsize=1)
    stop_event = mp_core.Event()

    print("="*60)
    print("ADVANCED DRIVER MONITORING SYSTEM v6.0")
    print("ENHANCED PHONE DETECTION WITH MULTIPLE METHODS")
    print("="*60)
    
    tts_proc = mp_core.Process(target=_gtts_alert_worker, args=(tts_queue, tts_active_flag), daemon=True)
    tts_proc.start()

    capture_proc = mp_core.Process(target=video_capture_worker, args=(frame_queue, stop_event, capture_ready_event))
    capture_proc.start()

    processing_proc = mp_core.Process(target=processing_worker, args=(frame_queue, result_queue, stop_event))
    processing_proc.start()
    
    # 4. Run Loading Screen (BLOCKING)
    run_loading_screen(capture_ready_event)

    # 5. Start Main UI Worker (Original CV Interface - BLOCKING)
    try:
        main_ui_worker(frame_queue, result_queue, stop_event, tts_queue, alert_history, tts_active_flag)
    except Exception as e:
        print(f"[ERROR] Main UI Worker encountered an exception: {e}")
    finally:
        stop_event.set()
        
        tts_queue.put(None) 
        
        print("[SYSTEM] Joining all worker processes...")
        tts_proc.join(timeout=2.0)
        capture_proc.join(timeout=2.0)
        processing_proc.join(timeout=2.0)
        
        final_cleanup_and_exit()

# --- Main Entry ---
if __name__ == '__main__':
    mp_core.set_start_method('spawn', force=True)
    mp_core.freeze_support()
    run_multi_process_system()