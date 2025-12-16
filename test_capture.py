import cv2
import numpy as np
import mediapipe as mp
import pyttsx3

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [263, 387, 385, 362, 380, 373]


def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)


def landmarks_to_np(landmarks, w, h):
    pts = []
    for lm in landmarks:
        pts.append(np.array([lm.x * w, lm.y * h], dtype=np.float32))
    return np.array(pts, dtype=np.float32)


def run_once():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Cannot open camera')
        return 2

    ret, frame = cap.read()
    if not ret:
        print('No frame captured')
        cap.release()
        return 3

    h, w = frame.shape[:2]
    mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_face.process(rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        pts = landmarks_to_np(landmarks, w, h)
        left_eye = pts[LEFT_EYE_IDX]
        right_eye = pts[RIGHT_EYE_IDX]
        leftEAR = eye_aspect_ratio(left_eye)
        rightEAR = eye_aspect_ratio(right_eye)
        print(f'leftEAR={leftEAR:.4f} rightEAR={rightEAR:.4f}')
    else:
        print('No face detected')

    # TTS test
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.say('This is a quick test alert. If you hear this, TTS is working.')
        engine.runAndWait()
        print('TTS ran')
    except Exception as e:
        print('TTS error:', e)

    mp_face.close()
    cap.release()
    return 0


if __name__ == '__main__':
    exit(run_once())
