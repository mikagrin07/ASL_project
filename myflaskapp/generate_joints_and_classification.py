import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import os
import urllib.request
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import threading


# Load MediaPipe Hand and Pose Landmarker
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

MODEL_PATH = "/home/mikagrin/ASL_project/myflaskapp/weights/asl_lstm_single_layer.h5"

# Define constants
LEFT_HAND_LANDMARKS = 21
RIGHT_HAND_LANDMARKS = 21
POSE_LANDMARKS = 33
FEATURES_PER_FRAME = (LEFT_HAND_LANDMARKS + RIGHT_HAND_LANDMARKS + POSE_LANDMARKS) * 2  # X, Y for each landmark
MAX_FRAMES = 15  # Number of frames used for classification

# Load label encoder
LABELS = ['book', 'computer_bk', 'drink', 'i', 'science', 'study']  # Same classes as in training

# === הורדת מודלים ===
#BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BASE_DIR = "/home/mikagrin/ASL_project/myflaskapp/"

POSE_MODEL_PATH = os.path.join(BASE_DIR, "pose_landmarker.task")

HAND_MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")


def download_model(url, path):
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)


# Global variables
pose_landmarker = None
pose_lock = threading.Lock()


model = None
hand_landmarker = None
pose_landmarker = None


#####################################################################################################
#   extract_landmark_matrix_full_video
#####################################################################################################

def extract_landmark_matrix_full_video(video_path, output_path=None):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    n_frames = min(total_frames, MAX_FRAMES)

    selected_indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)


    frames = []
    current_frame = 0
    selected_set = set(selected_indices)

    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    else:
        out = None


    ####### Init Models ################

    base_options_hand = python.BaseOptions(model_asset_path=HAND_MODEL_PATH)

    options_hand = vision.HandLandmarkerOptions(base_options=base_options_hand, num_hands=2)

    hand_landmarker = vision.HandLandmarker.create_from_options(options_hand)


    base_options_pose = python.BaseOptions(model_asset_path=POSE_MODEL_PATH)
    options_pose = vision.PoseLandmarkerOptions(base_options=base_options_pose)
    pose_landmarker = vision.PoseLandmarker.create_from_options(options_pose)

    ###################################

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or len(frames) >= n_frames:
            break

        if current_frame in selected_set:

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            hand_result = hand_landmarker.detect(mp_image)

            pose_result = pose_landmarker.detect(mp_image)

            frame_data = [0.0] * FEATURES_PER_FRAME

            if hand_result.hand_landmarks:
                for idx, hand in enumerate(hand_result.hand_landmarks):
                    base = 0 if hand_result.handedness[idx][0].category_name == "Left" else 21 * 2
                    for lm_idx, lm in enumerate(hand):
                        frame_data[base + lm_idx * 2] = lm.x
                        frame_data[base + lm_idx * 2 + 1] = lm.y

            if pose_result.pose_landmarks:
                base = (21 + 21) * 2
                for lm_idx, lm in enumerate(pose_result.pose_landmarks[0]):
                    frame_data[base + lm_idx * 2] = lm.x
                    frame_data[base + lm_idx * 2 + 1] = lm.y

            frames.append(frame_data)

        elif out:
            out.write(frame)

        current_frame += 1

    cap.release()
    if out:
        out.release()

    while len(frames) < n_frames:
        frames.append([0.0] * FEATURES_PER_FRAME)

    frames = np.array(frames[:n_frames])
    return frames


#####################################################################################################
#   classify_video
#####################################################################################################

def classify_video(video_path, outputs_dir):
    """
    Classifies a video based on extracted landmarks using the trained RNN model.
    """
    print(f"Start classify video {video_path} ....")

    model = tf.keras.models.load_model(MODEL_PATH)

    sequence = extract_landmark_matrix_full_video(video_path, outputs_dir)
    print(f" Processing video: {video_path}")
    if sequence is None:
        return "Error: Could not extract landmarks."

    sequence = np.expand_dims(sequence, axis=0)  # Add batch dimension

    print(f"Start predict ....")
    predictions = model.predict(sequence)
    predicted_label = LABELS[np.argmax(predictions)]

    return predicted_label


#####################################################################################################
#   generate_joints
#####################################################################################################

def generate_joints(input_video_path, output_video_path, outputs_dir):
    """
    This function only classifies the input video by extracting landmarks
    and predicting the sign using the trained model.
    It does not modify or write any new video files.
    """

    try:
        # פשוט מעבירים את הווידאו לפונקציית הסיווג
        predicted_sign = classify_video(input_video_path, outputs_dir)
        return predicted_sign

    except Exception as e:
        print(f"❌ [generate_joints] Exception: {str(e)}", flush=True)
        return "Error: failed to classify video"
