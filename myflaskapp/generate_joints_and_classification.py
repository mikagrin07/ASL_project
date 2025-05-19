import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import os
import urllib.request
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import threading
#from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.editor import VideoFileClip

# Load MediaPipe Hand and Pose Landmarker
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

MODEL_PATH = "/home/mikagrin/ASL_project1/myflaskapp/weights/ASL_final_model.h5"

# Define constants
LEFT_HAND_LANDMARKS = 21
RIGHT_HAND_LANDMARKS = 21
POSE_LANDMARKS = 33
FEATURES_PER_FRAME = (LEFT_HAND_LANDMARKS + RIGHT_HAND_LANDMARKS + POSE_LANDMARKS) * 2  # X, Y for each landmark
MAX_FRAMES = 15  # Number of frames used for classification

# Load label encoder
LABELS = ['book', 'computer', 'drink', 'i', 'other', 'read', 'science', 'study', 'water']  # Same classes as in training

# === הורדת מודלים ===
#BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BASE_DIR = "/home/mikagrin/ASL_project1/myflaskapp/"

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

def classify_single_word(video_path, outputs_dir):
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

    print(f"!!!@@@ Start predict ....")
    predictions = model.predict(sequence)
    predicted_label = LABELS[np.argmax(predictions)]

    return predicted_label

#####################################################################################################
#   split_video_chunks
#####################################################################################################

def split_video_chunks(video_path, chunk_duration):
    video = VideoFileClip(video_path)
    print("--------------------------------- video ----------------------------")
    duration = video.duration
    print(f"--------------------------------- duration: {duration} ----------------------------")
    chunks = []

    start = 0.0
    idx = 0
    while start < duration:
        end = min(start + chunk_duration, duration)
        #output_path = f"/content/chunk_{idx}_{int(chunk_duration * 100)}.mp4"
        output_path = f"chunk_{idx}_{int(chunk_duration * 100)}.mp4"
        video.subclip(start, end).write_videofile(output_path, codec="libx264", audio=False, verbose=False, logger=None)
        chunks.append(output_path)
        start += chunk_duration
        idx += 1

    return chunks

#####################################################################################################
#   predict_chunks
#####################################################################################################

def predict_chunks(chunks, model, LABELS, run_label=""):
    predictions = []

    for i, chunk in enumerate(chunks):
        matrix = extract_landmark_matrix_full_video(chunk)
        matrix = np.expand_dims(matrix, axis=0)

        pred = model.predict(matrix, verbose=0)
        label = np.argmax(pred, axis=1)[0]
        confidence = float(np.max(pred))
        word = LABELS[label]

        print(f"{run_label} matrix {i+1}: '{word}' (confidence: {confidence:.2%})")
        predictions.append(word)

    return predictions


#####################################################################################################
#   merge_predictions_with_time_v2
#####################################################################################################

def merge_predictions_with_time_v2(preds_07, preds_09, chunk_duration_07=0.7, chunk_duration_09=0.9, max_time_diff=1):
    final_sentence = []
    max_len = max(len(preds_07), len(preds_09))

    for i in range(max_len):
        word_07 = preds_07[i] if i < len(preds_07) else None
        word_09 = preds_09[i] if i < len(preds_09) else None

        # Remove 'other' as it doesn't contribute meaningfully
        if word_07 == 'other': word_07 = None
        if word_09 == 'other': word_09 = None

        chosen = None

        # If both words are the same and not None, keep it
        if word_07 and word_07 == word_09:
            chosen = word_07
        # If both exist but are different, prefer the one closest in time
        elif word_07 and word_09:
            # Check the time difference between the two words
            time_07 = i * chunk_duration_07
            time_09 = i * chunk_duration_09
            time_diff = abs(time_07 - time_09)

            if time_diff <= max_time_diff:
                chosen = word_09
            else:
                chosen = word_07  # Default to 0.7s prediction if time difference is too large
        # If only one word exists, take it only if it is surrounded by words from the other prediction
        elif word_07:
            # Check if this word has a corresponding match before or after in the 0.9s prediction
            idx_in_09 = None
            for j in range(i, len(preds_09)):
                if preds_09[j] == word_07:
                    idx_in_09 = j
                    break

            if idx_in_09 is not None:
                time_09 = idx_in_09 * chunk_duration_09
                time_07 = i * chunk_duration_07
                time_diff = abs(time_07 - time_09)

                # If it is too far apart, discard the word
                if time_diff <= max_time_diff:
                    chosen = word_07

        elif word_09:
            # Check if this word has a corresponding match before or after in the 0.7s prediction
            idx_in_07 = None
            for j in range(i, len(preds_07)):
                if preds_07[j] == word_09:
                    idx_in_07 = j
                    break

            if idx_in_07 is not None:
                time_07 = idx_in_07 * chunk_duration_07
                time_09 = i * chunk_duration_09
                time_diff = abs(time_07 - time_09)

                # If it is too far apart, discard the word
                if time_diff <= max_time_diff:
                    chosen = word_09

        # Avoid consecutive duplicates in final sentence
        if chosen and (len(final_sentence) == 0 or final_sentence[-1] != chosen):
            final_sentence.append(chosen)

    return final_sentence

#####################################################################################################
#   dual_run_prediction_with_time_v2
#####################################################################################################

def dual_run_prediction_with_time_v2(video_path, LABELS):

    model = tf.keras.models.load_model(MODEL_PATH)

    chunks_07 = split_video_chunks(video_path, chunk_duration=0.7)
    preds_07 = predict_chunks(chunks_07, model, LABELS, run_label="0.7s")

    chunks_09 = split_video_chunks(video_path, chunk_duration=0.9)
    preds_09 = predict_chunks(chunks_09, model, LABELS, run_label="0.9s")

    final_sentence = merge_predictions_with_time_v2(preds_07, preds_09)

    print("\n📝 Full sentence (0.7s):", " ".join(preds_07))
    print("📝 Full sentence (0.9s):", " ".join(preds_09))
    print("✅ Final combined sentence:", " ".join([w for w in final_sentence if w]))
    return final_sentence


#####################################################################################################
#   generate_joints
#####################################################################################################

def generate_joints(input_video_path, output_video_path, outputs_dir, mode="single_word"):
    try:
        print("--------------------------------------------------------------------------------------------------------")
        if mode == "full_sentence":
            prediction = dual_run_prediction_with_time_v2(input_video_path, LABELS)
            return {
                "predicted_sentence": " ".join(prediction)
            }
        else:
            prediction = classify_single_word(input_video_path, outputs_dir)
            return {
                "predicted_sign": prediction,
            }
    except Exception as e:
        print(f"❌ [generate_joints] Exception: {str(e)}", flush=True)
        return {
            "predicted_sign": "Error: failed to classify video",
            "mode_tag": "error"
        }
