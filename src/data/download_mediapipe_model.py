import os
import subprocess
def download_landmark_model(
        landmark_model_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        landmark_model_file = "face_landmarker_v2_with_blendshapes.task"):
    """
    Downloads the landmark model for face landmark detection using OpenCV.

    Args:
        landmark_model_url (str): URL of the landmark model file.
        landmark_model_file (str): Local path to save the landmark model file.
    """
    if not os.path.isfile("face_landmarker_v2_with_blendshapes.task"):
        try:
            print("Downloading landmark model...")
            subprocess.run(["wget", "-O", landmark_model_file, landmark_model_url], check=False)
            print("Landmark model downloaded successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error downloading landmark model: {e}")
