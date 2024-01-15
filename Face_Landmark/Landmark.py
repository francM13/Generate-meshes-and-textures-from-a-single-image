import subprocess
import cv2
import matplotlib.pyplot as plt
import os
from Face_Landmark.Mediapipe import draw_landmarks_on_image
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def generate_face_landmarks(image_input, plot_image=False):
    """
    Generates face landmarks from an image using MediaPipe FaceLandmarker.

    Args:
        image_input (str or np.ndarray): Path to the image file or the image data itself.
        plot_image (bool): Whether to display the image or not.

    Returns:
        The face landmarks or an error occurs.
    """

    # Download the landmark model if it's not already available
    download_landmark_model()

    # Import the necessary modules.
    base_options = python.BaseOptions(model_asset_path="face_landmarker_v2_with_blendshapes.task")
    options = vision.FaceLandmarkerOptions(base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)

    # Load the input image.
    if isinstance(image_input, str):
        image = mp.Image.create_from_file(image_input)
    else:
        img = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)

    # Detect face landmarks from the input image.
    detection_result = detector.detect(image)

    # Process the detection result. In this case, visualize it.
    annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)

    # Display the annotated image
    if plot_image:
        plt.imshow(annotated_image)
        plt.show()

    return detection_result.face_landmarks[0]

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
            subprocess.run(["wget", "-O", landmark_model_file, landmark_model_url], check=True)
            print("Landmark model downloaded successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error downloading landmark model: {e}")