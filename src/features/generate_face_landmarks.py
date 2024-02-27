import subprocess
import cv2
import matplotlib.pyplot as plt
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
from typing import Union
from src.data.download_mediapipe_model import download_landmark_model
from src.visualization.visualize_landmark_on_image import visualize_landmarks_on_image

def generate_face_landmarks(image_input: Union[str, np.ndarray], plot_image: bool = False) -> landmark_pb2.NormalizedLandmarkList:
    """
    Generates face landmarks from an image using MediaPipe FaceLandmarker.

    Args:
        image_input (Union[str, np.ndarray]): The input image, either as a file path or a NumPy array.
        plot_image (bool, optional): Whether to display the image with landmarks (defaults to False).

    Returns:
        landmark_pb2.NormalizedLandmarkList: The detected face landmarks as a `NormalizedLandmarkList` object.

    Raises:
        TypeError: If the input types are incorrect.
    """

    # Input validation
    if not isinstance(image_input, (str, np.ndarray)):
        raise TypeError("image_input must be a string (file path) or a NumPy array.")
    if not isinstance(plot_image, bool):
        raise TypeError("plot_image must be a boolean.")

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
    annotated_image = visualize_landmarks_on_image(image.numpy_view(), detection_result)

    # Display the annotated image
    if plot_image:
        plt.imshow(annotated_image)
        plt.show()

    return detection_result.face_landmarks[0]
