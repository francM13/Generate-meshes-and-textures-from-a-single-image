# Import libraries
import matplotlib.pyplot as plt
from PIL import Image
import os

def image_plot(image_path,verbose=True):
    """
    Plots the image with the given path.

    Args:
        image_path (str): Path to the image file.
        verbose (bool, optional): If True, print the image dimensions. Defaults to True.
    """   
    
    if os.path.exists(image_path):
        try:
            # Open the image with Pillow
            image = Image.open(image_path)

            # Convert image to a format compatible with matplotlib
            image_rgb = image.convert("RGB")

            # Display the image inline
            plt.imshow(image_rgb)
            plt.axis('off')  # Hide unnecessary axis labels and ticks
            plt.show()

            if verbose:
                image_width, image_height = image.size
                print(f"Image size: {image_width} x {image_height} pixels")

        except FileNotFoundError:
            print(f"Error: Image file not found: {image_path}")
    else:
        print(f"Image file not found: {image_path}")