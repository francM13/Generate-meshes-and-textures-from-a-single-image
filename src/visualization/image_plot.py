# Import libraries
import matplotlib.pyplot as plt
from PIL import Image
import os

def image_plot(image, verbose=True):
    """
    Plot the image with the given path.

    Args:
        image (str,array): Path to the image file or array containing an image.
        verbose (bool, optional): If True, print the image dimensions. Defaults to True.
    """  
    
    # Check if the image path exists
    if os.path.exists(image):
        try:
            # Open the image with Pillow
            image = Image.open(image)

            # Convert image to a format compatible with matplotlib
            image_rgb = image.convert("RGB")

            # Display the image inline
            plt.imshow(image_rgb)
            plt.axis('off')
            plt.show()

            # Print image dimensions if verbose is True
            if verbose:
                image_width, image_height = image.size
                print(f"Image size: {image_width} x {image_height} pixels")

        except FileNotFoundError:
            print(f"Error: Image file not found: {image}")
    else:
        # Display the image directly if it's not a file
        plt.imshow(image)
        plt.axis('off')
        plt.show()
