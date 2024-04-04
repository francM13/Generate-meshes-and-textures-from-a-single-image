from super_image import EdsrModel, ImageLoader
from PIL import Image
import numpy as np
import os

def upscale(image, save=None):
    """
    Upscale the input image using the EDSR base model.

    Args:
        image (PIL.Image.Image or numpy.ndarray): The input image. If it is a path to an image, the function will open it.
        save (str, optional): The path to save the upscaled image. Defaults to None.

    Returns:
        The upscaled image that is saved as 'upscaled_texture.png' in the current directory too.
    """
    # Check if the input is an image path or a numpy array
    if isinstance(image, str):
        # If it is a path, open the image
        image = Image.open(image)
    else:
        if image.max() <= 1:
            image = Image.fromarray((image * 255).astype(np.uint8))

    # Load the EDSR base model with a scale of 4
    model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=4) 
    
    # Load the image into the model's input format
    inputs = ImageLoader.load_image(image)

    # Predict the upscaled image
    preds = model(inputs)

    # Save the upscaled image
    if save is not None:
        # Save the upscaled image to the specified path
        ImageLoader.save_image(preds, save)

    # Return the upscaled image
    return preds



