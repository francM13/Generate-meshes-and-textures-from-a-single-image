from PIL import Image
import os

def resize_and_save_image(image_path, output_folder, new_name, new_width=256, new_height=256, verbose=False):
  """
  Resizes an image, saves it to a new folder with a new name.

  Args:
      image_path: Path to the image file.
      output_folder: Path to the folder where the resized image will be saved.
      new_name: The new name for the resized image (without extension).
      new_width: The desired width of the resized image. Default is 256
      new_height: The desired height of the resized image. Default is 256
      verbose: If True, print the new file path. Default is False
  """
  # Open the image
  try:
    image = Image.open(image_path)
  except FileNotFoundError:
    print(f"Error: Image file not found: {image_path}")
    return

  # Create the output folder if it doesn't exist
  if not os.path.exists(output_folder):
    os.makedirs(output_folder)

  # Resize the image
  resized_image = image.resize((new_width, new_height))

  # Get the original file extension
  filename, extension = os.path.splitext(image_path)

  # Construct the new file path
  new_file_path = os.path.join(output_folder, f"{new_name}{extension}")

  # Save the resized image
  resized_image.save(new_file_path)
  if verbose: 
    print(f"Image resized and saved to: {new_file_path}")
