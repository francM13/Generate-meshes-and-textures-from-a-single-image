# Generate meshes and textures from a single image

The goal of the project is to easily create a head mesh for a 3D environment with an associated texture starting from a single image as input.<br>
The goal is achieved using the AI ​​method and various external tools.<br>

## Installation

### FLAME container

To generate the head mesh, [FLAME](https://flame.is.tue.mpg.de/index.html "FLAME") is used, a tool that parametrically generates the entire head.<br>
[FLAME](https://flame.is.tue.mpg.de/index.html "FLAME") is implemented within a Container to be used independently of the project environment.<br>

1. Download the [FLAME 2020 templates](https://flame.is.tue.mpg.de/download.php "FLAME 2020 templates") (authentication is required)
2. Paste the downloaded file into the "FLAME/Model" folder.
3. Build the Docker image inside the “FLAME” folder and run it on port “8081”.

### Environment

To use the tool we recommend installing the conda environment.<br>

```console".
conda env create -f environment.yml
conda activates texture-env
```

> [!WARNING]
> The Mediapipe model is downloaded automatically, the download link may be disabled, in this case manually download the [Mediapipe face Landmark model](https://developers.google.com/mediapipe/solutions/vision/face_landmarker "Mediapipe face reference model") and add it to the repository folder.<br>

## How to use

### Mesh

To generate the mesh, first open the jupyter notebook `notebooks/1.2-mesh-GA-generation.ipynb`, then provide the file path to the GA provider using the following method.<br>

```python
mesh_GA.setRefereneImage(image = "Your_File_Path")
```

Then run and save the mesh as specified in the notebook.<br>
Here is an example of the generated Mesh.<br>
![Mesh Example](reports\figures\Example_Mesh.jpg)

### Texture

To generate the texture, open the Jupyter notebook `notebooks/1.3-texture- generation.ipynb`, open the image file as follows.<br>

```python
input_img = tf_io.read_file("Your_file_path")
input_img = tf_io.decode_png(input_img, channels=3)
input_img = tf_image.convert_image_dtype(input_img, "float32")
input_img = tf_image.resize(input_img, [256,256])
texture_image(input_img)
input_img=expand_dims(input_img, axis=0)
```

Then serve as input the image to the neural network as shown in the notebook.<br>
Here is an example of the generated UV texture.<br>
![Example Mesh](reports\figures\Example_Texture.jpg)

### Final results

The application of the texture on the mesh is not performed independently, to apply it use an external tool such as [Blender](https://www.blender.org/ "Blender").<br>
Here is an example of the final result.<br>
![Example Result](reports\figures\Example_Result.jpg)

## Disclaimer

This project is under the MIT License, so it is provided "as is" without warranty of any kind.<br>
Future updates are not guaranteed.<br>
