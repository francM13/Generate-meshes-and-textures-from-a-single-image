import requests
import numpy as np
import pyrender
import trimesh

class FlameInterface:
    """Class for interacting with the Flame API.

    Attributes:
        endpoint_url (str): The URL of the Flame API endpoint.
        port (int): The port number of the endpoint (default: 8081).
        batch_size (int): The batch size for the request (default: 1).

    Methods:
        getFlame(self, shape_params=None, pose_params=None, expression_params=None):
            Sends a request to the Flame API and returns the vertices, faces,
            and landmarks data.

            Args:
                shape_params (array, optional): Shape parameters for the request.
                    Expected dimensions: (batch_size, 1, 100).
                    Defaults to None.
                pose_params (array, optional): Pose parameters for the request.
                    Expected dimensions: (batch_size, 1, 6).
                    Defaults to None.
                expression_params (array, optional): Expression parameters for the request.
                    Expected dimensions: (batch_size, 1, 50).
                    Defaults to None.


            Returns:
                tuple: A tuple containing three NumPy arrays:
                    - vertices (np.ndarray): The vertex data.
                    - faces (np.ndarray): The face data.
                    - landmarks (np.ndarray): The landmark data.

        
        Render(self, vertices, faces, landmarks, render_landmarks=True):
            Renders the mesh with optional landmark visualization.
            
            Args:
                vertices (np.ndarray): The vertex data.
                faces (np.ndarray): The face data.
                landmarks (np.ndarray): The landmark data.
                render_landmarks (bool, optional): Whether to render landmarks. Defaults to True.

            Returns:
                None (visualization is displayed using pyrender.Viewer).
            """

    def __init__(self, endpoint_url="http://127.0.0.1", port=8081, batch_size=1):
        """Initializes the FlameInterface object.

        Args:
            endpoint_url (str, optional): The URL of the Flame API endpoint.
                Defaults to "http://127.0.0.1".
            port (int, optional): The port number of the endpoint. Defaults to 8081.
            batch_size (int, optional): The batch size for the request. Defaults to 1.
        """

        self.endpoint_url = f"{endpoint_url}:{port}/getFlame"
        self.batch_size = batch_size

    def getFlame(self, shape_params=None, pose_params=None, expression_params=None):
        """Sends a GET request to the Flame API and returns the vertices, faces,
        and landmarks data.

        Args:
            shape_params (array, optional): Shape parameters for the request.
                Expected dimensions: (batch_size, 1, 100).
                Defaults to None.
            pose_params (array, optional): Pose parameters for the request.
                Expected dimensions: (batch_size, 1, 6).
                Defaults to None.
            expression_params (array, optional): Expression parameters for the request.
                Expected dimensions: (batch_size, 1, 50).
                Defaults to None.

        Returns:
            tuple: A tuple containing three NumPy arrays:
                - vertices (np.ndarray): The vertex data.
                - faces (np.ndarray): The face data.
                - landmarks (np.ndarray): The landmark data.

        Raises:
            ConnectionError: If an error occurs during the network request.
        """

        # Prepare the payload
        payload = {}
        if shape_params is not None:
            payload['shape_params'] = shape_params.tolist()
        if pose_params is not None:
            payload['pose_params'] = pose_params.tolist()
        if expression_params is not None:
            payload['expression_params'] = expression_params.tolist()
        if self.batch_size != 1:
            payload['batch_size'] = self.batch_size

        # Set the request headers
        headers = {
            'Content-Type': 'application/json',
            'accept': 'application/json',
        }

        try:
            # Send the GET request
            response = requests.request(method="GET", url=self.endpoint_url, json=payload, headers=headers)
            response.raise_for_status()  # Raise an error for non-200 status codes

            # Process the response
            if response.status_code == 200:
                output = response.json()
                vertices = np.array(output[0])
                faces = np.array(output[1])
                landmarks = np.array(output[2])
                return vertices, faces, landmarks
            else:
                print(f"Error: {response.status_code},{response.reason}")
                raise ConnectionError(f"Failed to get data from Flame API: {response.reason}")

        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Error connecting to Flame API: {e}")

    def Render(self, vertices, faces, landmarks, render_landmarks=True):
        """Renders the mesh with optional landmark visualization.

        Args:
            vertices (np.ndarray): The vertex data.
            faces (np.ndarray): The face data.
            landmarks (np.ndarray): The landmark data.
            render_landmarks (bool, optional): Whether to render landmarks. Defaults to True.

        Returns:
            None (visualization is displayed using pyrender.Viewer).
        """

        # Ensure vertices and landmarks are 1D arrays
        vertices = vertices.squeeze()
        landmarks = landmarks.squeeze()

        # Create vertex colors for the mesh
        vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 1]

        # Create a Trimesh object with vertex colors
        tri_mesh = trimesh.Trimesh(vertices, faces, vertex_colors=vertex_colors)

        # Convert the Trimesh object to a Mesh object
        mesh = pyrender.Mesh.from_trimesh(tri_mesh)

        # Create a new scene
        scene = pyrender.Scene()

        # Add the mesh to the scene
        scene.add(mesh)

        if (render_landmarks):
            # Create a small sphere for landmark visualization
            sm = trimesh.creation.uv_sphere(radius=0.005)
            sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
        
            # Create transformations for each landmark and add them to the scene
            tfs = np.tile(np.eye(4), (len(landmarks), 1, 1))
            tfs[:, :3, 3] = landmarks
            joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
            scene.add(joints_pcl)

        # Display the scene using pyrender.Viewer with raymond lighting
        pyrender.Viewer(scene, use_raymond_lighting=True)
