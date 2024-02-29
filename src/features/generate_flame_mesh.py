from src.models.flame_interface import flame_interface
import numpy as np
from src.features.flame_mesh import flame_mesh
from src.visualization.visualize_mesh import visualize_mesh


flame_interface_pointer=None

import numpy as np

def generate_flame_mesh(shape_params:np.ndarray=None, pose_params:np.ndarray=None, expression_params:np.ndarray=None, endpoint_url:str=None, port:int=None, visualize:bool=False) -> tuple:
    """
    Generate a flame mesh based on shape, pose, and expression parameters.

    Args:
        - shape_params (np.ndarray, optional): Array of shape parameters.
        - pose_params (np.ndarray, optional): Array of pose parameters.
        - expression_params (np.ndarray, optional): Array of expression parameters.
        - endpoint_url (str, optional): URL of the endpoint.
        - port (int, optional): Port number.
        - visualize (bool, optional): Whether to visualize the mesh.

    Returns:
        - tuple: Tuple containing the flame mesh vertices, faces, and landmarks.
    """

    global flame_interface_pointer

    # If flame interface pointer is not initialized, initialize it
    if flame_interface_pointer is None:
        flame_inputs = {}
        if endpoint_url is not None:
            flame_inputs["endpoint_url"] = endpoint_url
            
        if port is not None:
            flame_inputs["port"] = port
        
        flame_interface_pointer = flame_interface(**flame_inputs)
    
    # Prepare inputs for generating the flame mesh
    mesh_inputs = {}
    if shape_params is not None:
        mesh_inputs["shape_params"] = shape_params
    if pose_params is not None:
        mesh_inputs["pose_params"] = pose_params
    if expression_params is not None:
        mesh_inputs["expression_params"] = expression_params

    # Generate the flame mesh
    Flame_vertices, Flame_faces, Flame_landmarks = flame_interface_pointer.getFlame(**mesh_inputs)

    # Create a flame mesh object
    Mesh = flame_mesh(Flame_vertices, Flame_faces, Flame_landmarks)

    # Visualize the mesh if required
    if visualize:
        visualize_mesh([Mesh[0], Mesh[1]])

    return Mesh
