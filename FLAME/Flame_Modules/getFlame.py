import numpy as np
import torch

from flame_pytorch import FLAME, get_config

# Create a cache dictionary to store computed models
flame_cache = {}

def getFlame(shape_params=torch.rand(1,100,dtype=torch.float32),# shape parameters
             pose_params = torch.tensor([[1, 0, 1, torch.rand(1),0,0]], dtype=torch.float32), # pose parameters
             expression_params = torch.rand(1, 50, dtype=torch.float32),# expression parameters
             batch_size=1,# batch size
             ):
    """
    This function returns the vertices, landmarks, and faces of the mesh generated using FLAME 
    based on the given shape, pose, and expression parameters. It also has 
    options to visualize the model and landmarks. 

    Parameters:
    - shape_params: torch.Tensor, optional, default=torch.rand(1,100,dtype=torch.float32)
        The shape parameters for the FLAME model.
    - pose_params: torch.Tensor, optional, default=torch.tensor([[1, 0, 1, torch.rand(1),0,0]], dtype=torch.float32)
        The pose parameters for the FLAME model.
    - expression_params: torch.Tensor, optional, default=torch.rand(1, 50, dtype=torch.float32)
        The expression parameters for the FLAME model.
    - batch_size: int, optional, default=1
        The batch size for the FLAME model.

    Returns:
    - vertices: torch.Tensor
        The vertices of the FLAME model.
    - faces: torch.Tensor
        The faces of the FLAME model.
    - landmarks: torch.Tensor
        The landmarks of the FLAME model.
    """

    # Convert input parameters to torch tensors
    shape_params = torch.as_tensor(shape_params, dtype=torch.float32)
    pose_params = torch.as_tensor(pose_params, dtype=torch.float32)
    expression_params = torch.as_tensor(expression_params, dtype=torch.float32)

    # Get configuration and set batch size
    config = get_config()
    config.batch_size=batch_size

    
    # Check if the model for the given batch_size is already initialize and cached
    if batch_size in flame_cache:
        flamelayer = flame_cache[batch_size]
    else:
        # Initialize FLAME model
        flamelayer = FLAME(config)
        # Cache the model for the given batch_size
        flame_cache[batch_size] = flamelayer

    # Generate vertices and landmarks
    vertice, landmark = flamelayer(
        shape_params, expression_params, pose_params
    )
    faces = flamelayer.faces


    return vertice,faces,landmark
