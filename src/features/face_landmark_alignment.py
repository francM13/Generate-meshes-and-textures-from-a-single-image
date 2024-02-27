import numpy as np
import copy
import open3d as o3d
from scipy.optimize import minimize

def face_landmark_alignment(flame_landmark: o3d.geometry.PointCloud, landmark: o3d.geometry.PointCloud) -> None:
    """
    A function to align the flame mesh and the landmark.

    Parameters:
        flame_landmark (o3d.geometry.PointCloud): The flame landmark.
        landmark (o3d.geometry.PointCloud): The landmark.

    Returns:
        o3d.geometry.TriangleMesh: The aligned flame mesh.
    """
    ## Target Points ([left eye - right eye - nose])
    flame_target = np.array([36,45,30])
    landmark_target = np.array([33,263,4])

    def Loss_alignment(x):
        """
        A function to calculate the loss for alignment.
        
        Parameters:
            x (numpy.array): The input array for transformation.

        Returns:
            numpy.float64: The calculated loss.
        """
        # Copy the landmark
        landmark_copy=copy.deepcopy(landmark)
        x=x.reshape(3,3)
        
        # Rotation
        rotate_x=o3d.geometry.get_rotation_matrix_from_xyz([x[0,0],0,0])
        rotate_y=o3d.geometry.get_rotation_matrix_from_xyz([0,x[0,1],0])
        rotate_z=o3d.geometry.get_rotation_matrix_from_xyz([0,0,x[0,2]])

        # Translation and Scale
        translate=x[1,:3]
        scale=x[2,0]

        # Alignment
        landmark_copy.scale(scale,center=(0,0,0))
        landmark_copy.translate(translate.reshape(-1))
        landmark_copy.rotate(rotate_x)
        landmark_copy.rotate(rotate_y)
        landmark_copy.rotate(rotate_z)
        
        # Calculate Loss
        Loss=np.sum([np.linalg.norm(landmark_copy.points[landmark_target[0]]-flame_landmark.points[flame_target[0]]),
                    np.linalg.norm(landmark_copy.points[landmark_target[1]]-flame_landmark.points[flame_target[1]]),
                    np.linalg.norm(landmark_copy.points[landmark_target[2]]-flame_landmark.points[flame_target[2]])])
        return Loss
    
    # Initial guess
    x0 = np.zeros([3,3]).reshape(-1)  

    # Optimize
    res = minimize(Loss_alignment, x0, tol=0.000001)

    # Result
    optimal_x = res.x
    optimal_x=optimal_x.reshape(3,3)

    # Get Alignment parameters
    rotate_x=o3d.geometry.get_rotation_matrix_from_xyz([optimal_x[0,0],0,0])
    rotate_y=o3d.geometry.get_rotation_matrix_from_xyz([0,optimal_x[0,1],0])
    rotate_z=o3d.geometry.get_rotation_matrix_from_xyz([0,0,optimal_x[0,2]])
    translate=optimal_x[1,:3]
    scale=optimal_x[2,0]

    # Align
    # landmark=copy.deepcopy(landmark)
    landmark.scale(scale,center=(0,0,0))
    landmark.translate(translate.reshape(-1))
    landmark.rotate(rotate_x)
    landmark.rotate(rotate_y)
    landmark.rotate(rotate_z)

    return