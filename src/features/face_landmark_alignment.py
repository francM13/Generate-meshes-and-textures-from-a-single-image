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
    ## Target Points ([left eye - right eye - nose - left mouth corner - right mouth corner])
    flame_target = np.array([36,45,30,48,54])
    landmark_target = np.array([33,263,4,78,308])

    #Center
    center=flame_landmark.points[flame_target[2]]

    # Translation
    translation_vector = flame_landmark.points[flame_target[2]] - landmark.points[landmark_target[2]]
    landmark.translate(translation_vector)

    #Scale
    distance_1 = np.linalg.norm(flame_landmark.points[flame_target[0]] - flame_landmark.points[flame_target[1]])
    distance_2 = np.linalg.norm(landmark.points[landmark_target[0]] - landmark.points[landmark_target[1]])

    scale_factor = distance_1 / distance_2
    landmark.scale(scale_factor, center=center)


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
        
        # Rotation
        rotate_x=o3d.geometry.get_rotation_matrix_from_xyz([x[0],0,0])
        rotate_y=o3d.geometry.get_rotation_matrix_from_xyz([0,x[1],0])
        rotate_z=o3d.geometry.get_rotation_matrix_from_xyz([0,0,x[2]])

        # Alignment
        landmark_copy.rotate(rotate_x,center=center)
        landmark_copy.rotate(rotate_y,center=center)
        landmark_copy.rotate(rotate_z,center=center)
        
        # Calculate Loss
        Loss=np.sum([np.linalg.norm(landmark_copy.points[landmark_target[0]]-flame_landmark.points[flame_target[0]]),
                    np.linalg.norm(landmark_copy.points[landmark_target[1]]-flame_landmark.points[flame_target[1]]),
                    np.linalg.norm(landmark_copy.points[landmark_target[2]]-flame_landmark.points[flame_target[2]]),
                    np.linalg.norm(landmark_copy.points[landmark_target[3]]-flame_landmark.points[flame_target[3]]),
                    np.linalg.norm(landmark_copy.points[landmark_target[4]]-flame_landmark.points[flame_target[4]])])
        return Loss
    
    # Initial guess
    x0 = np.zeros(3).reshape(-1)  

    # Optimize
    res = minimize(Loss_alignment, x0, tol=0.000001)

    # Result
    optimal_x = res.x

    # Get Alignment parameters
    rotate_x=o3d.geometry.get_rotation_matrix_from_xyz([optimal_x[0],0,0])
    rotate_y=o3d.geometry.get_rotation_matrix_from_xyz([0,optimal_x[1],0])
    rotate_z=o3d.geometry.get_rotation_matrix_from_xyz([0,0,optimal_x[2]])

    # Align
    landmark.rotate(rotate_x,center=center)
    landmark.rotate(rotate_y,center=center)
    landmark.rotate(rotate_z,center=center)

    return