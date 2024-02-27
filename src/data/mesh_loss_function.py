import open3d as o3d
import numpy as np
from src.features.face_landmark_alignment import face_landmark_alignment
from src.visualization.visualize_mesh import visualize_mesh


def mesh_loss_function(mesh:o3d.geometry.TriangleMesh,flame_landmark:o3d.geometry.PointCloud,landmark:o3d.geometry.PointCloud,visualize:bool = False) -> float:
    """	
    Calculate the loss function for the given mesh and landmarks.
     
	Parameters:
	    mesh (o3d.geometry.TriangleMesh): The input triangle mesh.
	    flame_landmark (o3d.geometry.PointCloud): The flame landmark point cloud.
	    landmark (o3d.geometry.PointCloud): The landmark point cloud.
	    visualize (bool, optional): Whether to visualize the process. Defaults to False.
         
	Returns:
	    float: The calculated loss function value.
    """
    # Align mesh and landmark
    face_landmark_alignment(flame_landmark,landmark)

    # Convert mesh to pointcloud
    mesh_pountcloud = o3d.geometry.PointCloud()
    mesh_pountcloud.points = mesh.vertices     

    # Calculate distances
    distances = o3d.geometry.PointCloud.compute_point_cloud_distance(mesh_pountcloud,landmark)
    distances=np.array(distances)

    if(visualize):
        visualize_mesh([mesh,landmark])

    return distances.mean()