import open3d as o3d
import numpy as np
from src.features.face_landmark_alignment import face_landmark_alignment
from src.visualization.visualize_mesh import visualize_mesh
from src.data.landmark_correspondences import landmark_points_68


def mesh_loss_function(mesh:o3d.geometry.TriangleMesh,flame_landmark:o3d.geometry.PointCloud,landmark:o3d.geometry.PointCloud,visualize:bool = False):
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
    
    # Calculate distances between mesh landmarks and landmarks
    all_distances = []
    for i in range(len(landmark_points_68)):
        distance = np.linalg.norm(flame_landmark.points[i] - landmark.points[landmark_points_68[i]])
        all_distances.append(distance)

    # Convert mesh and landmark to numpy arrays
    Mesh_Vertex=np.array(mesh.vertices)
    landmarks_Vertex=np.array(landmark.points)
    landmarks_Vertex=np.delete(landmarks_Vertex, landmark_points_68, axis=0)

    
    # Calculate distances between mesh vertices and landmarks
    distances = np.linalg.norm(Mesh_Vertex[:, None] - landmarks_Vertex[None, :], axis=2)
    
    # Find the closest landmark for each mesh vertex
    closest_indices = np.argmin(distances, axis=0)
    
    # Calculate the distances to the closest landmarks
    closest_distances = distances[closest_indices, range(len(landmarks_Vertex))]
    
    # Visualize the mesh and landmarks if required
    if(visualize):
        visualize_mesh([mesh,landmark])
    
    # Return the mean of the closest distances
    
    return np.array(all_distances).mean() + closest_distances.mean()