import open3d as o3d
import numpy as np
from src.features.face_landmark_alignment import face_landmark_alignment
from src.visualization.visualize_mesh import visualize_mesh
from src.data.landmark_correspondences import landmark_points_68


def mesh_loss_function(
    mesh: o3d.geometry.TriangleMesh,
    flame_landmark: o3d.geometry.PointCloud,
    landmark: o3d.geometry.PointCloud,
    visualize: bool = False,
    apply_loss: list[str] = ["landmarks", "mesh"],
    reduce_landmarks: bool = True,
) -> dict[str, float]:
    """
    Calculate the loss function for the given mesh and landmarks.

    Parameters:
        mesh (o3d.geometry.TriangleMesh): The input triangle mesh.
        flame_landmark (o3d.geometry.PointCloud): The flame landmark point cloud.
        landmark (o3d.geometry.PointCloud): The landmark point cloud.
        visualize (bool, optional): Whether to visualize the process. Defaults to False.
        apply_loss (List[str], optional): The loss functions to apply. Defaults to ["landmarks", "mesh"].
            options: ["landmarks", "mesh", "pointclouds"]
        reduce_landmarks (bool, optional): Whether to reduce the landmarks. Defaults to True.

    Returns:
        Dict[str, float]: The calculated loss function values.
    """

    # Align mesh and landmark
    face_landmark_alignment(flame_landmark, landmark)

    # Convert apply_loss to lower case for consistent comparison
    apply_loss = [x.lower() for x in apply_loss]

    loss = {}

    # Calculate landmarks loss if specified
    if "landmarks" in apply_loss:
        loss["landmarks"] = landmark_to_landmark_distances(
            flame_landmark, landmark
        ).mean()

    # Reduce landmarks if specified
    if reduce_landmarks and ("mesh" in apply_loss or "pointclouds" in apply_loss):
        landmarks_Vertex = np.array(landmark.points)
        landmarks_Vertex = np.delete(landmarks_Vertex, landmark_points_68, axis=0)
        landmark = o3d.geometry.PointCloud()
        landmark.points = o3d.utility.Vector3dVector(landmarks_Vertex.reshape(-1, 3))
        landmark.colors = o3d.utility.Vector3dVector(
            np.ones((landmarks_Vertex.shape[0], 3)) * [0.0, 1.0, 0.0]
        )

    # Calculate mesh loss if specified
    if "mesh" in apply_loss:
        loss["mesh"] = mesh_to_pointcloud_distances(mesh, landmark).mean()

    # Calculate pointclouds loss if specified
    if "pointclouds" in apply_loss:
        loss["pointclouds"] = pointcloud_to_pointcloud_distances(
            mesh, landmark
        ).mean()

    # Visualize mesh if specified
    if visualize:
        visualize_mesh([mesh, landmark])

    return loss

def mesh_to_pointcloud_distances(o3d_mesh: o3d.t.geometry.TriangleMesh,
                                   cloud: o3d.t.geometry.PointCloud) -> np.ndarray:
    """
    Compute signed distances from a mesh to a point cloud.

    Args:
        o3d_mesh (o3d.t.geometry.TriangleMesh): The input mesh.
        cloud (o3d.t.geometry.PointCloud): The input point cloud.

    Returns:
        np.ndarray: Array of signed distances from the mesh to each point in the cloud.
    """
    # Convert Open3D legacy meshes to tensors
    tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(o3d_mesh)
    tensor_pointcloud = o3d.t.geometry.PointCloud.from_legacy(cloud)

    # Create a raycasting scene
    scene = o3d.t.geometry.RaycastingScene()

    # Add the mesh to the scene
    _ = scene.add_triangles(tensor_mesh)

    # Compute signed distances from the mesh to the point cloud
    sdf = scene.compute_distance(tensor_pointcloud.point.positions)

    # Convert the tensor to a numpy array and return the result
    return sdf.numpy()

def landmark_to_landmark_distances(flame_landmark: o3d.geometry.PointCloud,
                                   landmark: o3d.geometry.PointCloud) -> np.ndarray:
    """
    Calculate distances between landmarks in two point clouds.

    Args:
        flame_landmark (o3d.geometry.PointCloud): The flame landmark point cloud.
        landmark (o3d.geometry.PointCloud): The mediapipe landmark point cloud.

    Returns:
        np.ndarray: Array of distances between corresponding points in the two point clouds.
    """

    # Calculate distances between landmarks in two point clouds.
    # Loop over landmarks and calculate distances between corresponding points.
    landmark_distances = []  # List to store distances.

    # Loop over landmarks and calculate distances between corresponding points.
    for i in range(len(landmark_points_68)):
        # Calculate distance between two points.
        distance = np.linalg.norm(flame_landmark.points[i] - landmark.points[landmark_points_68[i]])

        # Append distance to list.
        landmark_distances.append(distance)
    
    return np.array(landmark_distances)

def pointcloud_to_pointcloud_distances(mesh: o3d.geometry.TriangleMesh, landmarks: o3d.geometry.PointCloud) -> np.ndarray:
    """
    Calculate the distances between each vertex of a mesh and each landmark in a point cloud.

    Args:
        mesh (o3d.geometry.TriangleMesh): The input mesh.
        landmarks (o3d.geometry.PointCloud): The input point cloud.

    Returns:
        np.ndarray: Array of distances between each vertex and each landmark.
    """
    
    # Convert mesh and landmark vertices to numpy arrays
    Mesh_Vertex = np.array(mesh.vertices)
    landmarks_Vertex = np.array(landmarks.points)

    # Calculate distances between mesh vertices and landmarks
    distances = np.linalg.norm(Mesh_Vertex[:, None, :] - landmarks_Vertex[None, :, :], axis=2)
    
    # Find the indices of the closest landmark for each mesh vertex
    closest_indices = np.argmin(distances, axis=1)
    
    # Calculate the distances to the closest landmarks
    closest_distances = distances[range(len(Mesh_Vertex)), closest_indices]

    return closest_distances
