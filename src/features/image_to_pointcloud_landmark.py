from src.features.generate_face_landmarks import generate_face_landmarks
import numpy as np
import open3d as o3d

def image_to_pointcloud_landmark(image, plot_image=False,plot_pointcloud=False):
    """
    Converts image to 3D point-cloud landmark coordinates.

    Args:
        image (str or np.ndarray): Path to the image file or the image data itself.
        plot_image (bool): Whether to display the image or not.
        plot_pointcloud (bool): Whether to display the point-cloud or not.

    Returns:
        The 3D point-cloud coordinates or an error occurs.
    """

    # Generate face landmarks
    face_landmarks = generate_face_landmarks(image, plot_image)

    # Convert face landmarks to 3D object coordinates
    face_landmarks_points = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks])

    # Create point cloud with green color
    color_array = np.ones((face_landmarks_points.shape[0], 3)) * [0.0, 1.0, 0.0] 
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(face_landmarks_points.reshape(-1,3))
    point_cloud.colors = o3d.utility.Vector3dVector(color_array)

    if(plot_pointcloud):
        o3d.visualization.draw_plotly([point_cloud])

    return point_cloud