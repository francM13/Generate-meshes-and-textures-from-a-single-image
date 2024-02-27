import numpy as np
import open3d as o3d
from typing import Optional,Union

def flame_mesh(vertices:np.ndarray, faces:np.ndarray, landmarks:Optional[np.ndarray] = None,flame_landmark:bool=True,plot_mesh:bool=False) -> Union[o3d.geometry.TriangleMesh, tuple[o3d.geometry.TriangleMesh, o3d.geometry.PointCloud]]:
    """
    Generate a triangle mesh from the given vertices and faces. Optionally, landmarks can be added to the mesh. 
    
    Args:
        vertices: An ndarray of vertices.
        faces: An ndarray of faces.
        landmarks: An optional ndarray of landmarks.
        flame_landmark: A boolean indicating whether to include flame landmarks. Default is True.
        plot_mesh: A boolean indicating whether to plot the mesh. Default is False.
    
    Returns:
        Either a TriangleMesh object or a tuple of TriangleMesh and PointCloud objects.
    """
    # Generate mesh
    vertices = o3d.utility.Vector3dVector(vertices.reshape(-1,3))
    faces = o3d.utility.Vector3iVector(faces.reshape(-1,3))

    mesh = o3d.geometry.TriangleMesh(vertices, faces)
    
    if(flame_landmark):
        # Generate flame landmarks
        landmark_color_array = np.ones((landmarks.shape[1], 3)) * [1.0, 0.0, 0.0] 
        landmark_pointcloud = o3d.geometry.PointCloud()
        landmark_pointcloud.points = o3d.utility.Vector3dVector(landmarks.reshape(-1,3))
        landmark_pointcloud.colors = o3d.utility.Vector3dVector(landmark_color_array)
        if plot_mesh:
            mesh_plot((mesh, landmark_pointcloud))
        return(mesh, landmark_pointcloud)
    else:
        if plot_mesh:
            mesh_plot(mesh)
        return mesh
    
def mesh_plot(mesh):
    if isinstance(mesh, o3d.geometry.TriangleMesh):
        o3d.visualization.draw_plotly([mesh])
    else:
        o3d.visualization.draw_plotly([mesh[0],mesh[1]])
