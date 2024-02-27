import open3d as o3d
from typing import Union

def visualize_mesh(x: Union[o3d.geometry.TriangleMesh, o3d.geometry.PointCloud,list]) -> None:
    """
    A function to visualize the mesh.
    
    Parameters:
        x: The input mesh for transformation.
    """
    if isinstance(x, o3d.geometry.TriangleMesh) or isinstance(x, o3d.geometry.PointCloud):
        o3d.visualization.draw_plotly([x])
    else :
        o3d.visualization.draw_plotly(x)