import numpy as np
import pyvista as pv
from geoUtils import *

import numpy as np
import pyvista as pv

resolution = 5
def create_colored_pyvista_mesh(height_array, rgb_image_array):
    """
    Creates a colored PyVista mesh from height and an RGB image array.

    Args:
        height_array (np.ndarray): A 2D NumPy array representing heights (DEM).
        rgb_image_array (np.ndarray): A 3D NumPy array representing RGB colors
                                       with shape (rows, cols, 3), values typically 0-255.
    """

    rows, cols = height_array.shape

    # Check if height_array and rgb_image_array dimensions match
    if rgb_image_array.shape[0] != rows or rgb_image_array.shape[1] != cols or rgb_image_array.shape[2] != 3:
        raise ValueError(
            f"RGB image array shape {rgb_image_array.shape} does not match "
            f"expected (height_rows, height_cols, 3) or is not an RGB image."
        )

    # 1. Create X, Y coordinates for the grid
    x_coords = np.arange(cols) * resolution
    y_coords = np.arange(rows) * resolution
    X, Y = np.meshgrid(x_coords, y_coords)

    # 2. Create the 3D points (X, Y, Z)
    points = np.c_[X.ravel(), Y.ravel(), height_array.ravel()]

    # 3. Create a StructuredGrid
    grid = pv.StructuredGrid(X, Y, np.flip(height_array,0))

    # 4. Prepare Colors from the RGB image array
    # Flatten the RGB image array from (rows, cols, 3) to (N, 3)
    #colors = rgb_image_array.reshape(-1, 3) # -1 infers the first dimension (rows*cols)
    # For grayscale, R=G=B. So, we repeat the normalized grayscale value three times.
    t_rgb_image_array = np.flip(np.flip(rgb_image_array.transpose((1,0,2)),0),1)
    colors = np.vstack((np.flip(t_rgb_image_array[:,:,0],0).flatten(),
                        np.flip(t_rgb_image_array[:,:,1],0).flatten(),
                        np.flip(t_rgb_image_array[:,:,2],0).flatten())).transpose()
    # Normalize colors to 0.0-1.0 range if they are 0-255
    if colors.max() > 1.0:
        colors = colors / 255.0

    # Ensure colors are float32 as typically preferred by VTK/PyVista for textures/colors
    colors = colors.astype(np.float32)
    
    # 5. Add colors to the grid's point data
    grid["colors"] = colors

    

    # 6. Create a plotter and add the mesh
    plotter = pv.Plotter(window_size=[1024, 768])
    plotter.set_background(color='000000')#'78d5e1')
    # Use rgb=True because we are providing per-point RGB colors
    plotter.add_mesh(grid, scalars="colors", rgb=True, show_edges=False, smooth_shading=True)

    # Add a title
    plotter.add_title("PyVista Mesh from DEM with RGB Image Colors")

    # Show the axes
    plotter.add_axes()

    # Display the plot
    plotter.show()

# --- Sample Usage ---
if __name__ == "__main__":

    rgbFile = './dem_data/L2_RGB.tif'
    demFile = './dem_data/L2_DEM.tif'

    rgbTiff = Geotiff(rgbFile)
    demTiff = Geotiff(demFile)

    height_data  = demTiff.tiff.read(1)
    rgb_image = np.stack([rgbTiff.tiff.read(1),rgbTiff.tiff.read(2),rgbTiff.tiff.read(2)]).transpose((1,2,0))
    rgb_image = (rgb_image/rgb_image.max()).astype('float32')
 
    create_colored_pyvista_mesh(height_data, rgb_image)