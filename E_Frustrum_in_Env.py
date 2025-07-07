import numpy as np
import pyvista as pv
import rasterio.transform
from geoUtils import *

import numpy as np
import pyvista as pv

resolution = 5

class WASDCameraControl:
    """
    A class to manage WASD camera movement in a PyVista plotter.
    """
    def __init__(self, plotter, move_speed=0.1, rotate_speed=5.0):
        self.plotter = plotter
        self.move_speed = move_speed
        self.rotate_speed = rotate_speed
        self.camera = plotter.camera
        self._bind_keys()
        self.is_moving = False # To control continuous movement if desired (more advanced)

        # Print initial camera state for debugging/understanding
        print(f"Initial Camera Position: {self.camera.position}")
        print(f"Initial Camera Focal Point: {self.camera.focal_point}")
        print(f"Initial Camera View Up: {self.camera.GetViewUp()}")

    def _bind_keys(self):
        """Binds WASD and other control keys to camera movement methods."""
        print("Binding WASD, QE (roll), RF (up/down) keys for camera control.")
        print("Press 'C' to print current camera coordinates.")
        self.plotter.add_key_event('w', self._move_forward)
        self.plotter.add_key_event('s', self._move_backward)
        self.plotter.add_key_event('a', self._strafe_left)
        self.plotter.add_key_event('d', self._strafe_right)
        self.plotter.add_key_event('q', self._roll_left)
        self.plotter.add_key_event('e', self._roll_right)
        self.plotter.add_key_event('r', self._move_up)
        self.plotter.add_key_event('f', self._move_down)
        self.plotter.add_key_event('c', self._print_camera_coords)

    def _get_camera_vectors(self):
        """Helper to get normalized camera direction and right vectors."""
        pos = np.array(self.camera.position)
        foc = np.array(self.camera.focal_point)
        up = np.array(self.camera.GetViewUp())

        # Direction vector (from position to focal point)
        direction = foc - pos
        direction = direction / np.linalg.norm(direction)

        # Right vector (perpendicular to direction and up)
        # Using cross product of direction and up to get the right vector
        right = np.cross(direction, up)
        right = right / np.linalg.norm(right)

        # Recalculate true up vector after getting right vector
        # This ensures 'up' is orthogonal to 'direction' and 'right'
        # which can prevent 'roll' drift when moving.
        true_up = np.cross(right, direction)
        true_up = true_up / np.linalg.norm(true_up)

        return direction, right, true_up

    def _update_camera(self, delta_pos):
        """Applies a translation vector to both position and focal point."""
        self.camera.position = np.array(self.camera.position) + delta_pos
        self.camera.focal_point = np.array(self.camera.focal_point) + delta_pos
        self.plotter.render() # Re-render the scene to show changes

    def _move_forward(self):
        direction, _, _ = self._get_camera_vectors()
        self._update_camera(direction * self.move_speed)

    def _move_backward(self):
        direction, _, _ = self._get_camera_vectors()
        self._update_camera(-direction * self.move_speed)

    def _strafe_left(self):
        _, right, _ = self._get_camera_vectors()
        self._update_camera(-right * self.move_speed)

    def _strafe_right(self):
        _, right, _ = self._get_camera_vectors()
        self._update_camera(right * self.move_speed)

    def _move_up(self):
        # Move along the world's Z-axis (or whichever is 'up' in your scene)
        self._update_camera(np.array([0, 0, 1]) * self.move_speed)

    def _move_down(self):
        self._update_camera(np.array([0, 0, -1]) * self.move_speed)

    def _roll_left(self):
        # Roll involves rotating the camera's 'view_up' vector
        self.camera.roll(self.rotate_speed) # Rolls by rotating around the view vector
        self.plotter.render()

    def _roll_right(self):
        self.camera.roll(-self.rotate_speed) # Rolls by rotating around the view vector
        self.plotter.render()

    def _print_camera_coords(self):
        print(f"\n--- Current Camera Coords ---")
        print(f"Position: {self.camera.position}")
        print(f"Focal Pt: {self.camera.focal_point}")
        print(f"View Up:  {self.camera.GetViewUp()}")
        print(f"-----------------------------\n")

def create_colored_pyvista_mesh(demFile, rgbFile, cameraPos, cameraDir):
    """
    Creates a colored PyVista mesh from height and an RGB image array.

    Args:
        height_array (np.ndarray): A 2D NumPy array representing heights (DEM).
        rgb_image_array (np.ndarray): A 3D NumPy array representing RGB colors
                                       with shape (rows, cols, 3), values typically 0-255.
    """

    ########################################
    ############################ Import tiff
    ########################################
    rgbTiff = Geotiff(rgbFile)
    demTiff = Geotiff(demFile)

    height_array  = demTiff.tiff.read(1)
    rgb_image_array = np.stack([rgbTiff.tiff.read(1),rgbTiff.tiff.read(2),rgbTiff.tiff.read(2)]).transpose((1,2,0))
    rgb_image_array = (rgb_image/rgb_image.max()).astype('float32')

    rows, cols = height_array.shape

    # Check if height_array and rgb_image_array dimensions match
    if rgb_image_array.shape[0] != rows or rgb_image_array.shape[1] != cols or rgb_image_array.shape[2] != 3:
        raise ValueError(
            f"RGB image array shape {rgb_image_array.shape} does not match "
            f"expected (height_rows, height_cols, 3) or is not an RGB image."
        )

    ########################################
    #########################  Setup surface
    ########################################
    # 1. Create X, Y coordinates for the grid
    x_coords = np.arange(cols) * resolution
    y_coords = np.arange(rows) * resolution
    X, Y = np.meshgrid(x_coords, y_coords)

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

    ########################################
    #########################  Setup Camera
    ########################################

    lat,lon = cvtCoord_deg2dec(cameraPos[0],cameraPos[1])
    print(lat,lon)
    #openPin(lat,lon)
    Cy,Cx = rasterio.transform.rowcol(demTiff.transform,lon,lat) # In image space Row:y Col:x
    Ch = height_array[-Cy,Cx] + 1
    print(Cy,Cx,Ch)

    sphere = pv.Sphere(radius=1.0, phi_resolution=20, theta_resolution=20)
    sphere.translate([Cx*resolution,(rows+Cy)*resolution,Ch], inplace=True) # Move 2 units along X-axis
    print([Cx*resolution,(rows+Cy)*resolution,Ch])
    ########################################
    ########################### Render Scene
    ########################################
    # 6. Create a plotter and add the mesh
    

    # Set a very small near clipping plane and a large far clipping plane
    # This gives you a wide range to work with.
    # Note: Too small a near plane can cause Z-fighting with very distant objects.
    
    plotter = pv.Plotter(window_size=[1024, 768])
    plotter.set_background(color='ffffff')#'78d5e1')
    # Use rgb=True because we are providing per-point RGB colors
    plotter.add_mesh(grid, scalars="colors", rgb=True, show_edges=False, smooth_shading=True)
    plotter.add_mesh(sphere, color='red', show_edges=False, smooth_shading=True)
    # Add a title
    plotter.add_title("PyVista Mesh from DEM with RGB Image Colors")

    # Show the axes
    plotter.add_axes()

    camera_controller = WASDCameraControl(plotter, move_speed=0.2, rotate_speed=2.0)
    # camera = plotter.camera
    # camera.clipping_range = (0.001, 100000.0)
    # Display the plot
    # plotter.set_interaction_slice_orthogonal()
    plotter.disable()
    plotter.show()

# --- Sample Usage ---
if __name__ == "__main__":
    camera_coord_deg =  ("37-50-44N","23-48-27E")
    rgbFile = './data_ignore/L2_RGB.tif'
    demFile = './data_ignore/L2_DEM.tif'

    rgbTiff = Geotiff(rgbFile)
    demTiff = Geotiff(demFile)

    height_data  = demTiff.tiff.read(1)
    rgb_image = np.stack([rgbTiff.tiff.read(1),rgbTiff.tiff.read(2),rgbTiff.tiff.read(2)]).transpose((1,2,0))
    rgb_image = (rgb_image/rgb_image.max()).astype('float32')
 
    create_colored_pyvista_mesh(demFile, rgbFile, camera_coord_deg, [0,0,0])