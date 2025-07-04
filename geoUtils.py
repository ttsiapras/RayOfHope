import rasterio
import webbrowser
from rasterio.warp import reproject, Resampling, calculate_default_transform
import numpy as np
import os

def openPin(latitude,longitude):
    # Create the Google Maps URL with a pin
    url = f"https://www.google.com/maps?q={latitude},{longitude}"
    # Open in the default web browser
    webbrowser.open(url)


class Geotiff():
    def __init__(self,path):
        self.tiff = rasterio.open(path)
        self.path = path
        self.transform          = self.tiff.transform
        self.inv_transform      = ~self.transform
        self.bounds             = self.tiff.bounds
        self.width, self.height = self.tiff.width, self.tiff.height

        # Reconstruct bounding box from transform
        left, top = self.transform * (0, 0)
        right, bottom = self.transform * (self.width, self.height)

        # Compare
        print("Imported GeoTiff:",path)
        print("W=",self.width,"H=",self.height)
        print("Bounds","l:",left,"b:", bottom,"r:", right,"t:", top)

    def coord2pixel(self,lat,long):
        return(self.inv_transform * (long, lat))

    def pixel2coord(self,X,Y):
        """
        @brief Reurns the Decimal coordinates of the pixel X,y with X=Column and Y=Row
        """
        return(self.transform * (X, Y))
    

def reproject_to_meters(input_geotiff_path, output_geotiff_path, target_crs_epsg='auto', target_resolution=None):
    """
    Reprojects a GeoTIFF to a Projected Coordinate System (meters) for accurate 3D aspect ratio.

    Args:
        input_geotiff_path (str): Path to the input GeoTIFF file (e.g., in degrees).
        output_geotiff_path (str): Path for the output reprojected GeoTIFF.
        target_crs_epsg (str or int): The EPSG code (e.g., 'EPSG:32634' for UTM Zone 34N,
                                      or 'auto' to let rasterio try to find a suitable UTM zone).
                                      For Greece, common options include UTM Zone 34N (EPSG:32634)
                                      or UTM Zone 35N (EPSG:32635), or a local grid like
                                      Hellenic Geodetic Reference System 1987 (EPSG:2100).
        target_resolution (tuple or float, optional): Desired output resolution in meters (e.g., (10, 10)
                                                    for 10x10 meter pixels, or 10 for square pixels).
                                                    If None, rasterio will calculate an appropriate default.
    Returns:
        str: Path to the newly created reprojected GeoTIFF.
    """
    with rasterio.open(input_geotiff_path) as src:
        # Determine if reprojection is necessary
        if src.crs.is_projected and src.crs.linear_units == 'metre':
            print(f"Input image '{input_geotiff_path}' is already in a projected CRS with meters ({src.crs.to_string()}).")
            # You might still want to reproject if 'target_crs_epsg' is different
            # or if 'target_resolution' is specified and needs to be enforced.
            if src.crs.to_string() == rasterio.crs.CRS.from_string(target_crs_epsg).to_string() and target_resolution is None:
                print("No reprojection needed. Exiting.")
                return input_geotiff_path

        print(f"Reprojecting '{input_geotiff_path}'...")

        # Determine the target CRS
        if target_crs_epsg == 'auto':
            # This is a very convenient feature of rasterio/pyproj to find a
            # suitable UTM zone based on the center of the input data.
            bounds = src.bounds
            center_lon = (bounds.left + bounds.right) / 2
            center_lat = (bounds.bottom + bounds.top) / 2
            # Use pyproj to find the best UTM zone
            try:
                import pyproj
                transformer = pyproj.Transformer.from_crs("EPSG:4326", "utm", always_xy=True)
                _, _, zone, _ = transformer.transform(center_lon, center_lat)
                hemisphere = 'N' if center_lat >= 0 else 'S'
                target_crs = rasterio.crs.CRS.from_string(f"EPSG:326{zone}" if hemisphere == 'N' else f"EPSG:327{zone}")
                print(f"Automatically determined target CRS: {target_crs.to_string()} (UTM Zone {zone}{hemisphere})")
            except (ImportError, Exception) as e:
                print(f"Could not automatically determine UTM zone ({e}). Falling back to a general projected CRS if input is geographic.")
                # Fallback to Web Mercator if auto-detection fails and input is geographic
                if src.crs.is_geographic:
                    target_crs = 'EPSG:3857' # Web Mercator, generally useful but has distortions
                    print(f"Using default fallback CRS: {target_crs}")
                else:
                    # If input is already projected but not in meters, just use its own CRS
                    target_crs = src.crs
                    print(f"Input is already projected. Reprojecting to its own CRS ({src.crs.to_string()}) with specified resolution if any.")
        else:
            target_crs = rasterio.crs.CRS.from_string(str(target_crs_epsg))

        # Calculate the default transform and dimensions for the output
        # This function intelligently determines the output extent and resolution
        # to ensure the entire input raster is covered at the desired output CRS.
        # If target_resolution is None, it tries to preserve the relative resolution.
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds,
            resolution=target_resolution # This is where you pass your desired output resolution
        )

        # Prepare the output profile
        profile = src.profile.copy()
        profile.update({
            'crs': target_crs,
            'transform': dst_transform,
            'width': dst_width,
            'height': dst_height,
            'dtype': src.dtype, # Keep original data type unless specific conversion needed
            'count': src.count # Ensure all bands are handled
        })

        # Create an empty array for the reprojected data
        # For multiple bands, it will be (count, height, width)
        reprojected_data = np.empty((src.count, dst_height, dst_width), dtype=profile['dtype'])

        # Perform the actual reprojection and resampling for all bands
        for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, i),
                destination=reprojected_data[i-1], # Write to the correct band slice
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=profile['transform'],
                dst_crs=profile['crs'],
                resampling=Resampling.cubic, # Bicubic interpolation
                num_threads=4 # Optional: for performance
            )

        # Write the reprojected data to a new GeoTIFF
        with rasterio.open(output_geotiff_path, 'w', **profile) as dst:
            dst.write(reprojected_data) # Write all bands at once if profile['count'] > 1

        print(f"Reprojection complete. Output saved to: {output_geotiff_path}")
        with rasterio.open(output_geotiff_path) as reprojected_src:
            print(f"New resolution (pixel size) in meters: {reprojected_src.res}")
            print(f"New CRS: {reprojected_src.crs.to_string()}")
            print(f"Linear Units: {reprojected_src.crs.linear_units}")

    return output_geotiff_path


def create_colored_point_cloud_and_mesh(height_array, rgb_color_array,resolution,smooth=True):
    """
    Creates a colored Open3D point cloud and mesh from height and grayscale arrays.

    Args:
        height_array (np.ndarray): A 2D NumPy array representing heights (DEM).
        rgb_color_array (np.ndarray): A 2D NumPy array representing rgb colors
                                        (values typically 0-255 or 0.0-1.0).
    """
    import open3d as o3d
    
    rows, cols = height_array.shape

    # 1. Create X, Y coordinates for the grid
    # We create a grid where each point (i, j) corresponds to a height and color value.
    # The X and Y coordinates will simply be their indices in the grid.
    x_coords, y_coords = np.meshgrid(np.arange(cols), np.arange(rows))
    x_coords = x_coords*resolution
    y_coords = y_coords*resolution
    # 2. Combine X, Y, and Height (Z) into a single array of 3D points
    # Flatten all arrays to create a list of (x, y, z) points
    points = np.vstack((x_coords.flatten(), y_coords.flatten(), np.flip(height_array,0).flatten())).transpose()

    # 3. Prepare Colors from the grayscale array
    # Normalize grayscale values to 0.0-1.0 if they are in 0-255 range
    if rgb_color_array.max() > 1.0:
        normalized_gray = rgb_color_array / 255.0
    else:
        normalized_gray = rgb_color_array

    # For grayscale, R=G=B. So, we repeat the normalized grayscale value three times.
    colors = np.vstack((np.flip(normalized_gray[:,:,0],0).flatten(),
                        np.flip(normalized_gray[:,:,1],0).flatten(),
                        np.flip(normalized_gray[:,:,2],0).flatten())).transpose()

    # Ensure colors are float64 for Open3D
    colors = colors.astype(np.float64)

    # --- Create and Visualize Point Cloud ---
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    print("Visualizing Point Cloud...")
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1000.0, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd,coordinate_frame], window_name="Colored Point Cloud from DEM")

    # --- Create and Visualize Mesh ---
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)

    # Define triangles for the mesh
    # For a regular grid, each square formed by (i,j), (i+1,j), (i,j+1), (i+1,j+1)
    # can be divided into two triangles.
    triangles = []
    for i in range(rows - 1):
        for j in range(cols - 1):
            # Vertices for the current square:
            # (i, j)      -> v1 = i * cols + j
            # (i, j+1)    -> v2 = i * cols + j + 1
            # (i+1, j)    -> v3 = (i + 1) * cols + j
            # (i+1, j+1)  -> v4 = (i + 1) * cols + j + 1

            v1 = i * cols + j
            v2 = i * cols + (j + 1)
            v3 = (i + 1) * cols + j
            v4 = (i + 1) * cols + (j + 1)

            # Triangle 1 (top-left to bottom-right diagonal)
            triangles.append([v1, v2, v3])
            # Triangle 2 (bottom-right to top-left diagonal)
            triangles.append([v2, v4, v3])

    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(triangles))
    if(smooth):
        mesh = mesh.filter_smooth_laplacian(number_of_iterations=6, lambda_filter=0.5)
    # Assign vertex colors to the mesh (same as point cloud colors)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    # Optional: Compute normals for better shading (important for mesh visualization)
    mesh.compute_vertex_normals()

    print("Visualizing Mesh...")
    o3d.visualization.draw_geometries([mesh,coordinate_frame], window_name="Colored Mesh from DEM")


def create_geotiff_from_jpg_world(jpg_path, output_geotiff_path):
    """
    Creates a GeoTIFF from a JPG image, a world file (.wld), and an auxiliary file (.aux).

    Args:
        jpg_path (str): Path to the input .jpg file.
        output_geotiff_path (str): Path for the output GeoTIFF file.
    """
    # Ensure the world file and aux file exist alongside the JPG
    wld_path = jpg_path.replace('.jpg', '.wld')
    aux_path = jpg_path + '.aux.xml' # Or jpg_path.replace('.jpg', '.jpg.aux') depending on naming

    if not os.path.exists(wld_path):
        print(f"Error: World file not found at {wld_path}")
        return
    if not os.path.exists(aux_path):
        print(f"Warning: Auxiliary file not found at {aux_path}. CRS information might be missing.")
        # GDAL/Rasterio might still be able to infer some CRS, or default to a common one,
        # but it's best to have the .aux file for accurate CRS.

    try:
        # rasterio will automatically pick up the .wld and .aux files
        # if they are in the same directory and have the matching base name.
        with rasterio.open(jpg_path) as src:
            print(f"Successfully opened {jpg_path} with georeferencing.")

            # Get the profile (metadata) for the new GeoTIFF
            profile = src.profile.copy()

            # Update driver to GeoTIFF
            profile.update(driver='GTiff')

            # USGS data is often 8-bit, so float32 might be overkill unless you
            # plan to do complex processing later. Keep original dtype for now.
            # If your JPG is RGB, src.count will be 3, src.read() will return 3D array.
            # If your JPG is grayscale, src.count will be 1, src.read() will return 2D array.
            # profile.update(dtype=src.dtype) # Ensure original dtype is used

            # Read the image data
            # If it's RGB, it will be (bands, height, width) automatically
            image_data = src.read()

            # Write the data to a new GeoTIFF file
            with rasterio.open(output_geotiff_path, 'w', **profile) as dst:
                dst.write(image_data)

            print(f"\nGeoTIFF created successfully: {output_geotiff_path}")
            print(f"Output CRS: {src.crs}")
            print(f"Output Transform: {src.transform}")
            print(f"Output Resolution: {src.res}")

    except rasterio.errors.RasterioIOError as e:
        print(f"Error reading or writing raster: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def reproject_utm_to_latlon(input_utm_geotiff_path, output_latlon_geotiff_path, target_resolution_degrees=None):
    """
    Reprojects a GeoTIFF from a UTM projection to Latitude/Longitude (WGS84, EPSG:4326).

    Args:
        input_utm_geotiff_path (str): Path to the input GeoTIFF file (expected to be in UTM).
        output_latlon_geotiff_path (str): Path for the output GeoTIFF file in Lat/Lon.
        target_resolution_degrees (float or tuple, optional): Desired output resolution in degrees.
                                                            If None, rasterio calculates a default
                                                            that attempts to preserve relative resolution.
                                                            For example, 0.00009 for approx 10m at equator.
    Returns:
        str: Path to the newly created reprojected GeoTIFF.
    """
    # The target CRS for Lat/Lon is almost always WGS84, which is EPSG:4326
    target_crs = 'EPSG:4326'

    with rasterio.open(input_utm_geotiff_path) as src:
        # Verify the input CRS is actually UTM or projected
        if not src.crs.is_projected:
            print(f"Warning: Input GeoTIFF '{input_utm_geotiff_path}' does not appear to be in a projected CRS.")
            print(f"Its CRS is: {src.crs.to_string()}")
            # You might want to add a check here to ensure it's actually UTM if strict validation is needed
            # For example: if not "UTM" in src.crs.name: ...
        elif src.crs.to_string() == target_crs:
            print(f"Input image '{input_utm_geotiff_path}' is already in {target_crs}. No reprojection needed.")
            return input_utm_geotiff_path

        print(f"Reprojecting '{input_utm_geotiff_path}' from {src.crs.to_string()} to {target_crs}...")

        # Calculate the default transform and dimensions for the output
        # This will figure out the appropriate pixel size in degrees to cover the extent.
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds,
            resolution=target_resolution_degrees # Optional: control output resolution in degrees
        )

        # Prepare the output profile
        profile = src.profile.copy()
        profile.update({
            'crs': target_crs,
            'transform': dst_transform,
            'width': dst_width,
            'height': dst_height,
            'dtype': 'float32', # Keep original data type
            'count': src.count  # Ensure all bands are handled
        })

        # Create an empty array for the reprojected data
        reprojected_data = np.empty((src.count, dst_height, dst_width), dtype=profile['dtype'])

        # Perform the actual reprojection and resampling for all bands
        for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, i),
                destination=reprojected_data[i-1], # Write to the correct band slice
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=profile['transform'],
                dst_crs=profile['crs'],
                resampling=Resampling.cubic, # Bicubic interpolation is good for continuous data
                num_threads=4 # Optional: for performance
            )

        # Write the reprojected data to a new GeoTIFF
        with rasterio.open(output_latlon_geotiff_path, 'w', **profile) as dst:
            dst.write(reprojected_data)

        print(f"\nReprojection complete. Output saved to: {output_latlon_geotiff_path}")
        with rasterio.open(output_latlon_geotiff_path) as reprojected_src:
            print(f"New resolution (pixel size) in degrees: {reprojected_src.res}")
            print(f"New CRS: {reprojected_src.crs.to_string()}")
            #print(f"Angular Units: {reprojected_src.crs.angular_units}")

    return output_latlon_geotiff_path