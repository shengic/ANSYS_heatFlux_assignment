import numpy as np
import os

def read_geometry(file_path):
    """
    Reads the geometric parameters (s1, s2, c) from the geometry file
    and calculates the circle radius automatically using both s1-c and s2-c.
    
    Expected file format (BM_coordinate2.txt):
        s1=598.89,0,3356.86
        s2=527.44,0,2346.43
        c=8930.38,0,2260
    """
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    # Parse s1
    s1_vals = lines[0].split('=')[1].split(',')
    s1 = tuple(float(val) for val in s1_vals)

    # Parse s2
    s2_vals = lines[1].split('=')[1].split(',')
    s2 = tuple(float(val) for val in s2_vals)

    # Parse c (circle center)
    c_vals = lines[2].split('=')[1].split(',')
    c = tuple(float(val) for val in c_vals)

    # Calculate radius from both s1-c and s2-c
    s1_arr = np.array(s1)
    s2_arr = np.array(s2)
    c_arr  = np.array(c)
    r1 = np.linalg.norm(s1_arr - c_arr)
    r2 = np.linalg.norm(s2_arr - c_arr)

    # If the two radii differ by more than 1% (relative difference), take the average
    avg_radius = (r1 + r2) / 2.0
    if abs(r1 - r2) / avg_radius > 0.01:
        radius = avg_radius
    else:
        radius = r1  # They are nearly identical

    return s1, s2, c, radius

def define_plane_axes(s1, s2, c):
    """
    Defines the plane axes using:
      - v1 = s1 - c and v2 = s2 - c.
      - z_dir: the unit vector from v2 x v1 (acts as the plane normal).
      - y_dir: the unit vector in the direction of (s1 - c).
      - x_dir: the unit vector determined by (y_dir x z_dir).

    Returns:
      n, x_dir, y_dir  (with n set equal to z_dir)
    """
    s1 = np.array(s1)
    s2 = np.array(s2)
    c  = np.array(c)
    
    # v1 and v2 from center to s1 and s2, respectively
    v1 = s1 - c
    v2 = s2 - c

    # y_dir: unit vector along v1 (i.e., s1 - c)
    y_dir = v1 / np.linalg.norm(v1)
    
    # z_dir: unit vector perpendicular to the plane (from v2 x v1)
    z_dir = np.cross(v2, v1)
    z_dir = z_dir / np.linalg.norm(z_dir)
    
    # x_dir: unit vector perpendicular to y_dir in the plane (y_dir x z_dir)
    x_dir = np.cross(y_dir, z_dir)
    x_dir = x_dir / np.linalg.norm(x_dir)
    
    # For clarity, we use n to denote the plane normal (z_dir)
    n = z_dir
    return n, x_dir, y_dir

def project_to_2d(point, origin, x_dir, y_dir):
    """
    Projects a 3D point onto a 2D plane defined by x_dir and y_dir,
    using 'origin' (in 3D) as the reference (0,0) in plane coordinates.
    """
    translated_point = point - origin
    x_coord = np.dot(translated_point, x_dir)
    y_coord = np.dot(translated_point, y_dir)
    return np.array([x_coord, y_coord])

def map_to_3d(point_2d, origin, x_dir, y_dir):
    """
    Maps a 2D point back to its original 3D coordinate system,
    using 'origin' as (0,0) in plane coordinates.
    """
    x_component = point_2d[0] * x_dir
    y_component = point_2d[1] * y_dir
    return origin + x_component + y_component

def find_tangent_point(cp, pp, radius):
    """
    Finds the two tangent points on a circle (center cp, radius)
    from an external point pp in 2D.
    
    Returns:
       tp1, tp2
    If the point is inside the circle (no tangents), returns (None, None).
    """
    cp = np.array(cp)
    pp = np.array(pp)

    d = np.linalg.norm(pp - cp)
    if d < radius:
        return None, None

    # Compute angles
    delta = pp - cp
    alpha = np.arctan2(delta[1], delta[0])
    beta = np.arccos(radius / d)

    # Tangent point angles
    theta_tp1 = alpha + beta
    theta_tp2 = alpha - beta

    tp1 = cp + radius * np.array([np.cos(theta_tp1), np.sin(theta_tp1)])
    tp2 = cp + radius * np.array([np.cos(theta_tp2), np.sin(theta_tp2)])

    return tp1, tp2

def find_tangent_and_offset(point, c, x_dir, y_dir, radius, n):
    """
    Given a 3D point, find:
      1) The distance from the point to one tangent point on the circle,
      2) The 3D coordinates of that tangent point,
      3) The offset of the point from the plane.
      
    Returns:
      distance, tangent_point_3d, offset
    or
      None, None, None if no tangent is possible.
    """
    # Project the 3D point onto the 2D plane (using c as the origin)
    point_2d = project_to_2d(point, c, x_dir, y_dir)

    # Find tangent points in 2D (circle center at [0, 0] in plane coordinates)
    tp1_2d, tp2_2d = find_tangent_point([0, 0], point_2d, radius)
    if tp1_2d is None:
        return None, None, None

    # Select the first tangent point
    tp_3d = map_to_3d(tp1_2d, c, x_dir, y_dir)

    # Compute the distance from the point to the chosen tangent point
    distance = np.linalg.norm(tp_3d - point)

    # Compute the offset from the plane (absolute distance along the normal)
    offset = np.abs(np.dot(point - c, n))

    return distance, tp_3d, offset

def process_points(points_file, geometry_file):
    """
    Reads s1, s2, c from the geometry file, computes the plane normal,
    and processes each point from points_file to find the tangent point,
    distance, and offset.
    """
    # Read geometry from the file
    s1, s2, c, radius = read_geometry(geometry_file)

    # Convert to numpy arrays
    s1_arr = np.array(s1)
    s2_arr = np.array(s2)
    c_arr  = np.array(c)

    # Print out the calculated radius
    print(f"Calculated radius: {radius:.4f}")
    print("========================================")

    # Define the plane normal and in-plane axes using the new definitions:
    # v1 = s1 - c, v2 = s2 - c, z_dir = v2 x v1, y_dir = normalized(v1), and
    # x_dir = y_dir x z_dir.
    n, x_dir, y_dir = define_plane_axes(s1_arr, s2_arr, c_arr)

    # Read and process each point from points.txt
    with open(points_file, 'r') as f:
        for line in f:
            x, y, z = map(float, line.strip().split(','))
            point = np.array([x, y, z])

            distance, tp_3d, offset = find_tangent_and_offset(
                point, c_arr, x_dir, y_dir, radius, n
            )

            if tp_3d is not None:
                print(f"Point: ({x:.3f}, {y:.3f}, {z:.3f})")
                print(f"  Tangent point: ({tp_3d[0]:.4f}, {tp_3d[1]:.4f}, {tp_3d[2]:.4f})")
                print(f"  Distance to tangent: {distance:.4f}")
                print(f"  Offset from plane: {offset:.4f}")
                print("----------------------------------------")
            else:
                print(f"Point: ({x:.3f}, {y:.3f}, {z:.3f}) is inside the circle. No tangent available.")
                print("----------------------------------------")

if __name__ == "__main__":
    # Assume that points.txt and BM_coordinate2.txt are in the same directory as this script.
    base_path = os.path.dirname(os.path.abspath(__file__))
    points_file = os.path.join(base_path, "points.txt")
    geometry_file = os.path.join(base_path, "BM_coordinate2.txt")
    
    process_points(points_file, geometry_file)
