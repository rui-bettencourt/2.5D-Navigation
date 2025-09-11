import numpy as np

def wrap_angle(angle):
    """
    Wraps an angle in radians to the range [-pi, pi].
    
    Args:
        angle (float): Angle in radians.
    
    Returns:
        float: Wrapped angle in radians.
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi

def interpolate_path(path, start, goal, dof=6):
    """
    Interpolates roll, pitch, and arm configuration angles (q1 to q7) for a given path.
    
    Parameters:
    - path: List of dictionaries with 'x', 'y', 'z', 'yaw' as keys.
    - start: Dictionary with 'x', 'y', 'z', 'yaw', 'roll', 'pitch', and 'q1' to 'q7'.
    - goal: Dictionary with 'x', 'y', 'z', 'yaw', 'roll', 'pitch', and 'q1' to 'q7'.
    
    Returns:
    - List of dictionaries (path) with interpolated values for roll, pitch, q1 to q7.
    """
    # Extract roll, pitch, and arm configuration angles from start and goal
    start_angles = np.array([start['roll'], start['pitch']] + [start[f'q{i}'] for i in range(1, dof+1)])
    goal_angles = np.array([goal['roll'], goal['pitch']] + [goal[f'q{i}'] for i in range(1, dof+1)])
    
    # Total steps in the path
    n_steps = len(path)
    
    # Generate interpolation factors for each step
    interpolation_factors = np.linspace(0, 1, n_steps)
    
    # Interpolated path
    interpolated_path = []
    for i, step in enumerate(path):
        # Interpolation factor for the current step
        factor = interpolation_factors[i]
        
        # Linearly interpolate roll, pitch, and arm angles
        interpolated_angles = start_angles + factor * (goal_angles - start_angles)
        
        # Update step with interpolated values
        updated_step = {
            **step,
            'roll': interpolated_angles[0],
            'pitch': interpolated_angles[1],
            **{f'q{j}': interpolated_angles[j + 1] for j in range(1, dof+1)}
        }
        
        interpolated_path.append(updated_step)
    
    return interpolated_path

def sample_path(path, num_samples):
    """
    Samples a path by evenly selecting points from the input path.
    
    Args:
        path (List[List[float]]): Input path.
        num_samples (int): Number of samples in the final path.
    
    Returns:
        List[List[float]]: Sampled path.
    """
    # Number of points in the input path
    n_points = len(path)
    
    # Indices of the selected points
    indices = np.linspace(0, n_points - 1, num_samples, dtype=int)
    
    # Sampled path
    sampled_path = [path[i] for i in indices]
    
    return sampled_path

def create_bounding_box_corners(center, obstacle_threshold):
    # Calculate the bounding box half-width for each dimension (using obstacle_threshold)
    half_size = obstacle_threshold  # This is the half-length in each direction
    
    # Define the min and max corners of the bounding box
    min_corner = np.array(center) - half_size
    max_corner = np.array(center) + half_size
    
    return min_corner, max_corner