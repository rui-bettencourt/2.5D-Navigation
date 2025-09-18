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

def interpolate_path(path, start, goal, dof):
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
    start_angles = np.array([start['roll'], start['pitch']] + [start[f'q{i}'] for i in range(1, 8)])
    goal_angles = np.array([goal['roll'], goal['pitch']] + [goal[f'q{i}'] for i in range(1, 8)])
    
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

def calculate_path_distance(path):
    """
    Calculates the total Euclidean distance covered by a path.

    Args:
        path (List[Dict]): List of dictionaries with keys 'x', 'y', 'z', and 'yaw'.

    Returns:
        float: Total distance covered by the path.
    """
    if len(path) < 2:
        return 0.0

    distance = 0.0
    for i in range(1, len(path)):
        p1 = path[i - 1]
        p2 = path[i]
        segment = np.sqrt(
            (p2['x'] - p1['x']) ** 2 +
            (p2['y'] - p1['y']) ** 2 +
            (p2['z'] - p1['z']) ** 2
        )
        distance += segment
    return distance

def apply_safety_configuration_to_path(path, safety_config, dof):
    """
    Replaces the values of 'q1' to 'q{qdof}' in all path steps except the first and last
    with the values from safety_config.

    Args:
        path (List[Dict]): List of dictionaries with keys including 'q1' to 'q{qdof}'.
        safety_config (Dict): Dictionary with keys 'q1' to 'q{qdof}' specifying safe joint values.
        qdof (int): Number of joints (degrees of freedom).

    Returns:
        List[Dict]: Modified path with safety configuration applied.
    """
    if len(path) < 3:
        return path.copy()

    new_path = []
    for i, step in enumerate(path):
        if i == 0 or i == len(path) - 1:
            new_path.append(step.copy())
        else:
            updated_step = step.copy()
            for j in range(1, dof):
                key = f'q{j}'
                if key in safety_config:
                    updated_step[key] = safety_config[key]
            new_path.append(updated_step)
    return new_path