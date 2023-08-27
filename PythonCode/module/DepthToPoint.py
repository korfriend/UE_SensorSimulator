import numpy as np


def toPoints(channel: int, res: int, vfov: float, hfov: float, depthmap: np.array):
    # Compute deltas
    delta_deg = hfov / res
    v_rot_delta = vfov / channel

    results = []

    for ch_idx in range(channel):
        v_rot_deg = -vfov / 2 + ch_idx * v_rot_delta

        for x in range(res):
            # Determine the depth from the depthmap
            depth = depthmap[ch_idx, x]

            # Check if depth is less than 0, if so, skip the current iteration
            if depth < 0:
                continue

            rot_deg = delta_deg * x - hfov / 2

            # Calculating the direction vector components using trigonometric functions
            dir_x_ue = np.cos(np.radians(v_rot_deg)) * np.cos(np.radians(rot_deg))
            dir_y_ue = np.cos(np.radians(v_rot_deg)) * np.sin(np.radians(rot_deg))
            dir_z_ue = np.sin(np.radians(v_rot_deg))

            # Convert from Unreal Engine coordinates to Panda3D coordinates
            dir_x_pd = dir_y_ue
            dir_y_pd = dir_x_ue
            dir_z_pd = dir_z_ue

            # Rotate around the z-axis by 90 degrees (counter-clockwise)
            theta = np.radians(-90)
            rotated_x = dir_x_pd * np.cos(theta) - dir_y_pd * np.sin(theta)
            rotated_y = dir_x_pd * np.sin(theta) + dir_y_pd * np.cos(theta)

            # Calculate the hit position based on the direction and depth
            hit_position_x = rotated_x * depth
            hit_position_y = rotated_y * depth
            hit_position_z = dir_z_pd * depth

            results.append((hit_position_x, hit_position_y, hit_position_z, depth))

    return results
