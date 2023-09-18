import numpy as np


def toPoints(channel, res, vfov, hfov, depthmap, origin=(0, 0, 0)):
    # Initialize an empty list to store the 3D points
    points = []

    # Calculate the angular increments in horizontal and vertical directions
    delta_hfov = hfov / res
    delta_vfov = vfov / (channel - 1)

    # Iterate through each channel and each point in the resolution
    for ch in range(channel):
        v_angle = -vfov / 2 + ch * delta_vfov
        for r in range(res):
            h_angle = -hfov / 2 + r * delta_hfov

            # Get the depth value from the depth map
            depth = depthmap[ch * res + r]

            if depth >= 0:  # Only consider valid depth values
                # Calculate the 3D coordinates of the point in the sensor's local coordinate system
                x = depth * np.cos(np.radians(v_angle)) * np.cos(np.radians(h_angle))
                y = depth * np.cos(np.radians(v_angle)) * np.sin(np.radians(h_angle))
                z = depth * np.sin(np.radians(v_angle))

                # Transform the coordinates to the global coordinate system using the origin
                x += origin[0]
                y += origin[1]
                z += origin[2]

                # Append the point to the list of points
                points.append([x, -y, z])

    return points
