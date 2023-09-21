# standard library imports
import pickle
import time

# third-party imports
import keyboard
import numpy as np
from scipy.interpolate import CubicSpline

# local imports
import config.config_constants as cfg
from custom.utils.tools import TM2020OpenPlanetClient
import logging

PATH_REWARD = cfg.REWARD_PATH
DATASET_PATH = cfg.DATASET_PATH


def record_reward_dist(path_reward=PATH_REWARD):
    positions = []
    client = TM2020OpenPlanetClient()
    path = path_reward

    is_recording = False
    while True:
        if keyboard.is_pressed('e'):
            logging.info(f"start recording")
            is_recording = True
        if is_recording:
            data = client.retrieve_data(sleep_if_empty=0.01)  # we need many points to build a smooth curve
            terminated = bool(data[8])
            if keyboard.is_pressed('q') or terminated:
                logging.info(f"Computing reward function checkpoints from captured positions...")
                logging.info(f"Initial number of captured positions: {len(positions)}")
                positions = np.array(positions)

                final_positions = [positions[0]]
                dist_between_points = 0.1
                j = 1
                move_by = dist_between_points
                pt1 = final_positions[-1]
                while j < len(positions):
                    pt2 = positions[j]
                    pt, dst = line(pt1, pt2, move_by)
                    if pt is not None:  # a point was created
                        final_positions.append(pt)  # add the point to the list
                        move_by = dist_between_points
                        pt1 = pt
                    else:  # we passed pt2 without creating a new point
                        pt1 = pt2
                        j += 1
                        move_by = dst  # remaining distance

                final_positions = np.array(final_positions)
                upsampled_arr = interp_points_with_cubic_spline(final_positions)
                # print(final_positions)
                # print()
                # print(upsampled_arr)
                logging.info(f"Final number of checkpoints in the reward function: {len(upsampled_arr)}")

                pickle.dump(upsampled_arr, open(path, "wb"))
                logging.info(f"All done")
                return
            else:
                positions.append([data[2], data[3], data[4]])
        else:
            time.sleep(0.05)  # waiting for user to press E


def interp_points_with_cubic_spline(sub_array, data_density: int = 3):
    original_x, original_y, original_z = sub_array.T

    # Calculate the new x-values based on data density (e.g., double the points)
    original_i = np.arange(0, data_density * len(original_x), step=data_density)
    new_i = np.arange(0, data_density * len(original_x) - 1)

    # Perform cubic spline interpolation for each vector (x, y, z)
    cs_x = CubicSpline(original_i, original_x)
    cs_y = CubicSpline(original_i, original_y)
    cs_z = CubicSpline(original_i, original_z)

    # Interpolate the y-values for the new_x values for each vector
    new_x_values = cs_x(new_i)
    new_y_values = cs_y(new_i)
    new_z_values = cs_z(new_i)

    # Combine the new x, y, and z values into a single NumPy array
    new_data = np.array([new_x_values, new_y_values, new_z_values])

    # Transpose the new_data array to have x, y, z as rows
    new_data = new_data.T

    return new_data


def line(pt1, pt2, dist):
    """
    Creates a point between pt1 and pt2, at distance dist from pt1.

    If dist is too large, returns None and the remaining distance (> 0.0).
    Else, returns the point and 0.0 as remaining distance.
    """
    vec = pt2 - pt1
    norm = np.linalg.norm(vec)
    if norm < dist:
        return None, dist - norm  # we couldn't create a new point but we moved by a distance of norm
    else:
        vec_unit = vec / norm
        pt = pt1 + vec_unit * dist
        return pt, 0.0


if __name__ == "__main__":
    record_reward_dist(path_reward=PATH_REWARD)
