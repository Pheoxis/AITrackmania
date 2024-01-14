# standard library imports
import platform

import numpy as np

if platform.system() == "Windows":

    import time


    def control_gamepad(gamepad, control):
        '''
        Controls a gamepad for driving simulation.
        Actions:
        Maps control inputs to specific gamepad actions for gas, brake, and turning.
        Handles triggers and joystick movements accordingly.
        Updates the gamepad state.
        '''
        assert all(-1.0 <= c <= 1.0 for c in control), "This function accepts only controls between -1.0 and 1.0"
        if control[0] > 0.0:  # gas
            mapped_value = 0.165 * control[0] + 0.835  # f(-0.515)=0.75
            gamepad.right_trigger_float(value_float=mapped_value)  # car starts driving from 0.75
        else:
            gamepad.right_trigger_float(value_float=0.0)
        if control[1] > 0.75:  # break
            # mapped_value = 0.5 * control[0] + 0.5  # x0 = 1/2
            gamepad.left_trigger_float(value_float=control[1])
        else:
            gamepad.left_trigger_float(value_float=0.0)

        # gamepad.left_joystick_float(control[2], 0.0)  # turn
        gamepad.left_joystick_float(mapped_steering(control[2]), 0.0)  # turn
        gamepad.update()


    # def upside_down_normal_y(x, mu=0, sigma=0.3, epsilon=0.45):
    #     """
    #     Transforms a joystick input for vertical (Y-axis) movement.
    #     Actions:
    #     Adjusts the input based on a standard normal distribution to invert or normalize the joystick input.
    #     Returns the transformed input within the range of -1 to 1.
    #     """
    #
    #     # Calculate the PDF of the standard normal distribution
    #     pdf = 1.05 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    #
    #     # Take the negative of the standard normal PDF to flip it
    #     upside_down_pdf = -pdf + 1.045
    #
    #     # Soft transition using a sigmoid-like function
    #     # It gradually transitions the value to 0 as x approaches -epsilon and epsilon
    #     transition = 1 / (1 + np.exp(-10 * (abs(x) - epsilon)))
    #
    #     # Apply the transition to the upside_down_pdf
    #     result = transition * upside_down_pdf
    #
    #     if x < 0:
    #         return max([-result, -1.])
    #     else:
    #         return min([result, 1.])
    def mapped_steering(x, k=15):
        sigmoid = 1 / (1 + np.exp(-k * (x + 0.4))) * 1 / (1 + np.exp(-k * (0.4 - x)))
        return np.tan(x) / np.tan(1) * (1 - sigmoid)


    # def upside_down_normal_y(x, mu=0, sigma=0.4):
    #     """
    #     Transforms a joystick input for vertical (Y-axis) movement.
    #     Actions:
    #     Adjusts the input based on a standard normal distribution to invert or normalize the joystick input.
    #     Returns the transformed input within the range of -1 to 1.
    #     """
    #     if -0.2 < x < 0.2:
    #         return 0.
    #     # Calculate the PDF of the standard normal distribution
    #     pdf = 1.05 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    #
    #     # Take the negative of the standard normal PDF to flip it
    #     upside_down_pdf = -pdf + 1.045
    #
    #     if x < 0:
    #         return max([-upside_down_pdf, -1.])
    #     else:
    #         return min([upside_down_pdf, 1.])

    def gamepad_reset(gamepad):
        """
        Resets the gamepad state and performs button presses for reset purposes.
        Actions:
        Resets the gamepad state.
        Simulates pressing and releasing a specific button (B button) on the gamepad.
        """
        gamepad.reset()
        gamepad.press_button(button=0x2000)  # press B button
        gamepad.update()
        time.sleep(0.1)
        gamepad.release_button(button=0x2000)  # release B button
        gamepad.update()


    def gamepad_save_replay_tm20(gamepad):
        '''
        Initiates specific actions on the gamepad to save a replay in a simulated environment.
        Actions:
        Simulates a series of button presses (D-pad down, A, D-pad up, A) to trigger the replay saving functionality.
        '''
        time.sleep(5.0)
        gamepad.reset()
        gamepad.press_button(0x0002)  # dpad down
        gamepad.update()
        time.sleep(0.1)
        gamepad.release_button(0x0002)  # dpad down
        gamepad.update()
        time.sleep(0.2)
        gamepad.press_button(0x1000)  # A
        gamepad.update()
        time.sleep(0.1)
        gamepad.release_button(0x1000)  # A
        gamepad.update()
        time.sleep(0.2)
        gamepad.press_button(0x0001)  # dpad up
        gamepad.update()
        time.sleep(0.1)
        gamepad.release_button(0x0001)  # dpad up
        gamepad.update()
        time.sleep(0.2)
        gamepad.press_button(0x1000)  # A
        gamepad.update()
        time.sleep(0.1)
        gamepad.release_button(0x1000)  # A
        gamepad.update()


    def gamepad_close_finish_pop_up_tm20(gamepad):
        '''
        Simulates pressing a button to close a finish pop-up window in a simulated environment.
        Actions:
        Simulates pressing and releasing the A button on the gamepad to close the pop-up window.
        '''
        gamepad.reset()
        gamepad.press_button(0x1000)  # A
        gamepad.update()
        time.sleep(0.1)
        gamepad.release_button(0x1000)  # A
        gamepad.update()

else:

    def control_gamepad(gamepad, control):
        pass


    def gamepad_reset(gamepad):
        pass


    def gamepad_save_replay_tm20(gamepad):
        pass


    def gamepad_close_finish_pop_up_tm20(gamepad):
        pass
