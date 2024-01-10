# http://www.flint.jp/misc/?q=dik&lang=en  key indicator

# standard library imports
import platform
import time

# local imports
from custom.utils.control_mouse import (mouse_change_name_replay_tm20,
                                                                mouse_close_replay_window_tm20,
                                                                mouse_save_replay_tm20)

if platform.system() == "Windows":
    # standard library imports
    import ctypes

    # third-party imports
    import keyboard

    SendInput = ctypes.windll.user32.SendInput

    # constants:

    W = 0x11
    A = 0x1E
    S = 0x1F
    D = 0x20
    DEL = 0xD3
    R = 0x13

    # C struct redefinitions

    PUL = ctypes.POINTER(ctypes.c_ulong)

    class KeyBdInput(ctypes.Structure):
        _fields_ = [("wVk", ctypes.c_ushort), ("wScan", ctypes.c_ushort), ("dwFlags", ctypes.c_ulong), ("time", ctypes.c_ulong), ("dwExtraInfo", PUL)]

    class HardwareInput(ctypes.Structure):
        _fields_ = [("uMsg", ctypes.c_ulong), ("wParamL", ctypes.c_short), ("wParamH", ctypes.c_ushort)]

    class MouseInput(ctypes.Structure):
        _fields_ = [("dx", ctypes.c_long), ("dy", ctypes.c_long), ("mouseData", ctypes.c_ulong), ("dwFlags", ctypes.c_ulong), ("time", ctypes.c_ulong), ("dwExtraInfo", PUL)]

    class Input_I(ctypes.Union):
        _fields_ = [("ki", KeyBdInput), ("mi", MouseInput), ("hi", HardwareInput)]

    class Input(ctypes.Structure):
        _fields_ = [("type", ctypes.c_ulong), ("ii", Input_I)]

    # Key Functions

    def PressKey(hexKeyCode):
        '''
        Simulates pressing a key on the keyboard.
        Actions:
        Creates a keyboard input event using the SendInput function from user32.dll.
        Sends a key press event using the given key code.
        '''
        extra = ctypes.c_ulong(0)
        ii_ = Input_I()
        ii_.ki = KeyBdInput(0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra))
        x = Input(ctypes.c_ulong(1), ii_)
        ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

    def ReleaseKey(hexKeyCode):
        '''
        Simulates releasing a key on the keyboard.
        Actions:
        Creates a keyboard input event using the SendInput function from user32.dll.
        Sends a key release event using the given key code.
        '''
        extra = ctypes.c_ulong(0)
        ii_ = Input_I()
        ii_.ki = KeyBdInput(0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra))
        x = Input(ctypes.c_ulong(1), ii_)
        ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

    def apply_control(action):  # move_fast
        '''
        Applies control actions based on specific key presses for movement.
        Actions:
        Determines which keys (W, A, S, D) to press or release based on the provided action string.
        Simulates key presses or releases accordingly for forward (W), backward (S), left (A), and right (D) movement.
        '''
        if 'f' in action:
            PressKey(W)
        else:
            ReleaseKey(W)
        if 'b' in action:
            PressKey(S)
        else:
            ReleaseKey(S)
        if 'l' in action:
            PressKey(A)
        else:
            ReleaseKey(A)
        if 'r' in action:
            PressKey(D)
        else:
            ReleaseKey(D)

    def keyres():
        '''
        Triggers a key press and release for the DEL key.
        Actions:
        Simulates a press and release of the DEL key.
        '''
        PressKey(DEL)
        ReleaseKey(DEL)

    def keysavereplay():  # TODO: debug
        '''
        Saves a replay with specific key sequences and mouse actions.
        Actions:
        Simulates pressing the R key, waits, changes the replay name, writes a timestamp, saves the replay, and closes the replay window.
        Utilizes mouse-related functions for changing replay names, saving the replay, and closing the window.
        '''
        PressKey(R)
        time.sleep(0.1)
        ReleaseKey(R)
        time.sleep(1.0)
        mouse_change_name_replay_tm20()
        time.sleep(1.0)
        keyboard.write(str(time.time_ns()))
        time.sleep(1.0)
        mouse_save_replay_tm20()
        time.sleep(1.0)
        mouse_close_replay_window_tm20()
        time.sleep(1.0)

else:

    def apply_control(action):  # move_fast
        pass

    def keyres():
        pass

    def keysavereplay():
        pass
