
import numpy as np
import matplotlib.pyplot as plt

from gym import Wrapper


class HumanEnv(Wrapper):
    """Environment that renders the environment in human visual mode.

    NB: This class heavily use code from minihack repo (fixes a couple of bugs/deprecated code)
    https://github.com/facebookresearch/minihack/blob/main/minihack/tiles/window.py
    """

    def __init__(self, env):
        super().__init__(env)

        self.imshow_obj = None

        # Create the figure and axes
        self.fig, self.ax = plt.subplots()

        # Show the env name in the window title
        self.canvas_manager = plt.get_current_fig_manager()
        self.canvas_manager.set_window_title("MiniHack the Planet")

        # Turn off x/y axis numbering/ticks
        self.ax.xaxis.set_ticks_position("none")
        self.ax.yaxis.set_ticks_position("none")
        _ = self.ax.set_xticklabels([])
        _ = self.ax.set_yticklabels([])

        # Flag indicating the window was closed
        self.closed = False
        self.title = ""

        def close_handler(evt):
            self.closed = True

        self.fig.canvas.mpl_connect("close_event", close_handler)

        # Remove all keymaps
        for key in plt.rcParams.keys():
            if "keymap" in key:
                plt.rcParams[key].clear()

    def reset(self):
        observation = self.env.reset()
        self.redraw(observation)
        return observation

    def step(self, action):
        observation, reward, terminated, info = self.env.step(action)

        if terminated:
            self.reset()
        else:
            self.redraw(observation)

        return observation, reward, terminated, info

    def render(self, block=True):
        # If not blocking, trigger interactive mode
        if not block:
            plt.ion()

        # Show the plot
        # In non-interactive mode, this enters the matplotlib event loop
        # In interactive mode, this call does not block
        plt.show()

    ################################################################
    ############################ window ############################
    ################################################################
    def redraw(self, observation):
        img = observation["pixel"]
        msg = observation["message"]
        msg = msg[: np.where(msg == 0)[0][0]].tobytes().decode("utf-8")

        # Show the first image of the environment
        if self.imshow_obj is None:
            self.imshow_obj = self.ax.imshow(img, interpolation="bilinear")

        self.imshow_obj.set_data(img)
        self.fig.canvas.draw()

        if msg != self.title:
            self.title = msg
            plt.title(msg, loc="left")

        # Let matplotlib process UI events
        # This is needed for interactive mode to work properly
        plt.pause(0.001)

    def set_caption(self, text):
        """
        Set/update the caption text below the image
        """

        plt.xlabel(text)

    def reg_key_handler(self, key_handler):
        """
        Register a keyboard event handler
        """

        # Keyboard handler
        self.fig.canvas.mpl_connect("key_press_event", key_handler)

    def close(self):
        """
        Close the window
        """

        plt.close()
        self.closed = True
