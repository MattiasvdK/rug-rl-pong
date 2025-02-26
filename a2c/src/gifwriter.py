from PIL import Image
from typing import List
import numpy as np
import os
import shutil


class GifWriter:
    """Class to write RL trajectories to a GIF.

    This class writes trajectories, i.e. sequences of states to GIF
    files. It also writes the first frame to a JPEG file.

    This class was mainly written to creating visualizations for the
    presentation.
    """
    def __init__(
            self,
            data_dir: str,
            duration: int = 100,
    ) -> None:
        """Initialise the GifWriter class.

        @param data_dir (str): The directory to store the files.
        @param duration (int): The duration of each frame in milliseconds.
        """

        self.data_dir = data_dir
        self.duration = duration
        self.written = 0
        self.agent_dir = None
        self.gif_dir = None
        self.jpg_dir = None

    def write_trajectory(
        self,
        trajectory: List[np.ndarray],
        trajectory_index: int
    ) -> None:
        """Writes a trajectory to a GIF.

        This function writes a trajectory into a GIF file and saves the
        first frame as a JPEG.

        @param trajectory (List[np.ndarray]): the sequence of states.
        @param trajectory_index (int): the index of the trajectory,
            used for naming the files.
        """

        file_name = f'episode_{trajectory_index}'

        gif_path = os.path.join(self.gif_dir, f'{file_name}.gif')
        thumbnail_path = os.path.join(self.jpg_dir, f'{file_name}.jpg')
        images = self._trajectory_to_images(trajectory)
        first = images[0]
        first_jpg = first.convert('L')
        first_jpg.save(thumbnail_path)
        first.save(gif_path, save_all=True, append_images=images[1:],
                   duration=self.duration, loop=0)

    def new_agent(self, agent_name: str) -> None:
        """Prepares directories for a new agent.

        @param agent_name (str): The name of the agent.
        """
        self.agent_dir = os.path.join(self.data_dir, agent_name)

        if os.path.isdir(self.agent_dir):
            shutil.rmtree(self.agent_dir)
        os.makedirs(self.agent_dir)

    def new_simulation(self, sim_index: int) -> None:
        """Prepares directories for a new simulation.

        @param sim_index (int): The index of the simulation.
        """
        sim_dir = os.path.join(self.agent_dir, f'sim_{sim_index}')
        self.gif_dir = os.path.join(sim_dir, 'gifs')
        self.jpg_dir = os.path.join(sim_dir, 'jpgs')

        os.makedirs(sim_dir)
        os.makedirs(self.gif_dir)
        os.makedirs(self.jpg_dir)

    @staticmethod
    def _trajectory_to_images(trajectory: List[np.ndarray]) -> List[Image.Image]:
        """Converts a trajectory into images.

        This function converts a trajectory of states into images to
        be used as frames for the GIF.

        @param trajectory (List[np.ndarray]): the sequence of states.

        @return List[Image.Image]: the sequence of images.
        """
        images = []
        for idx, step in enumerate(trajectory):
            image = GifWriter._state_to_image(step)
            images.append(image)
        return images

    @staticmethod
    def _state_to_image(
            state: np.ndarray,
    ) -> Image.Image:
        """Converts a state into an image.

        This function converts a state into an image to be used
        as frame in the GIF.
        The last channel of the state is selected.

        @param state (numpy.ndarray): the state.

        @return Image.Image: the image.
        """
        state = state.transpose(2, 0, 1)
        frame = np.where(state[-1] > 75, 255, 0)
        return Image.fromarray(frame)

