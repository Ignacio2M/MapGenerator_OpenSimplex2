import asyncio

from matplotlib import pyplot as plt
import numpy as np
from worldMap import generateTerrain


async def visualize_windows():
    queue = asyncio.Queue(maxsize=10)
    fig, ax = plt.subplots()

    generator = asyncio.create_task(generateTerrain((600, 600), (60, 60), queue, 123456789,
                                                    octaves=3, amplitude=1, gain=0.15, frequency=1 / 500, lacunarity=6))
    sup_matrix = None

    while not (generator.done() and queue.empty()):
        try:
            sup_matrix = await asyncio.wait_for(queue.get(), 0.1)
            ax.clear()
            ax.imshow(sup_matrix, cmap='viridis', interpolation='none')
            plt.draw()
            plt.pause(0.1)
            queue.task_done()
        except asyncio.TimeoutError:
            pass
    await generator

    if sup_matrix is not None:
        ax.imshow(sup_matrix, cmap='viridis', interpolation='none')

        x = np.arange(0, sup_matrix.shape[0])
        y = np.arange(0, sup_matrix.shape[1])
        x, y = np.meshgrid(x, y)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.plot_surface(x, y, sup_matrix, cmap='viridis')
        plt.draw()
        plt.show()


async def main():
    await visualize_windows()


if __name__ == '__main__':
    asyncio.run(main())
