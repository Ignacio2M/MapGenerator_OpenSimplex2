import asyncio
import random
import numpy as np
import concurrent.futures

from OpenSimplex2 import OpenSimplexNoise


async def generateTerrain(map_shape, chunk_shape, queue, seed=None, frequency=5 / 500, octaves=1, amplitude=1,
                          lacunarity=2, gain=0.5):
    '''
    Generates a terrain matrix using the OpenSimplexNoise algorithm. The terrain is generated in chunks
    and the progress is reported through an asynchronous queue.

    Parameters:
    - map_shape (tuple of int): The shape of the entire terrain map (height, width).
    - chunk_shape (tuple of int): The shape of each terrain chunk (height, width).
    - queue (asyncio.Queue): The queue to report progress of the terrain generation.
    - seed (int, optional): The seed for the noise generation. Default is None.
    - frequency (float, optional): The frequency of the noise. Default is 5 / 500.
    - octaves (int, optional): The number of octaves to use in the noise generation. Default is 1.
    - amplitude (float, optional): The amplitude of the noise. Default is 1.
    - lacunarity (float, optional): The lacunarity of the noise. Default is 2.
    - gain (float, optional): The gain of the noise. Default is 0.5.

    Returns:
    - None: This function does not return anything. It generates terrain asynchronously and updates the provided queue.

    Raises:
    - Exception: Any exception raised during the terrain generation process.

    Example usage:
    ```python
    import asyncio
    import numpy as np
    from opensimplex import OpenSimplexNoise
    import concurrent.futures
    import random

    async def main():
        map_shape = (100, 100)
        chunk_shape = (10, 10)
        queue = asyncio.Queue()
        await generateTerrain(map_shape, chunk_shape, queue, seed=42)

    asyncio.run(main())
    ```

    Note:
    This function uses multithreading to generate terrain chunks concurrently and asyncio to handle asynchronous
    updates to the queue.
    '''

    # create end matrix
    terrain = np.zeros(map_shape, dtype=float)

    # Calculate chunks shape
    x_chunk = list(range(map_shape[0] // chunk_shape[0]))
    y_chunk = list(range(map_shape[1] // chunk_shape[1]))
    chunks_index = [(x, y) for x in x_chunk for y in y_chunk]
    random.shuffle(chunks_index)
    noise = OpenSimplexNoise(seed)

    loop = asyncio.get_running_loop()

    # Multithreading calculation
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            loop.run_in_executor(
                executor, generateTerrainChunk, noise.noise2_ImproveX, chunk_shape, index_chunk, frequency,
                octaves, amplitude, lacunarity, gain
            )
            for index_chunk in chunks_index
        ]

        # Check for updates and put in queue
        for future in asyncio.as_completed(futures):
            sup_matrix, start_pints, end_points = await future
            terrain[start_pints[0]:end_points[0], start_pints[1]:end_points[1]] = sup_matrix
            await queue.put(terrain.copy())

        await asyncio.gather(*futures)


def generateTerrainChunk(noise, chunk_shape, chunk_index, frequency=5 / 10, octaves=1, amplitude=1, lacunarity=2,
                         gain=0.5):
    sup_matrix = np.zeros(chunk_shape, dtype=float)
    start_pints = [chunk_index[0] * chunk_shape[0], chunk_index[1] * chunk_shape[1]]
    end_points = [start_pints[0] + chunk_shape[0], start_pints[1] + chunk_shape[1]]

    for i in range(octaves):
        for dx, x in enumerate(range(start_pints[0], end_points[0])):
            for dy, y in enumerate(range(start_pints[1], end_points[1])):
                sup_matrix[dx, dy] = sup_matrix[dx, dy] + amplitude * noise(x * frequency, y * frequency)
        frequency *= lacunarity
        amplitude *= gain
    return sup_matrix, start_pints, end_points
