import numpy as np

from ex2_utils import generate_responses_1, get_patch, show_image
from math import floor


def create_kernel(kernel_size):
    # make sure kernel size is odd
    kernel_size = kernel_size - 1 if kernel_size % 2 == 0 else kernel_size

    kernel_y = np.zeros((kernel_size, kernel_size))

    for row, value in enumerate(range(-(kernel_size // 2), (kernel_size // 2) + 1, 1)):
        kernel_y[row] = value

    kernel_x = np.transpose(kernel_y)

    return kernel_x, kernel_y


def mean_shift(responses, position, kernel_size, epsilon):
    kernel_x, kernel_y = create_kernel(kernel_size)
    patch_size = kernel_x.shape

    converged = False

    while not converged:
        w, inliers = get_patch(responses, position, patch_size)

        x_change = np.divide(
            np.sum(np.multiply(kernel_x, w)),
            np.sum(w)
        )

        y_change = np.divide(
            np.sum(np.multiply(kernel_y, w)),
            np.sum(w)
        )

        print([floor(n) for n in position], x_change, y_change)

        if abs(x_change) < epsilon and abs(y_change) < epsilon:
            converged = True

        position = position[0] + x_change, position[1] + y_change

    return int(floor(position[0])), int(floor(position[1]))


if __name__ == "__main__":
    responses = generate_responses_1()
    # show_image(responses * 400, 0, "responses")
    kernel_size = 5
    epsilon = 0.05
    starting_position = (30, 70)
    max_x, max_y = mean_shift(responses, starting_position, kernel_size, epsilon)
    print(max_x, max_y, responses[max_x][max_y])