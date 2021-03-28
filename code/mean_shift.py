import numpy as np

from ex2_utils import generate_responses_1, get_patch, show_image
from math import floor


def create_kernel(kernel_size):
    # make sure kernel size is odd
    kernel_size = [int(ks - 1) if ks % 2 == 0 else int(ks) for ks in kernel_size]

    kernel_x = np.zeros((kernel_size[0], kernel_size[1]))

    for col, value in enumerate(range(-(kernel_size[1] // 2), (kernel_size[1] // 2) + 1, 1)):
        kernel_x[:, col] = value

    kernel_y = np.zeros((kernel_size[0], kernel_size[1]))

    for row, value in enumerate(range(-(kernel_size[0] // 2), (kernel_size[0] // 2) + 1, 1)):
        kernel_y[row] = value


    return kernel_x, kernel_y


def mean_shift(responses, position, kernel_size, epsilon):
    kernel_x, kernel_y = create_kernel(kernel_size[1])
    patch_size = (kernel_size[0], kernel_size[1])

    converged = False
    iters = 0

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
        iters += 1

        # if iters % 100 == 0:
            # print(iters)

    return int(floor(position[0])), int(floor(position[1])), iters


if __name__ == "__main__":
    responses = generate_responses_1()
    # show_image(responses * 400, 0, "responses")
    kernel_size = (5, 5)
    epsilon = 0.01
    starting_position = (50, 50)
    max_x, max_y, iters = mean_shift(responses, starting_position, kernel_size, epsilon)
    print(max_x, max_y, responses[max_x][max_y], iters)