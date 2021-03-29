from math import floor

import numpy as np
import cv2

from ex2_utils import get_patch, create_epanechnik_kernel, extract_histogram, backproject_histogram, Tracker, show_image
from mean_shift import create_kernel


def normalize_histogram(histogram):
    bin_sum = sum(histogram)
    return np.array([el/bin_sum for el in histogram])


def mean_shift(image, position, size, q, kernel, nbins, epsilon):

    k_size = [size[1], size[0]]

    if k_size[0] % 2 == 0:
        k_size[0] = k_size[0] - 1
    if k_size[1] % 2 == 0:
        k_size[1] = k_size[1] - 1

    kernel_x, kernel_y = create_kernel(k_size)
    patch_size = (k_size[1], k_size[0])

    converged = False
    iters = 0

    while not converged:
        patch, mask = get_patch(image, position, patch_size)
        p = normalize_histogram(extract_histogram(patch, nbins, weights=kernel))
        v = np.sqrt(np.divide(q, p + epsilon))
        w = backproject_histogram(patch, v, nbins, kernel)

        x_change = np.divide(
            np.sum(np.multiply(kernel_x, w)),
            np.sum(w)
        )

        y_change = np.divide(
            np.sum(np.multiply(kernel_y, w)),
            np.sum(w)
        )

        # print([floor(n) for n in position], x_change, y_change)

        if abs(x_change) < epsilon and abs(y_change) < epsilon:
            converged = True
        else:
            position = position[0] + x_change, position[1] + y_change
        iters += 1

        # if iters % 100 == 0:
            # print(iters)

    return int(floor(position[0])), int(floor(position[1]))


class MeanShiftTracker(Tracker):

    def initialize(self, image, region):

        if len(region) == 8:
            x_ = np.array(region[::2])
            y_ = np.array(region[1::2])
            region = [np.min(x_), np.min(y_), np.max(x_) - np.min(x_) + 1, np.max(y_) - np.min(y_) + 1]

        self.window = max(region[2], region[3]) * self.parameters.enlarge_factor

        left = max(region[0], 0)
        top = max(region[1], 0)

        right = min(region[0] + region[2], image.shape[1] - 1)
        bottom = min(region[1] + region[3], image.shape[0] - 1)

        self.position = (region[0] + region[2] / 2, region[1] + region[3] / 2)
        self.size = (round(region[2]), round(region[3]))
        self.template = image[int(top):int(bottom), int(left):int(right)]
        self.kernel = create_epanechnik_kernel(self.size[0], self.size[1], self.parameters.kernel_sigma)
        self.q = normalize_histogram(extract_histogram(self.template,
                                                       self.parameters.histogram_bins,
                                                       weights=self.kernel))

    def track(self, image):
        left = max(round(self.position[0] - float(self.window) / 2), 0)
        top = max(round(self.position[1] - float(self.window) / 2), 0)

        right = min(round(self.position[0] + float(self.window) / 2), image.shape[1] - 1)
        bottom = min(round(self.position[1] + float(self.window) / 2), image.shape[0] - 1)

        if right - left < self.template.shape[1] or bottom - top < self.template.shape[0]:
            return [self.position[0] + self.size[0] / 2, self.position[1] + self.size[1] / 2, self.size[0],
                    self.size[1]]

        new_x, new_y = mean_shift(image,
                                  self.position,
                                  self.size,
                                  self.q,
                                  self.kernel,
                                  self.parameters.histogram_bins,
                                  self.parameters.epsilon)

        # MODEL UPDATE
        self.template, _ = get_patch(image, (new_x, new_y), self.size)
        self.q = (1 - self.parameters.update_alpha) * self.q + self.parameters.update_alpha * normalize_histogram(extract_histogram(self.template, self.parameters.histogram_bins, weights=self.kernel))

        self.position = (new_x, new_y)
        return [new_x, new_y, self.size[0], self.size[1]]


class MSParams():
    def __init__(self):
        self.enlarge_factor = 2
        # self.mean_shift_kernel_size = 5
        self.kernel_sigma = 0.5
        self.histogram_bins = 16
        self.epsilon = 1
        self.update_alpha = 0
