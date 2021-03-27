import numpy as np

from ex2_utils import *


def normalize_histogram(histogram):
    bin_sum = sum(histogram)
    return np.array([el/bin_sum for el in histogram])


# def mean_shift(image, prevPos, template_histogram, size, kernel, nbins, eps):
#     currPos = prevPos
#     converged = False
#
#     while not converged:
#         patch, mask = get_patch(image, currPos, size)
#         histogram = normalize_histogram(extract_histogram(patch, nbins, weights=kernel))
#
#         show_histogram(template_histogram)
#         show_histogram(histogram)
#
#         weights = np.sqrt(np.divide(template_histogram, histogram + eps))
#         show_histogram(weights)
#         exit()


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
        self.size = (region[2], region[3])
        self.template = image[int(top):int(bottom), int(left):int(right)]
        self.kernel = create_epanechnik_kernel(self.size[0], self.size[1], self.parameters.kernel_sigma)
        self.template_histogram = normalize_histogram(extract_histogram(self.template,
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

        mean_shift(image,
                   self.position,
                   self.template_histogram,
                   self.size,
                   self.kernel,
                   self.parameters.histogram_bins,
                   self.parameters.epsilon)

        # self.position = (left + max_loc[0] + float(self.size[0]) / 2, top + max_loc[1] + float(self.size[1]) / 2)
        #
        # return [left + max_loc[0], top + max_loc[1], self.size[0], self.size[1]]


class MSParams():
    def __init__(self):
        self.enlarge_factor = 2
        # self.kernelW = 22
        # self.kernelH = 22
        self.kernel_sigma = 0.5
        self.histogram_bins = 16
        self.epsilon = 0.1
