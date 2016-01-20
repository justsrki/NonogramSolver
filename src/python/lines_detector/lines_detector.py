import numpy as np
import cv2
from util.benchmark import timeit
from config import Config


class LinesDetector:
    def __init__(self, line_detection_fn):
        self.image = None
        self.line_detection_fn = line_detection_fn

    def set_image(self, image):
        self.image = image

    @staticmethod
    def get_lines_coordinates(sum_line, length):
        result = []
        threshold = 255 * length / 3

        lines = []
        for x, val in enumerate(sum_line):
            if val > threshold:
                lines.append((x, val))

        eps = 2 * np.mean(np.diff([x for x, y in lines]))
        last = 0
        sum_line, center = 1, 0

        lines.append((16 * Config.image_width, 0))
        for line in lines:
            if (line[0] - last) < eps:
                sum_line += line[1]
                center += line[0] * line[1]
            else:
                result.append(center / sum_line)
                sum_line, center = line[1], line[0] * line[1]
            last = line[0]

        return result

    @timeit
    def get_result(self):
        img_v, img_h = self.line_detection_fn(self.image)
        coords_v = self.get_lines_coordinates(img_v.sum(axis=0), self.image.shape[0])
        coords_h = self.get_lines_coordinates(img_h.sum(axis=1), self.image.shape[1])
        return (img_v, img_h), (coords_v, coords_h)
