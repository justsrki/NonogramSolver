import cv2
import numpy as np

from util.benchmark import timeit


class PerspectiveTransformer:
    def __init__(self, binarize_fn):
        self.binarize_fn = binarize_fn
        self.image = None
        self.contour = None

    def set_image(self, image):
        self.image = image

    def set_contour(self, contour):
        self.contour = contour

    # http://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
    def create_rect(self):
        points = np.zeros((self.contour.shape[0], 2), 'float32')
        for i, contour_point in enumerate(self.contour):
            points[i] = (contour_point[0])

        rect = np.zeros((4, 2), dtype="float32")

        s = points.sum(axis=1)
        rect[0] = points[np.argmin(s)]
        rect[2] = points[np.argmax(s)]

        diff = np.diff(points, axis=1)
        rect[1] = points[np.argmin(diff)]
        rect[3] = points[np.argmax(diff)]

        return rect

    @timeit
    def get_result(self):
        rect = self.create_rect()
        (tl, tr, br, bl) = rect

        width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        max_width = max(int(width_a), int(width_b))

        height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        max_height = max(int(height_a), int(height_b))

        dst_coordinates = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]], dtype="float32")

        wrapped_img = cv2.warpPerspective(self.image, cv2.getPerspectiveTransform(rect, dst_coordinates), (max_width, max_height))
        return self.binarize_fn(wrapped_img)
