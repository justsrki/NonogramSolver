import cv2
from config import Config

from util.benchmark import timeit


class NonogramDetector:
    def __init__(self, binarize_fn, rectangle_fn):
        self.binarize_fn = binarize_fn
        self.rectangle_fn = rectangle_fn
        self.image = None

    def set_image(self, image):
        self.image = image

    @timeit
    def resize_image(self):
        factor = Config.image_width / self.image.shape[1]
        self.image = cv2.resize(self.image, (0, 0), fx=factor, fy=factor, interpolation=Config.interpolation)

    @timeit
    def change_color_channel(self):
        self.image = cv2.cvtColor(self.image, Config.cvt_color_code)

    @timeit
    def binarize(self):
        self.image = self.binarize_fn(self.image)

    # @timeit
    @staticmethod
    def check_minimal_size(contour):
        x, y, w, h = cv2.boundingRect(contour)
        if h > Config.d_min and w > Config.d_min and 1. / Config.aspect_ratio < 1.0 * h / w < Config.aspect_ratio:
            return True

    @timeit
    def find_rectangles(self):
        rects = []
        _, contours, _ = cv2.findContours(self.image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if self.check_minimal_size(contour):
                rect = self.rectangle_fn(contour)
                if rect is not None:
                    rects.append(contour)

        return rects

    @timeit
    def get_result(self):
        self.change_color_channel()
        self.resize_image()
        self.binarize()
        rects = self.find_rectangles()

        return self.image, rects
