import cv2

from util.benchmark import timeit


class Nonogram:
    @timeit
    def __init__(self, image):
        self.detector = None
        self.perspective_transformer = None
        self.image = cv2.imread(image) if isinstance(image, basestring) else image

    def set_detector(self, detector):
        self.detector = detector

    def set_perspective_transformer(self, transformer):
        self.perspective_transformer = transformer

    @timeit
    def solve(self):
        self.detector.set_image(self.image)
        img, rects = self.detector.get_result()

        for rect in rects:
            self.perspective_transformer.set_image(img)
            self.perspective_transformer.set_contour(rect)
            img = self.perspective_transformer.get_result()

        return img
