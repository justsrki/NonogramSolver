import cv2
from util.benchmark import timeit


class Nonogram:
    @timeit
    def __init__(self, image):
        self.detector = None
        self.perspective_transformer = None
        self.line_detector = None
        self.image = cv2.imread(image) if isinstance(image, basestring) else image

    def set_detector(self, detector):
        self.detector = detector

    def set_perspective_transformer(self, transformer):
        self.perspective_transformer = transformer

    def set_line_detector(self, detector):
        self.line_detector = detector

    @timeit
    def solve(self):
        sol = []
        self.detector.set_image(self.image)
        img, rects = self.detector.get_result()

        for rect in rects:
            self.perspective_transformer.set_image(img)
            self.perspective_transformer.set_contour(rect)
            img = self.perspective_transformer.get_result()

            self.line_detector.set_image(img)
            imgs, coords = self.line_detector.get_result()

            # img = cv2.add(img, cv2.add(imgs[0], imgs[1]))
            sol.append(img)

        return sol
