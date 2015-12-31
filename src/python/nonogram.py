import cv2
from util.benchmark import timeit


class Nonogram:
    @timeit
    def __init__(self, image):
        self.nonogram_detector = None
        self.perspective_transformer = None
        self.line_detector = None
        self.digit_detector = None
        self.digit_classifier = None
        self.nonogram_solver = None
        self.solution_creator = None
        self.image = cv2.imread(image) if isinstance(image, basestring) else image

    def set_nonogram_detector(self, detector):
        self.nonogram_detector = detector

    def set_perspective_transformer(self, transformer):
        self.perspective_transformer = transformer

    def set_line_detector(self, detector):
        self.line_detector = detector

    def set_digit_detector(self, detector):
        self.digit_detector = detector

    def set_digit_classifier(self, classifier):
        self.digit_classifier = classifier

    def set_nonogram_solver(self, solver):
        self.nonogram_solver = solver

    def set_solution_creator(self, creator):
        self.solution_creator = creator

    @timeit
    def solve(self):
        result = []
        self.nonogram_detector.set_image(self.image)
        img, rects = self.nonogram_detector.get_result()

        for rect in rects:
            self.perspective_transformer.set_image(img)
            self.perspective_transformer.set_contour(rect)
            img_wrapped = self.perspective_transformer.get_result()

            self.line_detector.set_image(img_wrapped)
            imgs, coords = self.line_detector.get_result()
            img = cv2.add(img_wrapped, cv2.add(imgs[0], imgs[1]))

            self.digit_detector.set_image(img)
            self.digit_detector.set_lines(coords)
            img, nonogram_values = self.digit_detector.get_result()

            if nonogram_values is None:
                result.append(img)
                continue

            self.nonogram_solver.set_values(nonogram_values)
            solution = self.nonogram_solver.get_result()

            self.solution_creator.set_image(img_wrapped)
            self.solution_creator.set_coordinates(coords)
            self.solution_creator.set_solution(solution)
            img = self.solution_creator.get_result()

            result.append(img)

        return result
