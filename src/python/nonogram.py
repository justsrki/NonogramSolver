import cv2
from util.benchmark import timeit


class Nonogram:
    @timeit
    def __init__(self, image, nonogram_detector, perspective_transformer, line_detector,
                 digit_detector, digit_classifier, nonogram_solver, solution_creator):
        self.nonogram_detector = nonogram_detector
        self.perspective_transformer = perspective_transformer
        self.line_detector = line_detector
        self.digit_detector = digit_detector
        self.digit_classifier = digit_classifier
        self.nonogram_solver = nonogram_solver
        self.solution_creator = solution_creator
        self.image = cv2.imread(image) if isinstance(image, basestring) else image

    @timeit
    def solve(self):
        result = []
        self.nonogram_detector.set_image(self.image)
        img_bin, rects = self.nonogram_detector.get_result()

        for rect in rects:
            self.perspective_transformer.set_image(self.image)
            self.perspective_transformer.set_contour(rect)
            img_wrapped = self.perspective_transformer.get_result()

            self.line_detector.set_image(img_wrapped)
            imgs, coords = self.line_detector.get_result()
            img = cv2.add(img_wrapped, cv2.add(imgs[0], imgs[1]))

            self.digit_detector.set_image(img)
            self.digit_detector.set_lines(coords)
            img, nonogram_values = self.digit_detector.get_result()

            if nonogram_values is None:
                # result.append(img)
                print "Nonogram can't be found or solved!"
                continue

            self.nonogram_solver.set_values(nonogram_values)
            solution = self.nonogram_solver.get_result()

            self.solution_creator.set_image(img_wrapped)
            self.solution_creator.set_coordinates(coords)
            self.solution_creator.set_solution(solution)
            img = self.solution_creator.get_result()

            result.append(img)

        return result
