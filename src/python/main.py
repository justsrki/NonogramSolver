from nonogram import Nonogram
from nonogram_detector.binarize_functions import *
from nonogram_detector.nonogram_detector import *
from nonogram_detector.rectangle_functions import *
from perspective_transformer.perspective_transformer import *
from lines_detector.lines_detector import *
from lines_detector.line_detection_functions import *
from digit_detector.digit_detector import *
from digit_classifier.mask_classifier import *
from nonogram_solver.nonogram_solver import *
from solution_creator.solution_creator import *

'''
mask_classifier = MaskClassifier(digits_path="..\\..\\res\\digits_00")
mask_classifier.test("..\\..\\res\\digits_00")
'''

path = "..\\..\\res\\imgs_00\\img_%2d.jpg"
cap = cv2.VideoCapture(path)
while True:
    print
    ret, frame = cap.read()
    if not ret:
        break

    solver = Nonogram(frame)

    nonogram_detector = NonogramDetector(binarize_adaptive_threshold, rectangle_approx_poly)
    solver.set_nonogram_detector(nonogram_detector)

    transformer = PerspectiveTransformer(binarize_fixed_threshold(96))
    solver.set_perspective_transformer(transformer)

    line_detector = LinesDetector(lines_nxm_kernel)
    solver.set_line_detector(line_detector)

    classifier = MaskClassifier(masks_path="..\\..\\res\\digits_00\\masks")
    solver.set_digit_classifier(classifier)

    digit_detector = DigitDetector(classifier)
    solver.set_digit_detector(digit_detector)

    nonogram_solver = NonogramSolver()
    solver.set_nonogram_solver(nonogram_solver)

    solution_creator = SolutionCreator()
    solver.set_solution_creator(solution_creator)

    imgs = solver.solve()
    for img in imgs:
        img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow('test', img)
        cv2.waitKey(0)
