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

    nonogram_detector = NonogramDetector(binarize_adaptive_threshold, rectangle_approx_poly)
    transformer = PerspectiveTransformer(binarize_fixed_threshold(105), binarize_adaptive_threshold)
    line_detector = LinesDetector(lines_nxm_kernel)
    classifier = MaskClassifier(masks_path="..\\..\\res\\digits_00\\masks")
    digit_detector = DigitDetector(classifier)
    nonogram_solver = NonogramSolver()
    solution_creator = SolutionCreator()

    solver = Nonogram(frame, nonogram_detector=nonogram_detector, perspective_transformer=transformer,
                      line_detector=line_detector, digit_classifier=classifier, digit_detector=digit_detector,
                      nonogram_solver=nonogram_solver, solution_creator=solution_creator)

    imgs = solver.solve()
    for img in imgs:
        img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow('test', img)
        cv2.waitKey(0)
