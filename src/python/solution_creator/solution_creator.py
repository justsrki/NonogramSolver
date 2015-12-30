import cv2


class SolutionCreator:

    def __init__(self):
        self.image = None
        self.coordinates = None
        self.solution = None

    def set_image(self, image):
        self.image = image

    def set_coordinates(self, coordinates):
        self.coordinates = coordinates

    def set_solution(self, solution):
        self.solution = solution

    def get_result(self):
        self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)
        lines_v = self.coordinates[0]
        lines_h = self.coordinates[1]
        for i in xrange(1, len(lines_v) - 1):
            for j in xrange(1, len(lines_h) - 1):
                x1, x2 = lines_v[i], lines_v[i + 1]
                y1, y2 = lines_h[j], lines_h[j + 1]
                if self.solution[j - 1, i - 1] != 0:
                    cv2.rectangle(self.image, (x1, y1), (x2, y2), (0, 0, 0), thickness=cv2.FILLED)

        return self.image
