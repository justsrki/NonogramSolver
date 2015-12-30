import cv2
import numpy as np
from util.benchmark import timeit
from os import listdir
from os.path import isfile, join
from config import Config


class MaskClassifier:
    @timeit
    def __init__(self, digits_path=None, masks_path=None):
        self.masks = [None] * 10

        if digits_path is not None:
            self.path = digits_path
            self.masks_path = self.path + '\\masks\\'
            self.create_masks()
        elif masks_path is not None:
            self.masks_path = masks_path
            self.load_masks()
            pass

    @staticmethod
    def __get_files(path):
        file_paths = []
        for f in listdir(path):
            if isfile(join(path, f)):
                file_paths.append(join(path, f))
            else:
                for digit_path in MaskClassifier.__get_files(join(path, f)):
                    file_paths.append(digit_path)
        return file_paths

    @staticmethod
    def __create_mask(paths):
        mask = np.zeros(Config.digit_size, 'float32')
        for path in paths:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, Config.digit_size)
            mask += img

        mask /= mask.max()
        return mask

    def create_masks(self):
        for i in xrange(0, 10):
            paths = MaskClassifier.__get_files(self.path + '\\' + str(i))
            mask = MaskClassifier.__create_mask(paths) * 255
            cv2.imwrite(self.masks_path + '\\' + str(i) + '.png', mask)
            self.masks[i] = mask.astype('int')

    def load_masks(self):
        for i in xrange(0, 10):
            mask = np.zeros(Config.digit_size, 'int')
            mask += cv2.resize(cv2.imread(self.masks_path + '\\' + str(i) + '.png', cv2.IMREAD_GRAYSCALE), Config.digit_size)
            self.masks[i] = mask

    def classify(self, img):
        img = cv2.resize(img, Config.digit_size)
        return [(1.0 * np.abs(self.masks[i] - img).sum() / self.masks[i].sum(), i) for i, mask in enumerate(self.masks)]

    def test(self, root_path, full_log=False):
        for i in xrange(0, 10):
            print '---------- ', i, ' ----------'
            count, error = 0, 0

            for path in MaskClassifier.__get_files(root_path + '\\' + str(i)):
                count += 1
                digits = sorted(self.classify(cv2.imread(path, cv2.IMREAD_GRAYSCALE)))

                if full_log:
                    print "{} - {:.4f}  -  {} - {:.4f}".format(digits[0][1], digits[0][0], digits[1][1], digits[1][0])

                if i != digits[0][1]:
                    error += 1
                    print digits
            print "Successful: {:.2f}% - Count: {}, Error: {}".format(100.0 * (count - error) / count, count, error)
