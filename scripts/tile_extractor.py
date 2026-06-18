import cv2
import numpy as np

class ITileExtractor(object):

    debug = False

    def __init__(self):
        pass

    def extract(self, frame: cv2.Mat) -> cv2.Mat:
        raise Exception("NotImplementedException")