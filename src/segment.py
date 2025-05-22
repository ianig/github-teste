import cv2 as cv
from skimage.util import img_as_float

class SegmentationHelper:
    def __init__(self, img=None):  # já está certo assim
        self._img = img

    def read_img(self, path):
        img = cv.imread(path, cv.IMREAD_GRAYSCALE)
        self._img = img_as_float(img)
        if self._img is None:
            raise ValueError("Erro ao ler a imagem")

    def roi(self):
        self._recorte = cv.selectROI("segment", self._img)
        (x, y, w, h) = self._recorte
        cv.destroyAllWindows()
        return self._recorte, self._img[y:y+h, x:x+w]

    def get_img(self):
        return self._img

