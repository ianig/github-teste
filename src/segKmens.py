
import polars as pl
from sklearn.cluster import KMeans
from skimage.util import img_as_float
import numpy as np

class SegKMens:
    def __init__(self, img):
        self._img = img
        radio = self._radio(np.array(img))
        radio = radio.flatten()

        df_pl = pl.DataFrame({
            "intensity": img.flatten(),
            "raio": radio
        })
        df_pd = df_pl.to_pandas()
        self._df = df_pd

        self._model = KMeans(6, random_state=123)
        self._model.fit(df_pd)

    def _radio(self, img):
        h, w = img.shape  # altura e largura da imagem

        # Coordenadas do centro da imagem
        centro_y, centro_x = (h - 1) / 2, (w - 1) / 2

        # Cria um grid com coordenadas (x, y)
        y_indices, x_indices = np.indices((h, w))

        # Calcula a distância euclidiana até o centro
        distancias = np.sqrt((x_indices - centro_x)**2 + (y_indices - centro_y)**2)

        # Normaliza entre 0 e 1
        raio_max = distancias.max()
        distancias_normalizadas = distancias / raio_max if raio_max != 0 else distancias

        return distancias_normalizadas

    def clusters(self):
        y_pred = self._model.predict(self._df)
        self._y_pred = y_pred
        return np.unique(y_pred)
    
    def run(self, cluster):
        zeros = np.zeros(shape=self._img.shape)
        pred_image = self._y_pred.reshape(self._img.shape)
        pred_image = np.where(pred_image != cluster, 0, pred_image) 
        pred_image = np.where(pred_image == cluster, 1, pred_image) 
        return pred_image + zeros