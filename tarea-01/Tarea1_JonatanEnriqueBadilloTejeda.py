import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse


"""Calcula la función de distribución acumulativa (CDF) normalizada."""
def compute_cdf(hist):
    cdf = hist.cumsum()
    cdf_normalized = cdf / float(cdf[-1]) # Normalización dividiendo entre el máximo
    return cdf_normalized

"""Aplica la especificación del histograma a una imagen utilizando la CDF de referencia."""
def match_histogram(image, ref_cdf):
    # Calcular el histograma y la CDF de la imagen original
    hist, bins = np.histogram(image.flatten(), 256, [0,256])
    cdf_image = compute_cdf(hist)
    
    # Crear la función de mapeo
    mapping = np.interp(cdf_image, ref_cdf, np.arange(256))
    
    # Aplicar la transformación
    corrected_image = np.interp(image.flatten(), np.arange(256), mapping).reshape(image.shape)
    
    return corrected_image.astype(np.uint8)

