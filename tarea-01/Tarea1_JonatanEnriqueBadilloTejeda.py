import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse


"""Calcula la función de distribución acumulativa (CDF) normalizada."""
def compute_cdf(hist):
    cdf = hist.cumsum()
    cdf_normalized = cdf / float(cdf[-1]) # Normalización dividiendo entre el máximo
    return cdf_normalized

