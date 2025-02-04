import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse


"""calcula la función de distribución acumulativa (CDF) normalizada."""
def compute_cdf(hist):
    cdf = hist.cumsum() # suma acumulativa
    cdf_normalized = cdf / float(cdf[-1]) # normalización dividiendo entre el total de píxeles
    return cdf_normalized

"""aplica la especificación del histograma a una imagen utilizando la CDF de referencia."""
def match_histogram(image, ref_cdf):
    # calcular el histograma y la CDF de la imagen original
    hist, bins = np.histogram(image.flatten(), 256, [0,256]) # histograma de la imagen original
    cdf_image = compute_cdf(hist) # CDF de la imagen original
    
    # crear la función de mapeo de intensidades 
    mapping = np.interp(cdf_image, ref_cdf, np.arange(256)) # mapeo de intensidades (0-255)
    
    # aplicar la transformación de intensidades a la imagen
    corrected_image = np.interp(image.flatten(), np.arange(256), mapping).reshape(image.shape) # imagen corregida con mapeo
    
    return corrected_image.astype(np.uint8) # convertir a uint8 para mostrar la imagen


"""generar y mostrar el histograma acumulativo normalizado de una imagen."""
def plot_histogram(image, title):
    hist, _ = np.histogram(image.flatten(), 256, [0,256]) # histograma de la imagen 
    cdf = compute_cdf(hist) # CDF de la imagen original
    plt.plot(cdf, color='blue') # graficar la CDF
    plt.title(title) # título del gráfico
    plt.xlabel("Intensidad") # etiqueta eje x
    plt.ylabel("Frecuencia Acumulada")  # etiqueta eje y
    plt.grid() # mostrar cuadrícula


def main():
    # argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Especificación del Histograma en Imágenes en Escala de Grises')
    # argumentos obligatorios para la imagen de entrada y el archivo de referencia
    parser.add_argument('-i', '--input', required=True, help='Ruta de la imagen de entrada en escala de grises')
    parser.add_argument('-o', '--output', required=True, help='Archivo de referencia con la CDF objetivo')
    # analizar los argumentos
    args = parser.parse_args()
    
    # cargar la imagen en escala de grises
    image = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE) 
    if image is None:
        print("Error al cargar la imagen.")
        return
    
    # cargar la CDF de referencia desde el archivo
    ref_points = np.loadtxt(args.output)

    # si hay dos columnas, extraer valores (a_k) y CDF (q_k)
    if ref_points.ndim == 2 and ref_points.shape[1] == 2:
        x_ref = ref_points[:, 0]  # valores de intensidad (0-255)
        y_ref = ref_points[:, 1]  # CDF de referencia (0-1)
        
        # interpolación lineal para obtener 256 valores de la CDF
        ref_cdf = np.interp(np.arange(256), x_ref, y_ref) 
    else:
        ref_cdf = ref_points  # si ya tiene 256 valores, usarlo directamente

    
    # aplicar la especificación del histograma a la imagen
    corrected_image = match_histogram(image, ref_cdf) # imagen corregida
    
    # mostrar resultados
    plt.figure(figsize=(12, 6))
    
    # imagen original con su histograma
    plt.subplot(2, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Imagen Original")
    plt.axis("off")
    
    plt.subplot(2, 2, 2)
    plot_histogram(image, "Histograma Acumulativo - Original")
    
    # imagen corregida con su histograma
    plt.subplot(2, 2, 3)
    plt.imshow(corrected_image, cmap='gray')
    plt.title("Imagen Corregida")
    plt.axis("off")
    
    plt.subplot(2, 2, 4)
    plot_histogram(corrected_image, "Histograma Acumulativo - Corregido")
    
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    main()