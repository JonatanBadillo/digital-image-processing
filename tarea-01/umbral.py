import cv2
import numpy as np


# Cargar la imagen en escala de grises
img = cv2.imread('input.jpg', cv2.IMREAD_GRAYSCALE)

# Crear una ventana
cv2.namedWindow('input')

# Crear una barra de desplazamiento (trackbar) para el umbral
cv2.createTrackbar('Umbral', 'input', 0, 255, nothing)

while True:
    # Obtener la posición actual de la barra de desplazamiento
    umbral = cv2.getTrackbarPos('Umbral', 'input')
    
    # Aplicar el umbral a la imagen
    _, thresh_img = cv2.threshold(img, umbral, 255, cv2.THRESH_BINARY)
    
    # Mostrar la imagen umbralizada
    cv2.imshow('input', thresh_img)
    
    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cerrar todas las ventanas
cv2.destroyAllWindows()