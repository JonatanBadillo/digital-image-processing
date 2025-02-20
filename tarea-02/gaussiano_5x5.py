# Filtro Gaussiano 5x5 - gaussiano_5x5.py
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ruta de la imagen
ruta = "carretera1.png"

# cargar imagen con validación
image = cv2.imread(ruta)
if image is None:
    print("⚠️ Error: No se pudo cargar la imagen. Verifica la ruta.")
    exit()

# convertir a RGB para visualización
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# definir kernel gaussiano 5x5
kernel_gaussian_5x5 = np.array([[1, 4, 7, 4, 1],
                                [4, 16, 26, 16, 4],
                                [7, 26, 41, 26, 7],
                                [4, 16, 26, 16, 4],
                                [1, 4, 7, 4, 1]]) * (1/273)

# aplicar filtro gaussiano 5x5
filtered_image = cv2.filter2D(image, -1, kernel_gaussian_5x5)

# visualizar comparativa
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(filtered_image)
plt.title("Gaussiano 5x5")
plt.axis("off")

plt.tight_layout()
plt.show()
