# Filtro Promedio 3x3 - promedio_3x3.py
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Ruta de la imagen
ruta = "carretera1.png"

# Cargar imagen con validación
image = cv2.imread(ruta)
if image is None:
    print("⚠️ Error: No se pudo cargar la imagen. Verifica la ruta.")
    exit()

# Convertir a RGB para visualización
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Definir kernel de promedio 3x3
kernel_average_3x3 = np.ones((3,3), np.float32) / 9.0

# Aplicar filtro promedio 3x3
filtered_image = cv2.filter2D(image, -1, kernel_average_3x3)

# Visualizar comparativa
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(filtered_image)
plt.title("Promedio 3x3")
plt.axis("off")

plt.tight_layout()
plt.show()