import cv2
import numpy as np
import matplotlib.pyplot as plt

# 📌 Ruta de la imagen (ajusta según la ubicación correcta)
ruta = "carretera1.png"

# 📌 Cargar imagen con validación
image = cv2.imread(ruta)
if image is None:
    print("⚠️ Error: No se pudo cargar la imagen. Verifica la ruta.")
    exit()

# Convertir a RGB para visualización en matplotlib
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 📌 Definir kernels de suavizado
kernel_average_3x3 = np.ones((3,3), np.float32) / 9.0
kernel_average_5x5 = np.ones((5,5), np.float32) / 25.0
kernel_gaussian_3x3 = np.array([[1, 4, 1], [4, 12, 4], [1, 4, 1]]) * (1/32)
kernel_gaussian_5x5 = np.array([[1, 4, 7, 4, 1], 
                                [4, 16, 26, 16, 4], 
                                [7, 26, 41, 26, 7], 
                                [4, 16, 26, 16, 4], 
                                [1, 4, 7, 4, 1]]) * (1/273)

# 📌 Aplicar filtros mencionados en clase
filtered_images = {
    "Original": image,
    "Promedio 3x3": cv2.filter2D(image, -1, kernel_average_3x3),
    "Promedio 5x5": cv2.filter2D(image, -1, kernel_average_5x5),
    "Gaussiano 3x3": cv2.filter2D(image, -1, kernel_gaussian_3x3),
    "Gaussiano 5x5": cv2.filter2D(image, -1, kernel_gaussian_5x5),
    "Mediana 3x3": cv2.medianBlur(image, 3),
    "Mediana 5x5": cv2.medianBlur(image, 5),
}

# 📌 Visualizar resultados
plt.figure(figsize=(12, 8))
for i, (title, img) in enumerate(filtered_images.items()):
    plt.subplot(3, 3, i + 1)
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")
plt.tight_layout()
plt.show()
