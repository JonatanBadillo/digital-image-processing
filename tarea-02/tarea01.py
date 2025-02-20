import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ruta de la imagen (ajusta según la ubicación correcta)
ruta = "carretera1.png"

# cargar imagen con validación
image = cv2.imread(ruta)
if image is None:
    print("⚠️ Error: No se pudo cargar la imagen. Verifica la ruta.")
    exit()

# convertir a RGB para visualización en matplotlib
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# crear carpeta para guardar las imágenes procesadas
output_dir = "/tarea-02/imagenes_procesadas"
os.makedirs(output_dir, exist_ok=True)

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
    "Promedio_3x3": cv2.filter2D(image, -1, kernel_average_3x3),
    "Promedio_5x5": cv2.filter2D(image, -1, kernel_average_5x5),
    "Gaussiano_3x3": cv2.filter2D(image, -1, kernel_gaussian_3x3),
    "Gaussiano_5x5": cv2.filter2D(image, -1, kernel_gaussian_5x5),
    "Mediana_3x3": cv2.medianBlur(image, 3),
    "Mediana_5x5": cv2.medianBlur(image, 5),
}

# 📌 Guardar imágenes procesadas
for name, img in filtered_images.items():
    save_path = os.path.join(output_dir, f"{name}.png")
    cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))  # Convertir de RGB a BGR para OpenCV
    print(f"✅ Imagen guardada: {save_path}")

# 📌 Visualizar resultados
plt.figure(figsize=(12, 8))
for i, (title, img) in enumerate(filtered_images.items()):
    plt.subplot(3, 3, i + 1)
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")
plt.tight_layout()
plt.show()
