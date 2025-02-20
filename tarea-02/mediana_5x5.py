# Filtro Mediana 5x5 - mediana_5x5.py
import cv2
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

# aplicar filtro de mediana 5x5
filtered_image = cv2.medianBlur(image, 5)

# visualizar comparativa
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(filtered_image)
plt.title("Mediana 5x5")
plt.axis("off")

plt.tight_layout()
plt.show()
