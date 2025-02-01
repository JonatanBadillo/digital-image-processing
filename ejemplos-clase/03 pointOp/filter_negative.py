#list of liberies used in this proyect
import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np

# main program ---------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", required = True, help = "Path to the input image")
args = vars(parser.parse_args())

image_BGR = cv2.imread(args["image"])
image_RGB = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB)

#se calcula el valor máximo existente en los pixeles de la imagen
max = image_RGB.max()
image_RGB_negative = max - image_RGB

#se genera una figura para mostrar los resultados con matplotlib
fig=plt.figure(figsize=(14,10))
#se maqueta el diseño del grafico
ax1=fig.add_subplot(2,2,1)
ax2=fig.add_subplot(2,2,2)
ax3=fig.add_subplot(2,3,5)
#se dibuja la imagen original
ax1.imshow(image_RGB)
ax1.set_title('Original image')
#se dibuja la imagen con el operador
ax2.imshow(image_RGB_negative)
ax2.set_title('Negative')
#se dibujan las graficas de las funciones
x = np.linspace(0, 255, 255)
y1 = x
y2 = np.array(255-x)
ax3.plot(x, y1, color = "r", linewidth = 1, label = "Id. Func.")
msg = "Negative func."
ax3.plot(x, y2, color = "b", linewidth = 1, label = msg)
ax3.legend()
ax3.set_title('Negative function')
plt.show()


