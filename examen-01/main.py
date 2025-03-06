import numpy as np
import cv2
import matplotlib.pyplot as plt

# función para recortar la región de interés
def recorte(frame):
    x1, y1 = 280, 400  # coordenadas de la esquina superior izquierda
    x2, y2 = 1280, 720  # coordenadas de la esquina inferior derecha
    return frame[y1:y2, x1:x2]

# función para aplicar filtro gamma
# mejorar constraste de la imagen
# gamma = 2: aumenta el contraste
def filtro_gamma(frame, gamma=2.0):
    # oscurece las luces y resalta los detalles en las sombras.
    gamma_corrected = np.array(255 * (frame / 255) ** gamma, dtype="uint8") # corrección gamma, 255 es el valor máximo de intensidad
    return gamma_corrected

# función para cargar el video y procesar los frames
def cargar_video(ruta_video):
    cap = cv2.VideoCapture(ruta_video)
    
    # verificar si el video se abrió correctamente
    if not cap.isOpened():
        print("Error al abrir el video.")
        return
    
    # leer los frames del video
    # procesar cada frame, cada ciclo lee un frame
    # aplicar recorte y filtro gamma
    while cap.isOpened():
        ret, frame = cap.read()
        # verificar si se llegó al final del video
        if not ret:
            break

        
        
        # recorte de la carretera
        frame_recortado = recorte(frame)
        
        # aplicar filtro gamma
        frame_filtrado = filtro_gamma(frame_recortado)
        
        # mostrar resultados parciales
        cv2.imshow('Frame Original', frame)
        cv2.imshow('Frame Recortado', frame_recortado)
        cv2.imshow('Frame con Filtro Gamma', frame_filtrado)
        
        # salir con tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ejecutar el procesamiento con un video de prueba
if __name__ == "__main__":
    ruta_video = 'lineas.mp4'  # ruta del video de entrada
    cargar_video(ruta_video)
