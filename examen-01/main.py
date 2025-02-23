# Import libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Función para recortar la región de interés
def recorte(frame):
    x1, y1 = 280, 400  # Coordenadas de la esquina superior izquierda
    x2, y2 = 1280, 720  # Coordenadas de la esquina inferior derecha
    return frame[y1:y2, x1:x2]

# Función para aplicar filtro gamma
def filtro_gamma(frame, gamma=2.0):
    gamma_corrected = np.array(255 * (frame / 255) ** gamma, dtype="uint8")
    return gamma_corrected

# Función para cargar el video y procesar los frames
def cargar_video(ruta_video):
    cap = cv2.VideoCapture(ruta_video)
    
    if not cap.isOpened():
        print("Error al abrir el video.")
        return
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Recorte de la región de interés
        frame_recortado = recorte(frame)
        
        # Aplicar filtro gamma
        frame_filtrado = filtro_gamma(frame_recortado)
        
        # Mostrar resultados parciales
        cv2.imshow('Frame Original', frame)
        cv2.imshow('Frame Recortado', frame_recortado)
        cv2.imshow('Frame con Filtro Gamma', frame_filtrado)
        
        # Salir con tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Ejecutar el procesamiento con un video de prueba
if __name__ == "__main__":
    ruta_video = 'lineas.mp4'  # Ruta del video de entrada
    cargar_video(ruta_video)
