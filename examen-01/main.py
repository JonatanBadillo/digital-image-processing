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

# Función para detectar líneas amarillas y blancas en el camino
def detectar_lineas(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([15, 150, 150])
    upper_yellow = np.array([30, 255, 255])
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 50, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    combined_mask = cv2.bitwise_or(mask_yellow, mask_white)
    return combined_mask

# Función para enmascarar el resto del video en negro
def aplicar_fondo_negro(frame, mask_lineas):
    resultado = cv2.bitwise_and(frame, frame, mask=mask_lineas)
    fondo_negro = np.zeros_like(frame)
    frame_con_lineas = cv2.add(fondo_negro, resultado)
    return frame_con_lineas

# Función para procesar el frame completo
def procesar_frame(frame):
    frame_procesado = filtro_gamma(frame)
    mask_lineas = detectar_lineas(frame_procesado)
    frame_con_lineas = aplicar_fondo_negro(frame_procesado, mask_lineas)
    return frame_con_lineas

# Función para cargar el video y procesar los frames
def cargar_video(ruta_video, ruta_salida):
    cap = cv2.VideoCapture(ruta_video)
    if not cap.isOpened():
        print("Error al abrir el video.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(ruta_salida, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_recortado = recorte(frame)
        frame_procesado = procesar_frame(frame_recortado)
        out.write(frame_procesado)
        cv2.imshow('Frame Procesado', frame_procesado)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Función principal
if __name__ == "__main__":
    ruta_video = 'lineas.mp4'  # Ruta del video
    ruta_salida = 'lineas_procesado.mp4'  # Ruta para exportar video
    cargar_video(ruta_video, ruta_salida)