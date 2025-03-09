import numpy as np
import cv2

# función para recortar la región de interés
def recorte(frame):
    x1, y1 = 280, 400  # coordenadas de la esquina superior izquierda
    x2, y2 = 1280, 720  # coordenadas de la esquina inferior derecha

    if frame.shape[0] < y2 or frame.shape[1] < x2:
        return frame  # devuelve el frame original si las coordenadas no son válidas
    return frame[y1:y2, x1:x2]

# función para aplicar filtro gamma
def filtro_gamma(frame, gamma=2.0):
    # crea una tabla de búsqueda para mapear los valores de píxeles de entrada a los valores de píxeles de salida
    # utilizando la fórmula de corrección gamma
    lookup_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(frame, lookup_table)

# función para detectar líneas amarillas y blancas en el camino
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

# función para aplicar fondo negro
def aplicar_fondo_negro(frame, mask_lineas):
    return cv2.bitwise_and(frame, frame, mask=mask_lineas)

# función para procesar el frame completo
def procesar_frame(frame):
    frame_procesado = filtro_gamma(frame)
    mask_lineas = detectar_lineas(frame_procesado)
    return aplicar_fondo_negro(frame_procesado, mask_lineas)

# función para cargar el video y procesar los frames
def cargar_video(ruta_video, ruta_salida):
    cap = cv2.VideoCapture(ruta_video)
    if not cap.isOpened():
        print("Error: No se pudo abrir el video.")
        return
    print("Video cargado correctamente.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo leer el primer frame.")
        cap.release()
        return

    frame_recortado = recorte(frame)
    height, width = frame_recortado.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(ruta_salida, fourcc, fps, (width, height))

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_recortado = recorte(frame)
            frame_procesado = procesar_frame(frame_recortado)
            out.write(frame_procesado)
            cv2.imshow('Frame Procesado', frame_procesado)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print("Procesamiento finalizado.")

# Función principal
if __name__ == "__main__":
    ruta_video = 'lineas.mp4'
    ruta_salida = 'lineas_procesado.mp4'
    cargar_video(ruta_video, ruta_salida)
