import numpy as np
import cv2

# función para recortar la región de interés
def recorte(frame):
    x1, y1 = 280, 400  # Coordenadas de la esquina superior izquierda
    x2, y2 = 1280, 720  # Coordenadas de la esquina inferior derecha

    # Verifica que las dimensiones sean válidas antes de recortar
    if frame.shape[0] < y2 or frame.shape[1] < x2:
        return frame  # Devuelve el frame original si las coordenadas no son válidas
    return frame[y1:y2, x1:x2]

# función para aplicar filtro gamma
def filtro_gamma(frame, gamma=2.0):
    lookup_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(frame, lookup_table)

# función para detectar líneas amarillas y blancas en el camino
def detectar_lineas(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Definir rango de colores amarillo y blanco
    lower_yellow = np.array([15, 150, 150])
    upper_yellow = np.array([30, 255, 255])
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 50, 255])

    # Aplicar máscaras
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    combined_mask = cv2.bitwise_or(mask_yellow, mask_white)

    return combined_mask

# Función para aplicar fondo negro y mantener solo líneas detectadas
def aplicar_fondo_negro(frame, mask_lineas):
    resultado = cv2.bitwise_and(frame, frame, mask=mask_lineas)
    return resultado

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
        print("Error: No se pudo abrir el video.")
        return
    print("Video cargado correctamente.")

    # Obtener información del video
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"FPS del video: {fps}")
    print(f"Total de frames detectados: {total_frames}")

    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo leer el primer frame.")
        cap.release()
        return

    # Verificar dimensiones después del recorte
    frame_recortado = recorte(frame)
    height, width = frame_recortado.shape[:2]
    print(f"Tamaño del frame después del recorte: {width}x{height}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(ruta_salida, fourcc, fps, (width, height))

    frame_count = 0  # Contador de frames procesados

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Fin del video o error al leer el frame.")
                break

            frame_count += 1
            print(f"Procesando frame {frame_count}/{total_frames}")

            frame_recortado = recorte(frame)
            frame_procesado = procesar_frame(frame_recortado)

            # Verificar que el frame procesado tenga el mismo tamaño que el VideoWriter
            if frame_procesado.shape[:2] != (height, width):
                print(f"Error: Tamaño incorrecto en el frame procesado ({frame_procesado.shape[:2]})")
                break

            out.write(frame_procesado)

            cv2.imshow('Frame Procesado', frame_procesado)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Procesamiento detenido por el usuario.")
                break
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print("Recursos liberados correctamente.")

# Función principal
if __name__ == "__main__":
    ruta_video = 'lineas.mp4'  # Ruta del video
    ruta_salida = 'lineas_procesado.mp4'  # Ruta para exportar video
    cargar_video(ruta_video, ruta_salida)
