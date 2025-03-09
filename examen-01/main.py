import numpy as np
import cv2

# funcion para recortar la region de interes
# esto sirve para enfocarnos solo en una parte del frame

def recorte(frame):
    x1, y1 = 280, 400  # coordenadas de la esquina superior izquierda
    x2, y2 = 1280, 720  # coordenadas de la esquina inferior derecha
    if frame.shape[0] < y2 or frame.shape[1] < x2:
        return frame  # devuelve el frame original si las coordenadas no son validas
    return frame[y1:y2, x1:x2]  # devuelve solo la region de interes

# funcion para unir el frame procesado con el fondo negro
# se usa para conservar solo la parte procesada del frame

def union(frame, frame_recortado):
    x1, y1 = 280, 400  # coordenadas de la esquina superior izquierda
    x2, y2 = 1280, 720  # coordenadas de la esquina inferior derecha
    frame_negro = np.zeros_like(frame)  # creamos un frame negro del mismo tamaño
    frame_negro[y1:y2, x1:x2] = frame_recortado  # colocamos el frame procesado en la region correspondiente
    return frame_negro

# funcion para aplicar filtro gamma
# sirve para ajustar el brillo de la imagen

def filtro_gamma(frame):
     # Filtro Gamma
    gamma = 2
    image_RGB_gamma = np.array(255*(frame / 255)**gamma, dtype="uint8")
    return image_RGB_gamma



# funcion para aplicar suavizado gaussiano 5x5
# ayuda a reducir el ruido en la imagen

def suavizar(frame):
    # definir kernel gaussiano 5x5 para el filtro
    kernel_gaussian_5x5 = np.array([[1, 4, 7, 4, 1],
                                    [4, 16, 26, 16, 4],
                                    [7, 26, 41, 26, 7],
                                    [4, 16, 26, 16, 4],
                                    [1, 4, 7, 4, 1]]) * (1/273)
    
    # aplicar filtro gaussiano 5x5 para suavizar la imagen
    return cv2.filter2D(frame, -1, kernel_gaussian_5x5)

# funcion para detectar lineas amarillas y blancas en la imagen
# se usa para identificar carriles en la carretera

def detectar_lineas(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # convertimos la imagen a espacio de color hsv
    
    # definir rangos de color para amarillo y blanco
    lower_yellow = np.array([15, 150, 150]) # definimos el rango de color amarillo bajo en hsv
    upper_yellow = np.array([30, 255, 255]) # definimos el rango de color amarillo alto en hsv
    lower_white = np.array([0, 0, 200]) # definimos el rango de color blanco bajo en hsv
    upper_white = np.array([180, 50, 255]) # definimos el rango de color blanco alto en hsv
    
    # crear mascaras para detectar los colores
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow) # crear mascara para el color amarillo
    mask_white = cv2.inRange(hsv, lower_white, upper_white) # crear mascara para el color blanco
    
    combined_mask = cv2.bitwise_or(mask_yellow, mask_white)  # combinar ambas mascaras
    return combined_mask

# funcion para aplicar fondo negro a todo menos las lineas detectadas

def aplicar_fondo_negro(frame, mask_lineas):
    resultado = cv2.bitwise_and(frame, frame, mask=mask_lineas)  # aplicar la mascara al frame original
    fondo_negro = np.zeros_like(frame)  # crear un frame negro del mismo tamaño
    frame_con_lineas = cv2.add(fondo_negro, resultado)  # agregar las lineas al fondo negro
    return frame_con_lineas

# funcion para procesar el frame completo
# aplica suavizado, filtro gamma y deteccion de lineas

def procesar_frame(frame):
    frame_suavizado = suavizar(frame)  # aplicar suavizado gaussiano
    frame_procesado = filtro_gamma(frame_suavizado)  # aplicar filtro gamma
    mask_lineas = detectar_lineas(frame_procesado)  # detectar lineas amarillas y blancas
    frame_con_lineas = aplicar_fondo_negro(frame_procesado, mask_lineas)  # aplicar fondo negro
    return frame_con_lineas

# funcion para cargar el video y procesar los frames
# lee el video, procesa los frames y los guarda

def cargar_video(ruta_video, ruta_salida):
    cap = cv2.VideoCapture(ruta_video)  # abrir el video
    if not cap.isOpened():
        print("error: no se pudo abrir el video.")
        return
    
    # obtener propiedades del video

    # obtener fps del video original
    fps = cap.get(cv2.CAP_PROP_FPS)
    # obtener ancho y alto del video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # configurar videowriter para guardar el video procesado
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(ruta_salida, fourcc, fps, (width, height))
    
    try:
        while True:
            ret, frame = cap.read()  # leer un frame del video
            if not ret:
                break  # salir si no hay mas frames
            
            # procesar el frame
            
            # recortar la region de interes
            frame_recortado = recorte(frame)
            # procesar el frame rec
            frame_procesado = procesar_frame(frame_recortado)
            # unir el frame procesado
            frame_unido = union(frame, frame_procesado)
            
            # guardar el frame procesado en el video de salida
            out.write(frame_unido)
            
            # mostrar el frame procesado en una ventana
            cv2.imshow('frame procesado', frame_unido)
            
            # salir con tecla 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print("procesamiento finalizado.")

# funcion principal
if __name__ == "__main__":
    ruta_video = 'lineas.mp4'  # ruta del video de entrada
    ruta_salida = 'lineas_procesado.mp4'  # ruta para exportar video procesado
    
    # llamar a la funcion para procesar el video
    cargar_video(ruta_video, ruta_salida)