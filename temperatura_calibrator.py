# temperatura_calibrator.py
import cv2
import numpy as np
import thermal_utils # Importar nuestro módulo de utilidades térmicas

# --- Configuración de la Cámara ---
# ASEGÚRATE DE QUE ESTE ÍNDICE SEA EL CORRECTO PARA TU CÁMARA TÉRMICA
CAMERA_INDEX = 2
CAP = None

def init_camera():
    """Inicializa y abre la conexión con la cámara."""
    global CAP
    if CAP is None or not CAP.isOpened():
        print(f"Intentando abrir la cámara con índice {CAMERA_INDEX}...")
        CAP = cv2.VideoCapture(CAMERA_INDEX)
        if not CAP.isOpened():
            print(f"Error: No se pudo abrir la cámara con índice {CAMERA_INDEX}.")
            print("Verifica que la cámara esté conectada y que el índice sea el correcto.")
            return False
        print("Cámara abierta correctamente.")
    return True

def draw_calibration_points(frame):
    """Dibuja los puntos de calibración sobre un frame."""
    display_frame = frame.copy()
    for i, (x, y) in enumerate(thermal_utils.CALIBRATION_POINTS_PIXEL_COORDS):
        cv2.circle(display_frame, (x, y), 5, (0, 255, 0), -1) # Círculo verde relleno
        cv2.putText(display_frame, f"P{i+1}", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return display_frame

def get_1_channel_gray(frame_raw):
    """Asegura que la imagen de entrada sea de 1 canal en escala de grises."""
    if frame_raw.ndim == 3 and frame_raw.shape[2] == 3:
        return cv2.cvtColor(frame_raw, cv2.COLOR_BGR2GRAY)
    return frame_raw

def get_3_channel_bgr(frame_raw):
    """Asegura que la imagen de entrada sea de 3 canales BGR para dibujar."""
    if frame_raw.ndim == 1 or (frame_raw.ndim == 2):
        return cv2.cvtColor(frame_raw, cv2.COLOR_GRAY2BGR)
    return frame_raw

def capture_calibration_samples():
    """
    Función principal que guía al usuario para capturar una o más muestras de calibración.
    """
    if not init_camera():
        return

    collected_samples = []

    print("\n--- Asistente de Calibración Inicial ---")
    
    while True:
        print("\n--- Preparando para Capturar Nueva Muestra ---")
        print("1. Coloca tus termopares en las posiciones marcadas en verde.")
        print("2. Enfoca la ventana de video y presiona 's' para capturar y congelar la imagen.")
        print("3. Presiona 'q' para finalizar el asistente.")

        captured_ir_gray_image = None
        # Bucle de video en vivo
        while True:
            ret, frame = CAP.read()
            if not ret:
                print("Error al leer fotograma.")
                continue

            ir_image_raw = frame[0:thermal_utils.IR_IMAGE_HEIGHT, 0:thermal_utils.IR_IMAGE_WIDTH]
            display_frame_bgr = get_3_channel_bgr(ir_image_raw)
            display_frame_with_points = draw_calibration_points(display_frame_bgr)

            cv2.imshow('Video en Vivo - Presiona "s" para capturar', display_frame_with_points)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                # Capturamos la imagen en escala de grises de 1 canal para la lógica
                captured_ir_gray_image = get_1_channel_gray(ir_image_raw)
                print("\n¡Imagen congelada! Ahora introduce las temperaturas en la consola.")
                break
            elif key == ord('q'):
                break
        
        # Si se presionó 'q', salir del bucle principal
        if captured_ir_gray_image is None:
            break

        # Convertimos la imagen gris capturada a BGR para mostrarla congelada
        frozen_display_frame = get_3_channel_bgr(captured_ir_gray_image)
        frozen_display_frame_with_points = draw_calibration_points(frozen_display_frame)
        cv2.imshow('Video en Vivo - Presiona "s" para capturar', frozen_display_frame_with_points)
        cv2.waitKey(1)

        # Pedir los datos al usuario mientras ve la imagen congelada
        measured_temperatures = []
        for i in range(len(thermal_utils.CALIBRATION_POINTS_PIXEL_COORDS)):
            while True:
                try:
                    # Se mantiene mostrando la imagen congelada mientras espera el input
                    cv2.imshow('Video en Vivo - Presiona "s" para capturar', frozen_display_frame_with_points)
                    cv2.waitKey(1)
                    temp_str = input(f"  -> Temperatura para Punto {i+1} (C): ")
                    measured_temperatures.append(float(temp_str))
                    break
                except ValueError:
                    print("   Entrada inválida. Por favor, introduce un número (ej: 25.4).")

        pixel_values_at_points = np.array([captured_ir_gray_image[y, x] for x, y in thermal_utils.CALIBRATION_POINTS_PIXEL_COORDS])

        sample_data = {
            "coords": thermal_utils.CALIBRATION_POINTS_PIXEL_COORDS,
            "values": pixel_values_at_points,
            "temps": measured_temperatures
        }
        collected_samples.append(sample_data)
        print("\n¡Muestra añadida con éxito! Volviendo al video en vivo...")

    # Una vez que el usuario ha terminado (presionando 'q' en el bucle de video)
    cv2.destroyAllWindows()
    
    if collected_samples:
        print(f"\nSe han recolectado {len(collected_samples)} muestra(s).")
        print("Guardando los nuevos datos de calibración...")
        thermal_utils.clear_calibration_data()
        for sample in collected_samples:
            thermal_utils.add_calibration_sample(
                sample["coords"],
                sample["values"],
                sample["temps"]
            )
        print("\n¡Calibración inicial completada y guardada en 'calibration_data.json'!")
        print("Ahora puedes ejecutar 'app.py' para iniciar el monitor web.")
    else:
        print("\nNo se capturaron muestras. Saliendo sin cambios.")

    CAP.release()

if __name__ == '__main__':
    capture_calibration_samples()