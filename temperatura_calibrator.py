# temperatura_calibrator.py
import cv2
import numpy as np
import os
import json
import thermal_utils # Importar nuestro módulo de utilidades térmicas

# --- Configuración de la Cámara ---
CAMERA_INDEX = 2 # Asegúrate de que este índice sea el correcto para tu cámara térmica
CAP = None

# --- Puntos de Calibración ---
# Coordenadas de los 4 puntos en la imagen de 192x156 (aproximadas, ajustar si es necesario)
# Centro: IR_IMAGE_WIDTH // 2, IR_IMAGE_HEIGHT // 2
# Triángulo equilátero alrededor del centro. Distancia de 20 píxeles del centro.
# (Recuerda que la imagen de ejemplo es 192x156, la cámara es 192x256. Adaptar.)
# Si la imagen IR es de 192 de alto y 256 de ancho:
CENTER_X = thermal_utils.IR_IMAGE_WIDTH // 2
CENTER_Y = thermal_utils.IR_IMAGE_HEIGHT // 2
RADIUS = 40 # Distancia de los puntos periféricos al centro, ajusta según sea necesario para tu setup

# Coordenadas de los 4 puntos:
# 1. Punto central
# 2. Punto superior
# 3. Punto inferior-izquierdo
# 4. Punto inferior-derecho

# Usando trigonometría para un triángulo equilátero
# Ángulos: 90 grados (arriba), 210 grados (abajo-izquierda), 330 grados (abajo-derecha)
# Estos son los ángulos si el punto de arriba es 0 grados (eje Y negativo)
# O si es más intuitivo, 90 grados (arriba), 90+120=210, 90+240=330

CALIBRATION_POINTS_PIXEL_COORDS = np.array([
    [CENTER_X, CENTER_Y], # Centro
    [CENTER_X, int(CENTER_Y - RADIUS)], # Arriba
    [int(CENTER_X - RADIUS * np.cos(np.deg2rad(30))), int(CENTER_Y + RADIUS * np.sin(np.deg2rad(30)))], # Abajo-izquierda
    [int(CENTER_X + RADIUS * np.cos(np.deg2rad(30))), int(CENTER_Y + RADIUS * np.sin(np.deg2rad(30)))]  # Abajo-derecha
], dtype=int)


def init_camera():
    global CAP
    if CAP is None or not CAP.isOpened():
        print(f"Intentando abrir la cámara con índice {CAMERA_INDEX}...")
        CAP = cv2.VideoCapture(CAMERA_INDEX)
        if not CAP.isOpened():
            print("Error: No se pudo abrir la cámara. Asegúrate de que el índice sea correcto.")
            return False
        else:
            # Establecer resolución si es necesario (algunas cámaras lo permiten)
            # CAP.set(cv2.CAP_PROP_FRAME_WIDTH, thermal_utils.IR_IMAGE_WIDTH)
            # CAP.set(cv2.CAP_PROP_FRAME_HEIGHT, thermal_utils.IR_IMAGE_HEIGHT)
            print("Cámara abierta correctamente.")
            return True
    return True

def capture_and_display():
    if not init_camera():
        print("No se pudo iniciar la cámara. Saliendo.")
        return

    measured_temperatures = [0.0] * 4 # Para almacenar las temperaturas ingresadas por el usuario

    print("\n--- Modo de Calibración de Temperatura ---")
    print("Coloca tus termopares en los 4 puntos marcados en la pantalla.")
    print("Presiona 's' para guardar las temperaturas y calibrar.")
    print("Presiona 'q' para salir sin guardar.")

    while True:
        ret, frame = CAP.read()
        if not ret:
            print("Error al leer el fotograma. Reconectando...")
            CAP.release()
            init_camera()
            continue

        # Recortar la región de interés si la cámara captura un frame más grande
        ir_gray_image = frame[0:thermal_utils.IR_IMAGE_HEIGHT, 0:thermal_utils.IR_IMAGE_WIDTH]

        # Convertir a imagen de color para dibujar los puntos
        display_frame = cv2.cvtColor(ir_gray_image, cv2.COLOR_GRAY2BGR) if ir_gray_image.ndim == 2 else ir_gray_image.copy()

        # Dibujar los puntos de calibración
        for i, (x, y) in enumerate(CALIBRATION_POINTS_PIXEL_COORDS):
            cv2.circle(display_frame, (x, y), 5, (0, 255, 0), -1) # Círculo verde
            cv2.putText(display_frame, f"P{i+1}", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Mostrar el frame
        cv2.imshow('Calibracion de Camara Termica', display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            print("\nIngresa las temperaturas medidas por tus termopares para cada punto:")
            for i in range(4):
                while True:
                    try:
                        temp_str = input(f"Temperatura para el Punto {i+1} (C): ")
                        measured_temperatures[i] = float(temp_str)
                        break
                    except ValueError:
                        print("Entrada inválida. Por favor, ingresa un número.")

            # Guardar los datos de calibración
            calibration_data = {
                "pixel_coords": CALIBRATION_POINTS_PIXEL_COORDS.tolist(),
                "measured_temps": measured_temperatures
            }
            with open(thermal_utils.CALIBRATION_FILE, 'w') as f:
                json.dump(calibration_data, f, indent=4)
            print(f"Datos de calibración guardados en {thermal_utils.CALIBRATION_FILE}")
            
            # Recargar los datos de calibración en thermal_utils para que estén disponibles
            thermal_utils.load_calibration_data()
            print("Calibración completada y guardada.")
            break
        elif key == ord('q'):
            print("Saliendo sin guardar la calibración.")
            break

    CAP.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    capture_and_display()