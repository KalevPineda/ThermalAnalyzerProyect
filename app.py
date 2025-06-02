# app.py
from flask import Flask, Response, render_template, send_from_directory, jsonify
import cv2
import numpy as np
import time
import threading
import os
import glob
import thermal_utils

# --- Configuración del Servidor Flask ---
app = Flask(__name__)

# --- Configuración de la Cámara ---
CAMERA_INDEX = 2
CAP = None

# --- Configuración de Guardado Automático ---
SAVE_INTERVAL_SECONDS = 60
last_save_time = 0

# --- FACTOR DE ESCALADO PARA EL STREAMING ---
# 2.0 significa el doble de tamaño
STREAM_SCALE_FACTOR = 2.0

def init_camera():
    global CAP
    if CAP is None or not CAP.isOpened():
        print(f"Intentando abrir la cámara con índice {CAMERA_INDEX}...")
        CAP = cv2.VideoCapture(CAMERA_INDEX)
        if not CAP.isOpened():
            print("Error: No se pudo abrir la cámara. Asegúrate de que el índice sea correcto.")
            return False
        else:
            print("Cámara abierta correctamente.")
            return True
    return True

def generate_frames():
    global last_save_time
    if not init_camera():
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + b'' + b'\r\n')
        return

    while True:
        ret, frame = CAP.read()
        if not ret:
            print("Error al leer el fotograma del stream.")
            CAP.release()
            init_camera()
            time.sleep(1)
            continue

        # --- Procesamiento del fotograma ---
        ir_gray_image = frame[0:thermal_utils.IR_IMAGE_HEIGHT, 0:thermal_utils.IR_IMAGE_WIDTH]
        temperature_matrix = thermal_utils.calculate_temperature_matrix(ir_gray_image)
        min_temp_detected = np.min(temperature_matrix)
        max_temp_detected = np.max(temperature_matrix)

        colored_display_frame = thermal_utils.apply_rainbow_colormap(
            temperature_matrix,
            vmin=thermal_utils.MIN_TEMP_C,
            vmax=thermal_utils.MAX_TEMP_C
        )

        # --- Agregar texto de temperatura min/max al frame ---
        text_min = f"Min Temp: {min_temp_detected:.1f} C"
        text_max = f"Max Temp: {max_temp_detected:.1f} C"
        cv2.putText(colored_display_frame, text_min, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(colored_display_frame, text_max, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # --- REDIMENSIONAR EL FRAME PARA EL STREAMING ---
        # Calcula las nuevas dimensiones
        new_width = int(colored_display_frame.shape[1] * STREAM_SCALE_FACTOR)
        new_height = int(colored_display_frame.shape[0] * STREAM_SCALE_FACTOR)
        
        # Aplica el redimensionamiento
        # cv2.INTER_LINEAR es una buena opción para escalado hacia arriba
        resized_frame = cv2.resize(colored_display_frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        # --- Guardar datos periódicamente (usar el frame original o el procesado, NO el redimensionado para streaming) ---
        current_time = time.time()
        if current_time - last_save_time > SAVE_INTERVAL_SECONDS:
            thermal_utils.save_data_as_dataset(temperature_matrix, ir_gray_image)
            # Pasamos las temperaturas detectadas para el nombre del archivo de imagen
            thermal_utils.save_img(ir_gray_image, min_temp_detected, max_temp_detected)
            last_save_time = current_time

        # Codificar el frame REDIMENSIONADO para streaming
        ret, buffer = cv2.imencode('.jpg', resized_frame) # Usar 'resized_frame' aquí
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        time.sleep(0.03)

# El resto del código de app.py (endpoints, etc.) permanece igual
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/download_data')
def download_data():
    files_list = []
    for f in sorted(os.listdir(thermal_utils.DATASETS_FOLDER), reverse=True):
        if f.endswith('.h5'):
            files_list.append({'name': f, 'type': 'dataset', 'path': f'/datasets/{f}'})
    for f in sorted(os.listdir(thermal_utils.THERMAL_IMAGES_FOLDER), reverse=True):
        if f.endswith('.jpeg') or f.endswith('.jpg'):
            files_list.append({'name': f, 'type': 'image', 'path': f'/images/{f}'})
    return render_template('download.html', files=files_list)

@app.route('/datasets/<filename>')
def serve_dataset(filename):
    return send_from_directory(thermal_utils.DATASETS_FOLDER, filename, as_attachment=True)

@app.route('/images/<filename>')
def serve_image(filename):
    return send_from_directory(thermal_utils.THERMAL_IMAGES_FOLDER, filename)


@app.route('/api/latest_data')
def api_latest_data():
    temperature_matrix, ir_gray_image_float, error = thermal_utils.load_latest_dataset()
    if error:
        return jsonify({"status": "error", "message": error}), 500
    
    return jsonify({
        "status": "success",
        "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "temperature_matrix_shape": temperature_matrix.shape,
        "temperature_matrix_sample": temperature_matrix[0,0],
        "min_temp_detected": np.min(temperature_matrix),
        "max_temp_detected": np.max(temperature_matrix),
        "ir_gray_image_float_shape": ir_gray_image_float.shape,
        "ir_gray_image_float_sample": ir_gray_image_float[0,0]
    })

@app.route('/api/latest_rainbow_image')
def api_latest_rainbow_image():
    image_path = thermal_utils.load_latest_rainbow_image_path()
    if image_path:
        return send_from_directory(os.path.dirname(image_path), os.path.basename(image_path))
    return jsonify({"status": "error", "message": "No hay imágenes arcoíris disponibles."}), 404

@app.teardown_appcontext
def shutdown_camera(exception=None):
    global CAP
    if CAP is not None and CAP.isOpened():
        CAP.release()
        print("Cámara liberada al apagar la aplicación.")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)