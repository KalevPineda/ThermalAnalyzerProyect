# app.py
from flask import Flask, Response, render_template, send_from_directory, jsonify, request
import cv2
import numpy as np
import time
import os
import thermal_utils
from threading import Thread, Lock

# --- Configuración del Servidor Flask ---
app = Flask(__name__)

# --- Configuración de la Cámara ---
CAMERA_INDEX = 0
# --- Variables globales para el threading de la cámara ---
camera_thread = None
output_frame = None
lock = Lock()

# --- Estado de la UI ---
SHOW_CALIBRATION_POINTS = False

# --- Configuración de Guardado Automático ---
SAVE_INTERVAL_SECONDS = 60
last_save_time = 0

# --- FACTOR DE ESCALADO PARA EL STREAMING ---
STREAM_SCALE_FACTOR = 2.5


# --- HILO DEDICADO A LA CAPTURA DE LA CÁMARA ---
def camera_capture_thread():
    global output_frame
    
    print(f"Intentando abrir la cámara con índice {CAMERA_INDEX}...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara en el hilo.")
        return
    print("Cámara abierta correctamente en el hilo.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al leer el fotograma en el hilo. Saliendo del hilo de captura.")
            break
        
        # Usamos un Lock para asegurar que el fotograma se escriba de forma segura
        with lock:
            output_frame = frame.copy()
    
    cap.release()
    print("Hilo de cámara terminado y cámara liberada.")

# --- FUNCIÓN PRINCIPAL DE GENERACIÓN DE FRAMES (AHORA MÁS LIGERA) ---
def generate_frames():
    global last_save_time, output_frame

    while True:
        # Esperar hasta que el primer fotograma esté disponible
        if output_frame is None:
            time.sleep(0.1)
            continue

        with lock:
            # Hacemos una copia para procesar y evitar problemas de concurrencia
            frame = output_frame.copy()

        ir_gray_image = frame[0:thermal_utils.IR_IMAGE_HEIGHT, 0:thermal_utils.IR_IMAGE_WIDTH]
        temperature_matrix = thermal_utils.calculate_temperature_matrix(ir_gray_image)
        min_temp_detected = np.min(temperature_matrix)
        max_temp_detected = np.max(temperature_matrix)
        
        colored_display_frame = thermal_utils.apply_rainbow_colormap(
            temperature_matrix, vmin=min_temp_detected, vmax=max_temp_detected
        )

        text_min = f"Min: {min_temp_detected:.1f}C"
        text_max = f"Max: {max_temp_detected:.1f}C"
        cv2.putText(colored_display_frame, text_min, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(colored_display_frame, text_max, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if SHOW_CALIBRATION_POINTS:
            for i, (x, y) in enumerate(thermal_utils.CALIBRATION_POINTS_PIXEL_COORDS):
                cv2.circle(colored_display_frame, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(colored_display_frame, f"P{i+1}", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        new_width = int(colored_display_frame.shape[1] * STREAM_SCALE_FACTOR)
        new_height = int(colored_display_frame.shape[0] * STREAM_SCALE_FACTOR)
        # Usamos INTER_NEAREST que es más rápido y adecuado para imágenes térmicas
        resized_frame = cv2.resize(colored_display_frame, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

        current_time = time.time()
        if current_time - last_save_time > SAVE_INTERVAL_SECONDS:
            thermal_utils.save_data_as_dataset(temperature_matrix, ir_gray_image)
            thermal_utils.save_img(ir_gray_image, min_temp_detected, max_temp_detected)
            last_save_time = current_time

        ret, buffer = cv2.imencode('.jpg', resized_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03) # Pequeña pausa para no sobrecargar la CPU

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- Rutas de descarga y API (SIN CAMBIOS) ---
@app.route('/download_data')
def download_data():
    files_list = []
    try:
        for f in sorted(os.listdir(thermal_utils.DATASETS_FOLDER), key=lambda x: os.path.getmtime(os.path.join(thermal_utils.DATASETS_FOLDER, x)), reverse=True):
            if f.endswith('.h5'):
                files_list.append({'name': f, 'type': 'Dataset HDF5', 'path': f'/datasets/{f}'})
    except FileNotFoundError:
        pass
    try:
        for f in sorted(os.listdir(thermal_utils.THERMAL_IMAGES_FOLDER), key=lambda x: os.path.getmtime(os.path.join(thermal_utils.THERMAL_IMAGES_FOLDER, x)), reverse=True):
            if f.endswith('.jpeg') or f.endswith('.jpg'):
                file_type = 'Imagen Gris' if 'gray' in f else 'Imagen Arcoíris'
                files_list.append({'name': f, 'type': file_type, 'path': f'/images/{f}'})
    except FileNotFoundError:
        pass
    return render_template('download.html', files=files_list)

@app.route('/datasets/<filename>')
def serve_dataset(filename):
    return send_from_directory(thermal_utils.DATASETS_FOLDER, filename, as_attachment=True)

@app.route('/images/<filename>')
def serve_image(filename):
    return send_from_directory(thermal_utils.THERMAL_IMAGES_FOLDER, filename)

@app.route('/api/calibration_status', methods=['GET'])
def get_calibration_status():
    if thermal_utils.CALIBRATION_MODEL is not None:
        status = f"Polinomial Grado {thermal_utils.CALIBRATION_DEGREE} ({len(thermal_utils.ALL_CALIBRATION_SAMPLES)} muestras)"
    else:
        status = "No Calibrado (Usando escala lineal)"
    return jsonify({"status": "success", "message": status})

@app.route('/api/toggle_calibration_markers', methods=['POST'])
def toggle_markers():
    global SHOW_CALIBRATION_POINTS
    SHOW_CALIBRATION_POINTS = not SHOW_CALIBRATION_POINTS
    return jsonify({"status": "success", "showing_markers": SHOW_CALIBRATION_POINTS})

@app.route('/api/add_calibration_sample', methods=['POST'])
def add_sample_from_web():
    global output_frame
    data = request.get_json()
    if not data or 'temps' not in data or len(data['temps']) != 4:
        return jsonify({"status": "error", "message": "Datos inválidos. Se esperan 4 temperaturas."}), 400
    
    measured_temps = [float(t) for t in data['temps']]

    if output_frame is None:
        return jsonify({"status": "error", "message": "No se pudo acceder a la cámara."}), 500

    with lock:
        frame = output_frame.copy()
    
    ir_gray_image = frame[0:thermal_utils.IR_IMAGE_HEIGHT, 0:thermal_utils.IR_IMAGE_WIDTH]
    
    if ir_gray_image.ndim == 3:
        ir_gray_1_channel = ir_gray_image[:, :, 0]
    else:
        ir_gray_1_channel = ir_gray_image
        
    pixel_values = np.array([ir_gray_1_channel[y, x] for x, y in thermal_utils.CALIBRATION_POINTS_PIXEL_COORDS])

    success = thermal_utils.add_calibration_sample(
        thermal_utils.CALIBRATION_POINTS_PIXEL_COORDS,
        pixel_values,
        measured_temps
    )
    if success:
        return jsonify({"status": "success", "message": "Muestra de calibración añadida correctamente."})
    else:
        return jsonify({"status": "error", "message": "No se pudo guardar la muestra de calibración."}), 500

@app.route('/api/clear_calibration', methods=['POST'])
def clear_calibration_from_web():
    if thermal_utils.clear_calibration_data():
        return jsonify({"status": "success", "message": "Datos de calibración eliminados."})
    else:
        return jsonify({"status": "error", "message": "Error al eliminar los datos de calibración."}), 500

# --- SE ELIMINÓ LA FUNCIÓN @app.teardown_appcontext ---

if __name__ == '__main__':
    # --- Iniciar el hilo de la cámara ---
    camera_thread = Thread(target=camera_capture_thread)
    camera_thread.daemon = True
    camera_thread.start()
    
    # Iniciar el servidor Flask
    app.run(host='0.0.0.0', port=5000, debug=False)
