
# app.py
from flask import Flask, Response, render_template, send_from_directory, jsonify
import cv2
import numpy as np
import time
import threading
import os
import glob # Para listar archivos fácilmente
import thermal_utils # Importamos nuestras funciones

# --- Configuración del Servidor Flask ---
app = Flask(__name__)

# --- Configuración de la Cámara ---
CAMERA_INDEX = 2
CAP = None # Objeto VideoCapture global para ser accedido por múltiples funciones

# --- Configuración de Guardado Automático ---
SAVE_INTERVAL_SECONDS = 60 # Guardar datos cada 60 segundos
last_save_time = 0

# --- Inicialización de la Cámara ---
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
            # Intentar establecer propiedades si es necesario, aunque no es crítico para este caso
            # CAP.set(cv2.CAP_PROP_FRAME_WIDTH, thermal_utils.IR_IMAGE_WIDTH)
            # CAP.set(cv2.CAP_PROP_FRAME_HEIGHT, thermal_utils.IR_IMAGE_HEIGHT * 2) # Altura total
            return True
    return True

# --- Función para generar frames para el streaming ---
def generate_frames():
    global last_save_time
    # Asegurarse de que la cámara esté inicializada
    if not init_camera():
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + b'' + b'\r\n')
        return

    while True:
        ret, frame = CAP.read()
        if not ret:
            print("Error al leer el fotograma del stream.")
            CAP.release()
            init_camera() # Intentar reiniciar la cámara
            time.sleep(1) # Esperar un poco antes de reintentar
            continue

        # --- Procesamiento del fotograma ---
        # Recortar la parte IR gris
        ir_gray_image = frame[0:thermal_utils.IR_IMAGE_HEIGHT, 0:thermal_utils.IR_IMAGE_WIDTH]

        # Calcular matriz de temperatura
        temperature_matrix = thermal_utils.calculate_temperature_matrix(ir_gray_image)
        min_temp_detected = np.min(temperature_matrix)
        max_temp_detected = np.max(temperature_matrix)

        # Aplicar filtro arcoíris para el streaming
        colored_display_frame = thermal_utils.apply_rainbow_colormap(
            temperature_matrix,
            vmin=thermal_utils.MIN_TEMP_C, # Usar el rango completo de la cámara para el colormap
            vmax=thermal_utils.MAX_TEMP_C
        )

        # --- Agregar texto de temperatura min/max al frame ---
        text_min = f"Min Temp: {min_temp_detected:.1f} C"
        text_max = f"Max Temp: {max_temp_detected:.1f} C"
        cv2.putText(colored_display_frame, text_min, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(colored_display_frame, text_max, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


        # --- Guardar datos periódicamente ---
        current_time = time.time()
        if current_time - last_save_time > SAVE_INTERVAL_SECONDS:
            thermal_utils.save_data_as_dataset(temperature_matrix, ir_gray_image)
            thermal_utils.save_img(ir_gray_image, min_temp_detected, max_temp_detected)
            last_save_time = current_time

        # Codificar el frame para streaming
        ret, buffer = cv2.imencode('.jpg', colored_display_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        # Pequeña pausa para no saturar el CPU
        time.sleep(0.03)

# --- Endpoint 1: Streaming de video de la cámara ---
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- Página principal para visualizar el streaming ---
@app.route('/')
def index():
    return render_template('index.html')

# --- Endpoint 2: Descarga de datos ---
@app.route('/download_data')
def download_data():
    files_list = []
    # Listar archivos de datasets
    for f in sorted(os.listdir(thermal_utils.DATASETS_FOLDER), reverse=True):
        if f.endswith('.h5'):
            files_list.append({'name': f, 'type': 'dataset', 'path': f'/datasets/{f}'})
    # Listar archivos de imágenes
    for f in sorted(os.listdir(thermal_utils.THERMAL_IMAGES_FOLDER), reverse=True):
        if f.endswith('.jpeg') or f.endswith('.jpg'):
            files_list.append({'name': f, 'type': 'image', 'path': f'/images/{f}'})
    return render_template('download.html', files=files_list)

# Endpoints para servir archivos estáticos (descargables)
@app.route('/datasets/<filename>')
def serve_dataset(filename):
    return send_from_directory(thermal_utils.DATASETS_FOLDER, filename, as_attachment=True)

@app.route('/images/<filename>')
def serve_image(filename):
    return send_from_directory(thermal_utils.THERMAL_IMAGES_FOLDER, filename)


# Endpoint para cargar el último dataset (API)
@app.route('/api/latest_data')
def api_latest_data():
    temperature_matrix, ir_gray_image_float, error = thermal_utils.load_latest_dataset()
    if error:
        return jsonify({"status": "error", "message": error}), 500
    
    # Para JSON, numpy arrays deben convertirse a listas
    # OJO: arrays grandes pueden hacer el JSON muy pesado. Considera si necesitas todo el array aquí.
    return jsonify({
        "status": "success",
        "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "temperature_matrix_shape": temperature_matrix.shape,
        "temperature_matrix_sample": temperature_matrix[0,0], # Ejemplo de un solo valor
        "min_temp_detected": np.min(temperature_matrix),
        "max_temp_detected": np.max(temperature_matrix),
        "ir_gray_image_float_shape": ir_gray_image_float.shape,
        "ir_gray_image_float_sample": ir_gray_image_float[0,0] # Ejemplo de un solo valor
        # Si realmente necesitas el array completo en JSON (puede ser lento y grande):
        # "temperature_matrix": temperature_matrix.tolist(),
        # "ir_gray_image_float": ir_gray_image_float.tolist()
    })

# Endpoint para obtener la última imagen arcoíris (API)
@app.route('/api/latest_rainbow_image')
def api_latest_rainbow_image():
    image_path = thermal_utils.load_latest_rainbow_image_path()
    if image_path:
        return send_from_directory(os.path.dirname(image_path), os.path.basename(image_path))
    return jsonify({"status": "error", "message": "No hay imágenes arcoíris disponibles."}), 404


# --- Función para liberar la cámara al apagar el servidor ---
@app.teardown_appcontext
def shutdown_camera(exception=None):
    global CAP
    if CAP is not None and CAP.isOpened():
        CAP.release()
        print("Cámara liberada al apagar la aplicación.")

# --- Ejecutar el servidor ---
if __name__ == '__main__':
    # La cámara se inicializará la primera vez que se solicite un frame
    # (o podrías llamarlo aquí si quieres asegurar que esté lista al inicio)
    # init_camera()
    app.run(host='0.0.0.0', port=5000, debug=False) # host='0.0.0.0' para acceso remoto

    
