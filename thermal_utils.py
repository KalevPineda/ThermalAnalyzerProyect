

# thermal_utils.py
import cv2
import numpy as np
import os
import datetime
import h5py # Para guardar datasets
import matplotlib.pyplot as plt

# --- Configuración de la Cámara y Rango de Temperatura (pueden ser parámetros de funciones o globales) ---
# Los definimos aquí para que las funciones los tengan a mano, pero podrían ser pasados como argumentos
IR_IMAGE_HEIGHT = 192
IR_IMAGE_WIDTH = 256
MIN_TEMP_C = -20.0
MAX_TEMP_C = 200.0

# --- Rutas de Carpetas ---
DATASETS_FOLDER = 'DataSets'
THERMAL_IMAGES_FOLDER = 'ThermalImages'

# Asegurarse de que las carpetas existan
os.makedirs(DATASETS_FOLDER, exist_ok=True)
os.makedirs(THERMAL_IMAGES_FOLDER, exist_ok=True)

# --- Función para Calcular la Matriz de Temperatura ---
def calculate_temperature_matrix(ir_gray_image_8bit):
    """
    Calcula la matriz de temperatura a partir de una imagen IR en escala de grises (uint8).
    Asume que la imagen es (H, W, 3) con R=G=B o (H, W).
    """
    if ir_gray_image_8bit.ndim == 3:
        ir_gray_1_channel = ir_gray_image_8bit[:, :, 0]
    else:
        ir_gray_1_channel = ir_gray_image_8bit

    pixel_values_float = ir_gray_1_channel.astype(np.float32)
    temperature_matrix = MIN_TEMP_C + (pixel_values_float / 255.0) * (MAX_TEMP_C - MIN_TEMP_C)
    return temperature_matrix

# --- Función para aplicar paleta de colores arcoíris ---
def apply_rainbow_colormap(temperature_matrix, vmin=MIN_TEMP_C, vmax=MAX_TEMP_C):
    """
    Aplica una paleta de colores arcoíris (JET o similar) a la matriz de temperatura.
    Retorna una imagen BGR uint8.
    """
    # Normalizar la matriz de temperatura al rango 0-255
    normalized_temp = ((temperature_matrix - vmin) / (vmax - vmin) * 255).astype(np.uint8)

    # Aplicar un colormap común en cámaras térmicas (JET en matplotlib es similar a arcoíris)
    # Matplotlib genera RGBA, necesitamos convertir a BGR para OpenCV
    cmap = plt.get_cmap('jet') # Puedes probar 'viridis', 'inferno', 'magma', 'plasma'
    colored_image_rgba = cmap(normalized_temp)
    colored_image_bgr = (colored_image_rgba[:, :, :3] * 255).astype(np.uint8) # Quitar alfa y convertir a BGR
    colored_image_bgr = cv2.cvtColor(colored_image_bgr, cv2.COLOR_RGB2BGR) # Convertir de RGB (matplotlib) a BGR (OpenCV)

    return colored_image_bgr

# --- Función para guardar dataset ---
def save_data_as_dataset(temperature_matrix, ir_gray_image):
    """
    Guarda las matrices de temperatura y la imagen IR (convertida a float32)
    en un archivo HDF5 con marca de tiempo.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(DATASETS_FOLDER, f"thermal_data_{timestamp}.h5")

    # Convertir ir_gray_image a float32 si no lo está ya
    # Asumimos que ir_gray_image es (H,W,3) con R=G=B o (H,W)
    if ir_gray_image.dtype != np.float32:
        if ir_gray_image.ndim == 3:
            ir_gray_image_float = ir_gray_image[:, :, 0].astype(np.float32) # Tomar un canal y float32
        else:
            ir_gray_image_float = ir_gray_image.astype(np.float32)
    else:
        ir_gray_image_float = ir_gray_image

    with h5py.File(filename, 'w') as f:
        f.create_dataset('temperature_matrix', data=temperature_matrix)
        f.create_dataset('ir_gray_image_float', data=ir_gray_image_float)
        # Opcional: guardar metadatos
        f.attrs['timestamp'] = timestamp
        f.attrs['min_temp_c'] = MIN_TEMP_C
        f.attrs['max_temp_c'] = MAX_TEMP_C
        f.attrs['ir_image_height'] = IR_IMAGE_HEIGHT
        f.attrs['ir_image_width'] = IR_IMAGE_WIDTH
        print(f"Dataset guardado: {filename}")
    return filename

# --- Función para guardar imágenes ---
def save_img(ir_gray_image, min_temp_detected=None, max_temp_detected=None):
    """
    Guarda la imagen IR en escala de grises y una versión con filtro arcoíris.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- Guardar imagen IR en escala de grises original (uint8) ---
    # Asegurarse de que sea de 1 canal para guardar como JPEG en escala de grises
    if ir_gray_image.ndim == 3:
        gray_output_image = cv2.cvtColor(ir_gray_image, cv2.COLOR_BGR2GRAY)
    else:
        gray_output_image = ir_gray_image # Ya es de 1 canal

    gray_filename = os.path.join(THERMAL_IMAGES_FOLDER, f"ir_gray_{timestamp}.jpeg")
    cv2.imwrite(gray_filename, gray_output_image)
    print(f"Imagen IR gris guardada: {gray_filename}")

    # --- Guardar imagen con filtro arcoíris ---
    # Primero calcular la matriz de temperatura
    temperature_matrix = calculate_temperature_matrix(ir_gray_image)
    
    # Si no se proporcionan, calcular min/max de la matriz de temperatura actual
    if min_temp_detected is None:
        min_temp_detected = np.min(temperature_matrix)
    if max_temp_detected is None:
        max_temp_detected = np.max(temperature_matrix)

    # Aplicar el colormap
    colored_image = apply_rainbow_colormap(temperature_matrix, vmin=MIN_TEMP_C, vmax=MAX_TEMP_C)
    
    # Asegurarse de que los valores de temperatura en el nombre no tengan demasiados decimales
    min_temp_str = f"{min_temp_detected:.1f}".replace('.', '_')
    max_temp_str = f"{max_temp_detected:.1f}".replace('.', '_')

    colored_filename = os.path.join(THERMAL_IMAGES_FOLDER,
                                   f"ir_rainbow_{timestamp}_min{min_temp_str}_max{max_temp_str}.jpeg")
    cv2.imwrite(colored_filename, colored_image)
    print(f"Imagen IR arcoíris guardada: {colored_filename}")

    return gray_filename, colored_filename

# --- Función para leer el último dataset guardado ---
def load_latest_dataset():
    """
    Carga el dataset HDF5 más reciente de la carpeta DataSets.
    """
    list_of_files = os.listdir(DATASETS_FOLDER)
    h5_files = [f for f in list_of_files if f.endswith('.h5')]
    if not h5_files:
        return None, None, "No se encontraron archivos .h5 en la carpeta DataSets."

    # Ordenar por fecha para obtener el más reciente
    h5_files.sort(key=lambda x: os.path.getmtime(os.path.join(DATASETS_FOLDER, x)), reverse=True)
    latest_file = os.path.join(DATASETS_FOLDER, h5_files[0])

    try:
        with h5py.File(latest_file, 'r') as f:
            temperature_matrix = f['temperature_matrix'][()]
            ir_gray_image_float = f['ir_gray_image_float'][()]
        print(f"Dataset cargado: {latest_file}")
        return temperature_matrix, ir_gray_image_float, None
    except Exception as e:
        return None, None, f"Error al cargar el dataset {latest_file}: {e}"

# --- Función para leer la última imagen arcoíris guardada ---
def load_latest_rainbow_image_path():
    """
    Retorna la ruta del archivo de imagen arcoíris más reciente.
    """
    list_of_files = os.listdir(THERMAL_IMAGES_FOLDER)
    rainbow_images = [f for f in list_of_files if f.startswith('ir_rainbow_') and f.endswith('.jpeg')]
    if not rainbow_images:
        return None

    # Ordenar por fecha para obtener el más reciente
    rainbow_images.sort(key=lambda x: os.path.getmtime(os.path.join(THERMAL_IMAGES_FOLDER, x)), reverse=True)
    return os.path.join(THERMAL_IMAGES_FOLDER, rainbow_images[0])

    