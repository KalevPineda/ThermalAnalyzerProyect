# thermal_utils.py
import cv2
import numpy as np
import os
import datetime
import h5py
import matplotlib.pyplot as plt
import json

# --- Configuración General ---
IR_IMAGE_HEIGHT = 192
IR_IMAGE_WIDTH = 256
MIN_TEMP_C = -20.0
MAX_TEMP_C = 200.0

# --- Rutas ---
DATASETS_FOLDER = 'DataSets'
THERMAL_IMAGES_FOLDER = 'ThermalImages'
CALIBRATION_FILE = 'calibration_data.json'

os.makedirs(DATASETS_FOLDER, exist_ok=True)
os.makedirs(THERMAL_IMAGES_FOLDER, exist_ok=True)

# --- Puntos de Calibración Fijos ---
# Coordenadas de los 4 puntos en la imagen de 192x256
CENTER_X = IR_IMAGE_WIDTH // 2
CENTER_Y = IR_IMAGE_HEIGHT // 2
RADIUS = 60 # Aumentado para mayor separación en 192x256
CALIBRATION_POINTS_PIXEL_COORDS = np.array([
    [CENTER_X, CENTER_Y],
    [CENTER_X, int(CENTER_Y - RADIUS)],
    [int(CENTER_X - RADIUS * np.cos(np.deg2rad(30))), int(CENTER_Y + RADIUS * np.sin(np.deg2rad(30)))],
    [int(CENTER_X + RADIUS * np.cos(np.deg2rad(30))), int(CENTER_Y + RADIUS * np.sin(np.deg2rad(30)))]
], dtype=int)

# --- Variables Globales de Calibración NUC ---
ALL_CALIBRATION_SAMPLES = []  # Lista para almacenar todas las muestras de calibración
CALIBRATION_MODEL = None      # Almacenará los coeficientes del modelo polinomial
CALIBRATION_DEGREE = 2        # Grado del polinomio para la calibración NUC

def perform_nuc_calibration():
    """
    Realiza la calibración NUC usando todas las muestras acumuladas.
    Calcula un modelo polinomial y lo almacena en CALIBRATION_MODEL.
    """
    global CALIBRATION_MODEL
    if not ALL_CALIBRATION_SAMPLES:
        print("No hay muestras de calibración para procesar.")
        CALIBRATION_MODEL = None
        return

    all_pixel_values = []
    all_measured_temps = []

    # Recopilar todos los puntos de todas las muestras
    for sample in ALL_CALIBRATION_SAMPLES:
        all_pixel_values.extend(sample['pixel_values'])
        all_measured_temps.extend(sample['measured_temps'])

    # Se necesita al menos (grado + 1) puntos para el ajuste
    if len(all_pixel_values) < CALIBRATION_DEGREE + 1:
        print(f"No hay suficientes puntos ({len(all_pixel_values)}) para una calibración de grado {CALIBRATION_DEGREE}. Se necesitan al menos {CALIBRATION_DEGREE + 1}.")
        CALIBRATION_MODEL = None
        return

    try:
        # np.polyfit devuelve [p_n, p_{n-1}, ..., p_0] para p(x) = p_n*x^n + ... + p_0
        model = np.polyfit(all_pixel_values, all_measured_temps, CALIBRATION_DEGREE)
        CALIBRATION_MODEL = model
        print(f"Calibración NUC (polinomial grado {CALIBRATION_DEGREE}) completada con {len(ALL_CALIBRATION_SAMPLES)} muestra(s) y {len(all_pixel_values)} puntos.")
        print(f"Modelo: {CALIBRATION_MODEL}")
    except Exception as e:
        print(f"Error durante la calibración NUC: {e}")
        CALIBRATION_MODEL = None

def load_calibration_data():
    """
    Carga las muestras de calibración desde el archivo JSON y realiza la calibración.
    """
    global ALL_CALIBRATION_SAMPLES
    if os.path.exists(CALIBRATION_FILE):
        try:
            with open(CALIBRATION_FILE, 'r') as f:
                data = json.load(f)
                # La nueva estructura es una lista de muestras bajo la clave "samples"
                ALL_CALIBRATION_SAMPLES = data.get('samples', [])
            print(f"Cargadas {len(ALL_CALIBRATION_SAMPLES)} muestras de calibración desde {CALIBRATION_FILE}")
            perform_nuc_calibration()
        except Exception as e:
            print(f"Error al cargar datos de calibración: {e}")
            ALL_CALIBRATION_SAMPLES = []
            perform_nuc_calibration()
    else:
        print("No se encontró el archivo de calibración. El sistema usará la escala lineal por defecto.")
        ALL_CALIBRATION_SAMPLES = []
        perform_nuc_calibration()

def add_calibration_sample(pixel_coords, pixel_values, measured_temps):
    """
    Añade una nueva muestra de calibración, la guarda en el archivo y recalibra.
    """
    global ALL_CALIBRATION_SAMPLES
    new_sample = {
        "timestamp": datetime.datetime.now().isoformat(),
        "pixel_coords": pixel_coords.tolist(),
        "pixel_values": pixel_values.tolist(),
        "measured_temps": measured_temps
    }
    ALL_CALIBRATION_SAMPLES.append(new_sample)
    
    # Guardar todas las muestras en el archivo
    try:
        with open(CALIBRATION_FILE, 'w') as f:
            json.dump({"samples": ALL_CALIBRATION_SAMPLES}, f, indent=4)
        print(f"Nueva muestra de calibración añadida. Total: {len(ALL_CALIBRATION_SAMPLES)}.")
        # Recalcular el modelo con la nueva información
        perform_nuc_calibration()
        return True
    except Exception as e:
        print(f"Error al guardar la nueva muestra de calibración: {e}")
        ALL_CALIBRATION_SAMPLES.pop() # Revertir si falla el guardado
        return False

def clear_calibration_data():
    """
    Borra el archivo de calibración y resetea las variables globales.
    """
    global ALL_CALIBRATION_SAMPLES, CALIBRATION_MODEL
    if os.path.exists(CALIBRATION_FILE):
        try:
            os.remove(CALIBRATION_FILE)
            print(f"Archivo de calibración '{CALIBRATION_FILE}' borrado.")
        except Exception as e:
            print(f"Error al borrar el archivo de calibración: {e}")
            return False
    
    ALL_CALIBRATION_SAMPLES = []
    CALIBRATION_MODEL = None
    print("Todos los datos de calibración han sido eliminados.")
    return True


# Cargar datos al iniciar
load_calibration_data()

def calculate_temperature_matrix(ir_gray_image_8bit):
    """
    Calcula la matriz de temperatura a partir de una imagen IR.
    Utiliza el modelo de calibración NUC si está disponible, si no, una escala lineal.
    """
    if ir_gray_image_8bit.ndim == 3:
        ir_gray_1_channel = ir_gray_image_8bit[:, :, 0]
    else:
        ir_gray_1_channel = ir_gray_image_8bit

    pixel_values_float = ir_gray_1_channel.astype(np.float32)

    if CALIBRATION_MODEL is not None:
        # Aplicar el modelo polinomial (NUC)
        temperature_matrix = np.polyval(CALIBRATION_MODEL, pixel_values_float)
    else:
        # Fallback: Usar la fórmula lineal por defecto si no hay calibración
        temperature_matrix = MIN_TEMP_C + (pixel_values_float / 255.0) * (MAX_TEMP_C - MIN_TEMP_C)

    # Asegurar que los valores queden dentro del rango global
    temperature_matrix = np.clip(temperature_matrix, MIN_TEMP_C, MAX_TEMP_C)
    return temperature_matrix

# --- El resto de las funciones de utilidad permanecen sin cambios significativos ---

def apply_rainbow_colormap(temperature_matrix, vmin=MIN_TEMP_C, vmax=MAX_TEMP_C):
    normalized_temp = np.clip((temperature_matrix - vmin) / (vmax - vmin), 0, 1)
    normalized_temp_uint8 = (normalized_temp * 255).astype(np.uint8)
    cmap = plt.get_cmap('jet')
    colored_image_rgba = cmap(normalized_temp_uint8)
    colored_image_bgr = cv2.cvtColor((colored_image_rgba[:, :, :3] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return colored_image_bgr

def save_data_as_dataset(temperature_matrix, ir_gray_image):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(DATASETS_FOLDER, f"thermal_data_{timestamp}.h5")
    if ir_gray_image.ndim == 3:
        ir_gray_image_float = ir_gray_image[:, :, 0].astype(np.float32)
    else:
        ir_gray_image_float = ir_gray_image.astype(np.float32)
    with h5py.File(filename, 'w') as f:
        f.create_dataset('temperature_matrix', data=temperature_matrix)
        f.create_dataset('ir_gray_image_float', data=ir_gray_image_float)
        f.attrs['timestamp'] = timestamp
        f.attrs['calibration_model'] = json.dumps(CALIBRATION_MODEL.tolist() if CALIBRATION_MODEL is not None else None)
    print(f"Dataset guardado: {filename}")
    return filename

def save_img(ir_gray_image, min_temp_detected=None, max_temp_detected=None):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if ir_gray_image.ndim == 3:
        gray_output_image = cv2.cvtColor(ir_gray_image, cv2.COLOR_BGR2GRAY)
    else:
        gray_output_image = ir_gray_image
    gray_filename = os.path.join(THERMAL_IMAGES_FOLDER, f"ir_gray_{timestamp}.jpeg")
    cv2.imwrite(gray_filename, gray_output_image)
    
    temperature_matrix = calculate_temperature_matrix(ir_gray_image)
    if min_temp_detected is None: min_temp_detected = np.min(temperature_matrix)
    if max_temp_detected is None: max_temp_detected = np.max(temperature_matrix)
    colored_image = apply_rainbow_colormap(temperature_matrix, vmin=MIN_TEMP_C, vmax=MAX_TEMP_C)
    min_temp_str = f"{min_temp_detected:.1f}".replace('.', '_')
    max_temp_str = f"{max_temp_detected:.1f}".replace('.', '_')
    colored_filename = os.path.join(THERMAL_IMAGES_FOLDER, f"ir_rainbow_{timestamp}_min{min_temp_str}_max{max_temp_str}.jpeg")
    cv2.imwrite(colored_filename, colored_image)
    print(f"Imágenes guardadas: {gray_filename}, {colored_filename}")
    return gray_filename, colored_filename

def load_latest_dataset():
    list_of_files = [f for f in os.listdir(DATASETS_FOLDER) if f.endswith('.h5')]
    if not list_of_files: return None, None, "No se encontraron archivos .h5"
    latest_file = max(list_of_files, key=lambda x: os.path.getmtime(os.path.join(DATASETS_FOLDER, x)))
    try:
        with h5py.File(os.path.join(DATASETS_FOLDER, latest_file), 'r') as f:
            return f['temperature_matrix'][()], f['ir_gray_image_float'][()], None
    except Exception as e:
        return None, None, f"Error al cargar {latest_file}: {e}"

def load_latest_rainbow_image_path():
    list_of_files = [f for f in os.listdir(THERMAL_IMAGES_FOLDER) if f.startswith('ir_rainbow_') and f.endswith('.jpeg')]
    if not list_of_files: return None
    latest_file = max(list_of_files, key=lambda x: os.path.getmtime(os.path.join(THERMAL_IMAGES_FOLDER, x)))
    return os.path.join(THERMAL_IMAGES_FOLDER, latest_file)