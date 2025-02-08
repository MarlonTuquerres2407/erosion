import cv2
import numpy as np
from ultralytics import YOLO
import os
import shutil
from IPython.display import Image

# Cargar tu modelo entrenado de YOLOv11
modelo = YOLO('C:\\Users\\matuq\\OneDrive\\Documentos\\erosion_app\\runs\\segment\\train\\weights\\best.pt')  # Cambia esta ruta por la de tu modelo entrenado

# Crear un directorio para guardar las máscaras
output_dir = 'C:\\Users\\matuq\\OneDrive\\Documentos\\erosion_app\\result_mask'
os.makedirs(output_dir, exist_ok=True)

# Función para borrar todas las máscaras existentes
def borrar_mascaras(output_dir):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

# Borrar las máscaras existentes antes de procesar nuevas imágenes
borrar_mascaras(output_dir)

# Cargar la imagen que deseas procesar
 
# imagen_path = 'C:\\Users\\matuq\\OneDrive\\Documentos\\erosion_app\\ej2.jpg'  # Cambia esto por el path de tu imagen
imagen_path= Image.open()
imagen = cv2.imread(imagen_path)

# Verificar si la imagen se cargó correctamente
if imagen is None:
    raise ValueError(f"No se pudo cargar la imagen en la ruta: {imagen_path}")

# Realizar la predicción
resultados = modelo(imagen)  # Predicciones de la imagen

# Índice de la clase "erosión fluvial" (cambia esto según tu modelo)
indice_erosion_fluvial = 3  # Asegúrate de que este índice sea correcto

# Procesar los resultados
for i, result in enumerate(resultados):
    if result.masks is not None:
        for idx, (mask, cls) in enumerate(zip(result.masks.data, result.boxes.cls)):  # Acceder a la data de la máscara y la clase
            if cls == indice_erosion_fluvial:
                print(f"Procesando máscara {i} - Segmento {idx}, forma: {mask.shape}")

                # Convertir tensor de PyTorch a NumPy
                mascara_np = mask.cpu().numpy().astype(np.uint8) * 255  # Escalar a 255

                # Asegurar que la máscara tenga 2D (remover dimensión extra si es necesario)
                if len(mascara_np.shape) == 3:
                    mascara_np = mascara_np[0]

                # Redimensionar la máscara al tamaño de la imagen original
                mascara_resized = cv2.resize(mascara_np, (imagen.shape[1], imagen.shape[0]))

                # Guardar la máscara
                mascara_filename = os.path.join(output_dir, f'mascara_erosion_fluvial_{i}_{idx}.png')
                cv2.imwrite(mascara_filename, mascara_resized)
                print(f'Máscara guardada en: {mascara_filename}')

def calcular_area_mascara(mascara, resolucion_pixel=0.5):
    """
    Calcula el área de una máscara en píxeles y metros cuadrados.

    Args:
        mascara (np.array): Máscara binaria.
        resolucion_pixel (float, optional): Resolución del píxel en metros.
                                           Por defecto es 0.5 metros/píxel.

    Returns:
        tuple: Una tupla que contiene el área en píxeles y el área en metros cuadrados.
    """
    # Calcular el área en píxeles
    area_en_pixeles = np.sum(mascara == 255)

    # Calcular el área en metros cuadrados
    area_en_metros_cuadrados = area_en_pixeles * (resolucion_pixel ** 2)

    return area_en_pixeles, area_en_metros_cuadrados

# Directorio donde se guardan las máscaras
output_dir = 'C:\\Users\\matuq\\OneDrive\\Documentos\\erosion_app\\result_mask'

# Get dimensions of the first mask to initialize combined_mask
first_mask_path = next((os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.png')), None)
if first_mask_path:
    first_mask = cv2.imread(first_mask_path, cv2.IMREAD_GRAYSCALE)
    combined_mask = np.zeros_like(first_mask)
else:
    raise ValueError("No mask files found in the directory.")

# Procesar todas las máscaras en el directorio
for mascara_filename in os.listdir(output_dir):
    if mascara_filename.endswith('.png'):
        ruta_mascara = os.path.join(output_dir, mascara_filename)
        mascara = cv2.imread(ruta_mascara, cv2.IMREAD_GRAYSCALE)

        # Asegurar que la máscara sea binaria (0 o 255)
        _, mascara_binaria = cv2.threshold(mascara, 127, 255, cv2.THRESH_BINARY)

        # Resize the current mask to match the combined_mask shape
        mascara_binaria = cv2.resize(mascara_binaria, (combined_mask.shape[1], combined_mask.shape[0]))

        # Combinar la máscara actual con la máscara combinada
        combined_mask = cv2.bitwise_or(combined_mask, mascara_binaria)

# Calcular el área de la máscara combinada
total_area_pixeles, total_area_metros_cuadrados = calcular_area_mascara(combined_mask)

print(f"Área total de las máscaras en píxeles: {total_area_pixeles}")
print(f"Área total de las máscaras en metros cuadrados: {total_area_metros_cuadrados}")

# Guardar la máscara combinada
combined_mask_filename = os.path.join(output_dir, 'mascara_erosion_combinada.png')
cv2.imwrite(combined_mask_filename, combined_mask)
print(f'Máscara combinada guardada en: {combined_mask_filename}')