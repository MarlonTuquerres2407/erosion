import cv2
import numpy as np
from ultralytics import YOLO
import os
import shutil
import streamlit as st

# Configurar Streamlit
st.title("Detección de Erosión Fluvial con YOLO")

# Cargar modelo YOLO
modelo_path = './runs/segment/train/weights/best.pt'
modelo = YOLO(modelo_path)

# Directorio para guardar las máscaras
output_dir = './result_mask'
os.makedirs(output_dir, exist_ok=True)

# Función para borrar todas las máscaras existentes
def borrar_mascaras(output_dir):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

# Borrar las máscaras antes de procesar nuevas imágenes
borrar_mascaras(output_dir)

# Cargar la imagen con Streamlit
imagen_subida = st.file_uploader("Sube una imagen", type=["jpg", "png", "jpeg"])

if imagen_subida is not None:
    # Convertir la imagen a formato OpenCV
    file_bytes = np.asarray(bytearray(imagen_subida.read()), dtype=np.uint8)
    imagen = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Verificar si la imagen se cargó correctamente
    if imagen is None:
        st.error("Error al cargar la imagen.")
    else:
        st.image(imagen, caption="Entrada", use_container_width=True)

        # Realizar la predicción
        resultados = modelo(imagen)

        imagen_resultado = resultados[0].plot()  # Generar imagen con resultados
        st.image(imagen_resultado, caption=f"Detección de objetos", use_container_width=True)

        # Índice de la clase "erosión fluvial" (ajustar según el modelo)
        indice_erosion_fluvial = 3

        # Procesar los resultados
        for i, result in enumerate(resultados):
            if result.masks is not None:
                for idx, (mask, cls) in enumerate(zip(result.masks.data, result.boxes.cls)):
                    if cls == indice_erosion_fluvial:
                        st.write(f"Procesando máscara {i} - Segmento {idx}, forma: {mask.shape}")

                        # Convertir tensor de PyTorch a NumPy
                        mascara_np = mask.cpu().numpy().astype(np.uint8) * 255

                        # Asegurar que la máscara tenga 2D
                        if len(mascara_np.shape) == 3:
                            mascara_np = mascara_np[0]

                        # Redimensionar la máscara al tamaño de la imagen original
                        mascara_resized = cv2.resize(mascara_np, (imagen.shape[1], imagen.shape[0]))

                        # Guardar la máscara
                        mascara_filename = os.path.join(output_dir, f'mascara_erosion_fluvial_{i}_{idx}.png')
                        cv2.imwrite(mascara_filename, mascara_resized)
                        st.write(f'Máscara guardada en: {mascara_filename}')

        # Función para calcular área de la máscara
        def calcular_area_mascara(mascara, resolucion_pixel=0.5):
            area_en_pixeles = np.sum(mascara == 255)
            area_en_metros_cuadrados = area_en_pixeles * (resolucion_pixel ** 2)
            return area_en_pixeles, area_en_metros_cuadrados

        # Combinar todas las máscaras
        first_mask_path = next((os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.png')), None)
        if first_mask_path:
            first_mask = cv2.imread(first_mask_path, cv2.IMREAD_GRAYSCALE)
            combined_mask = np.zeros_like(first_mask)
        else:
            st.error("No se encontraron máscaras en el directorio.")
            combined_mask = None

        if combined_mask is not None:
            for mascara_filename in os.listdir(output_dir):
                if mascara_filename.endswith('.png'):
                    ruta_mascara = os.path.join(output_dir, mascara_filename)
                    mascara = cv2.imread(ruta_mascara, cv2.IMREAD_GRAYSCALE)

                    _, mascara_binaria = cv2.threshold(mascara, 127, 255, cv2.THRESH_BINARY)
                    mascara_binaria = cv2.resize(mascara_binaria, (combined_mask.shape[1], combined_mask.shape[0]))
                    combined_mask = cv2.bitwise_or(combined_mask, mascara_binaria)

            # Calcular el área de la máscara combinada
            total_area_pixeles, total_area_metros_cuadrados = calcular_area_mascara(combined_mask)

            st.write(f"Área total de las máscaras en píxeles: {total_area_pixeles}")
            st.write(f"Área total de las máscaras en metros cuadrados: {total_area_metros_cuadrados}")

            # Guardar la máscara combinada
            combined_mask_filename = os.path.join(output_dir, 'mascara_erosion_combinada.png')
            cv2.imwrite(combined_mask_filename, combined_mask)
            st.write(f'Máscara combinada guardada en: {combined_mask_filename}')
            st.image(combined_mask, caption="Salida - Máscara combinada", use_container_width=True)

            # Agregar botón de descarga
            with open(combined_mask_filename, "rb") as file:
                btn = st.download_button(
                    label="Descargar Máscara",
                    data=file,
                    file_name="mascara_erosion_combinada.png",
                    mime="image/png"
                )
