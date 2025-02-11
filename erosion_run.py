import cv2
import numpy as np
import os
import shutil
import streamlit as st
from ultralytics import YOLO

# Configurar Streamlit
st.title("Erosión fluvial")

# Cargar modelo YOLO
modelo_path = './runs/segment/train/weights/best.pt'
modelo = YOLO(modelo_path)

# Directorio para guardar las máscaras
output_dir_detect = './result_detect'
output_dir = './result_mask'
os.makedirs(output_dir, exist_ok=True)

# Función para borrar todas las máscaras previas
def borrar_mascaras(output_dir):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

borrar_mascaras(output_dir)

# Función para calcular el área de la máscara
def calcular_area_mascara(mascara, resolucion_pixel=None, dimensiones_imagen=None):
    area_en_pixeles = np.sum(mascara == 255)
    if resolucion_pixel:
        area_en_metros_cuadrados = area_en_pixeles * (resolucion_pixel ** 2)
    elif dimensiones_imagen:
        largo, ancho = dimensiones_imagen
        total_pixeles = mascara.shape[0] * mascara.shape[1]
        area_en_metros_cuadrados = (area_en_pixeles / total_pixeles) * (largo * ancho)
    else:
        area_en_metros_cuadrados = None
    return area_en_pixeles, area_en_metros_cuadrados

# Cargar imagen en Streamlit
imagen_subida = st.file_uploader("Sube una imagen", type=["jpg", "png", "jpeg"])

if imagen_subida is not None:
    file_bytes = np.asarray(bytearray(imagen_subida.read()), dtype=np.uint8)
    imagen = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if imagen is None:
        st.error("Error al cargar la imagen.")
    else:
        st.image(imagen, caption="Entrada", use_container_width=True)
        resultados = modelo(imagen)

        imagen_resultado = resultados[0].plot()
        st.image(imagen_resultado, caption="Detección de erosión fluvial", use_container_width=True)

        # Guardar la imagen de detección
        resultado_filename = os.path.join(output_dir_detect, "deteccion_erosion.png")
        cv2.imwrite(resultado_filename, imagen_resultado)

        # Agregar botón de descarga para la imagen de detección
        with open(resultado_filename, "rb") as file:
            st.download_button(
                label="Descargar Imagen de Detección",
                data=file,
                file_name="deteccion_erosion.png",
                mime="image/png"
            )

        indice_erosion_fluvial = 3  # Ajustar según el modelo

        for i, result in enumerate(resultados):
            if result.masks is not None:
                for idx, (mask, cls) in enumerate(zip(result.masks.data, result.boxes.cls)):
                    if cls == indice_erosion_fluvial:
                        mascara_np = mask.cpu().numpy().astype(np.uint8) * 255
                        if len(mascara_np.shape) == 3:
                            mascara_np = mascara_np[0]
                        mascara_resized = cv2.resize(mascara_np, (imagen.shape[1], imagen.shape[0]))

                        mascara_filename = os.path.join(output_dir, f'mascara_erosion_{i}_{idx}.png')
                        cv2.imwrite(mascara_filename, mascara_resized)

        # Combinar todas las máscaras
        first_mask_path = next((os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.png')), None)
        if first_mask_path:
            first_mask = cv2.imread(first_mask_path, cv2.IMREAD_GRAYSCALE)
            combined_mask = np.zeros_like(first_mask)
        else:
            st.error("No se encontraron máscaras generadas.")
            combined_mask = None

        if combined_mask is not None:
            for mascara_filename in os.listdir(output_dir):
                if mascara_filename.endswith('.png'):
                    ruta_mascara = os.path.join(output_dir, mascara_filename)
                    mascara = cv2.imread(ruta_mascara, cv2.IMREAD_GRAYSCALE)
                    _, mascara_binaria = cv2.threshold(mascara, 127, 255, cv2.THRESH_BINARY)
                    mascara_binaria = cv2.resize(mascara_binaria, (combined_mask.shape[1], combined_mask.shape[0]))
                    combined_mask = cv2.bitwise_or(combined_mask, mascara_binaria)

            # Selección del método para calcular el área
            metodo_area = st.radio(
                "¿Cómo deseas calcular el área?",
                ("Usando dimensiones de la imagen", "Usando tamaño de píxel", "No calcular área")
            )

            dimensiones = None
            resolucion_pixel = None

            if metodo_area == "Usando dimensiones de la imagen":
                largo = st.number_input("Longitud de la imagen en metros:", min_value=0.01, format="%.2f")
                ancho = st.number_input("Ancho de la imagen en metros:", min_value=0.01, format="%.2f")
                if largo > 0 and ancho > 0:
                    dimensiones = (largo, ancho)

            elif metodo_area == "Usando tamaño de píxel":
                resolucion_pixel = st.number_input("Tamaño del píxel en metros:", min_value=0.0001, format="%.4f")

            if metodo_area != "No calcular área":
                total_area_pixeles, total_area_metros_cuadrados = calcular_area_mascara(combined_mask, resolucion_pixel, dimensiones)
                st.write(f"Área total en píxeles: {total_area_pixeles}")
                if total_area_metros_cuadrados is not None:
                    st.write(f"Área total en metros cuadrados: {total_area_metros_cuadrados:.4f} m²")
                else:
                    st.write("No se pudo calcular el área en metros cuadrados debido a falta de datos.")

            # Guardar la máscara combinada
            combined_mask_filename = os.path.join(output_dir, 'mascara_erosion_combinada.png')
            cv2.imwrite(combined_mask_filename, combined_mask)
            st.image(combined_mask, caption="Máscara combinada", use_container_width=True)

            # Agregar botón de descarga
            with open(combined_mask_filename, "rb") as file:
                st.download_button(
                    label="Descargar Máscara Combinada",
                    data=file,
                    file_name="mascara_erosion_combinada.png",
                    mime="image/png"
                )
