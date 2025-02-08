# import yaml

# data = {
#     'path': 'C:\\Users\\matuq\\OneDrive\\Documentos\\erosion_app\\Dst2_Segmentacion',
#     'train': 'images/train',
#     'val': 'images/val',
#     'test': 'images/test',
#     'names': {
#   0: 'suelo',
#   1: 'vegetacion',
#   2: 'aluvial',
#   3: 'erosion fluvial',
#   4: 'rio'
#     }
# }

# # Change the file path to a .yaml file within the directory
# with open('C:\\Users\\matuq\\OneDrive\\Documentos\\erosion_app\\Dst2_Segmentacion\\data.yaml', 'w') as file:
#     yaml.dump(data, file,
#               default_flow_style=False,
#               sort_keys=False)

from ultralytics import YOLO

# # 1. Cargar el modelo (nano, el más pequeño)
# model = YOLO("yolo11n-seg.pt")

# # Afinar el modelo
# results = model.train(data="C:\\Users\\matuq\\OneDrive\\Documentos\\erosion_app\\Dst2_Segmentacion\\data.yaml",
#                       epochs=50, # Número de iteraciones
#                       imgsz=640, # Tamaño de las imágenes (máxima dimensión)
#                       plots=True, # Guardar gráficos de entrenamiento
#                       )

from IPython.display import Image as IPyImage

IPyImage(filename='C:\\Users\\matuq\\OneDrive\\Documentos\\erosion_app\\runs\\segment\\train\\results.png', width=600)

model = YOLO ('C:\\Users\\matuq\\OneDrive\\Documentos\\erosion_app\\runs\\segment\\train\\weights\\best.pt')

preds = model('C:\\Users\\matuq\\OneDrive\\Documentos\\erosion_app\\Dst2_Segmentacion\\images\\test')

# Imágenes con buenas segmentaciones:
preds[1].show()
