from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import time
import torch

# 1. Pon las rutas exactas de tu disco duro
ruta_modelo = 'C:/Users/benja/OneDrive - mail.pucv.cl/Laboratorio/Agregacion/Codigos/Segmentacion/Modelos/best (2).pt' # Cambia esto a tu ruta real
imagen_prueba = 'H:/Alerror/Experimentos/Datos 1-15-2026/60%/20260115/IMG-1300.png' # Cambia esto a tu ruta real

# 2. Cargar y forzar la imagen a escala de grises
imagen = cv2.imread(imagen_prueba)
imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
imagen_gris_rgb = cv2.cvtColor(imagen_gris, cv2.COLOR_GRAY2RGB)


# 3. Cargar el modelo YOLO en SAHI
print("Cargando el modelo en SAHI...")
detection_model = AutoDetectionModel.from_pretrained(
    model_type='ultralytics',
    model_path=ruta_modelo,
    confidence_threshold=0.5,
    device="cuda:0"  
)

# 4. Inferencia con Ventana Deslizante Optimizada
print("Aplicando ventana deslizante (SAHI)...")
inicio = time.time()

resultado_sahi = get_sliced_prediction(
    imagen_gris_rgb,
    detection_model,
    slice_height=1024,
    slice_width=1024,
    overlap_height_ratio=0.2,       # Mantenemos el solapamiento bajo para ahorrar 3 segundos
    overlap_width_ratio=0.2,
    postprocess_type="NMS",         # Supresión no máxima rápida en CPU
    postprocess_match_metric="IOU"
)

fin = time.time()
print(f"⏱️ Tiempo de inferencia (SAHI): {fin - inicio:.2f} segundos")

# 5. Exportar la imagen limpia usando la herramienta nativa de SAHI
print("Exportando visualización...")
ruta_exportada = "C:/Users/benja/Downloads/resultado_sahi_limpio"

# SAHI maneja bien la memoria al exportar, así que esto no colapsará la RAM
resultado_sahi.export_visuals(
    export_dir=".", 
    file_name=ruta_exportada,
    hide_labels=True,  
    hide_conf=True      
)

print(f"¡Imagen exportada con éxito en: {ruta_exportada}.png!")