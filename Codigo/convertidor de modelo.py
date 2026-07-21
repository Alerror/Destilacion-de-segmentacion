from ultralytics import YOLO

# ==========================================
# 1. Configuración
# ==========================================
ruta_modelo = 'C:/Users/benja/OneDrive - mail.pucv.cl/Laboratorio/Agregacion/Codigos/Segmentacion/Modelos/best (2).pt'

print("Cargando modelo original en PyTorch...")
modelo = YOLO(ruta_modelo)

# ==========================================
# 2. Exportación a ONNX (Precisión Completa)
# ==========================================
print("Iniciando exportación a ONNX (FP32)...")

ruta_onnx = modelo.export(
    format="onnx",
    imgsz=1024,      # Mantenemos tu resolución correcta
    half=False,      # CRÍTICO: Precisión completa (FP32) para no dañar las máscaras
    device="0",      # Usar la RTX 3060
    simplify=True    # Limpia y optimiza la arquitectura gráfica interna
)

print(f"¡Exportación exitosa! Nuevo modelo listo en: {ruta_onnx}")