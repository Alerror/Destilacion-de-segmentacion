from ultralytics import YOLO

# ==========================================
# 1. Configuración
# ==========================================
ruta_modelo = 'C:/Users/benja/OneDrive - mail.pucv.cl/Laboratorio/Agregacion/Codigos/Segmentacion/Modelos/best (2).pt'

print("Cargando modelo original en PyTorch...")
modelo = YOLO(ruta_modelo)

# ==========================================
# 2. Compilación Bare-Metal (TensorRT)
# ==========================================
print("Iniciando compilación a TensorRT...")
print("La GPU se acelerará y tomará unos minutos. No interrumpas el proceso.")

ruta_engine = modelo.export(
    format="engine",
    imgsz=1024,      # El tamaño exacto de entrada que configuramos en tu script anterior
    half=True,       # Cuantización FP16 (Clave para duplicar la velocidad)
    device="0",      # Forzar el uso de la RTX 3060
    workspace=4      # Asignar 4GB de VRAM como espacio de trabajo durante la compilación
)

print(f"¡Compilación exitosa! Nuevo modelo listo para producción en: {ruta_engine}")