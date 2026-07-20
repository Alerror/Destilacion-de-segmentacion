import cv2
import glob
import time

# ==========================================
# 1. Rutas
# ==========================================
ruta_imagenes = 'H:/Alerror/Experimentos/Datos 1-15-2026/60%/20260115/*.png' 
ruta_video_salida = 'C:/Users/benja/Downloads/experimento_base_1080p.mp4'
fps_video = 10  

# ==========================================
# 2. Inicialización
# ==========================================
lista_imagenes = sorted(glob.glob(ruta_imagenes))
total_frames = len(lista_imagenes)

if total_frames == 0:
    raise ValueError("No se encontraron imágenes en la ruta.")

print(f"Generando video base a 1080p con {total_frames} imágenes...")

# Video de salida a 1920x1080
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(ruta_video_salida, fourcc, fps_video, (1920, 1080))

inicio = time.time()

# ==========================================
# 3. Procesamiento y Recorte
# ==========================================
for i, ruta_img in enumerate(lista_imagenes):
    imagen = cv2.imread(ruta_img)
    if imagen is None:
        continue
        
    # Mantener consistencia pasando a gris y luego a RGB
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    imagen_gris_rgb = cv2.cvtColor(imagen_gris, cv2.COLOR_GRAY2RGB)
    
    # Recorte Central a 1920x1080
    alto_original, ancho_original = imagen_gris_rgb.shape[:2]
    centro_y, centro_x = alto_original // 2, ancho_original // 2
    
    y_inicio = centro_y - 540
    y_fin = centro_y + 540
    x_inicio = centro_x - 960
    x_fin = centro_x + 960
    
    imagen_recortada = imagen_gris_rgb[y_inicio:y_fin, x_inicio:x_fin]
    
    out.write(imagen_recortada)
    
    if (i + 1) % 20 == 0:
        print(f"Procesando: {i + 1}/{total_frames} frames...")

out.release()
print(f"¡Video base listo en {time.time() - inicio:.1f} segundos!\nGuardado en: {ruta_video_salida}")