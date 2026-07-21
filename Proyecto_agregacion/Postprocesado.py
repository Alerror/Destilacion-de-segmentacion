import sqlite3
import numpy as np
import cv2
import json
import pandas as pd  # <-- Faltaba esto
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

def procesar_json_a_db(datos_geometria, nombre_muestra, ruta_db):
    print(f"Cargando JSON desde RAM para la muestra: {nombre_muestra}...")
    
    # <-- Aquí borramos el 'with open(...)'. Ya no lo necesitamos.

    # 2. Conectar a SQLite
    conexion = sqlite3.connect(ruta_db)
    cursor = conexion.cursor()

    # Crear una única tabla resumen (si no existe)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS resumen_experimento (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            muestra TEXT,
            segundo INTEGER,
            celulas INTEGER,
            rulos INTEGER,
            rull_cel INTEGER,
            solapas INTEGER,
            area_total_rulos REAL
        )
    ''')
    conexion.commit()

    print("Procesando fotogramas e insertando en la base de datos...")

    # 3. Masticar los datos segundo a segundo
    for llave_tiempo, datos_frame in datos_geometria.items():
        # Extraer el número del segundo (ej: "segundo_4" -> 4)
        segundo_actual = int(llave_tiempo.split('_')[1])
        
        # Contadores en cero para este frame
        conteos = {"celula": 0, "rulo": 0, "rulcell": 0, "solapa": 0}
        area_total_rulos = 0.0

        for clase, objetos in datos_frame.items():
            if clase in conteos:
                # El número de elementos es simplemente cuántos objetos hay en el diccionario de esa clase
                conteos[clase] = len(objetos)
            
            # Si la clase es rulo, calculamos el área sumada de todos los rulos en este frame
            if clase == "rulo":
                for id_objeto, coords in objetos.items():
                    x_lista = coords["x"]
                    y_lista = coords["y"]
                    
                    # Reconstruir el contorno para OpenCV
                    puntos = np.array(list(zip(x_lista, y_lista)), dtype=np.float32)
                    
                    # Sumar al área total del frame
                    area_total_rulos += cv2.contourArea(puntos)

        # 4. Insertar la fila limpia y resumida en SQLite
        cursor.execute('''
            INSERT INTO resumen_experimento 
            (muestra, segundo, celulas, rulos, rull_cel, solapas, area_total_rulos)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (nombre_muestra, segundo_actual, conteos["celula"], conteos["rulo"], 
              conteos["rulcell"], conteos["solapa"], area_total_rulos))

    conexion.commit()
    conexion.close()
    print(f"¡Listo! Base de datos actualizada con los resultados de {nombre_muestra}.")

def graficar_resultados(nombre_muestra, ruta_db, ruta_png):
    print(f"Extrayendo datos de {nombre_muestra}...")
    
    # 1. Extraer los datos de SQLite
    conexion = sqlite3.connect(ruta_db)
    # Seleccionamos y ordenamos cronológicamente para asegurar que el primer registro sea el t=0
    query = f"SELECT * FROM resumen_experimento WHERE muestra = '{nombre_muestra}' ORDER BY segundo ASC"
    df = pd.read_sql_query(query, conexion)
    conexion.close()

    if df.empty:
        print("Error: No se encontraron datos para esa muestra en la base de datos.")
        return

    # 2. Tratamiento Matemático: Normalizar células sueltas al valor de 1
    celulas_iniciales = df['celulas'].iloc[0]
    if celulas_iniciales > 0:
        df['celulas_norm'] = df['celulas'] / celulas_iniciales
    else:
        df['celulas_norm'] = df['celulas']

    # 3. Configuración estética para publicación científica
    # Usamos un estilo limpio y definimos el tamaño de la figura (dos paneles apilados)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.subplots_adjust(hspace=0.1) # Reducir el espacio entre los dos gráficos

    # --- PANEL SUPERIOR: Cinética de Agregación ---
    color_cel = '#1f77b4'  # Azul oscuro
    ax1.plot(df['segundo'], df['celulas_norm'], color=color_cel, linewidth=2.5, label='Células Libres (Norm)')
    ax1.set_ylabel('Fracción de Células Libres', color=color_cel, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color_cel)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Creamos un eje Y gemelo para los conteos crudos que manejan otra escala
    ax1_twin = ax1.twinx()
    color_rulos = '#d62728' # Rojo
    color_rulcel = '#2ca02c' # Verde
    
    ax1_twin.plot(df['segundo'], df['rulos'], color=color_rulos, linestyle='--', linewidth=2, label='Rouleaux')
    ax1_twin.plot(df['segundo'], df['rull_cel'], color=color_rulcel, linestyle='-.', linewidth=2, label='Células en Rouleaux')
    ax1_twin.set_ylabel('Conteo Absoluto', color='black', fontweight='bold')
    
    # Unificar las leyendas del panel superior
    lineas_1, eti_1 = ax1.get_legend_handles_labels()
    lineas_2, eti_2 = ax1_twin.get_legend_handles_labels()
    ax1_twin.legend(lineas_1 + lineas_2, eti_1 + eti_2, loc='center right')

    # --- PANEL INFERIOR: Evolución del Área ---
    color_area = '#9467bd' # Púrpura
    ax2.plot(df['segundo'], df['area_total_rulos'], color=color_area, linewidth=2.5, marker='o', markersize=4)
    ax2.set_ylabel('Área Total (px²)', color=color_area, fontweight='bold')
    ax2.set_xlabel('Tiempo (segundos)', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color_area)
    ax2.grid(True, linestyle='--', alpha=0.5)

    # Añadir título general
    fig.suptitle(f'Agregación Eritrocitaria - Muestra: {nombre_muestra}', fontsize=14, y=0.95)

    # 4. Guardar en alta resolución (300 dpi es el estándar de las revistas)
    nombre_archivo = f"grafico_{nombre_muestra}.png"
    plt.savefig(ruta_png, dpi=300, bbox_inches='tight')
    print(f"Gráfico generado y guardado en: {nombre_archivo}")
    
    plt.close(fig)