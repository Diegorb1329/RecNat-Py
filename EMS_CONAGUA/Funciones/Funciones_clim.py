import geopandas as gpd
import pandas as pd
import os
import matplotlib.pyplot as plt
from datetime import datetime
import shutil
import matplotlib.pyplot as plt
import seaborn as sns

def generar_boxplot(df, variable, output_path):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Mes', y=variable, data=df)
    plt.title(f'Boxplot de {variable} por Mes')
    plt.savefig(output_path)
    plt.close()


def extraer_datos_climatologicos(file_path):

    with open(file_path, 'r', encoding='ISO-8859-1') as file:
        # Leer las líneas del archivo
        lines = file.readlines()
    
    # Encontrar el inicio de los datos climatológicos
    inicio_datos = None
    for i, line in enumerate(lines):
        if line.strip().startswith('Fecha'):
            inicio_datos = i + 1
            break
    
    if inicio_datos is None:
        return pd.DataFrame()  # Retornar un DataFrame vacío si no se encuentra el inicio de los datos
    
    # Extraer y procesar los datos climatológicos
    data = []
    for line in lines[inicio_datos:]:
        parts = line.split()
        if len(parts) >= 5:  # Asegurarse de que la línea tiene suficientes partes para los datos
            fecha = parts[0]
            precip = parts[1]
            evap = parts[2] if parts[2] != 'Nulo' else None
            tmax = parts[3] if parts[3] != 'Nulo' else None
            tmin = parts[4] if parts[4] != 'Nulo' else None
            data.append([fecha, precip, evap, tmax, tmin])
    
    # Convertir a DataFrame
    df = pd.DataFrame(data, columns=['Fecha', 'Precip', 'Evap', 'TMax', 'TMin'])
    
    # Convertir columnas al tipo correcto
    df['Fecha'] = pd.to_datetime(df['Fecha'], format='%d/%m/%Y', errors='coerce')
    df['Precip'] = pd.to_numeric(df['Precip'], errors='coerce')
    df['Evap'] = pd.to_numeric(df['Evap'], errors='coerce')
    df['TMax'] = pd.to_numeric(df['TMax'], errors='coerce')
    df['TMin'] = pd.to_numeric(df['TMin'], errors='coerce')
    
    return df


def calcular_matriz_calidad(shp_path, df_metadatos, carpeta_datos, fecha_inicio, fecha_fin, estado_operacion): 
    # Cargar el shapefile

    gdf_shp = gpd.read_file(shp_path)
    gdf_shp = gdf_shp.to_crs(epsg=4326)
    
    # Convertir latitud y longitud en puntos geométricos
    gdf_metadatos = gpd.GeoDataFrame(df_metadatos, geometry=gpd.points_from_xy(df_metadatos.longitud, df_metadatos.latitud))
    gdf_metadatos.crs = 'epsg:4326'
    
    # Filtrar estaciones dentro del área del SHP y por estado de operación
    estaciones_en_shp = gpd.sjoin(gdf_metadatos, gdf_shp, op='within')
    estaciones_filtradas = estaciones_en_shp[estaciones_en_shp['situacion'] == estado_operacion]
    
    resultados = []

    # Procesar cada estación filtrada
    for _, estacion in estaciones_filtradas.iterrows():
        estacion_id = estacion['estacion']
        archivo_datos = f"dia{estacion_id}.TXT"
        ruta_completa = os.path.join(carpeta_datos, archivo_datos)
        
        # Extraer los datos climatológicos usando la función previamente definida
        df_datos_estacion = extraer_datos_climatologicos(ruta_completa)
        
        # Filtrar por el rango de fechas
        df_datos_estacion = df_datos_estacion[(df_datos_estacion['Fecha'] >= fecha_inicio) & (df_datos_estacion['Fecha'] <= fecha_fin)]
        
        # Calcular el porcentaje de completitud para cada parámetro
        porcentaje_completitud = df_datos_estacion.notna().mean() * 100
        
        # Incluir información adicional de la estación
        resultado = {
            'estacion_id': estacion_id,
            'nombre': estacion['nombre'],
            'fecha_inicial_datos': df_datos_estacion['Fecha'].min(),
            'fecha_final_datos': df_datos_estacion['Fecha'].max(),
            'completitud_precip': porcentaje_completitud['Precip'],
            'completitud_evap': porcentaje_completitud['Evap'],
            'completitud_tmax': porcentaje_completitud['TMax'],
            'completitud_tmin': porcentaje_completitud['TMin']
        }
        
        resultados.append(resultado)
    
    # Convertir los resultados en un DataFrame para análisis
    df_resultados = pd.DataFrame(resultados)
    return df_resultados


def mapa_estaciones (shp_path, df_metadatos, estado_operacion):
    # Cargar el shapefile y asegurarse de que esté en WGS84
    gdf_area = gpd.read_file(shp_path)
    gdf_area = gdf_area.to_crs(epsg=4326)
    
    # Filtrar df_metadatos por estado de operación antes de convertirlo a GeoDataFrame
    df_metadatos_filtrado = df_metadatos[df_metadatos['situacion'] == estado_operacion]
    gdf_estaciones_filtradas = gpd.GeoDataFrame(df_metadatos_filtrado,
                                                geometry=gpd.points_from_xy(df_metadatos_filtrado.longitud, df_metadatos_filtrado.latitud))
    gdf_estaciones_filtradas.crs = "EPSG:4326"
    
    # Filtrar solo las estaciones dentro del área del shapefile
    estaciones_dentro_de_area_y_estado = gpd.sjoin(gdf_estaciones_filtradas, gdf_area, how="inner", op='within')
    
    # Visualización ajustada para incluir solo estaciones dentro del área y con el estado de operación deseado
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf_area.plot(ax=ax, color='lightblue', edgecolor='black')  # Dibuja el área
    estaciones_dentro_de_area_y_estado.plot(ax=ax, color='red', markersize=5)  # Dibuja solo las estaciones filtradas
    
    # Añadir etiquetas de "estacion" para cada punto
    for x, y, label in zip(estaciones_dentro_de_area_y_estado.geometry.x, estaciones_dentro_de_area_y_estado.geometry.y, estaciones_dentro_de_area_y_estado['estacion']):
        ax.text(x, y, label, fontsize=9)
    
    plt.show()

# Inconcsistencias

def contar_anomalias_especificas(df):
    resultado = {}
    for variable in ['Precip', 'Evap', 'TMax', 'TMin']:
        serie = df[variable]
        # Detectar datos iguales consecutivos
        datos_iguales_consecutivos = (serie.diff() == 0).astype(int).groupby(serie.ne(serie.shift()).cumsum()).cumsum().max()
        # Detectar saltos de datos incompletos
        saltos_incompletos = serie.isnull().astype(int).groupby(serie.notnull().astype(int).cumsum()).cumsum().max()
        
        resultado[variable] = {
            'datos_iguales_consecutivos': datos_iguales_consecutivos > 5,
            'saltos_incompletos': saltos_incompletos > 7
        }
    return resultado

def copiar_archivo(src, dst):
    with open(src, 'rb') as fsrc:
        with open(dst, 'wb') as fdst:
            fdst.write(fsrc.read())


def encontrar_maximo_datos_nulos(df):
    resultado = {}
    for variable in ['Precip', 'Evap', 'TMax', 'TMin']:
        serie = df[variable]
        # Calcular el máximo de datos nulos consecutivos
        max_nulos = serie.isnull().astype(int).groupby(serie.notnull().astype(int).cumsum()).cumsum().max()
        
        resultado[variable] = {'max_nulos': max_nulos}
    return resultado


def ajustar_inconsistencias(df):
    # Asegurar que las precipitaciones sean no negativas
    df.loc[df['Precip'] < 0, 'Precip'] = None
    
    # Asegurar que TMax sea mayor o igual que TMin y ambos en el rango aceptable
    condiciones = (df['TMax'] < df['TMin']) | (df['TMax'] > 100) | (df['TMax'] < -100)
    df.loc[condiciones, 'TMax'] = None
    
    condiciones = (df['TMin'] > df['TMax']) | (df['TMin'] > 100) | (df['TMin'] < -100)
    df.loc[condiciones, 'TMin'] = None
    
    return df

# Asume que las funciones extraer_datos_climatologicos, ajustar_inconsistencias, 
# contar_anomalias_especificas, y encontrar_maximo_datos_nulos están definidas como se proporcionó anteriormente.

def reporte_calidad(file_paths, output_dir):
    
    carpeta_datos_crudos = os.path.join(output_dir, "datos_crudos")
    carpeta_datos_procesados = os.path.join(output_dir, "datos_procesados")
    carpeta_reportes = os.path.join(output_dir, "reportes")

    os.makedirs(carpeta_datos_crudos, exist_ok=True)
    os.makedirs(carpeta_datos_procesados, exist_ok=True)
    os.makedirs(carpeta_reportes, exist_ok=True)

    for file_path in file_paths:
        df = extraer_datos_climatologicos(file_path)
        if df.empty:
            print(f"No se encontraron datos en {file_path}.")
            continue

        destino_crudo = os.path.join(carpeta_datos_crudos, os.path.basename(file_path))
        copiar_archivo(file_path, destino_crudo)

        df = ajustar_inconsistencias(df)

        base_name = os.path.splitext(os.path.basename(file_path))[0]
        df.to_csv(os.path.join(carpeta_datos_procesados, f"{base_name}_procesados.csv"), index=False)

        df['Mes'] = df['Fecha'].dt.month
        completitud_por_mes = df.drop('Fecha', axis=1).groupby('Mes').apply(lambda x: x.notna().mean() * 100)

        anomalias_especificas = contar_anomalias_especificas(df)
        maximo_datos_nulos = encontrar_maximo_datos_nulos(df)

        output_path = os.path.join(carpeta_reportes, f"reporte_calidad_{base_name}.txt")
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(f"Reporte de Calidad de Datos - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            file.write(f"Archivo Analizado: {os.path.basename(file_path)}\n\n")
            file.write("Este reporte de calidad presenta el porcentaje de completitud de datos por mes y algunas de las principales anomalías de datos presentes. Se producen diferentes carpetas que almacenan gráficos sobre la calidad de datos, una copia de los datos originales y un archivo .csv que contiene los datos, excluyendo fehcas fuplicadas, casos donde TMAX > TMIN, registros con precipitaciones negativas y temperaturas superiores a 100 ºC y menores a -100 ºC")
            file.write("Porcentaje de Completitud por Mes y Variable:\n")
            file.write(completitud_por_mes.to_string() + "\n\n")
            file.write("Resumen de Anomalías:\n")
            file.write("Variable\tDatos Iguales Consecutivos\tSaltos de Datos Incompletos\tMáximo de Datos Nulos Consecutivos\n")
            for var in ['Precip', 'Evap', 'TMax', 'TMin']:
                file.write(f"{var}\t{anomalias_especificas[var]['datos_iguales_consecutivos']}\t{anomalias_especificas[var]['saltos_incompletos']}\t{maximo_datos_nulos[var]['max_nulos']}\n")

        
        print(f"Reporte generado: {output_path}")


def shp_resumen(csv_directory, metadata_file, output_directory):
    # Leer metadatos
    metadata = metadata_file
    metadata['latitud'] = metadata['latitud'].str.replace(' msnm', '').astype(float)
    metadata['longitud'] = metadata['longitud'].astype(float)
    
    # Leer archivos CSV de datos climáticos
    csv_files = [f for f in os.listdir(csv_directory) if f.endswith('.csv')]
    all_data = []
    
    for file in csv_files:
        data = pd.read_csv(os.path.join(csv_directory, file))
        # Extraer ID de estación del nombre del archivo
        estacion_id = int(file.split('_')[0].replace('dia', ''))
        data['estacion'] = estacion_id
        all_data.append(data)
    
    # Combinar todos los datos climáticos
    combined_data = pd.concat(all_data)
    
    # Filtrar datos para el rango de años 2000 a 2020
    combined_data['Año'] = combined_data['Año'].astype(int)
    filtered_data = combined_data[(combined_data['Año'] >= 2000) & (combined_data['Año'] <= 2020)]
    
    # Calcular resúmenes mensuales
    summary = filtered_data.groupby(['estacion', 'Mes']).agg({
        'Precip_Acumulada': 'sum',
        'TMin_Media': 'mean',
        'TMax_Media': 'mean',
        'Evap_Media': 'mean'
    }).reset_index()
    
    # Unir los resúmenes con la información de ubicación
    summary = summary.merge(metadata, left_on='estacion', right_on='estacion')
    
    # Crear una geometría de puntos a partir de la latitud y longitud
    geometry = [Point(xy) for xy in zip(summary['longitud'], summary['latitud'])]
    gdf = gpd.GeoDataFrame(summary, geometry=geometry)
    
    # Definir el sistema de referencia espacial (WGS84)
    gdf.set_crs(epsg=4326, inplace=True)
    
    # Guardar el geodataframe como un shapefile
    output_file = os.path.join(output_directory, 'summary_climate_data.shp')
    gdf.to_file(output_file)








