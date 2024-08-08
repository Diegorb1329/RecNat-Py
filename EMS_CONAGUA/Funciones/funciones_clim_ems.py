import geopandas as gpd
import pandas as pd
import os
import matplotlib.pyplot as plt
from datetime import datetime
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Polygon
from tqdm import tqdm

def generar_histogramas_redondeo(df, variables, output_path):
    fig, axs = plt.subplots(1, len(variables), figsize=(20, 5), sharey=True)

    # Diccionario para asignar colores a las variables
    colors = ['skyblue', 'red', 'green']
    color_dict = dict(zip(variables, colors))
    for ax, variable in zip(axs, variables):
        data = df[variable].dropna()

        if variable == 'PRECIP':
            # Filtrar valores que no son cero
            data = data[data != 0]

        data = abs(data - round(data, 0))
        unique_values = data.nunique()
        bins = unique_values if unique_values <= 10 else 10

        sns.histplot(data, bins=bins, kde=False, ax=ax, 
                     color=color_dict.get(variable, 'grey'),  # Color gris por defecto
                     edgecolor='black')
        ax.set_title(f'{variable} ROUNDING')
        ax.set_xlabel('Absolute Difference after Rounding')

    # Ajustar el límite x para todos los subplots
    for ax in axs:
        ax.set_xlim(0.0, 0.5)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def generar_analisis_mensual(df, archivo_nombre, carpeta_destino):

    df['Temp_Media'] = (df['TMax'] + df['TMin']) / 2
    
    df_mensual = df.groupby(['Año', 'Mes']).agg(
        Precip_Acumulada=('Precip', 'sum'),
        Temp_Media=('Temp_Media', 'mean'),  # Agregar el promedio mensual de la temperatura media diaria
        Evap_Media=('Evap', 'mean'),
        Evap_Max=('Evap', 'max'),
        Evap_Min=('Evap', 'min'),
        TMax_Media=('TMax', 'mean'),
        TMax_Max=('TMax', 'max'),
        TMax_Min=('TMax', 'min'),
        TMin_Media=('TMin', 'mean'),
        TMin_Max=('TMin', 'max'),
        TMin_Min=('TMin', 'min'),
        Dias_Registro=('Fecha', 'count')
    ).reset_index()
    
    # Calcular el porcentaje de completitud para cada variable
    for var in ['Precip', 'Evap', 'TMax', 'TMin']:
        df_mensual[f'Porc_Completo_{var}'] = df.groupby(['Año', 'Mes'])[var].apply(
            lambda x: x.notnull().mean() * 100).reset_index(drop=True)
    
    # Guardar el resultado
    resultado_path = os.path.join(carpeta_destino, f"analisis_mensual_{archivo_nombre}.csv")
    df_mensual.to_csv(resultado_path, index=False)
    return resultado_path




def generar_boxplot(df, variables, output_path):
    plt.figure(figsize=(20, 12))  # Ajusta el tamaño de la figura general

    # Asegurarse de que todos los meses están representados en el DataFrame
    # Esto es necesario para evitar el ValueError al graficar
    meses = pd.DataFrame({'Mes': range(1, 13)})
    df['Mes'] = df['Mes'].astype(int)  # Asegurarse de que 'Mes' es entero para el merge
    df = pd.merge(meses, df, on='Mes', how='left')

    for i, variable in enumerate(variables, 1):
        ax = plt.subplot(2, 2, i)  # Ajusta la disposición de los subplots
        df_filtrado = df.copy()  # Copia el DataFrame para manipulación

        if variable == 'Precip':
            df_filtrado = df_filtrado[df_filtrado[variable] > 0]
        df_filtrado = df_filtrado.dropna(subset=[variable])  # Elimina NaN específicamente para la variable actual

        # Proceder a graficar si hay datos disponibles después de filtrar
        if not df_filtrado.empty:
            sns.boxplot(x='Mes', y=variable, data=df_filtrado, ax=ax, showfliers=True)
            ax.set_title(f'Boxplot de {variable} por Mes')
        else:
            ax.text(0.5, 0.5, f'No hay datos disponibles para {variable}', 
                    horizontalalignment='center', verticalalignment='center', 
                    transform=ax.transAxes)
    
    plt.tight_layout()
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

    # Asegurarse de que los identificadores de estación sean cadenas
    df_metadatos['estacion'] = df_metadatos['estacion'].apply(lambda x: str(x).zfill(5))
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

def calcular_matriz_calidad_2 (shp_path, df_metadatos, carpeta_datos, fecha_inicio, fecha_fin, estado_operacion): 
    # Cargar el shapefile
    gdf_shp = gpd.read_file(shp_path)
    gdf_shp = gdf_shp.to_crs(epsg=4326)

    # Asegurarse de que los identificadores de estación sean cadenas y estén bien formateados
    df_metadatos['estacion'] = df_metadatos['estacion'].astype(str).str.zfill(5)
    gdf_metadatos = gpd.GeoDataFrame(df_metadatos, geometry=gpd.points_from_xy(df_metadatos.longitud, df_metadatos.latitud))
    gdf_metadatos.crs = 'epsg:4326'

    # Filtrar estaciones dentro del área del SHP y por estado de operación
    estaciones_en_shp = gpd.sjoin(gdf_metadatos, gdf_shp, op='within')
    estaciones_filtradas = estaciones_en_shp[estaciones_en_shp['situacion'] == estado_operacion]

    resultados = []
    file_paths = []  # Lista para guardar las rutas de los archivos de datos de las estaciones

    # Procesar cada estación filtrada
    for _, estacion in estaciones_filtradas.iterrows():
        estacion_id = estacion['estacion']
        archivo_datos = f"dia{estacion_id}.TXT"
        ruta_completa = os.path.join(carpeta_datos, archivo_datos)
        
        # Agregar la ruta del archivo a la lista de rutas de archivo
        file_paths.append(ruta_completa)
        
        try:
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
        except FileNotFoundError:
            # Si el archivo no se encuentra, se puede registrar un error o continuar con la siguiente iteración
            print(f"No se encontró el archivo para la estación: {estacion_id}")

    resultados_df = pd.DataFrame(resultados)

    # Devolver tanto los resultados como la lista de file_paths
    return resultados_df, file_paths
    
    # Convertir los resultados en un DataFrame para análisis
    df_resultados = pd.DataFrame(resultados)
    return df_resultados

def mapa_estaciones (shp_path, df_metadatos, estado_operacion, zona_utm, mostrar_buffer=False, tamaño_buffer=None):
    zona_a_epsg = {
        "11N": "EPSG:32611",
        "12N": "EPSG:32612",
        "13N": "EPSG:32613",
        "14N": "EPSG:32614",
        "15N": "EPSG:32615",
        "16N": "EPSG:32616",
    }
    
    if zona_utm not in zona_a_epsg:
        raise ValueError(f"Zona UTM '{zona_utm}' no es válida.")
    
    epsg_utm = zona_a_epsg[zona_utm]
    
    gdf_area = gpd.read_file(shp_path).to_crs(epsg=4326)
    
    nueva_area = gdf_area  # Inicializar con el área original
    
    # Verificar si se ha especificado un tamaño de buffer y es mayor que cero
    if mostrar_buffer and tamaño_buffer is not None and tamaño_buffer > 0:
        gdf_area_utm = gdf_area.to_crs(epsg_utm)
        buffer = gdf_area_utm.buffer(tamaño_buffer)
        buffer_wgs84 = buffer.to_crs(epsg=4326)
        nueva_area = buffer_wgs84.unary_union  # Unir el buffer con el área original
        
        # Crear y guardar un nuevo shapefile con el área extendida por el buffer
        nuevo_shp_path = os.path.join(os.path.dirname(shp_path), 'AOI_buffer.shp')
        nueva_area_gdf = gpd.GeoDataFrame(geometry=[nueva_area], crs='EPSG:4326')
        nueva_area_gdf.to_file(nuevo_shp_path)
    
    df_metadatos_filtrado = df_metadatos[df_metadatos['situacion'] == estado_operacion]
    gdf_estaciones = gpd.GeoDataFrame(df_metadatos_filtrado, 
                                      geometry=gpd.points_from_xy(df_metadatos_filtrado.longitud, 
                                                                  df_metadatos_filtrado.latitud))
    gdf_estaciones.crs = "EPSG:4326"
    
    # Filtrar estaciones que están dentro del área (y buffer si se especificó)
    gdf_estaciones_filtradas = gdf_estaciones[gdf_estaciones.geometry.within(nueva_area)]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Dibujar el área original
    gdf_area.plot(ax=ax, color='lightblue', edgecolor='black', zorder=2)
    
    # Dibujar el buffer con un color gris fijo si se especificó
    if mostrar_buffer and tamaño_buffer is not None and tamaño_buffer > 0:
        gpd.GeoSeries(buffer_wgs84).plot(ax=ax, color='gray', alpha=0.5, zorder=1)
    
    # Dibujar las estaciones con un zorder alto para que estén en el frente
    gdf_estaciones_filtradas.plot(ax=ax, color='red', markersize=5, zorder=3)
    
    # Añadir etiquetas de estaciones
    for x, y, label in zip(gdf_estaciones_filtradas.geometry.x, gdf_estaciones_filtradas.geometry.y, gdf_estaciones_filtradas['estacion']):
        ax.text(x, y, label, fontsize=9, zorder=4)
    
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
# Modificamos la función para generar los plots de series temporales con una disposición vertical

def generar_series(df, output_dir):
    # Asegurarse de que 'Fecha' es un datetime
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    
    # Definir las variables para iterar
    variables = ['Precip', 'Evap', 'TMax', 'TMin']
    
    # Generar un plot para cada variable
    for variable in variables:
        # Crear una figura con 12 subplots (1 por cada mes) en un arreglo vertical
        fig, axs = plt.subplots(12, 1, figsize=(5, 30), sharex=True, sharey=True)
        
        # Iterar a través de cada mes para el subplot
        for i in range(1, 13):
            # Filtrar el dataframe por mes
            df_mes = df[df['Mes'] == i]
            media_mes = df_mes[variable].mean()
        
            # Graficar la serie temporal para el mes actual
            sns.lineplot(x='Fecha', y=variable, data=df_mes, ax=axs[i-1], legend=False)
            axs[i-1].axhline(media_mes, color='red', linestyle='--', linewidth=1)
            
            # Título para cada subplot con el nombre del mes
            axs[i-1].set_title(f'Month: {i}')
            
            # Formatear la fecha del eje X para que sea más legible
            axs[i-1].xaxis.set_major_locator(plt.MaxNLocator(3))
            plt.setp(axs[i-1].get_xticklabels(), rotation=45, ha="right")

        # Ajustar el layout para evitar superposición de etiquetas
        plt.tight_layout()
        
        # Verificamos si el directorio de salida existe, si no, lo creamos
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Guardar la figura en el directorio de salida
        output_path = f"{output_dir}/{variable}_series_temporales_por_mes_vertical.png"
        plt.savefig(output_path)
        plt.close()


def reporte_calidad(file_paths, output_dir, fecha_inicio=None, fecha_fin=None):
    carpeta_datos_crudos = os.path.join(output_dir, "datos_crudos")
    carpeta_datos_procesados = os.path.join(output_dir, "datos_procesados")
    carpeta_reportes = os.path.join(output_dir, "reportes")
    carpeta_plots = os.path.join(output_dir, "plots")

    os.makedirs(carpeta_datos_crudos, exist_ok=True)
    os.makedirs(carpeta_datos_procesados, exist_ok=True)
    os.makedirs(carpeta_reportes, exist_ok=True)
    os.makedirs(carpeta_plots, exist_ok=True)

    variables = ['Precip', 'Evap', 'TMax', 'TMin']

    for file_path in tqdm (file_paths, desc="Procesando archivos"):
        
        df = extraer_datos_climatologicos(file_path)
        if df.empty:
            print(f"No se encontraron datos en {file_path}.")
            continue

        destino_crudo = os.path.join(carpeta_datos_crudos, os.path.basename(file_path))
        copiar_archivo(file_path, destino_crudo)

        df = ajustar_inconsistencias(df)
        df['Fecha'] = pd.to_datetime(df['Fecha'])
        df['Año'] = df['Fecha'].dt.year
        df['Mes'] = df['Fecha'].dt.month

        meses = pd.DataFrame({'Mes': range(1, 13)})
        df = pd.merge(meses, df, on='Mes', how='left')
        
        if fecha_inicio and fecha_fin:
            df = df[(df['Fecha'] >= fecha_inicio) & (df['Fecha'] <= fecha_fin)]

        base_name = os.path.splitext(os.path.basename(file_path))[0]
        subcarpeta_boxplots = os.path.join(carpeta_plots, base_name, "boxplots_outliers")
        os.makedirs(subcarpeta_boxplots, exist_ok=True)
        
        output_histogramas_path = os.path.join(subcarpeta_boxplots, f"histogramas_redondeo_{base_name}.png")
        generar_histogramas_redondeo(df, variables, output_histogramas_path)

        output_path = os.path.join(subcarpeta_boxplots, f"boxplots_{base_name}.png")
        generar_boxplot(df, variables, output_path)

        output_path_series = os.path.join(subcarpeta_boxplots, f"Serie_{base_name}.png")
        generar_series(df, output_path_series)

        carpeta_procesados_mensuales = os.path.join(output_dir, "procesados_mensuales")
        os.makedirs(carpeta_procesados_mensuales, exist_ok=True)

        df.to_csv(os.path.join(carpeta_datos_procesados, f"{base_name}_procesados.csv"), index=False)

        df = pd.read_csv(os.path.join(carpeta_datos_procesados, f"{base_name}_procesados.csv"))
        archivo_nombre = os.path.splitext(os.path.basename(file_path))[0]
        generar_analisis_mensual(df, archivo_nombre, carpeta_procesados_mensuales)

        anomalias_especificas = contar_anomalias_especificas(df)
        maximo_datos_nulos = encontrar_maximo_datos_nulos(df)

        report_path = os.path.join(carpeta_reportes, f"reporte_calidad_{base_name}.txt")
        with open(report_path, 'w', encoding='utf-8') as report_file:
            report_file.write(f"Reporte de Calidad de Datos - {datetime.now().strftime('%Y-%m-%d')}\n")
            report_file.write(f"Archivo Analizado: {os.path.basename(file_path)}\n\n")
            report_file.write("Este reporte de calidad presenta el porcentaje de completitud de datos por mes y algunas de las principales anomalías de datos presentes. Se producen diferentes carpetas que almacenan gráficos sobre la calidad de datos, una copia de los datos originales y un archivo .csv que contiene los datos, excluyendo fehcas fuplicadas, casos donde TMAX > TMIN, registros con precipitaciones negativas y temperaturas superiores a 100 ºC y menores a -100 ºC"
                             )
            completitud_por_mes = df.groupby('Mes').apply(lambda x: x.notnull().mean() * 100)
            completitud_transpuesta = completitud_por_mes.transpose()
            report_file.write("Completitud de Datos por Mes y Variable:\n")
            report_file.write(f"{completitud_transpuesta}\n\n")
            report_file.write("Boxplots Generados:\n")
            report_file.write(f"Todos los boxplots se han generado en un único archivo: Ver boxplot en {output_path}\n")
            report_file.write("\nHistogramas de Análisis de Redondeo Generados:\n")
            report_file.write(f"Todos los histogramas de análisis de redondeo se han generado en un único archivo: Ver histogramas en {output_histogramas_path}\n")
            report_file.write("\nAnálisis de Anomalías Específicas:\n")
            report_file.write("Variable\tDatos Iguales Consecutivos\tSaltos de Datos Incompletos\tMáximo de Datos Nulos Consecutivos\n")
            for var in variables:
                report_file.write(f"{var}\t{anomalias_especificas[var]['datos_iguales_consecutivos']}\t{anomalias_especificas[var]['saltos_incompletos']}\t{maximo_datos_nulos[var]}\n")

        print(f"Reporte de calidad generado con éxito: {report_path}")


def pr(folder_path, metadata_path, output_folder):
    # Cargar los metadatos
    metadata_df = pd.read_csv(metadata_path)
    
    # Crear una lista para guardar los datos
    data_frames = []
    
    # Leer todos los archivos en la carpeta
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            temp_df = pd.read_csv(file_path)
            # Filtrar por año
            temp_df = temp_df[temp_df['Año'] >= 2000]
            data_frames.append(temp_df)
    
    # Concatenar todos los dataframes
    full_data = pd.concat(data_frames, ignore_index=True)
    
    # Filtrar los datos de las estaciones con los metadatos
    full_data = full_data.merge(metadata_df, left_on='estacion_id', right_on='estacion', how='left')
    
    # Calcular estadísticas mensuales
    monthly_stats = full_data.groupby(['Nombre_Estacion', 'Mes']).agg({
        'Precip_Acumulada': 'mean',
        'Temp_Media': 'mean',
        'TMax_Media': 'mean',
        'TMin_Media': 'mean'
    }).pivot_table(index='Nombre_Estacion', columns='Mes', values=['Precip_Acumulada', 'Temp_Media', 'TMax_Media', 'TMin_Media'])
    
    # Aplanar MultiIndex en columnas
    monthly_stats.columns = ['_'.join(map(str, col)).strip() for col in monthly_stats.columns.values]
    monthly_stats.reset_index(inplace=True)
    
    # Guardar la tabla resultante
    output_path = os.path.join(output_folder, 'estadisticas_mensuales.csv')
    monthly_stats.to_csv(output_path, index=False)
    
    # Crear tabla de cobertura de datos
    data_coverage = full_data.groupby('estacion_id').agg({
        'Año': ['min', 'max'],
        'Precip_Acumulada': 'count',
        'Temp_Media': 'count',
        'TMax_Media': 'count',
        'TMin_Media': 'count'
    }).reset_index()
    data_coverage.columns = ['estacion', 'Año_inicial', 'Año_final', 'Datos_Precip', 'Datos_Temp_Media', 'Datos_TMax_Media', 'Datos_TMin_Media']
    
    # Guardar la tabla de cobertura
    coverage_path = os.path.join(output_folder, 'cobertura_datos.csv')
    data_coverage.to_csv(coverage_label, index=False)

    return monthly_stats, data_coverage


from sklearn.metrics import r2_score

def plot_combined_pp_comparison(station_filename):
    folder_path = '/content/RecNat-Py/Varios/CHIRPSvsEM_CONAGUA/precip_merged_data/'
    file_path = folder_path + station_filename
    
    try:
        data = pd.read_csv(file_path)
        data['Fecha'] = pd.to_datetime(data['Fecha'])
        
        # Limpieza de datos: Eliminar filas con valores nulos o infinitos
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(subset=['pp_em', 'pp_chirps'], inplace=True)
        
        monthly_data = data.groupby(pd.Grouper(key='Fecha', freq='M')).sum()

        fig, ax = plt.subplots(1, 2, figsize=(20, 8))
        
        # Diario
        ax[0].scatter(data['pp_em'], data['pp_chirps'], alpha=0.6, edgecolors='w')
        coef = np.polyfit(data['pp_em'], data['pp_chirps'], 1)
        poly1d_fn = np.poly1d(coef)
        r_squared = r2_score(data['pp_chirps'], poly1d_fn(data['pp_em']))
        ax[0].plot(data['pp_em'], poly1d_fn(data['pp_em']), color="red")
        ax[0].text(0.05, 0.95, f'R² = {r_squared:.2f}', transform=ax[0].transAxes, fontsize=12, verticalalignment='top')
        ax[0].set_title(f"{station_filename} - Comparación Diaria")
        ax[0].set_xlabel('pp_em (mm)')
        ax[0].set_ylabel('pp_chirps (mm)')
        
        # Mensual
        ax[1].scatter(monthly_data['pp_em'], monthly_data['pp_chirps'], alpha=0.6, edgecolors='w')
        coef_m = np.polyfit(monthly_data['pp_em'], monthly_data['pp_chirps'], 1)
        poly1d_fn_m = np.poly1d(coef_m)
        r_squared_m = r2_score(monthly_data['pp_chirps'], poly1d_fn_m(monthly_data['pp_em']))
        ax[1].plot(monthly_data['pp_em'], poly1d_fn_m(monthly_data['pp_em']), color="red")
        ax[1].text(0.05, 0.95, f'R² = {r_squared_m:.2f}', transform=ax[1].transAxes, fontsize=12, verticalalignment='top')
        ax[1].set_title(f"{station_filename} - Comparación Mensual")
        ax[1].set_xlabel('Suma Mensual pp_em (mm)')
        ax[1].set_ylabel('Suma Mensual pp_chirps (mm)')
        
        plt.show()
    
    except FileNotFoundError:
        print("El archivo especificado no se encontró en la carpeta designada.")
    except Exception as e:
        print(f"Ocurrió un error al procesar los datos: {e}")