import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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
