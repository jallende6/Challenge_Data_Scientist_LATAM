import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def countplot(df,variables,xi,yi,hue=None):
    
    """
    Permite agregar diferentes variables para graficar countplot.

    Parámetros:
    df: dataframe donde se encuentras las variables.
    variables: Lista de variables del dataframe para graficar.
    xi: Ancho de los graficos
    yi: Largo de los graficos

    Retorna:
    Graficos en dos columnas diferentes.
    """
    filas = 2
    cols = (len(variables) + 1) // 2

    fig, axs = plt.subplots(cols, filas, figsize=(xi,yi))
    axs = axs.ravel()
    day_order = [ 'Lunes', 'Martes', 'Miercoles', 'Jueves','Viernes', 'Sabado',
       'Viernes','Domingo']     
    for i, var in enumerate(variables):
        if var == 'DIANOM':
            sns.countplot(data=df, x=var, hue=hue, ax=axs[i],order=day_order)
            axs[i].set_title(f'Countplot de {var}')
            axs[i].set_xticklabels(axs[i].get_xticklabels(), rotation=90)  # Girar las etiquetas del eje x
            axs[i].set_ylabel('Frecuencia')
        else:
            sns.countplot(data=df, x=var, hue=hue, ax=axs[i])
            axs[i].set_title(f'Countplot de {var}')
            axs[i].set_xticklabels(axs[i].get_xticklabels(), rotation=90)  # Girar las etiquetas del eje x
            axs[i].set_ylabel('Frecuencia')

    plt.tight_layout()
    return plt.show()

def histplot(df,variables,xi,yi):
    
    """
    Permite agregar diferentes variables para graficar countplot.

    Parámetros:
    df: dataframe donde se encuentras las variables.
    variables: Lista de variables del dataframe para graficar.
    xi: Ancho de los graficos
    yi: Largo de los graficos

    Retorna:
    Graficos en dos columnas diferentes.
    """
    filas = 2
    cols = (len(variables) + 1) // 2
    
    fig, axs = plt.subplots(cols, filas, figsize=(xi,yi))
    axs = axs.ravel() 
    for i, var in enumerate(variables):
        sns.histplot(data=df, x=var, ax=axs[i])
        axs[i].set_title(f'Countplot de {var}')
        axs[i].set_xticklabels(axs[i].get_xticklabels(), rotation=90)  # Girar las etiquetas del eje x
        axs[i].set_ylabel('Frecuencia')
    plt.tight_layout()
    return plt.show()    

    
def distribucion_porcentual(df,cols):
     
     """
    Permite conocer la distribución porcentual de una variable de los primeros 10 datos.

    Parámetros:
    df: dataframe donde se encuentras las variables.
    cols: columna del dataframe.
    

    Retorna:
    Graficos en dos columnas diferentes.
    """
     for colname in cols:
        valor_counts = df[colname].value_counts('%')*100
        print(f"Distribución porcentual de la variable '{colname}':")
        print(valor_counts.head(10))

def validador_high_season(fecha):
     
    """
    Función que permite reconocer si es o no high_season. La lista es_high_season, son los parametros del mes, dia de inicio y fin (mes,dia_inicio,dia_fin)


    """
   
    es_high_season = [
        (12, 15, 31),  # Diciembre
        (1, 1, 31),     # Enero
        (2, 1, 31),     # Febrero
        (3, 1, 3),     # Marzo
        (7, 15, 31),   # Julio
        (9, 11, 30)]   # Septiembre
    for mes, dia_i, dia_f in es_high_season:
        if fecha.month == mes and dia_i <= fecha.day <= dia_f:
            return 1
    
    return 0

def boxplot(df,variables,xi,yi,var_y=None,hue=None):
    """
    Grafico Boxplot

    Parametros:

    df: dataframe donde se encuentras las variables.
    variables: Lista de variables del dataframe para graficar.
    xi: Ancho de los graficos
    yi: Largo de los graficos
    var_y: variable cuantitativa
    hue: variable binaria.

    Retorna: 
    Grafico de boxplot a variable sele
    """
    filas = 2
    cols = (len(variables) + 1) // 2

    fig, axs = plt.subplots(cols, filas, figsize=(xi,yi))
    axs = axs.ravel() 
    day_order = [ 'Lunes', 'Martes', 'Miercoles', 'Jueves','Viernes', 'Sabado',
       'Viernes','Domingo']
    for i, var in enumerate(variables):
        if var == 'DIANOM':
            sns.boxplot(data=df, x=var, y=var_y ,hue=hue, ax=axs[i],order=day_order)
            axs[i].set_title(f'Boxplot de {var} en minutos y diferenciado por la existencia de delay ')
            axs[i].set_xticklabels(axs[i].get_xticklabels(), rotation=90)  # Girar las etiquetas del eje x
        
        else:
            sns.boxplot(data=df, x=var, y=var_y ,hue=hue, ax=axs[i])
            axs[i].set_title(f'Boxplot de {var} en minutos y diferenciado por la existencia de delay ')
            axs[i].set_xticklabels(axs[i].get_xticklabels(), rotation=90)  # Girar las etiquetas del eje x

    plt.tight_layout()
    return plt.show()


def minutos_diff(dfs,var):
    """
    Histogramas para observar el comportamiento de la diferencia de minutos desde la hora inicio de la operacion de vuelo vs la agendada.
    
    Parametros:

    dfs: Lista de dataframes que filtran por salidas previas al horario acordado y post.
    var: variable en cuestión min_diff

    Retorna:

    Graficos de cada uno de los filtros realizados en los dataframe.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    titles = ['min_diff <= 0','min_diff', 'min_diff > 0']
    for i, df in enumerate(dfs):
        media = np.mean(df[var])

        sns.histplot(data=df, x=var, bins=45, kde=True, ax=axes[i])
        axes[i].axvline(x=media, color='red', linestyle='dashed', linewidth=2, label='Media')
        axes[i].axvline(x=0, color='green', linestyle='dashed', linewidth=2, label='Media')
        axes[i].set_title(f'Histograma de {titles[i]}')
        axes[i].set_xlabel('Minutos')
        axes[i].set_ylabel('Frecuencia')

    plt.tight_layout()
    return plt.show()
    
