import seaborn as sns
import matplotlib.pyplot as plt

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
    sbplots_x_row = 2
    sbplots_x_col = (len(variables) + 1) // 2

    fig, axs = plt.subplots(sbplots_x_col, sbplots_x_row, figsize=(xi,yi))
    axs = axs.ravel() 
    for i, var in enumerate(variables):
        sns.countplot(data=df, x=var, hue=hue, ax=axs[i])
        axs[i].set_title(f'Countplot de {var}')
        axs[i].set_xticklabels(axs[i].get_xticklabels(), rotation=90)  # Girar las etiquetas del eje x

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
    sbplots_x_row = 2
    sbplots_x_col = (len(variables) + 1) // 2

    fig, axs = plt.subplots(sbplots_x_col, sbplots_x_row, figsize=(xi,yi))
    axs = axs.ravel() 
    for i, var in enumerate(variables):
        sns.histplot(data=df, x=var, ax=axs[i])
        axs[i].set_title(f'Countplot de {var}')
        axs[i].set_xticklabels(axs[i].get_xticklabels(), rotation=90)  # Girar las etiquetas del eje x

    plt.tight_layout()
    return plt.show()    

    
def distribucion_porcentual(df,cols):
     
     """
    Permite conocer la distribución porcentual de una variable.

    Parámetros:
    df: dataframe donde se encuentras las variables.
    cols: columna del dataframe.
    

    Retorna:
    Graficos en dos columnas diferentes.
    """
     for colname in cols:
        valor_counts = df[colname].value_counts('%')*100
        print(f"Distribución porcentual de la variable '{colname}':")
        print(valor_counts)

def validador_high_season(fecha):
     
    """
    Función que permite reconocer si es o no high_season


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

def boxplot(var_bx):
    """
    Grafico Boxplot

    Parametros:
    var_bx = Variable a graficar

    Retorna: 
    Grafico de boxplot a variable sele
    """
    sns.boxplot(data=var_bx)

    plt.title('Boxplot min_diff')
    plt.ylabel('Minutos')

    plt.show()

    
