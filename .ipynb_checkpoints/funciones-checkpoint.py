import seaborn as sns
import matplotlib.pyplot as plt

def countplot(df,variables,xi,yi):
    
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
        sns.countplot(data=df, x=var, ax=axs[i])
        axs[i].set_title(f'Countplot de {var}')
        axs[i].set_xticklabels(axs[i].get_xticklabels(), rotation=90)  # Girar las etiquetas del eje x

    plt.tight_layout()
    return plt.show()
    
def countplot_solo(df,var):
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
    plt.figure(figsize=(6, 4))  
    sns.countplot(data=df, x='high_season') 
    plt.title(f'Countplot de {var}')  
    plt.xlabel('Categoría')  
    plt.ylabel('Conteo')  
    return plt.show()
    
def distribucion_porcentual(cols):
    for colname in cols:
        valor_counts = df[colname].value_counts('%')*100
        print(f"Distribución porcentual de la variable '{colname}':")
        print(valor_counts)