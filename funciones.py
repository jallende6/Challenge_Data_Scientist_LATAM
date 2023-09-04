
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as TTS,cross_val_predict,GridSearchCV
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.model_selection import cross_val_score


#metricas
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.inspection import permutation_importance

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
    cols = (len(variables)+1) // 2

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
    cols = (len(variables)+1) // 2

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

def modelamiento(df,var,modelo,param_grid,cv=None,n_jobs=None,verbose=None):

    """
    Función que procesa diferentes modelos y entrega sus metricas respectivas

    Parámetros:
    df: Dataframe con el que se modelará.
    var: variable objetivo.
    modelo: Tipo de modelo a utilizar
    param_grid: Hiperparametros para utilizar en la grilla.
    cv: Cantidad de pliegues a usar en la validación cruzada
    n_jobs: Cantidad de núcleos del sistema para el procesamiento del modelo.
    verbose: Monitorear el progreso de un algoritmo de machine learning.

    Retorno:

    Metricas de los modelos predictivos, es decir, Precisión, recall, F1, accurancy, matriz de confusion y curva de roc.
    """
    X_train, X_test, y_train, y_test = TTS(df.drop(columns=[var]), df[var], test_size=0.33, random_state=2208)

    grilla = GridSearchCV(modelo, param_grid=param_grid, cv=cv, n_jobs=n_jobs, verbose=verbose)
    #y_LG_cv_predict = cross_val_predict(grilla, X_train, y_train, cv=cv, n_jobs=n_jobs, verbose=verbose)
    grilla.fit(X_train, y_train)
    y_predict = grilla.best_estimator_.predict(X_test)
    
    conf_matrix = confusion_matrix(y_test, y_predict)
    print('Mejores parámetros: ', grilla.best_params_)
    print('Mejor puntaje     : ', grilla.best_score_)
    print('Metricas a evaluar: \n',classification_report(y_test, y_predict))
    plt.figure(figsize=(8, 6))
    class_names = ["Negativo", "Positivo"]
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Matriz de Confusión')
    plt.show()
    
    fpr, tpr, thresholds = roc_curve(y_test, y_predict)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='Curva ROC (AUC = %0.2f)' % auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.title('Curva ROC')
    plt.legend(loc='lower right')
    plt.show()
    
    if isinstance(modelo, LogisticRegression):
        coeficientes = grilla.best_estimator_.coef_
        coeficientes_absolutos = abs(coeficientes)
        nombres_caracteristicas = X_train.columns
        coeficientes_df = pd.DataFrame({'Caracteristica': nombres_caracteristicas, 'Coeficiente_Absoluto': coeficientes_absolutos[0]})
        coeficientes_ordenados = coeficientes_df.sort_values(by='Coeficiente_Absoluto', ascending=False)
        nombres_coeficientes = coeficientes_ordenados['Caracteristica'].tolist()
        coeficientes_coef = coeficientes_ordenados['Coeficiente_Absoluto'].tolist()
        for nombre, coef in zip(nombres_coeficientes, coeficientes_coef):
            print(f'Característica: {nombre}, Coeficiente Absoluto: {coef}')
    
    else:
        importancia_caracteristicas = grilla.best_estimator_.feature_importances_
        nombres_caracteristicas = X_train.columns
        importancia_df = pd.DataFrame({'Caracteristica': nombres_caracteristicas, 'Importancia': importancia_caracteristicas})
        importancia_ordenada = importancia_df.sort_values(by='Importancia', ascending=False)
        nombres_importancia = importancia_ordenada['Caracteristica'].tolist()
        importancia_valores = importancia_ordenada['Importancia'].tolist()
        for nombre, importancia in zip(nombres_importancia, importancia_valores):
            print(f'Característica: {nombre}, Importancia: {importancia}')
