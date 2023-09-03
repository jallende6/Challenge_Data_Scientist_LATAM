def modelamiento(df,var):

    return X_train, X_test, y_train, y_test = TTS(df.drop(columns=[var]), df[var], test_size=0.33, random_state=2208)

def grilla(modelo,param_grid,cv=None,n_jobs=None,verbose=None,scoring=None)

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