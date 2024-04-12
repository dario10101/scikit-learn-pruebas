import pandas as pd

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor # metodo de ensamble

if __name__ == "__main__":

    dataset = pd.read_csv('./data/felicidad.csv')

    print(dataset)

    X = dataset.drop(['country', 'rank', 'score'], axis=1)
    y = dataset[['score']]

    reg = RandomForestRegressor()

    # grilla de parametros
    parametros = {
        'n_estimators' : range(4,16), # Cuantos arboles componen el bosque aleatorio
        'criterion' : ['friedman_mse', 'squared_error', 'poisson', 'absolute_error'], # Medida de calidad de los split del arbol
        'max_depth' : range(2,11) # Profundidad del arbol
    }

    rand_est = RandomizedSearchCV(
        reg, 
        parametros, # grilla de parametros
        n_iter=10, # 10 diferentes posibilidades para cada parametro
        cv=3, # 3 pliegues, 2 training y 1 de test (cross validation)
        scoring='neg_mean_absolute_error' # error absoluto minimo
    ).fit(X,y)

    print(rand_est.best_estimator_)
    print(rand_est.best_params_)
    print(rand_est.predict(X.loc[[0]]))

    #implmentacion_randomizedSearchCV