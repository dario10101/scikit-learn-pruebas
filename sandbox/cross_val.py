import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import cross_val_score, KFold

if __name__ == "__main__":

    dataset = pd.read_csv('./data/felicidad.csv')

    X = dataset.drop(['country', 'score'], axis=1)
    y = dataset['score']

    model = DecisionTreeRegressor()

    # Cambiar .fit por cross_val_score
    # cross_val_score: Configuración rapida por defecto
    score = cross_val_score(
        model,
        X, 
        y, 
        cv= 3, # Cantidad de pliegues
        scoring='neg_mean_squared_error' # usar error medio para optimizar
    )
    print(np.abs(np.mean(score)))

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    i = 0
    for train, test in kf.split(dataset):
        print(f'Conjunto {i} de folders ==============')
        print('Train:')
        print(train)
        print('Test:')
        print(test)
        i = i + 1

    #implementacion_cross_validation

