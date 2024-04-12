import pandas as pd
import sklearn

# Modelos lineales
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet

# Herramientas adicionales para cargar datos y medirlos
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


if __name__ == "__main__":
    dataset = pd.read_csv('./data/felicidad.csv')
    print(dataset.describe())

    X = dataset[['gdp', 'family', 'lifexp', 'freedom', 'corruption', 'generosity', 'dystopia']]
    Y = dataset[['score']]
    print(X.shape)
    print(Y.shape)

    # partir en trainig y test (25% para test)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

    # entrenar modelos de regularizacion ========================
    model_linear = LinearRegression().fit(x_train, y_train)
    y_predict_linear = model_linear.predict(x_test)
    
    model_lasso = Lasso(alpha=0.02).fit(x_train, y_train) # alpha: que tanta penalización
    y_predict_lasso = model_lasso.predict(x_test)

    model_ridge = Ridge(alpha=1).fit(x_train, y_train)
    y_predict_ridge = model_ridge.predict(x_test)

    model_elasticnet = ElasticNet(random_state=0, alpha=0.005).fit(x_train, y_train)
    y_predict_elasticnet = model_elasticnet.predict(x_test)
    # ===========================================================

    # Comparar los modelos de regularización
    linear_loss = mean_squared_error(y_test, y_predict_linear)
    print(f'Linear loss: {linear_loss}')
    lasso_loss = mean_squared_error(y_test, y_predict_lasso)
    print(f'Lasso loss: {lasso_loss}')
    ridge_loss = mean_squared_error(y_test, y_predict_ridge)
    print(f'Ridge loss: {ridge_loss}')
    elasticnet_loss = mean_squared_error(y_test, y_predict_elasticnet)
    print(f'Elastic Net loss: {elasticnet_loss}')

    # Que tanto se afectaron las columnas (peso)
    print("="*32)
    print('Coef LASSO')
    print(model_lasso.coef_)

    print('Coef RIDGE')
    print(model_ridge.coef_)

    print('Coef ELASTIC NET')
    print(model_elasticnet.coef_)
