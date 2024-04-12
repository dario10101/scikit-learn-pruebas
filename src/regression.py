# Modelos lineales
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RANSACRegressor, HuberRegressor
from sklearn.svm import SVR

from sklearn.decomposition import KernelPCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error

def run(x, y, is_categorical:bool):
    # TODO Optimizar parametros    
    if is_categorical:
        run_categorical(x, y)
    else:
        run_continuous(x, y)


def run_categorical(x, y):
    # Normalizamos los datos  
    dt_features = StandardScaler().fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(dt_features, y, test_size=0.3, random_state=42) 
    logistic_regression(x_train, x_test, y_train, y_test)  
    

def run_continuous(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35)

    linear_regression(x_train, x_test, y_train, y_test)    
    lasso(x_train, x_test, y_train, y_test)
    ridge(x_train, x_test, y_train, y_test)
    elastic_net(x_train, x_test, y_train, y_test)
    robust_algorithms(x_train, x_test, y_train, y_test)


def linear_regression(x_train, x_test, y_train, y_test):
    model = LinearRegression().fit(x_train, y_train)
    pred = model.predict(x_test)

    print(f'LinearRegression:')
    print(f' -> mean_squared_error: {mean_squared_error(y_test, pred)}')


def logistic_regression(x_train, x_test, y_train, y_test):
    solver='lbfgs'

    #Aplicamos la función de kernel de tipo polinomial
    kpca = KernelPCA(n_components=4, kernel='poly')
    
    #Vamos a ajustar los datos
    kpca.fit(x_train)

    #Aplicamos el algoritmo a nuestros datos de prueba y de entrenamiento
    dt_train = kpca.transform(x_train)
    dt_test = kpca.transform(x_test)

    #Aplicamos la regresión logística un vez que reducimos su dimensionalidad
    logistic = LogisticRegression(solver=solver)

    #Entrenamos los datos
    logistic.fit(dt_train, y_train)

    print(f'LogisticRegression:')
    print(f' -> solver: {solver}')
    print(f' -> accuracy: {logistic.score(dt_test, y_test)}')


def lasso(x_train, x_test, y_train, y_test):
    alpha=0.02
    model = Lasso(alpha=alpha).fit(x_train, y_train) # alpha: que tanta penalización
    pred = model.predict(x_test)

    print(f'Lasso:')
    print(f' -> alpha: {alpha}')
    print(f' -> mean_squared_error: {mean_squared_error(y_test, pred)}')


def ridge(x_train, x_test, y_train, y_test):
    alpha=1
    model = Ridge(alpha=1).fit(x_train, y_train)
    pred = model.predict(x_test)

    print(f'Ridge:')
    print(f' -> alpha: {alpha}')
    print(f' -> mean_squared_error: {mean_squared_error(y_test, pred)}')


def elastic_net(x_train, x_test, y_train, y_test):
    solver='lbfgs'
    model = ElasticNet(random_state=0, alpha=0.005).fit(x_train, y_train)
    pred = model.predict(x_test)

    print(f'ElasticNet:')
    print(f' -> random_state: {solver}')
    print(f' -> alpha: {solver}')
    print(f' -> mean_squared_error: {mean_squared_error(y_test, pred)}')


def robust_algorithms(x_train, x_test, y_train, y_test):
    estimadores = {
        'SVR' : SVR(gamma= 'auto', C=1.0, epsilon=0.1),
        'RANSAC' : RANSACRegressor(),
        'HUBER' : HuberRegressor(epsilon=1.35)
    }

    for name, estimador in estimadores.items():
        estimador.fit(x_train, y_train)
        predictions = estimador.predict(x_test)

        print(f'Robust with {name}:')
        print(f' -> mean_squared_error: {mean_squared_error(y_test, predictions)}')