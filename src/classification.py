from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def run(x, y):
    """ Runs classification algorithms
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35)
    
    gradient_boosting(x_train, x_test, y_train, y_test)
    k_neighbors(x_train, x_test, y_train, y_test)
    linear_support_vector(x_train, x_test, y_train, y_test)
    support_vector(x_train, x_test, y_train, y_test)
    linear_stochastic_gradient(x_train, x_test, y_train, y_test)
    decision_tree(x_train, x_test, y_train, y_test)
    bagging1(x_train, x_test, y_train, y_test)    


def gradient_boosting(x_train, x_test, y_train, y_test):
    # TODO que es n_estimators y optimizarlo
    n_estimators=50
    boost = GradientBoostingClassifier(n_estimators = n_estimators).fit(x_train, y_train)
    boost_pred = boost.predict(x_test)

    print(f'GradientBoostingClassifier:')
    print(f' -> n_estimators: {n_estimators}')
    print(f' -> accuracy: {accuracy_score(boost_pred, y_test)}')


def k_neighbors(x_train, x_test, y_train, y_test):
    # TODO Ver posibles parametros de KNeighborsClassifier y optimizarlos
    knn_class = KNeighborsClassifier().fit(x_train, y_train)
    knn_prediction = knn_class.predict(x_test)

    print(f'KNeighborsClassifier (KNN):')
    print(f' -> accuracy: {accuracy_score(knn_prediction, y_test)}')


def linear_support_vector(x_train, x_test, y_train, y_test):
    # TODO que es dual=False y optimizarlo
    dual=False
    model = LinearSVC(dual = dual).fit(x_train, y_train)
    pred = model.predict(x_test)

    print(f'LinearSVC:')
    print(f' -> dual: {dual}')
    print(f' -> accuracy: {accuracy_score(pred, y_test)}')


def support_vector(x_train, x_test, y_train, y_test):
    # TODO Ver posibles parametros de SVC y optimizarlos
    model = SVC().fit(x_train, y_train)
    pred = model.predict(x_test)

    print(f'SVC:')
    print(f' -> accuracy: {accuracy_score(pred, y_test)}')


def linear_stochastic_gradient(x_train, x_test, y_train, y_test):
    # TODO Ver posibles parametros de SGDClassifier y optimizarlos
    model = SGDClassifier().fit(x_train, y_train)
    pred = model.predict(x_test)

    print(f'SGDClassifier:')
    print(f' -> accuracy: {accuracy_score(pred, y_test)}')


def decision_tree(x_train, x_test, y_train, y_test):
    # TODO Ver posibles parametros de SGDClassifier y optimizarlos
    model = DecisionTreeClassifier().fit(x_train, y_train)
    pred = model.predict(x_test)

    print(f'DecisionTreeClassifier:')
    print(f' -> accuracy: {accuracy_score(pred, y_test)}')


def bagging1(x_train, x_test, y_train, y_test):
    classifier = {
        'KNeighbors': KNeighborsClassifier(),
        'LinearSCV': LinearSVC(dual=False),
        'SVC': SVC(),
        'SGDC': SGDClassifier(),
        'DecisionTree': DecisionTreeClassifier()
    }

    for name, estimator in classifier.items():
        # TODO que es n_estimators y optimizarlo
        bag_class = BaggingClassifier(estimator=estimator, n_estimators=5).fit(x_train, y_train)
        bag_pred = bag_class.predict(x_test)

        print(f'BaggingClassifier with {name}:')
        print(f' -> accuracy: {accuracy_score(bag_pred, y_test)}')


