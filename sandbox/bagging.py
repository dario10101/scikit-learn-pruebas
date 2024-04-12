import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    dt_heart = pd.read_csv('./data/heart.csv')
    print(dt_heart['target'].describe())

    # inplace=True no hace una copia, axis=1 se va por las columnas
    x = dt_heart.drop(['target'], axis=1)
    y = dt_heart['target']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35)

    knn_class = KNeighborsClassifier().fit(x_train, y_train)
    knn_prediction = knn_class.predict(x_test)
    print('KNN basico:')
    print(accuracy_score(knn_prediction, y_test))

    bag_class = BaggingClassifier(estimator=KNeighborsClassifier(), n_estimators=50).fit(x_train, y_train)
    bag_prediction = bag_class.predict(x_test)
    print('Bagging:')
    print(accuracy_score(bag_prediction, y_test))