import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    dt_heart = pd.read_csv('./data/heart.csv')
    print(dt_heart['target'].describe())

    # inplace=True no hace una copia, axis=1 se va por las columnas
    x = dt_heart.drop(['target'], axis=1)
    y = dt_heart['target']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35)

    # n_estimators: numero de arboles de decision a usar
    boost = GradientBoostingClassifier(n_estimators=50).fit(x_train, y_train)
    boost_pred = boost.predict(x_test)

    print(f'accuracy: {accuracy_score(boost_pred, y_test)}')



