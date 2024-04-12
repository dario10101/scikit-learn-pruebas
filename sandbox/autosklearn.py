import autosklearn.classification

# TODO Probar esta variante mas aotomatizada

cls = autosklearn.classification.AutoSklearnClassifier()

cls.fit(X_train, y_train)

predictions = cls.predict(X_test)