import pandas as pd

import src.classification as cls
import src.regression as reg

DATASETS = {
    "heart": {
        "path": "./data/heart.csv",
        "target": "target"
    },
    "candy": {
        "path": "./data/candy.csv",
        "target": "winpercent",
        "delete_columns": ['competitorname']
    },
    "felicidad": {
        "path": "./data/felicidad.csv",
        "target": "score",
        "delete_columns": ['country']
    },
    "felicidad_corrupt": {
        "path": "./data/felicidad_corrupt.csv",
        "target": "score",
        "delete_columns": ['country']
    }
}

CLASSIFICATION_DATASETS = {
    "heart": {
        "path": "./data/heart.csv",
        "target": "target"
    }
}

REGRESSION_DATASETS = {
    "heart": {
        "path": "./data/heart.csv",
        "target": "target",
        "is_categorical": True
    },
    "felicidad": {
        "path": "./data/felicidad.csv",
        "target": "score",
        "delete_columns": ['country'],
        "is_categorical": False
    },
    "felicidad_corrupt": {
        "path": "./data/felicidad_corrupt.csv",
        "target": "score",
        "delete_columns": ['country'],
        "is_categorical": False
    }    
}


def classification(datasets: dict):
    """"""
    print('')
    print('='*100)
    print('RUNNING CLASSIFICATION ALGOTITHMS...')

    for key in datasets:
        dataset_dict = datasets[key]  
        path = dataset_dict['path']
        target = dataset_dict['target']
        delete_columns = dataset_dict['delete_columns'] if 'delete_columns' in dataset_dict else []

        # read data
        dataset = pd.read_csv(path)

        # Drop categorical columns
        dataset = dataset.drop(delete_columns, axis=1, inplace=False)

        show_statistics(dataset, key, path, target)

        x = dataset.drop([target], axis=1)
        y = dataset[target]
        
        cls.run(x, y)


def regression(datasets: dict):
    print('')
    print('='*100)
    print('RUNNING REGRESSION ALGOTITHMS...')

    for key in datasets:
        dataset_dict = datasets[key]  
        path = dataset_dict['path']
        target = dataset_dict['target']
        is_categorical = dataset_dict['is_categorical']
        delete_columns = dataset_dict['delete_columns'] if 'delete_columns' in dataset_dict else []

        # read data
        dataset = pd.read_csv(path)

        # Drop categorical columns
        dataset = dataset.drop(delete_columns, axis=1, inplace=False)

        show_statistics(dataset, key, path, target)

        x = dataset.drop([target], axis=1)
        y = dataset[target]
        
        reg.run(x, y, is_categorical)


def show_statistics(dataset, name, path, target):
    print('='*100)
    print(f'Processing dataset: {name}\nPath: {path} \nTarget variable: {target}\nShape: {dataset.shape}')
    print('Show:')
    print(dataset.head(5))


if __name__ == "__main__":
    classification(CLASSIFICATION_DATASETS)
    regression(REGRESSION_DATASETS)

    # TODO
    #clustering(CLUSTERING_DATASETS)
    