"""
CLUSTERING, aprendizaje no supervisado con grupos fijos
"""
import pandas as pd

# Funciona mejor para PC con menos recursos
from sklearn.cluster import MiniBatchKMeans
import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == "__main__":
    dataset = pd.read_csv("./data/candy.csv")
    print(dataset.head(10))

    x = dataset.drop('competitorname', axis = 1)
    kmeans = MiniBatchKMeans(
        n_clusters=4, # k grupos de salida
        batch_size=8 # Pasar por n grupos los datos de entrenamiento
    ).fit(x)

    print(f"Total de centros: {len(kmeans.cluster_centers_)}")
    print("=" * 64)
    print(kmeans.predict(x))

    # Integrar dataset con la respuesta
    dataset['group'] = kmeans.predict(x)

    print(dataset)

    # Grafica
    print('Starts grapic')
    #sns.pairplot(dataset, hue='group')
    sns.pairplot(dataset[['sugarpercent','pricepercent','winpercent','group']], hue = 'group')
    plt.show()