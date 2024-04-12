import pandas as pd

from sklearn.cluster import MeanShift
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    dataset = pd.read_csv("./data/candy.csv")
    print(dataset.head(10))

    x = dataset.drop('competitorname', axis = 1)
    meanshift = MeanShift().fit(x)

    print(meanshift.labels_)
    print(max(meanshift.labels_))
    print("="*64)
    print(meanshift.cluster_centers_)

    dataset['meanshift'] = meanshift.labels_
    print("="*64)

    print(dataset)

    # Grafica
    print('Starts grapic')
    #sns.pairplot(dataset, hue='meanshift')
    sns.pairplot(dataset[['sugarpercent','pricepercent','winpercent','meanshift']], hue = 'meanshift')
    plt.show()