
<h2>DESCRIPCIÓN GENERAL</h2>
Mini-Framework para procesamiento de diferentes DataSets, bien sea clasificación, regresión o clustering, pasando por diferentes algoritmos.

<h2>TO-DO</h2>

* Agregar mas algoritmos
* Agregar forma de optimizar parámetros
* Guardar resultados en una lista y luego mostrarlos de mejor a peor
* Agregar autosklearn https://www.itmastersmag.com/noticias-analisis/que-es-automated-machine-learning-la-proxima-generacion-de-inteligencia-artificial/

<h2>SCIKIT-LEARN:</h2>

Buena para clasificación:
* ej: es cancer o nó? la imagen pertenece a perro, gato o ave?

Buena para regresión:
* ej: Predecir el precio del dolar, donde dentro de una imagen, encontrar un objeto

Buena para Clustering:
* Descubrir subconjuntos de datos similares dentro de un dataset
* Encontrar valores que se salen del comportamiento global
* ej: identificar productos similaes en un sistema de recomendaciones (ej recomendaciones de netflix)

<h2>CONCEPTOS GENERALES</h2>

PCA: (Principal Component Analysis), es una técnica en aprendizaje automático utilizada para la reducción de la dimensionalidad y la extracción de características de datos

KERNEL: Función matemática que sirve para aumentar la dimensión de un conjunto de datos

REGULARIZACION: Penalizar variables irrelevantes para que tengan menor incidencia en el modelo
* mas sesgo, menos varianza 
* Tipos:
** L1 Lasso: Eliminar features
** L2 Ridge: Reducir impacto de ciertos Features
** Elastic Net: Combinación de las anteriores

REGRESIONES ROBUSTAS: Sirven para manejar datos atípicos
* RANSAC: Se selecciona varias muestras aleatoriamente y selecciona la que menos datos atípicos tenga, para entrenar el modelo 
* Huber Reggresor: Dismunuye la influencia de los datos atípicos en el modelo (usa el error absoluto con un umbral epsilon, default 1,35)

METODOS DE ENSAMBLE: Combinar dirferentes métodos de ML con diferentes configuraciones y aplicar un método para logras un concenso
* Estrategia de Bagging: particionar un dataset (con posibles datos repetidos), entrenar un modelo por cada particion y con los resultados, llegar a un concenso
** Ejemplo: Random Forest
* Estrategia de Boosting: Método secuencial, pasa los datos por diferentes modelos, 1x1, con la información de error entre cada modelo.

CLUSTERING: estrategias para agrupar datos parecidos entre si
- Util cuando no conocemos las etiquetas de clasificación
- Descubrir patrones ocultos
- Identificar datos atipicos
- Dos casos de uso:
** Sabemos cuandos grupos "k" queremos como resultado (kmeans, Spectral Clustering)
** Queremos que el algoritmo los descubra (Meanshift, Clustering jerarquico, DBScan)

<h2>TIPOS DE VALIDACION</h2>
- Hold-On: Dividir datos en entrenamiento y pruebas
- Validacion cruzada 8K-Folds): Cubrir todos los datos como entrenamiento y test haciendo el proceso varias veces
- Validacion cruzada (LOOCV): Proceso mas intenso, entrenar con todos los datos menos uno, e ir iterando

<h2>ALGORITMOS:</h2>

Clasificacion:
* GradientBoostingClassifier
* KNeighborsClassifier
* LinearSVC
* SGDClassifier
* DecisionTreeClassifier
* BaggingClassifier (para unir varios clasificadores)

Regresion:
* LinearRegression: Variable objetivo continua
* Lasso: Variable objetivo continua
* Ridge: Variable objetivo continua
* ElasticNet: Variable objetivo continua
* LogisticRegression: Variable objetivo categórica
* DecisionTreeRegressor
* RandomForestRegressor

Regresión (robustas):
* SVR
* RANSAC
* HUBER

Clustering:
* MeanShift
* MiniBatchKMeans

<h2>OPTIMIZACION PARAMETRICA</h2>

Manual: 
* es costosa y tediosa

GridSearchCV: 
* Es una forma organizada, exhaustiva y sistematica de probar todos los parametros que le digamos que tenga que probar, con los respectivos rangos de valores que le aportemos.
* Definir una o varias métricas que queremos optimizar.
* Identificar los posibles valores que pueden tener los parámetros.
* Crear un diccionario de parámetros.
* Usar Cross Validation y entrenar el modelo
* usar cuando se quiera realizar un estudio a fondo sobre las implicaciones de los parámetros.

RandomizedSearchCV: 
* Si no tenemos tanto tiempo para una prueba tan exhaustiva o queremos combinaciones aleatorias usaremos este metodo. 
* Es lo mismo que el caso anterior, pero busca de forma aleatoria los parametros y Scikit Learn selecciona los mejores de las combinaciones aleatorias que se hicieron.
* En este método, definimos escalas de valores para cada uno de los parámetros seleccionados, el sistema probará varias iteraciones (Configurables según los recursos) y mostrará la mejor combinación encontrada.
* Usar cuando se quiera explorar posibles optimizaciones.

<h2>LIBRERIAS</h2>

- numpy: operaciones sobre tensores(estructura de datos para representar y manipular datos multidimensionales)
- scipy: realizar cálculos y análisis científicos y matemáticos avanzados, incluyendo optimización, álgebra lineal, interpolación, estadísticas, procesamiento de señales, resolución de ecuaciones diferenciales, procesamiento de imágenes, y mucho más.
- joblib: facilitar la paralelización y el almacenamiento en caché de tareas, lo que acelera la ejecución de código y el manejo eficiente de datos en aplicaciones de aprendizaje automático y procesamiento intensivo
- pandas: manipular y analizar datos tabulares (como hojas de cálculo) de manera eficiente
- matplotlib: crear gráficos y visualizaciones de datos de manera sencilla y personalizable
- scikit-learn: realizar tareas de aprendizaje automático, como clasificación, regresión, clustering y más, de manera simple y efectiva