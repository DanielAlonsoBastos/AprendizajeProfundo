# MTGNNe: Una extensión a la arquitectura MTGNN para el pronóstico de series de tiempo multivariantes utilizando Graph Neural Networks (GNN)

Daniel Alejandro Alonso Bastos

Proyecto final del curso Aprendizaje Profundo 2023-I / IIMAS / Dr. Gibran

En el repositorio [MTGNNe](https://github.com/DanielAlonsoBastos/AprendizajeProfundo/tree/main/mtgnne) se encuentra todo lo referente a la implementación de la arquitectura MTGNNe.

La arquitectura MTGNNe está motivada en extender el _framework_ propuesto en el trabajo [Connecting the Dots: Multivariate Time Series Forecasting with Graph Neuronal Networks](https://paperswithcode.com/paper/connecting-the-dots-multivariate-time-series) al aplicar mecanismos de atención cuando un nodo _i_ agrega información de sus nodos vecinos _j_ en una GCN, y también agregando mecanismos de atención en el paso de información entre las capas de la GCN. Además, se implementa una librería para realizar optimización de hiperparámetros.

![propuesta](https://user-images.githubusercontent.com/32237029/205446600-d8a44294-6704-43f5-8d38-a24ad39df171.png)

## Descripción de directorios

* En el directorio `experiments/exchange_rate` se encuentran una serie de carpetas, cada carpeta corresponde a las arquitecturas probadas, i.e., MTGNN, MTGAN, MTGNNAH, MTGANAH, MTGNN-Opt. Dentro de cada carpeta, en el proceso de experimentación se guardan los mejores modelos y la gráfica de pérdidas del conjunto de entrenamiento y validación para cada ejecución, i.e., si para cada experimento se ejecutan 10 veces cada modelo, en cada carpeta existirán 10 gráficas y 10 modelos. Solamente se guardaron las gráficas en el repositorio por temas de capacidad de memoria.

![experimentos](https://user-images.githubusercontent.com/32237029/205446587-3db1b5e5-a89d-48c5-b8a2-d8607bde6e1d.png)
  
* En el directorio `notebooks` se encuentran los notebooks y archivos `.py` que contienen las arquitecturas, a continuación se describen cada uno a alto nivel:
  * `1-mtgnne-complete.ipynb` es una libreta que describe de manera detallada cada paso: descargar y descomprimir la información, generar un DataLoader, descripción de las arquitecturas implementadas y sus parámetros, validación e impresión de información de las arquitecturas, entrenamiento y validación de resultados. Nótese que es un notebook introductorio que tiene como objetivo mostrar el flujo completo, por lo que, el entrenamiento no es exhaustivo, i.e., se utilizaron pocas épocas de entrenamiento, y por lo tanto los resultados son meramente ilustrativos.
  
  * `2-mtgnne-experiments.ipynb` es una libreta que tiene como objetivo realizar los experimentos, i.e., ejecuta el flujo completo 10 veces por cada arquitectura y regresa el promedio y desviación estándar de las métricas y el tiempo de ejecución. Esta libreta implementa todo de la libreta anterior, por lo que no se encuetran detallada en varias de las funciones, para ello referirse a la libreta `1-mtgnne-complete.ipynb`, pero si agrega un par de funciones que, una es para realizar la ejecución del pipeline de entrenamiento completo y la segunda es para realizar los experimentos.
  
  * `3-mtgnne-opt0.ipynb` es una libreta que tiene como objetivo emplear la librería Optuna para optimizar los hiperparámetros de las arquitecturas. En esta libreta se optimizan 10 hiperparámetros, por lo que se define el espacio de búsqueda de cada parámetro, y después se define una función objetivo para que Optuna minimice el RMSE del conjunto de validación.
  
  * `3-mtgnne-opt1.ipynb` es una libreta similar a `3-mtgnne-opt0.ipynb`, pero con la diferencia de que ahora se busca optimizar una menor cantidad de hiperparámetros basándose en la importancia de los mismos en el ejercicio de optimización anterior, dado que en el estudio de optimización anterior se obtuvieron peores resultados que MTGNN y MTGNNAH, como hipótesis se piensa que el espacio de búsqueda es muy grande, y que no se le dio el tiempo adecuado para encontrar un mejor mínimo.
  
  * `MTGNN.py` en este archivo, se encuentra la arquitectura original propuesta en el artículo [Connecting the Dots: Multivariate Time Series Forecasting with Graph Neuronal Networks](https://paperswithcode.com/paper/connecting-the-dots-multivariate-time-series). Para utilizar dicha arquitectura basta con `from MTGNN import MTGNN`
  
  * `MTGNNe.py` en este archivo se realizaron las modificaciones propuestas para extender la arquitectura MTGNN, para ello se implementó la clase `GraphAttentionLayer` y se modificó la clase `MixProp`, esto con el objetivo de introducir los mecanismos de atención en el vecindario de un nodo _i_, y en el paso de información entre las capas de la GCN. La arquitectura propuesta se encuentra en la clase `MTGNNe`. Además se incorporaron un par de clases que permiten la optimización de hiperparámetros `MTGNNe_Opt` que se utilizan en el notebook `3-mtgnne-opt0.ipynb` y `MTGNNe_Opt2` que se utilizan en el notebook `3-mtgnne-opt1.ipynb`.
  ```
  from MTGNNe import MTGNNe
  from MTGNNe import MTGNNe_Opt
  from MTGNNe import MTGNNe_Opt2
  ```
* En el directorio `results-experiments` se encuentra el archivo `results.csv` en donde están guardadas las métricas de cada iteración de cada modelo en el proceso de experimentación.
* `20221202_Connecting The Dots - MTGNNe.pdf` es la presentación expuesta en clase.

## Instalación de paqueterías

```
! pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113 
# inspeccionar arquitecturas
! pip install torchinfo
# barras de progreso
! pip install tqdm
# optuna
! pip install optuna
# visualizacion de datos -> plotly
! pip install plotly
```

Nota: Las siguientes librerías son para utilizar [PyTorch Geometric Temporal](https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/root.html), que tiene cargado los [modelos](https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/root.html) ST-GNN, incluido el de [MTGNN](https://pytorch-geometric-temporal.readthedocs.io/en/latest/_modules/torch_geometric_temporal/nn/attention/mtgnn.html#MTGNN), **pero no es necesario instalarlo** dado que todo el proyecto está realizado con PyTorch y el modelo MTGNN se tiene en el archivo `MTGNN.py` en el directorio de este notebook. Además la instalación es tardada y a veces complicada.

```
! pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
! pip install torch-sparse -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
! pip install torch-geometric
! pip install torch-geometric-temporal 
```
