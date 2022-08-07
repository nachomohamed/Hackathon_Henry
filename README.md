# House Prices - Advanced Regression Techniques

The first thing we do is import the dataset and the multiple libraries that we are going to use during this process.

When doing an analysis of the data we realize that there are several columns with null values, therefore we make the decision of the columns that will have more than 3% of null values will be eliminated.

Also with createDummies we will handle and replace the columns of categorical variables with corresponding numerical variables, that is, if the categorical variable has 4 different types, only 3 columns will be created on that category.

At the end we create a connection matrix to analyze which variables are more dependent on the price variable.

## Linear Regression

The first thing we are going to do is apply a linear regression model.

But we are not going to take all the columns, but we are going to select. On what number of columns the linear regression model gives us the best result. Therefore, we have made the different correlation values in a range and according to the correlation values we will be left with more or fewer columns.

Also in the same interaction process we are going to apply the StandardScaler to normalize the data and put them on a mean 0 and a deviation 1.

Obtaining the value of 0.1 as the best correlation value. We perform cross validation. Getting root_mean_squared_log_erro 0.66.
Understanding that we have found a good model. But that can be greatly improved we decided to use decision trees to refine the model.

## Decision Trees

Using the default values of the decision tree. And a cross validation of 5 separations we obtain a substantial improvement of the error that takes it to 0.20.

Even so, we consider that the model can still be greatly improved. So we decided to walk through many models and many GridSearchCV parameters. In this way to find the best model and approach the best parameters for that situation.

## Model Choice

For this case we will test the Linear Regression, SVR, Decision Tree, Random Forest, Ridge, GradientBoostingRegressor models.
For each model we have injured approximately 3 parameters. To be tested and in this way get closer to the best values.

For this case and for reasons of not being able to use a cross-validation due to having negative values ​​in the predictions and not being able to make the corresponding modifications. I decide to perform the analysis with the X and the Y. The predictions with respect to the prediction and the values ​​in Y. Knowing that this will not give me the best values ​​or real values. But considering that it will only be to choose the most suitable model to work.

After the whole process and observing the error values ​​that we obtain, I take the Random Forest and the Gradient Boosting Regressor as the best. Because of how similar both models are. He considered that the most robust model to work with is the Gradient Boosting Regressor model. And so he went on to work with that model and thus optimize its hyper parameters.


## Gradient Boosting Regressor and its hyperparameters

In this almost performing an iteration between different values of quantity of estimates. So that the model achieves its best results after several attempts. He concluded that the best value is 100 estimates. As the graph indicates

In this section we carry out the same process as in the previous section but working with the learning rate. Therefore, in this case we will carry out 2 interactions so that with values of the number of estimates close to the previous ones and with different values of the learning ratio. Let's get the best possible model for this situation.

In this way we obtain that the best number of n_estimators is 100 estimates. That the best learning ratio of the model is 0.23. In addition, we add the maximum depth level of the tree 6 and the minimum number of examples for a division 6, which we obtained from the iteration of the models with GridSearch.

## Final model and its training

Obtaining an RMSLE value of 0.1654. We proceed to train the model.

## Aggregating and working with test data

Now we carry out the import of the test data set and we carry out the corresponding transformations that were already explained above. Normalization and column selection.

With everything done by we proceed to make the prediction of the test values and then save them in a csv file.

## Final prediction and creation of the CSV

We uploaded the final prediction to Kaggle to compare the results and we arrived at Score: 0.17213, ranking 2985th in the competition


# House Prices - Advanced Regression Techniques

Lo primero que hacemos es importar el data set y las múltiples librerías que vamos está utilizando durante este proceso.

A la hora de hacer un análisis de los datos nos damos cuenta que hay varias columnas con valores nulos por lo tanto tomamos la decisión de las columnas que tengan un más de 3% de valores nulos serán eliminadas.

También con la createDummies manejaremos y reemplazaremos las columnas de variables categóricas por variables numéricas correspondientes, es decir si la variable categórica tienen 4 tipos distintos se crearán sólo 3 columnas sobre esa categoría.

Al final creamos una matriz de correlación para analizar qué variables son más dependientes de la variable de precio.

## Regresion Lineal

Lo primero que vamos a hacer es aplicar un modelo de regresión lineal.

Pero no vamos a tomar todas las columnas, sino que vamos a seleccionar. Sobre qué cantidad de columnas el modelo de regresión lineal nos arroja el mejor resultado. Por lo tanto hemos hecho en un rango los diferentes valores de correlación y de acuerdo a los valores de correlación nos vamos a quedar con más o menos columnas. 

También en el mismo proceso de interacción vamos a aplicar el StandardScaler para normalizar los datos y ponerlos sobre una media 0 y una desviacion 1.

Obteniendo el valor de 0.1 como mejor valor de correlación. Realizamo la validación cruzada. Obteniendo root_mean_squared_log_erro 0,66.
Entendiendo que hemos encontrado un buen modelo. Pero que pueden mejorarse ampliamente decidimos utilizar árboles de decisión para perfeccionar el modelo.

## Arboles de Desicion

Utilizando los valores por defecto del árbol de decisión. Y una validación cruzada de 5 separaciones obtenemos una mejora del error sustancial que lo lleva a 0,20.

Aún así consideramos que el modelo puede mejorarse ampliamente todavía. Por lo que decidimos hacer un recorrido sobre muchos modelos y sobre muchos parámetros GridSearchCV. De esta forma en contrar el mejor modelo y acercarnos al los mejores parámetros para dicha situación.

## Eleccion de modelo

Para este caso probaremos los modelos de Regresion Lineal, SVR, Arbol de desicion, Random Forest, Ridge, GradientBoostingRegressor.
Para cada modelo hemos lesionado 3 parámetros aproximadamente. Para ser testiados y de esta forma acercarnos a los mejores valores.

Para este caso y por motivos de no poder utilizar una validación cruzada por tener valores negativos en las predicciones y no poder hacer las modificaciones correspondientes. Decido realizar el análisis con el X y el Y. Las predicciones respecto a la predicción y los valores en Y. Sabiendo que esto no me entregará los mejores valores o valores reales. Pero considerando que solamente sera para elegir el modelo más adecuado para trabajar.

Después de todo el proceso y observando los valores de error que obtenemos tomo como los mejores a los del Random Forest y a los del Gradient Boosting Regressor. Por los similares que son ambos modelos. Consideró que el modelo más robusto para trabajar es el modelo de Gradient Boosting Regressor. Y por lo tanto pasó a trabajar con ese modelo y de esta forma optimizar sus hiper parámetros.

## Gradient Boosting Regressor y sus hiperparaemtros

En este casi realizando una interacion entre diferentes valores de cantidad de estimaciones. Para que el modelo consiga sus mejores resultados después de varios intentos. Concluyó que el mejor valor es de 100 estimaciones. Como indica el grafico

En esta sección realizamos el mismo proceso que en la sección anterior pero trabajando con la tasa de aprendizaje. Por lo tanto realizaremos en este caso 2 interacciones para que con valores de cantidad de estimaciones cercanos a los anteiores y con diferentes valores de de ratio de aprendizaje. Consigamos el mejor modelo posible para esta situación.

De esta forma obtenemos que la mejor cantidad de n_estimators son 100 estimaciones. Que el mejor ratio de aprendizaje del modelo es 0,23. Ademas sumamos el nivel maximo de profundidad del arbol 6 y la minima cantidad de ejemplos para una division 6, que obtubimos de la interacion de los modelos con GridSearch.

## Modelo final y su entrenamiento

Obteniendo un valor de RMSLE de 0,1654. Procedemos a entrenar el modelo.

## Agregando y trabajando con los datos de test

Ahora realizamos la importación del data set del test y le realizamos las transformaciones correspondientes que ya fueron explicadas anteriormente. La normalización y la selección de columnas.

Con todos realizado por procedemos a hacer la predicción de los valores de test para luego guardarlos en un archivo csv.

## Prediccion final y creacion del csv

Subimos la prediccion final a Kaggle para comparar los resultados y llegamos a Score: 0.17213 quedando en el puesto 2985 de la competicion