# Evaluación de la Predicción en RLM
# Instalación de paquetes
# install.packages("caret", "ggplot2", "Kknn")

# Especificando el directorio de trabajo
dir <- "/path/project_directory/"
setwd(dir)

# Cargamos el conjunto de datos a la variable dataset
path_dataset <- "datasets/datos2.csv"
dataset <- read.csv(path_dataset)

# Analizando estructura del dataset
str(dataset)
head(dataset)
summary(dataset)
# 26 observaciones con 5 variables.

# Valores de configuración para la partición de datos
test_seed <- 240388
data_train_p <- 0.75
data_train_times <- 1

### -----------------------------------------------------------------
### Armando los conjuntos de datos para el entrenamiento y validación
### -----------------------------------------------------------------

library(caret)
set.seed(test_seed)
# Creando la partición de los datos.
data_partition <- createDataPartition(y = dataset$tejados, p = data_train_p, list = FALSE, times = data_train_times)

# creando conjunto de datos para entrenamiento
data_train <- dataset[data_partition, ]
# 22 observaciones

# creando conjunto de datos para validación
data_test  <- dataset[-data_partition, ]
# 4 observaciones

# Información del conjunto de datos de entrenamiento para la variable Tejados
summary(data_train$tejados)

# Información del conjunto de datos de validación para la variable Tejados
summary(data_test$tejados)

### -----------------------------------------------------
### Construcción de modelos y validación Cruzada
### -----------------------------------------------------

crossModel <- tejados~gastos+clientes+marcas+potencial

# Variables de configuración
repetitions <- 5
k <- 10
metric <- "RMSE"

## Construcción de modelos

### -----------------------------------------------------
### Modelo Lineal Normal
### -----------------------------------------------------
set.seed(test_seed)
normal_linear_model<-train(crossModel,data = data_train,method ="glm", trControl = trainControl(method = "repeatedcv",number = k, repeats = repetitions, returnResamp = "all"))

# Información del Modelo Normal Lineal
normal_linear_model
summary(normal_linear_model)

# Resultado del Modelo Normal Lineal
normal_linear_model$results

# parameter     RMSE Rsquared      MAE   RMSESD RsquaredSD    MAESD
#     1      none 9.147033 0.992309 8.029442 4.118859 0.03105983 3.859554


normal_linear_model$resample

# Información del Error Cuadrático Medio (RMSE) - Modelo Normal Lineal
summary(normal_linear_model$resample$RMSE)
#   Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 1.024   6.509   9.119   9.147  12.509  18.797


### -----------------------------------------------------
### Modelo Gamma con enlace logarítmico
### -----------------------------------------------------

set.seed(test_seed)
gamma_model<-train(crossModel,data = data_train,method ="glm", family = Gamma(link=log),trControl = trainControl(method = "repeatedcv",number = k, repeats = repetitions, returnResamp = "all"))

# Información del Modelo Gamma
gamma_model
summary(gamma_model)

# Resultado del Modelo Gamma
gamma_model$results

#  parameter     RMSE  Rsquared      MAE   RMSESD RsquaredSD    MAESD
#      1      none 41.34139 0.9691626 33.37374 34.59123   0.120714 25.64806

gamma_model$resample

# Información del Error Cuadrático Medio (RMSE) - Modelo Gamma
summary(gamma_model$resample$RMSE)
#  Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 9.481  18.957  27.212  41.341  54.015 146.064


### -----------------------------------------------------
### Modelo Normal Inversa con enlace logaritmico
### -----------------------------------------------------

set.seed(test_seed)
normal_inverse_model<-train(crossModel,data = data_train,method ="glm", family = inverse.gaussian(link=log),trControl = trainControl(method = "repeatedcv",number = k, repeats = repetitions, returnResamp = "all"))

# Información del Modelo Normal Inversa
normal_inverse_model
summary(normal_inverse_model)

# Resultado del Modelo Normal Inversa
normal_inverse_model$results

# parameter     RMSE  Rsquared      MAE   RMSESD RsquaredSD  MAESD
#    1      none 71.43486 0.9526925 56.43195 69.43507   0.162222 49.277

normal_inverse_model$resample

# Información del Error Cuadrático Medio (RMSE) - Modelo Normal Inversa
summary(normal_inverse_model$resample$RMSE)
#  Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 4.837  26.297  52.455  71.435  78.669 271.215

### -----------------------------------------------------
### Modelo KNN
### -----------------------------------------------------

library(kknn)

hyperparameters <- expand.grid(kmax = seq(from = 1,to = 10,by = 1),distance = 2,kernel = c("optimal", "epanechnikov","gaussian"))

knn_train_control <- trainControl(method = "repeatedcv", number = k,repeats = repetitions, returnResamp = "final", verboseIter = FALSE)

set.seed(test_seed)
knn_model <- train(crossModel, data = data_train,method = "kknn",tuneGrid = hyperparameters,metric = metric,trControl = knn_train_control)

# Información del Modelo KNN
knn_model

# Resultado del Modelo KNN
knn_model$results

# Gráfica
ggplot(knn_model , highlight = TRUE) + scale_x_discrete(breaks = hyperparameters$kmax) +
  labs(title = "Gráfica. Raíz del Error Cuadrático Medio - Modelo KNN", x = "K") + theme_bw()
