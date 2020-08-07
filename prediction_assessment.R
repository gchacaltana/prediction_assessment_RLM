# Evaluación de la Predicción en RLM
# Instalación de paquetes
# install.packages("caret", "ggplot2")

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
hyperparameters <- as.data.frame(1)

## Construcción de modelos

## Modelo Lineal Normal
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

## Modelo Gamma con enlace logarítmico
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
