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

# Valores de configuración para la predicción
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
