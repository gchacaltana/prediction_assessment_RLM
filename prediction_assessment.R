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

### -----------------------------------------------------
### Construcción del Modelo Lineal Normal
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

library(ggpubr)
p1 <- ggplot(data = normal_linear_model$resample, aes(x = RMSE)) +
  geom_density(alpha = 0.5, fill = "gray50") +
  geom_vline(xintercept = mean(normal_linear_model$resample$RMSE),
             linetype = "dashed") +
  theme_bw() 
p2 <- ggplot(data = normal_linear_model$resample, aes(x = 1, y = RMSE)) +
  geom_boxplot(outlier.shape = NA, alpha = 0.5, fill = "gray50") +
  geom_jitter(width = 0.05) +
  labs(x = "") +
  theme_bw() +
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank())
p3 <- ggplot(data = normal_linear_model$resample, aes(x = Rsquared)) +
  geom_density(alpha = 0.5, fill = "gray50") +
  geom_vline(xintercept = mean(normal_linear_model$resample$Rsquared),
             linetype = "dashed") +
  theme_bw() 
p4 <- ggplot(data = normal_linear_model$resample, aes(x = 1, y = Rsquared)) +
  geom_boxplot(outlier.shape = NA, alpha = 0.5, fill = "gray50") +
  geom_jitter(width = 0.05) +
  labs(x = "") +
  theme_bw() +
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank())
p5 <- ggplot(data = normal_linear_model$resample, aes(x = MAE)) +
  geom_density(alpha = 0.5, fill = "gray50") +
  geom_vline(xintercept = mean(normal_linear_model$resample$MAE),
             linetype = "dashed") +
  theme_bw() 
p6 <- ggplot(data = normal_linear_model$resample, aes(x = 1, y = MAE)) +
  geom_boxplot(outlier.shape = NA, alpha = 0.5, fill = "gray50") +
  geom_jitter(width = 0.05) +
  labs(x = "") +
  theme_bw() +
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank())

nlm_plot <- ggarrange(p1, p2, p3, p4, p5, p6, ncol = 2, nrow = 3)
nlm_plot <- annotate_figure(nlm_plot,top = text_grob("Evaluación del Modelo Lineal Normal", size = 15))
nlm_plot

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

p1 <- ggplot(data = gamma_model$resample, aes(x = RMSE)) +
  geom_density(alpha = 0.5, fill = "gray50") +
  geom_vline(xintercept = mean(gamma_model$resample$RMSE),
             linetype = "dashed") +
  theme_bw()

p2 <- ggplot(data = gamma_model$resample, aes(x = 1, y = RMSE)) +
  geom_boxplot(outlier.shape = NA, alpha = 0.5, fill = "gray50") +
  geom_jitter(width = 0.05) +
  labs(x = "") +
  theme_bw() +
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank())

p3 <- ggplot(data = gamma_model$resample, aes(x = Rsquared)) +
  geom_density(alpha = 0.5, fill = "gray50") +
  geom_vline(xintercept = mean(gamma_model$resample$Rsquared),
             linetype = "dashed") +
  theme_bw()

p4 <- ggplot(data = gamma_model$resample, aes(x = 1, y = Rsquared)) +
  geom_boxplot(outlier.shape = NA, alpha = 0.5, fill = "gray50") +
  geom_jitter(width = 0.05) +
  labs(x = "") +
  theme_bw() +
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank())

p5 <- ggplot(data = gamma_model$resample, aes(x = MAE)) +
  geom_density(alpha = 0.5, fill = "gray50") +
  geom_vline(xintercept = mean(gamma_model$resample$MAE),
             linetype = "dashed") +
  theme_bw()

p6 <- ggplot(data = gamma_model$resample, aes(x = 1, y = MAE)) +
  geom_boxplot(outlier.shape = NA, alpha = 0.5, fill = "gray50") +
  geom_jitter(width = 0.05) +
  labs(x = "") +
  theme_bw() +
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank())

gamma_plot <- ggarrange(p1, p2, p3, p4, p5, p6, ncol = 2, nrow = 3)
gamma_plot <- annotate_figure(gamma_plot, top = text_grob("Evaluación del Modelo Gamma (enlace logarítmico)", size = 15))
gamma_plot


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

## Gráficas de evaluación de la precisión
p1 <- ggplot(data = normal_inverse_model$resample, aes(x = RMSE)) +
  geom_density(alpha = 0.5, fill = "gray50") +
  geom_vline(xintercept = mean(normal_inverse_model$resample$RMSE),
             linetype = "dashed") +
  theme_bw() 
p2 <- ggplot(data = normal_inverse_model$resample, aes(x = 1, y = RMSE)) +
  geom_boxplot(outlier.shape = NA, alpha = 0.5, fill = "gray50") +
  geom_jitter(width = 0.05) +
  labs(x = "") +
  theme_bw() +
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank())
p3 <- ggplot(data = normal_inverse_model$resample, aes(x = Rsquared)) +
  geom_density(alpha = 0.5, fill = "gray50") +
  geom_vline(xintercept = mean(normal_inverse_model$resample$Rsquared),
             linetype = "dashed") +
  theme_bw() 
p4 <- ggplot(data = normal_inverse_model$resample, aes(x = 1, y = Rsquared)) +
  geom_boxplot(outlier.shape = NA, alpha = 0.5, fill = "gray50") +
  geom_jitter(width = 0.05) +
  labs(x = "") +
  theme_bw() +
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank())
p5 <- ggplot(data = normal_inverse_model$resample, aes(x = MAE)) +
  geom_density(alpha = 0.5, fill = "gray50") +
  geom_vline(xintercept = mean(normal_inverse_model$resample$MAE),
             linetype = "dashed") +
  theme_bw() 
p6 <- ggplot(data = normal_inverse_model$resample, aes(x = 1, y = MAE)) +
  geom_boxplot(outlier.shape = NA, alpha = 0.5, fill = "gray50") +
  geom_jitter(width = 0.05) +
  labs(x = "") +
  theme_bw() +
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank())

nim_plot <- ggarrange(p1, p2, p3, p4, p5, p6, ncol = 2, nrow = 3)
nim_plot <- annotate_figure(nim_plot,top = text_grob("Evaluación del Modelo Normal Inversa con enlace logarítmico", size = 15))
nim_plot

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
