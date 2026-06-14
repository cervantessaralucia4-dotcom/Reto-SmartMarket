# 📊 SmartMarket Pipeline Inteligente

## 🚀 Descripción del Proyecto

**SmartMarket Pipeline Inteligente** es un proyecto de Ingeniería de Datos y Machine Learning desarrollado como proyecto final de formación, cuyo objetivo es automatizar el procesamiento de información de productos mediante un pipeline ETL inteligente.

El sistema permite extraer datos desde archivos CSV, realizar procesos de limpieza y validación, ampliar conjuntos de datos mediante técnicas de Data Augmentation, generar nuevas variables de análisis y entrenar un modelo predictivo capaz de estimar precios con alta precisión.

---

# 🎯 Objetivo General

Diseñar e implementar un pipeline de datos inteligente que automatice el proceso de:

* Extracción de datos.
* Transformación y limpieza.
* Validación de calidad.
* Expansión del dataset.
* Ingeniería de características.
* Entrenamiento de modelos predictivos.
* Generación de reportes y visualizaciones.

---

# ⚙️ Arquitectura del Pipeline

## 1. Extracción (Extract)

El pipeline inicia leyendo información desde un archivo CSV:

```text
datos_consolidados_40_registros.csv
```

Información inicial:

* 40 registros
* 6 columnas
* Datos de productos
* Categorías y origen de los productos

---

## 2. Validación

Se verifica que la estructura del archivo sea correcta.

Validaciones realizadas:

✅ Columnas obligatorias presentes

✅ Integridad del dataset

✅ Tipos de datos válidos

✅ Ausencia de columnas faltantes

---

## 3. Limpieza de Datos

Durante esta etapa se realiza:

* Eliminación de registros inválidos.
* Corrección de inconsistencias.
* Validación de nulos.
* Control de duplicados.

### Resultado

| Métrica    | Valor |
| ---------- | ----- |
| Nulos      | 0     |
| Duplicados | 0     |

---

## 4. Data Augmentation

Para mejorar la capacidad de entrenamiento del modelo se implementó una expansión inteligente del dataset.

### Resultado

| Dataset   | Registros |
| --------- | --------- |
| Original  | 40        |
| Expandido | 500       |

Factor de expansión:

```text
12x
```

---

## 5. Ingeniería de Características

Se generaron nuevas variables para enriquecer el análisis y aumentar el rendimiento del modelo.

### Variables creadas

```text
dia_semana
mes
dias_desde_registro
precio_log
rango_precio
```

Estas variables permiten capturar patrones temporales y comportamientos ocultos dentro del dataset.

---

## 6. Codificación de Variables

Las variables categóricas fueron transformadas a formato numérico mediante técnicas de encoding.

### Variables codificadas

```text
categoria_encoded
origen_encoded
```

---

# 🤖 Modelo de Machine Learning

Se implementó un modelo de:

## Regresión Lineal

Objetivo:

Predecir el precio de productos a partir de las variables generadas durante el pipeline.

---

## Variables Utilizadas

```text
categoria_encoded
origen_encoded
dia_semana
mes
dias_desde_registro
precio_log
```

---

# 📈 Entrenamiento

División de datos:

| Conjunto | Registros |
| -------- | --------- |
| Train    | 400       |
| Test     | 100       |

---

# 📊 Resultados del Modelo

## Métricas de Rendimiento

| Métrica  | Resultado |
| -------- | --------- |
| R² Train | 0.9589    |
| R² Test  | 0.9597    |
| RMSE     | 10.1825   |
| MAE      | 7.3727    |

---

## Evaluación

El modelo obtuvo:

```text
R² = 0.9597
```

Lo que indica que explica aproximadamente el:

```text
95.97%
```

de la variabilidad de los datos.

### Clasificación

✅ MODELO EXCELENTE

Criterio:

```text
R² ≥ 0.85
```

---

# 📋 Estadísticas del Dataset Final

| Métrica    | Valor |
| ---------- | ----- |
| Registros  | 500   |
| Columnas   | 13    |
| Categorías | 18    |
| Nulos      | 0     |
| Duplicados | 0     |
| Outliers   | 0     |

---

## Estadísticas de Precio

| Métrica  | Valor   |
| -------- | ------- |
| Promedio | $70.02  |
| Mínimo   | $12.99  |
| Máximo   | $180.99 |

---

# 💡 Insights de Negocio

### Categoría Más Costosa

```text
Datos
```

Precio promedio:

```text
$179.38
```

---

### Categoría Más Económica

```text
Libro
```

Precio promedio:

```text
$22.96
```

---

### Precio Promedio Global

```text
$70.02
```

---

# 📊 Visualizaciones Generadas

El pipeline genera automáticamente las siguientes gráficas:

```text
01_distribucion.png
02_correlaciones.png
03_resultados_modelo.png
04_dashboard_ejecutivo.png
```

Estas visualizaciones permiten analizar:

* Distribución de precios.
* Correlaciones entre variables.
* Resultados del modelo.
* Indicadores ejecutivos.

---

# 🛠 Tecnologías Utilizadas

### Lenguajes

* Python
* HTML
* CSS
* JavaScript

### Librerías de Datos

* Pandas
* NumPy

### Machine Learning

* Scikit-Learn

### Visualización

* Matplotlib
* Seaborn
* Chart.js

---

# 📂 Estructura del Proyecto

```text
SmartMarket/
│
├── pipeline.py
├── datos_consolidados_40_registros.csv
├── datos_500_registros.csv
│
├── 01_distribucion.png
├── 02_correlaciones.png
├── 03_resultados_modelo.png
├── 04_dashboard_ejecutivo.png
│
├── dashboard.html
├── README.md
│
└── requirements.txt
```

---

# ✅ Conclusión

SmartMarket Pipeline Inteligente demuestra la implementación completa de un flujo moderno de Ingeniería de Datos y Machine Learning, integrando procesos ETL, expansión de datos, generación de variables, entrenamiento de modelos predictivos y visualización de resultados.

El proyecto logró expandir exitosamente un dataset de 40 a 500 registros conservando su distribución estadística y alcanzó un desempeño sobresaliente mediante un modelo de Regresión Lineal con un **R² de 0.9597**, evidenciando una alta capacidad predictiva para la estimación de precios.
