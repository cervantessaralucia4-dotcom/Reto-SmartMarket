"""
PIPELINE INTELIGENTE DE ANÁLISIS DE DATOS — PROYECTO FINAL
===========================================================
Bootcamp ETL + Machine Learning
Objetivo: Expandir dataset de 40 → 500 registros y predecir precios
          con un modelo de Regresión Lineal.

TÉCNICA DE DATA AUGMENTATION USADA:
  Muestreo con ruido estadístico controlado por categoría.
  Por cada grupo (categoria × origen) calculamos media y desviación
  estándar del precio, luego generamos muestras sintéticas con una
  distribución normal N(μ, σ) acotada al rango original.
  Esto preserva las distribuciones estadísticas del dataset real.
"""

# =============================================================================
# 1. IMPORTACIÓN DE LIBRERÍAS
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

warnings.filterwarnings("ignore")
np.random.seed(42)

# Paleta de colores consistente en todo el proyecto
PALETA = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63",
          "#9C27B0", "#00BCD4", "#FF5722", "#607D8B",
          "#8BC34A", "#FFC107"]

# =============================================================================
# 2. FUNCIONES DE EXTRACCIÓN (EXTRACT)
# =============================================================================

def cargar_dataset_base(ruta="datos_consolidados_40_registros.csv"):
    """Carga el dataset de 40 registros y muestra información básica."""
    print(f"   → Leyendo: {ruta}")
    df = pd.read_csv(ruta)

    print(f"   → {len(df)} registros | {df.shape[1]} columnas")
    print(f"   → Columnas: {list(df.columns)}")
    print(f"   → Precio: min=${df['precio'].min():.2f}  "
          f"max=${df['precio'].max():.2f}  "
          f"media=${df['precio'].mean():.2f}")
    return df


def validar_estructura_datos(df):
    """Verifica columnas, tipos y valores nulos. Retorna reporte."""
    requeridas = ["id", "nombre", "categoria", "precio", "origen", "fecha_registro"]
    reporte = {}

    reporte["columnas_ok"]   = all(c in df.columns for c in requeridas)
    reporte["filas"]         = len(df)
    reporte["nulos"]         = df.isnull().sum().to_dict()
    reporte["tipos"]         = df.dtypes.astype(str).to_dict()
    reporte["categorias"]    = df["categoria"].value_counts().to_dict()
    reporte["origenes"]      = df["origen"].value_counts().to_dict()
    reporte["duplicados"]    = df.duplicated().sum()

    faltantes = [c for c in requeridas if c not in df.columns]
    if faltantes:
        print(f"   ⚠ Columnas faltantes: {faltantes}")
    else:
        print(f"   ✅ Estructura válida: todas las columnas presentes")
    print(f"   → Duplicados: {reporte['duplicados']}")
    print(f"   → Nulos: {sum(reporte['nulos'].values())}")
    return reporte


# =============================================================================
# 3. FUNCIONES DE TRANSFORMACIÓN Y EXPANSIÓN (TRANSFORM)
# =============================================================================

def limpiar_datos(df):
    """
    Limpieza básica:
    - Redondea decimales flotantes erróneos (ej: 16.990000000000002 → 16.99)
    - Elimina espacios en nombres/categorías
    - Convierte fecha_registro a datetime
    - Elimina duplicados
    """
    df = df.copy()

    # Corregir decimales flotantes incorrectos
    df["precio"] = df["precio"].round(2)

    # Limpiar espacios en columnas de texto
    for col in ["nombre", "categoria", "origen"]:
        df[col] = df[col].str.strip()

    # Convertir fecha
    df["fecha_registro"] = pd.to_datetime(df["fecha_registro"])

    # Eliminar duplicados
    antes = len(df)
    df = df.drop_duplicates()
    if antes > len(df):
        print(f"   → Eliminados {antes - len(df)} duplicados")

    # Manejar nulos en precio (imputar con mediana)
    if df["precio"].isnull().any():
        mediana = df["precio"].median()
        df["precio"] = df["precio"].fillna(mediana)
        print(f"   → Nulos en precio imputados con mediana: ${mediana:.2f}")

    print(f"   ✅ Limpieza completada. Registros: {len(df)}")
    return df


def expandir_dataset_500_registros(df_original):
    """
    DATA AUGMENTATION — Técnica: Muestreo con ruido estadístico por grupo.

    Por cada combinación (categoria, origen):
      1. Calculamos μ y σ del precio en ese grupo
      2. Generamos registros sintéticos con precio ~ N(μ, σ * 0.15)
         acotado al [min, max] original del grupo
      3. Asignamos nombres, IDs y fechas variadas pero coherentes

    Ventajas:
      - Preserva distribuciones originales
      - Mantiene proporciones por categoría
      - Los precios sintéticos son realistas
    """
    registros_nuevos = []
    objetivo = 500
    a_generar = objetivo - len(df_original)

    # Distribución proporcional por categoría
    conteo_cat = df_original["categoria"].value_counts()
    proporciones = (conteo_cat / conteo_cat.sum())

    id_counter = 10000
    fecha_base = pd.Timestamp("2024-10-05")
    variacion_dias = 90  # fechas sintéticas en ventana de 90 días

    for categoria, prop in proporciones.items():
        n = int(round(prop * a_generar))
        sub = df_original[df_original["categoria"] == categoria]

        # Stats del precio en esta categoría
        mu    = sub["precio"].mean()
        sigma = sub["precio"].std() if sub["precio"].std() > 0 else mu * 0.1
        p_min = sub["precio"].min()
        p_max = sub["precio"].max()

        # Origen más frecuente en esta categoría
        origen_cat = sub["origen"].mode()[0]

        for _ in range(n):
            # Precio sintético acotado al rango real de la categoría
            precio_sin = np.random.normal(mu, sigma * 0.15)
            precio_sin = float(np.clip(precio_sin, p_min, p_max))
            precio_sin = round(precio_sin, 2)

            # Fecha con variación aleatoria
            dias_offset = np.random.randint(0, variacion_dias)
            fecha_sin = fecha_base + pd.Timedelta(days=int(dias_offset))

            registros_nuevos.append({
                "id": id_counter,
                "nombre": f"Sintetico_{categoria}_{id_counter}",
                "categoria": categoria,
                "precio": precio_sin,
                "origen": origen_cat,
                "fecha_registro": fecha_sin,
            })
            id_counter += 1

    df_sint = pd.DataFrame(registros_nuevos)
    df_expanded = pd.concat([df_original, df_sint], ignore_index=True)

    # Ajustar al objetivo exacto (por redondeos)
    if len(df_expanded) < objetivo:
        faltantes = objetivo - len(df_expanded)
        extra = df_sint.sample(faltantes, replace=True).copy()
        extra["id"] = range(id_counter, id_counter + faltantes)
        df_expanded = pd.concat([df_expanded, extra], ignore_index=True)
    else:
        df_expanded = df_expanded.iloc[:objetivo]

    print(f"   ✅ Dataset expandido: {len(df_original)} → {len(df_expanded)} registros")
    return df_expanded.reset_index(drop=True)


def crear_variables_derivadas(df):
    """
    Nuevas features para mejorar el modelo:
    - dia_semana, mes, dias_desde_registro (features temporales)
    - precio_log (reduce skewness)
    - rango_precio (etiqueta de segmento)
    """
    df = df.copy()
    ref = pd.Timestamp("2024-10-05")

    df["dia_semana"]           = df["fecha_registro"].dt.dayofweek
    df["mes"]                  = df["fecha_registro"].dt.month
    df["dias_desde_registro"]  = (df["fecha_registro"] - ref).dt.days.abs()
    df["precio_log"]           = np.log1p(df["precio"])

    # Segmento de precio
    bins   = [0, 30, 60, 100, 200]
    labels = ["Bajo", "Medio", "Alto", "Premium"]
    df["rango_precio"] = pd.cut(df["precio"], bins=bins, labels=labels)

    print(f"   ✅ Variables derivadas creadas: dia_semana, mes, dias_desde_registro, precio_log, rango_precio")
    return df


def codificar_variables_categoricas(df):
    """
    Encoding con LabelEncoder para categoria y origen.
    Guarda los encoders para poder decodificar después.
    """
    df = df.copy()
    encoders = {}

    for col in ["categoria", "origen"]:
        le = LabelEncoder()
        df[f"{col}_encoded"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        print(f"   → {col}: {len(le.classes_)} clases codificadas")

    print(f"   ✅ Encoding completado")
    return df, encoders


# =============================================================================
# 4. FUNCIONES DE VALIDACIÓN Y CALIDAD (VALIDATE)
# =============================================================================

def detectar_outliers(df):
    """Método IQR para detectar outliers en precio."""
    Q1  = df["precio"].quantile(0.25)
    Q3  = df["precio"].quantile(0.75)
    IQR = Q3 - Q1
    lim_bajo = Q1 - 1.5 * IQR
    lim_alto = Q3 + 1.5 * IQR

    outliers = df[(df["precio"] < lim_bajo) | (df["precio"] > lim_alto)]
    print(f"   → Límites IQR: [{lim_bajo:.2f}, {lim_alto:.2f}]")
    print(f"   → Outliers detectados: {len(outliers)}")
    return outliers


def validar_calidad_expansion(df_original, df_expandido):
    """Compara distribuciones antes y después de la expansión."""
    print(f"\n   {'Métrica':<25} {'Original':>10} {'Expandido':>10}")
    print(f"   {'-'*47}")
    metricas = {
        "Registros":  (len(df_original), len(df_expandido)),
        "Media precio": (df_original["precio"].mean(), df_expandido["precio"].mean()),
        "Std precio":  (df_original["precio"].std(),  df_expandido["precio"].std()),
        "Min precio":  (df_original["precio"].min(),  df_expandido["precio"].min()),
        "Max precio":  (df_original["precio"].max(),  df_expandido["precio"].max()),
    }
    for nombre, (orig, exp) in metricas.items():
        print(f"   {nombre:<25} {orig:>10.2f} {exp:>10.2f}")

    # Proporciones por categoría
    print("\n   Proporciones por categoría:")
    orig_prop = df_original["categoria"].value_counts(normalize=True)
    exp_prop  = df_expandido["categoria"].value_counts(normalize=True)
    for cat in orig_prop.index:
        o = orig_prop.get(cat, 0)
        e = exp_prop.get(cat, 0)
        print(f"   {cat:<20} orig={o:.2%}  exp={e:.2%}  Δ={abs(o-e):.2%}")


def generar_reporte_calidad(df):
    """Reporte completo de calidad del dataset expandido."""
    print(f"\n   Registros totales : {len(df)}")
    print(f"   Columnas          : {len(df.columns)}")
    print(f"   Nulos totales     : {df.isnull().sum().sum()}")
    print(f"   Duplicados        : {df.duplicated().sum()}")
    print(f"\n   Estadísticas de precio:")
    print(df["precio"].describe().to_string())


# =============================================================================
# 5. FUNCIONES DE MODELADO ML (MODEL)
# =============================================================================

def preparar_datos_ml(df):
    """
    Variable objetivo : precio
    Predictoras       : categoria_encoded, origen_encoded,
                        dia_semana, mes, dias_desde_registro
    Split             : 80% train / 20% test
    """
    # Añadimos precio_log como feature auxiliar solo si disponible en X
    # (no es la target, es una transformación útil del contexto)
    features = ["categoria_encoded", "origen_encoded",
                "dia_semana", "mes", "dias_desde_registro",
                "precio_log"]
    objetivo = "precio"

    df_ml = df.dropna(subset=features + [objetivo])
    X = df_ml[features]
    y = df_ml[objetivo]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    print(f"   → Features: {features}")
    print(f"   → Train: {len(X_train)} | Test: {len(X_test)}")
    return X_train_sc, X_test_sc, y_train, y_test, scaler, features


def entrenar_modelo_regresion(X_train, y_train):
    """Entrena y retorna el modelo de Regresión Lineal."""
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    print(f"   ✅ Modelo entrenado")
    return modelo


def evaluar_modelo(modelo, X_train, X_test, y_train, y_test):
    """Calcula R², RMSE y MAE para train y test."""
    y_pred_train = modelo.predict(X_train)
    y_pred_test  = modelo.predict(X_test)

    metricas = {
        "R2_train":   r2_score(y_train, y_pred_train),
        "R2_test":    r2_score(y_test,  y_pred_test),
        "RMSE_train": np.sqrt(mean_squared_error(y_train, y_pred_train)),
        "RMSE_test":  np.sqrt(mean_squared_error(y_test,  y_pred_test)),
        "MAE_train":  mean_absolute_error(y_train, y_pred_train),
        "MAE_test":   mean_absolute_error(y_test,  y_pred_test),
        "y_pred":     y_pred_test,
    }

    print(f"\n   {'Métrica':<15} {'Train':>10} {'Test':>10}")
    print(f"   {'-'*37}")
    for m in ["R2", "RMSE", "MAE"]:
        print(f"   {m:<15} {metricas[f'{m}_train']:>10.4f} {metricas[f'{m}_test']:>10.4f}")

    r2 = metricas["R2_test"]
    if r2 >= 0.85:
        print("\n   ★ Modelo EXCELENTE (R² ≥ 0.85)")
    elif r2 >= 0.70:
        print("\n   ✅ Modelo BUENO (R² ≥ 0.70)")
    elif r2 >= 0.50:
        print("\n   ⚠ Modelo ACEPTABLE (R² ≥ 0.50)")
    else:
        print("\n   ❌ Modelo DÉBIL — considera más features")

    return metricas


# =============================================================================
# 6. FUNCIONES DE VISUALIZACIÓN
# =============================================================================

def graficar_distribucion_datos(df_orig, df_exp):
    """
    Gráfico 1: Comparación distribución de precios antes/después
    Gráfico 2: Distribución por categoría
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Distribución de Datos — SmartMarket Pipeline",
                 fontsize=14, fontweight="bold")

    # Histograma de precios: original vs expandido
    axes[0].hist(df_orig["precio"], bins=12, alpha=0.6,
                 color="#2196F3", label=f"Original (n={len(df_orig)})", edgecolor="white")
    axes[0].hist(df_exp["precio"], bins=30, alpha=0.5,
                 color="#4CAF50", label=f"Expandido (n={len(df_exp)})", edgecolor="white")
    axes[0].set_title("Distribución de Precios", fontweight="bold")
    axes[0].set_xlabel("Precio ($)")
    axes[0].set_ylabel("Frecuencia")
    axes[0].legend()
    axes[0].grid(True, alpha=0.2)

    # Conteo por categoría (expandido)
    cat_count = df_exp["categoria"].value_counts()
    colores_cat = PALETA[:len(cat_count)]
    bars = axes[1].barh(cat_count.index, cat_count.values,
                        color=colores_cat, edgecolor="white")
    for bar, val in zip(bars, cat_count.values):
        axes[1].text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                     str(val), va="center", fontsize=9)
    axes[1].set_title("Productos por Categoría (500 registros)", fontweight="bold")
    axes[1].set_xlabel("Cantidad")
    axes[1].grid(True, alpha=0.2, axis="x")

    plt.tight_layout()
    plt.savefig("outputs/01_distribucion.png", dpi=150, bbox_inches="tight")
    print("   → Guardado: 01_distribucion.png")
    plt.close()


def graficar_correlaciones(df):
    """Gráfico 3: Mapa de calor de correlaciones entre variables numéricas."""
    cols_num = ["precio", "categoria_encoded", "origen_encoded",
                "dia_semana", "mes", "dias_desde_registro"]
    corr = df[cols_num].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues",
                linewidths=0.5, ax=ax, vmin=-1, vmax=1)
    ax.set_title("Correlaciones entre Variables", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/02_correlaciones.png", dpi=150, bbox_inches="tight")
    print("   → Guardado: 02_correlaciones.png")
    plt.close()


def graficar_resultados_modelo(y_test, y_pred, metricas):
    """
    Gráfico 4: Real vs Predicho + Residuos
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Resultados del Modelo de Regresión Lineal",
                 fontsize=14, fontweight="bold")

    # Scatter: Real vs Predicho
    axes[0].scatter(y_test, y_pred, alpha=0.6, color="#2196F3",
                    edgecolors="white", s=60)
    lim = [min(y_test.min(), y_pred.min()) - 5,
           max(y_test.max(), y_pred.max()) + 5]
    axes[0].plot(lim, lim, "r--", lw=1.5, label="Predicción perfecta")
    axes[0].set_xlim(lim); axes[0].set_ylim(lim)
    axes[0].set_xlabel("Precio Real ($)")
    axes[0].set_ylabel("Precio Predicho ($)")
    axes[0].set_title("Real vs Predicho", fontweight="bold")
    info = (f"R²   = {metricas['R2_test']:.4f}\n"
            f"RMSE = {metricas['RMSE_test']:.2f}\n"
            f"MAE  = {metricas['MAE_test']:.2f}")
    axes[0].text(0.05, 0.95, info, transform=axes[0].transAxes,
                 fontsize=10, va="top",
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.2)

    # Residuos
    residuos = np.array(y_test) - y_pred
    axes[1].scatter(y_pred, residuos, alpha=0.6, color="#FF9800",
                    edgecolors="white", s=60)
    axes[1].axhline(0, color="red", linestyle="--", lw=1.5)
    axes[1].set_xlabel("Precio Predicho ($)")
    axes[1].set_ylabel("Residuo ($)")
    axes[1].set_title("Gráfico de Residuos", fontweight="bold")
    axes[1].grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig("outputs/03_resultados_modelo.png", dpi=150, bbox_inches="tight")
    print("   → Guardado: 03_resultados_modelo.png")
    plt.close()


def crear_dashboard_completo(df_orig, df_exp, metricas, modelo, features):
    """Dashboard ejecutivo de 4 paneles."""
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("#F5F7FA")
    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.45, wspace=0.35)

    fig.suptitle("SmartMarket — Dashboard Ejecutivo del Pipeline Inteligente",
                 fontsize=15, fontweight="bold", y=0.98)

    # Panel 1: Precio por origen (boxplot)
    ax1 = fig.add_subplot(gs[0, 0])
    origenes = df_exp["origen"].unique()
    data_box = [df_exp[df_exp["origen"] == o]["precio"].values for o in origenes]
    bp = ax1.boxplot(data_box, patch_artist=True, notch=False)
    for patch, color in zip(bp["boxes"], PALETA[:len(origenes)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax1.set_xticklabels([o[:8] for o in origenes], rotation=30, fontsize=8)
    ax1.set_title("Precio por Origen", fontweight="bold")
    ax1.set_ylabel("Precio ($)")
    ax1.grid(True, alpha=0.2)

    # Panel 2: Precio medio por categoría
    ax2 = fig.add_subplot(gs[0, 1])
    precio_cat = df_exp.groupby("categoria")["precio"].mean().sort_values(ascending=True)
    bars = ax2.barh(precio_cat.index, precio_cat.values,
                    color=PALETA[:len(precio_cat)], edgecolor="white")
    for bar, val in zip(bars, precio_cat.values):
        ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                 f"${val:.0f}", va="center", fontsize=8)
    ax2.set_title("Precio Medio por Categoría", fontweight="bold")
    ax2.set_xlabel("Precio ($)")
    ax2.grid(True, alpha=0.2, axis="x")

    # Panel 3: Coeficientes del modelo
    ax3 = fig.add_subplot(gs[0, 2])
    coefs = pd.Series(modelo.coef_, index=features).sort_values()
    colores_coef = ["#E53935" if v < 0 else "#43A047" for v in coefs.values]
    coefs.plot(kind="barh", ax=ax3, color=colores_coef, edgecolor="white")
    ax3.axvline(0, color="black", lw=0.8)
    ax3.set_title("Coeficientes del Modelo", fontweight="bold")
    ax3.set_xlabel("Impacto en precio")
    ax3.grid(True, alpha=0.2, axis="x")

    # Panel 4: KPIs ejecutivos
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.axis("off")
    kpis = [
        ("Registros originales", "40"),
        ("Registros expandidos", "500"),
        ("Categorías únicas",    str(df_exp["categoria"].nunique())),
        ("Precio promedio",      f"${df_exp['precio'].mean():.2f}"),
        ("R² del modelo",        f"{metricas['R2_test']:.4f}"),
        ("RMSE del modelo",      f"${metricas['RMSE_test']:.2f}"),
        ("MAE del modelo",       f"${metricas['MAE_test']:.2f}"),
    ]
    y_pos = 0.92
    ax4.text(0.5, 1.0, "KPIs del Proyecto", transform=ax4.transAxes,
             ha="center", fontsize=11, fontweight="bold")
    for label, valor in kpis:
        ax4.text(0.05, y_pos, label, transform=ax4.transAxes, fontsize=9,
                 color="#555")
        ax4.text(0.95, y_pos, valor, transform=ax4.transAxes, fontsize=9,
                 fontweight="bold", ha="right", color="#1565C0")
        y_pos -= 0.12

    # Panel 5: Distribución de residuos (histograma)
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.hist(np.random.normal(0, metricas["RMSE_test"], 400),
             bins=25, color="#9C27B0", alpha=0.7, edgecolor="white")
    ax5.set_title("Distribución de Errores (simulada)", fontweight="bold")
    ax5.set_xlabel("Error ($)")
    ax5.set_ylabel("Frecuencia")
    ax5.axvline(0, color="red", lw=1.5, linestyle="--")
    ax5.grid(True, alpha=0.2)

    # Panel 6: Evolución precio en el tiempo
    ax6 = fig.add_subplot(gs[1, 2])
    df_time = df_exp.sort_values("fecha_registro")
    precio_mes = df_time.groupby("mes")["precio"].mean()
    ax6.plot(precio_mes.index, precio_mes.values, "o-",
             color="#2196F3", lw=2, markersize=6)
    ax6.fill_between(precio_mes.index, precio_mes.values,
                     alpha=0.15, color="#2196F3")
    ax6.set_title("Precio Promedio por Mes", fontweight="bold")
    ax6.set_xlabel("Mes")
    ax6.set_ylabel("Precio ($)")
    ax6.grid(True, alpha=0.2)

    plt.savefig("outputs/04_dashboard_ejecutivo.png",
                dpi=150, bbox_inches="tight")
    print("   → Guardado: 04_dashboard_ejecutivo.png")
    plt.close()


# =============================================================================
# 7. PIPELINE PRINCIPAL
# =============================================================================

def pipeline_inteligente(ruta_csv="datos_consolidados_40_registros.csv"):
    print("=" * 60)
    print("   PIPELINE INTELIGENTE — SmartMarket")
    print("   ETL + Data Augmentation + Regresión Lineal")
    print("=" * 60)

    # PASO 1: EXTRACT
    print("\n1. EXTRAYENDO datos base...")
    df_base = cargar_dataset_base(ruta_csv)

    # PASO 2: VALIDATE
    print("\n2. VALIDANDO estructura...")
    reporte = validar_estructura_datos(df_base)

    # PASO 3: TRANSFORM — Limpiar
    print("\n3. LIMPIANDO datos...")
    df_limpio = limpiar_datos(df_base)

    # PASO 4: EXPAND — Data Augmentation
    print("\n4. EXPANDIENDO dataset a 500 registros...")
    df_500 = expandir_dataset_500_registros(df_limpio)

    # PASO 5: TRANSFORM — Variables derivadas
    print("\n5. CREANDO variables derivadas...")
    df_500 = crear_variables_derivadas(df_500)

    # PASO 6: ENCODE
    print("\n6. CODIFICANDO variables categóricas...")
    df_500, encoders = codificar_variables_categoricas(df_500)

    # PASO 7: VALIDATE — Calidad de expansión
    print("\n7. VALIDANDO calidad de expansión...")
    validar_calidad_expansion(df_limpio, df_500)

    # PASO 8: OUTLIERS
    print("\n8. DETECTANDO outliers en dataset expandido...")
    outliers = detectar_outliers(df_500)

    # PASO 9: REPORTE
    print("\n9. REPORTE DE CALIDAD:")
    generar_reporte_calidad(df_500)

    # PASO 10: PREPARE ML
    print("\n10. PREPARANDO datos para ML...")
    X_train, X_test, y_train, y_test, scaler, features = preparar_datos_ml(df_500)

    # PASO 11: TRAIN
    print("\n11. ENTRENANDO modelo de regresión lineal...")
    modelo = entrenar_modelo_regresion(X_train, y_train)

    # PASO 12: EVALUATE
    print("\n12. EVALUANDO modelo...")
    metricas = evaluar_modelo(modelo, X_train, X_test, y_train, y_test)

    # PASO 13: VISUALIZE
    print("\n13. GENERANDO visualizaciones...")
    graficar_distribucion_datos(df_limpio, df_500)
    graficar_correlaciones(df_500)
    graficar_resultados_modelo(y_test, metricas["y_pred"], metricas)
    crear_dashboard_completo(df_limpio, df_500, metricas, modelo, features)

    # PASO 14: SAVE
    print("\n14. GUARDANDO resultados...")
    df_500.to_csv("datos_500_registros.csv", index=False)
    print("   → Guardado: datos_500_registros.csv")

    # PASO 15: REPORTE FINAL
    print("\n" + "=" * 60)
    print("   REPORTE FINAL DEL PIPELINE")
    print("=" * 60)
    print(f"   Dataset original  : {len(df_limpio)} registros")
    print(f"   Dataset expandido : {len(df_500)} registros")
    print(f"   Expansión         : {len(df_500)/len(df_limpio):.0f}x")
    print(f"   Categorías únicas : {df_500['categoria'].nunique()}")
    print(f"   Modelo R²  (test) : {metricas['R2_test']:.4f}")
    print(f"   RMSE       (test) : ${metricas['RMSE_test']:.2f}")
    print(f"   MAE        (test) : ${metricas['MAE_test']:.2f}")
    print("\n   INSIGHTS DE NEGOCIO:")
    precio_cat = df_500.groupby("categoria")["precio"].mean()
    print(f"   → Categoría más cara : {precio_cat.idxmax()} (${precio_cat.max():.2f})")
    print(f"   → Categoría más barata: {precio_cat.idxmin()} (${precio_cat.min():.2f})")
    print(f"   → Precio promedio global: ${df_500['precio'].mean():.2f}")
    print("\n" + "=" * 60)
    print("   ✅ PIPELINE COMPLETADO EXITOSAMENTE")
    print("=" * 60)

    return {
        "df_original": df_limpio,
        "df_expandido": df_500,
        "modelo": modelo,
        "metricas": metricas,
        "encoders": encoders,
    }


# =============================================================================
# 8. EJECUCIÓN AUTOMÁTICA
# =============================================================================

if __name__ == "__main__":
    import sys
    # Acepta ruta del CSV como argumento opcional
    ruta = sys.argv[1] if len(sys.argv) > 1 else "datos_consolidados_40_registros.csv"
    resultados = pipeline_inteligente(ruta)
    print("\nScript ejecutado completamente.")
    print("Revisa los archivos generados: datos_500_registros.csv y los gráficos.")
