# 📚 TI24 — Proyecto Final: K-Nearest Neighbors (KNN)

**Universidad del Valle — Cochabamba**  
**Materia:** Tecnologías de Inteligencia Artificial  
**Tema:** K-Nearest Neighbors (KNN)  
**Dataset:** AI Student Impact Dataset (50,000 × 16)  
**Autor:** Juan Pablo [Apellido]  
**Fecha:** Junio 2026

---

## 🎯 Objetivo

Clasificar el nivel de riesgo de burnout (`Burnout_Risk_Level`: Low / Medium / High) en estudiantes universitarios según su uso de herramientas de Inteligencia Artificial Generativa, aplicando el algoritmo K-Nearest Neighbors y comparándolo con un árbol de decisión.

---

## 📂 Estructura del Repositorio

```
repositorio-TI24-[Apellido]/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   │   └── ai_student_impact_dataset.csv
│   └── processed/
│       ├── X_train_sc.npy
│       ├── X_test_sc.npy
│       ├── y_train.npy
│       ├── y_test.npy
│       ├── scaler.pkl
│       ├── label_encoders.pkl
│       └── knn_model.pkl
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_main.ipynb
│   └── 04_model_comparison.ipynb
└── docs/
    ├── informe_final.pdf
    ├── presentacion.pdf
    └── figs/
        ├── target_distribution.png
        ├── num_distributions.png
        ├── boxplots_vs_target.png
        ├── correlation_matrix.png
        ├── categorical_vs_target.png
        ├── pairplot.png
        ├── scaling_comparison.png
        ├── knn_elbow.png
        ├── knn_confusion_matrix.png
        ├── knn_roc_curve.png
        ├── model_comparison_metrics.png
        ├── comparison_confusion_matrices.png
        ├── comparison_roc_curves.png
        └── dt_feature_importance.png
```

---

## 🗂️ Dataset

| Atributo | Detalle |
|---|---|
| **Nombre** | AI Student Impact Dataset |
| **Fuente** | Kaggle |
| **Filas** | 50,000 |
| **Columnas** | 16 |
| **Variable objetivo** | `Burnout_Risk_Level` (Low / Medium / High) |
| **Variables numéricas** | 10 (GPA, horas GenAI, retención, ansiedad, etc.) |
| **Variables categóricas** | 5 (carrera, año de estudio, política institucional, etc.) |
| **Valores nulos** | 0 |

---

## ⚙️ Pipeline de Preprocesamiento

1. **Eliminación** de `Student_ID` (identificador no predictivo)
2. **Encoding ordinal** para `Year_of_Study` y `Prompt_Engineering_Skill`
3. **LabelEncoder** para variables nominales (`Major_Category`, `Primary_Use_Case`, `Institutional_Policy`)
4. **StandardScaler** ajustado únicamente en train (evitar data leakage)
5. **Split 80/20** estratificado

---

## 🤖 Modelo Principal — KNN

**Fundamento:** KNN clasifica una nueva instancia por voto mayoritario entre sus K vecinos más cercanos, usando distancia Euclidiana:

$$d(\mathbf{x}, \mathbf{q}) = \sqrt{\sum_{i=1}^{n}(x_i - q_i)^2}$$

**Búsqueda del K óptimo:** Cross-validation de 5 folds evaluando K = 1…30.

| Hiperparámetro | Valor |
|---|---|
| `n_neighbors` | Determinado por CV |
| `metric` | euclidean |
| `weights` | uniform |

---

## ⚖️ Modelo Comparativo — Decision Tree

Se compara KNN con `DecisionTreeClassifier` de scikit-learn, con búsqueda de `max_depth` óptimo por CV.

---

## 📊 Resultados

> Ejecutar los notebooks en orden (01 → 02 → 03 → 04) para reproducir todos los resultados.

---

## 🚀 Reproducción

```bash
# 1. Clonar el repositorio
git clone https://github.com/[usuario]/repositorio-TI24-[Apellido].git
cd repositorio-TI24-[Apellido]

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Colocar el dataset en data/raw/
# 4. Ejecutar notebooks en orden en Google Colab o Jupyter
```

---

## 📦 Dependencias

Ver `requirements.txt`

---

## 📚 Referencias

1. Cover, T. M., & Hart, P. E. (1967). *Nearest neighbor pattern classification*. IEEE Transactions on Information Theory, 13(1), 21–27.
2. Mitchell, T. (1997). *Machine Learning*. McGraw-Hill.
3. Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.). O'Reilly Media.
