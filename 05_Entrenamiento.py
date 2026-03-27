import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Modelos
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# DNN
from sklearn.neural_network import MLPClassifier

# 5.1. Cargar datos
X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv').squeeze()

X_val = pd.read_csv('X_val.csv')
y_val = pd.read_csv('y_val.csv').squeeze()

X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv').squeeze()

# Función de evaluación
def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test):
    
    results = {}

    # Train
    y_pred_train = model.predict(X_train)
    results['train'] = {
        'accuracy': accuracy_score(y_train, y_pred_train),
        'precision': precision_score(y_train, y_pred_train),
        'recall': recall_score(y_train, y_pred_train),
        'f1': f1_score(y_train, y_pred_train)
    }

    # Validation
    y_pred_val = model.predict(X_val)
    results['val'] = {
        'accuracy': accuracy_score(y_val, y_pred_val),
        'precision': precision_score(y_val, y_pred_val),
        'recall': recall_score(y_val, y_pred_val),
        'f1': f1_score(y_val, y_pred_val)
    }

    # Test
    y_pred_test = model.predict(X_test)
    results['test'] = {
        'accuracy': accuracy_score(y_test, y_pred_test),
        'precision': precision_score(y_test, y_pred_test),
        'recall': recall_score(y_test, y_pred_test),
        'f1': f1_score(y_test, y_pred_test)
    }

    return results

# 5.2. kNN
print("\n===== kNN =====")

pipeline_knn = Pipeline([
    ('scaler', StandardScaler()),
    ('model', KNeighborsClassifier())
])

param_knn = {
    'model__n_neighbors': [3,5,7],
    'model__weights': ['uniform', 'distance']
}

grid_knn = GridSearchCV(pipeline_knn, param_knn, cv=3, scoring='f1')
grid_knn.fit(X_train, y_train)

knn_best = grid_knn.best_estimator_
knn_results = evaluate_model(knn_best, X_train, y_train, X_val, y_val, X_test, y_test)

print("Mejores parámetros:", grid_knn.best_params_)

# 5.3. Decision Tree 
print("\n===== Decision Tree =====")

dt = DecisionTreeClassifier(random_state=42)

param_dt = {
    'max_depth': [3,5,10,None],
    'min_samples_split': [2,5,10]
}

grid_dt = GridSearchCV(dt, param_dt, cv=3, scoring='f1')
grid_dt.fit(X_train, y_train)

dt_best = grid_dt.best_estimator_
dt_results = evaluate_model(dt_best, X_train, y_train, X_val, y_val, X_test, y_test)

print("Mejores parámetros:", grid_dt.best_params_)

# 5.4. Random Forest 
print("\n===== Random Forest =====")

rf = RandomForestClassifier(random_state=42)

param_rf = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10]
}

grid_rf = GridSearchCV(rf, param_rf, cv=3, scoring='f1')
grid_rf.fit(X_train, y_train)

rf_best = grid_rf.best_estimator_
rf_results = evaluate_model(rf_best, X_train, y_train, X_val, y_val, X_test, y_test)

print("Mejores parámetros:", grid_rf.best_params_)

# 5.5. Gradient Boosting 
print("\n===== Gradient Boosting =====")

gb = GradientBoostingClassifier()

param_gb = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1]
}

grid_gb = GridSearchCV(gb, param_gb, cv=3, scoring='f1')
grid_gb.fit(X_train, y_train)

gb_best = grid_gb.best_estimator_
gb_results = evaluate_model(gb_best, X_train, y_train, X_val, y_val, X_test, y_test)

print("Mejores parámetros:", grid_gb.best_params_)

# 5.6. DNN (MLP) 
print("\n===== DNN =====")

pipeline_mlp = Pipeline([
    ('scaler', StandardScaler()),
    ('model', MLPClassifier(max_iter=300))
])

param_mlp = {
    'model__hidden_layer_sizes': [(64,32,16), (128,64,32)],
    'model__alpha': [0.0001, 0.001]
}

grid_mlp = GridSearchCV(pipeline_mlp, param_mlp, cv=3, scoring='f1')
grid_mlp.fit(X_train, y_train)

mlp_best = grid_mlp.best_estimator_
mlp_results = evaluate_model(mlp_best, X_train, y_train, X_val, y_val, X_test, y_test)

print("Mejores parámetros:", grid_mlp.best_params_)

plt.plot(mlp_best.named_steps['model'].loss_curve_)
plt.title('Curva de entrenamiento DNN')
plt.xlabel('Iteraciones')
plt.ylabel('Loss')
plt.show()

# 5.7. Comparativa de modelos
models = {
    'kNN': knn_results,
    'Decision Tree': dt_results,
    'Random Forest': rf_results,
    'Gradient Boosting': gb_results,
    'DNN': mlp_results
}

rows = []

for model_name, res in models.items():
    for split in ['train', 'val', 'test']:
        rows.append({
            'Model': model_name,
            'Dataset': split,
            'Accuracy': res[split]['accuracy'],
            'Precision': res[split]['precision'],
            'Recall': res[split]['recall'],
            'F1': res[split]['f1']
        })

df_results = pd.DataFrame(rows)

print("\n===== RESULTADOS =====")
print(df_results)