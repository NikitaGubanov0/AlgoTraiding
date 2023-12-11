from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Создаем синтетические данные для классификации
X, y = make_classification(n_samples=1000, n_features=4, n_informative=2, n_redundant=0, random_state=0, shuffle=False)

# Подготавливаем модель классификатора для оптимизации
rf = RandomForestClassifier(n_jobs=-1)

# Определяем параметры для оптимизации
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30]
}

# Используем поиск по сетке для определения лучших параметров
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5)
grid_search.fit(X, y)

# Выводим лучшие параметры
print(f"Лучшие параметры: {grid_search.best_params_}")
