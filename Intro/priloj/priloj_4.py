from scipy.optimize import minimize
import numpy as np

# Задаем функцию потерь
def loss_function(params):
    return (params - 2) ** 2

# Простая квадратичная функция

# Инициализируем параметры
initial_params = np.array([10])

# Начальная точка далеко от минимума

# Применяем минимизацию с использованием градиентного спуска
result = minimize(loss_function, initial_params, method='BFGS')

# Выводим результат оптимизации
print(f"Оптимизированные параметры: {result.x}")
print(f"Значение функции потерь: {result.fun}")
