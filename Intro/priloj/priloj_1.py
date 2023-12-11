import pandas as pd
import numpy as np

# Предположим, что у вас уже есть DataFrame `signals` с сигналами
# Параметры окон для оптимизации
short_window = best_windows[0]
long_window = best_windows[1]

# Создаем DataFrame с сигналами
signals = moving_average_strategy(train_data, short_window, long_window)

# Извлекаем столбцы Close и positions из train_data
data = train_data[['Close']].copy()
data['positions'] = signals['positions']

initial_capital = float(100000.0)

# Создание DataFrame `positions`
positions = pd.DataFrame(index=data.index).fillna(0.0)

# Покупаем 100 акций, когда сигнал 1
positions['stock'] = 100 * data['positions']

# Инициализируем портфель с капиталом
portfolio = positions.multiply(data['Close'], axis=0)

# Сохраняем разности стоимости портфеля
pos_diff = positions.diff()

# Добавляем `holdings` и `cash` в портфель
portfolio['holdings'] = (positions.multiply(data['Close'], axis=0)).sum(axis=1)
portfolio['cash'] = initial_capital - (pos_diff.multiply(data['Close'], axis=0)).sum(axis=1).cumsum()

# Добавляем `total` и `returns` в портфель
portfolio['total'] = portfolio['cash'] + portfolio['holdings']
portfolio['returns'] = portfolio['total'].pct_change()

# Выводим результаты бэктестинга
print(portfolio.tail())
