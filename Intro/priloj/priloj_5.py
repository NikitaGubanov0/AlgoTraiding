import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Загрузка исторических данных
data = pd.read_csv('historical_stock_prices.csv', index_col='Date')

# Разделим данные на обучающий и тестовый наборы
train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)

# Функция для вычисления стратегии с заданными окнами
def moving_average_strategy(train_data, short_window, long_window):
    signals = pd.DataFrame(index=train_data.index)
    signals['signal'] = 0.0

    # Создание короткой простой скользящей средней
    signals['short_mavg'] = train_data['Close'].rolling(window=short_window, min_periods=1).mean()

    # Создание длинной простой скользящей средней
    signals['long_mavg'] = train_data['Close'].rolling(window=long_window, min_periods=1).mean()

    # Сигналы для покупки/продажи
    signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1.0, 0.0)
    signals['positions'] = signals['signal'].diff()

    # Возвращаем DataFrame с сигналами
    return signals

# Параметры окон для оптимизации
windows = range(10, 200, 5)

best_profit = -np.inf
best_windows = None

for short_window in windows:
    for long_window in windows:
        if short_window >= long_window:
            continue
        signals = moving_average_strategy(train_data, short_window, long_window)

        # Рассчитываем доходность стратегии
        returns = (signals['positions'] * train_data['Close']).cumsum()
        profit = returns.iloc[-1]

        if profit > best_profit:
            best_profit = profit
            best_windows = (short_window, long_window)

# Выводим лучшие параметры и доходность
print(f"Лучшие окна: короткое - {best_windows[0]}, длинное - {best_windows[1]}")
print(f"Лучшая доходность: {best_profit}")
