
import pandas as pd
import numpy as np
import ccxt
from sklearn.model_selection import ParameterGrid

# Функция стратегии на основе скользящих средних
def moving_average_strategy(data, short_window, long_window):
    signals = pd.DataFrame(index=data.index)
    signals['signal'] = 0.0
    
    # Создаем короткую и длинную скользящие средние
    signals['short_mavg'] = data['Close'].rolling(window=short_window, min_periods=1).mean()
    signals['long_mavg'] = data['Close'].rolling(window=long_window, min_periods=1).mean()

    # Создаем сигналы
    signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1.0, 0.0)   
    signals['positions'] = signals['signal'].diff()

    return signals

# Инициализация API биржи
exchange = ccxt.binance({
    'apiKey': 'ВАШ_API_KEY',
    'secret': 'ВАШ_API_SECRET',
    'enableRateLimit': True,
})

# Функция для выполнения торговых ордеров
def execute_order(exchange, symbol, signal, amount):
    if signal == 1:  # Покупка
        order = exchange.create_market_buy_order(symbol, amount)
    elif signal == -1:  # Продажа
        order = exchange.create_market_sell_order(symbol, amount)
    else:
        order = None
    return order

# Функция для печати баланса
def print_balance(exchange, symbol):
    balance = exchange.fetch_balance()
    print(f"Баланс {symbol}: {balance['total'][symbol]}")

# Оптимизация параметров стратегии на исторических данных
param_grid = {
    'short_window': range(20, 60, 10),
    'long_window': range(60, 120, 20)
}
grid = ParameterGrid(param_grid)

# Функция для вычисления доходности стратегии
def evaluate_strategy(signals, data, initial_capital):
    portfolio = pd.DataFrame(index=signals.index)
    portfolio['holdings'] = (signals['positions'] * data['Close']).cumsum()
    portfolio['cash'] = initial_capital - (signals['positions'] * data['Close']).cumsum()
    portfolio['total'] = portfolio['holdings'] + portfolio['cash']
    return portfolio['total'][-1] - initial_capital

best_profit = -np.inf
best_params = None
initial_capital = 100000.0  # Пример начального капитала

# Предполагаем, что исторические данные загружены в DataFrame 'data'
data = pd.read_csv('historical_data.csv', index_col='Date', parse_dates=True)

for params in grid:
    signals = moving_average_strategy(data, params['short_window'], params['long_window'])
    profit = evaluate_strategy(signals, data, initial_capital)
    if profit > best_profit:
        best_profit = profit
        best_params = params

print(f"Лучшие параметры: {best_params}, доходность: {best_profit}")

# Рассчитываем сигналы на основе лучших параметров
final_signals = moving_average_strategy(data, best_params['short_window'], best_params['long_window'])

# Исполнение торговых ордеров в соответствии с сигналами
for index, signal in final_signals.iterrows():
    current_position = signal['positions']
    if current_position == 1:  # Сигнал к покупке
        print("Покупка")
        execute_order(exchange, 'BTC/USDT', current_position, amount=0.01)  # Пример объема позиции
    elif current_position == -1:  # Сигнал к продаже
        print("Продажа")
        execute_order(exchange, 'BTC/USDT', current_position, amount=0.01)  # Пример объема позиции
    # Выводим баланс после каждой транзакции
    print_balance(exchange, 'BTC')
