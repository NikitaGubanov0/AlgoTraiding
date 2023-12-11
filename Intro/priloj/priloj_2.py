def generate_orders(data):
    orders = []
    for index, row in data.iterrows():
        if row['positions'] == 1:
            orders.append({'Date': index, 'Type': 'BUY', 'Price': row['Close']})
        elif row['positions'] == -1:
            orders.append({'Date': index, 'Type': 'SELL', 'Price': row['Close']})
    return orders

# Генерируем ордера
trade_orders = generate_orders(data)

# Выводим первые 5 ордеров
print(trade_orders[:5])
