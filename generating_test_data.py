import numpy as np
import pandas as pd

# Параметры генерации
N = 2000  # Количество магазинов
min_sales = 1000  # Минимальная сумма продаж
max_sales = 10000  # Максимальная сумма продаж

# Создаем DataFrame
stores = [f"Store_{i}" for i in range(1, N+1)]
weeks = [f"Week_{i}" for i in range(1, 53)]

# Генерируем случайные данные
data = np.random.randint(min_sales, max_sales, size=(N, 52))
df = pd.DataFrame(data, index=stores, columns=weeks)
df.reset_index(inplace=True)
df.rename(columns={'index': 'Store_ID'}, inplace=True)
df.to_excel('static/stores.xlsx', index=False)