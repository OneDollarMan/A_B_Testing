import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.distance import cdist
from tqdm import tqdm
from config import GROUP_SIZE


def group_mean(df_sales_np: np.ndarray, idx_list: list, data_slice: slice):
    return np.mean(df_sales_np[idx_list][:, data_slice], axis=0)


def start_greedy_selection(df_sales_np: np.ndarray, len_store_ids: int, train_slice: slice) -> (list[int], list[int]):
    """
    Делит магазины на группы A и B по жадному алгоритму.

    Параметры:
        df_sales_np (np.ndarray): матрица [магазины x недели]
        len_store_ids (int): количество магазинов
        train_slice (slice): отрезок недель для анализа
    """
    np.random.seed(42)
    indices = np.arange(len_store_ids)
    np.random.shuffle(indices)

    group_a = [indices[0]]
    group_b = [indices[1]]
    used = set(group_a + group_b)

    with tqdm(total=GROUP_SIZE * 2) as pbar:
        while len(group_a) < GROUP_SIZE or len(group_b) < GROUP_SIZE:
            best_candidate = None
            min_diff = float('inf')

            for i in indices:
                if i in used:
                    continue

                if len(group_a) < GROUP_SIZE:
                    mean_A = group_mean(df_sales_np, group_a + [i], train_slice)
                    mean_B = group_mean(df_sales_np, group_b, train_slice)
                    diff_A = np.mean((mean_A - mean_B) ** 2)
                else:
                    diff_A = float('inf')

                if len(group_b) < GROUP_SIZE:
                    mean_A_alt = group_mean(df_sales_np, group_a, train_slice)
                    mean_B_alt = group_mean(df_sales_np, group_b + [i], train_slice)
                    diff_B = np.mean((mean_A_alt - mean_B_alt) ** 2)
                else:
                    diff_B = float('inf')

                if diff_A < diff_B and diff_A < min_diff:
                    best_candidate = (i, 'A')
                    min_diff = diff_A
                elif diff_B < min_diff:
                    best_candidate = (i, 'B')
                    min_diff = diff_B

            if best_candidate:
                idx, group = best_candidate
                used.add(idx)
                if group == 'A':
                    group_a.append(idx)
                else:
                    group_b.append(idx)
                pbar.update(1)
            else:
                break
    return list(map(int, group_a)), list(map(int, group_b))


def pairwise_nearest_groups(df_sales_np, train_slice) -> (list[int], list[int]):
    """
    Делит магазины на группы A и B по методу ближайших соседей.

    Параметры:
        df_sales_np (np.ndarray): матрица [магазины x недели]
        train_slice (slice): отрезок недель для анализа
    """
    data = df_sales_np[:, train_slice]  # Только обучающий период

    # Вычисляем матрицу расстояний (симметричная, 0 по диагонали)
    dists = cdist(data, data, metric='euclidean')
    # Чтобы не учитывать саму себя
    np.fill_diagonal(dists, np.inf)

    used = set()
    group_a = []
    group_b = []

    with tqdm(total=GROUP_SIZE * 2) as pbar:
        while len(group_a) < GROUP_SIZE and len(used) < len(df_sales_np):
            # Найдём ближайшую ещё неиспользованную пару
            min_dist = np.inf
            best_pair = None
            for i in range(len(df_sales_np)):
                if i in used:
                    continue
                # Индекс ближайшего неиспользованного соседа
                nearest = np.argsort(dists[i])
                for j in nearest:
                    if j not in used:
                        if dists[i, j] < min_dist:
                            best_pair = (i, j)
                            min_dist = dists[i, j]
                        break

            if best_pair is None:
                break  # Все пары разобраны

            i, j = best_pair
            used.update([i, j])
            group_a.append(i)
            group_b.append(int(j))
            pbar.update(2)

    return group_a, group_b


def check_group_deviation(df_sales_np: np.ndarray, group_a: list[int], group_b: list, train_slice: slice, valid_slice: slice):
    # --- Оценка на тренировке и валидации ---
    train_mean_A = group_mean(df_sales_np, group_a, train_slice)
    train_mean_B = group_mean(df_sales_np, group_b, train_slice)
    valid_mean_A = group_mean(df_sales_np, group_a, valid_slice)
    valid_mean_B = group_mean(df_sales_np, group_b, valid_slice)

    train_diff_pct = (np.mean(train_mean_A) - np.mean(train_mean_B)) / np.mean(train_mean_A)
    valid_diff_pct = (np.mean(valid_mean_A) - np.mean(valid_mean_B)) / np.mean(valid_mean_A)

    print(f'📊 Отклонение по тренировке: {train_diff_pct:.4f}')
    print(f'📊 Отклонение по валидации: {valid_diff_pct:.4f}')

    print(f'🟦 Группа A: {group_a}')
    print(f'🟥 Группа B: {group_b}')


def draw_groups_sales_plot(df_sales: pd.DataFrame, group_a: list[int], group_b: list[int], train_end: int):
    # --- График средних продаж ---
    weeks = df_sales.columns.tolist()
    df_sales_np = df_sales.to_numpy()
    mean_a_full = np.mean(df_sales_np[group_a], axis=0)
    mean_b_full = np.mean(df_sales_np[group_b], axis=0)

    plt.figure(figsize=(12, 6))
    plt.plot(weeks, mean_a_full, label='Группа A', linewidth=2)
    plt.plot(weeks, mean_b_full, label='Группа B', linewidth=2, linestyle='--')

    # Вертикальная линия границы train/val
    boundary_index = train_end
    plt.axvline(x=weeks[boundary_index], color='gray', linestyle=':', linewidth=2)
    plt.text(boundary_index + 0.5, max(mean_a_full.max(), mean_b_full.max()) * 0.95, 'валидация', color='gray')

    plt.title('Средние продажи по неделям: A vs B')
    plt.xlabel('Неделя')
    plt.ylabel('Продажи')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
