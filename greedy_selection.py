import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from config import GROUP_SIZE


def group_mean(df_sales_np: np.ndarray, idx_list: list, data_slice: slice):
    return np.mean(df_sales_np[idx_list][:, data_slice], axis=0)


def start_greedy_selection(df_sales_np: np.ndarray, len_store_ids: int, train_slice: slice) -> (list[int], list[int]):
    # --- 2. Жадный алгоритм по тренировочным данным ---
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


def check_group_deviation(df_sales_np: np.ndarray, group_a: list[int], group_b: list, train_slice: slice, valid_slice: slice):
    # --- 3. Оценка на тренировке и валидации ---
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
    # --- 4. График средних продаж ---
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
