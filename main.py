import pandas as pd
from config import FILENAME, ID_COLUMN_NAME, TRAIN_DEPTH_WEEK, VALID_DEPTH_WEEK, ALPHA, EFFECT_SIZE, AlgoEnum, ALGO
from test_errors import start_aa_test, start_ab_test
from groups_selection import start_greedy_selection, check_group_deviation, draw_groups_sales_plot, \
    pairwise_nearest_groups


def main():
    # Загрузка данных
    df = pd.read_excel(FILENAME)
    len_store_ids = len(df[ID_COLUMN_NAME].values)
    df_sales = df.drop(columns=[ID_COLUMN_NAME])
    df_sales_np = df_sales.to_numpy()
    n_weeks = df_sales_np.shape[1]

    # Проверка на достаточную длину
    assert TRAIN_DEPTH_WEEK + VALID_DEPTH_WEEK <= n_weeks, "Слишком большие TRAIN и VALID окна"

    # Индексы для тренировочного и валидационного периода
    train_start = n_weeks - TRAIN_DEPTH_WEEK - VALID_DEPTH_WEEK
    train_end = n_weeks - VALID_DEPTH_WEEK
    valid_start = train_end
    valid_end = n_weeks

    train_slice = slice(train_start, train_end)
    valid_slice = slice(valid_start, valid_end)

    # Расчет групп
    if ALGO == AlgoEnum.GREEDY:
        group_a, group_b = start_greedy_selection(df_sales_np, len_store_ids, train_slice)
    elif ALGO == AlgoEnum.PAIRWISE:
        group_a, group_b = pairwise_nearest_groups(df_sales_np, train_slice)
    else:
        raise Exception("Выберите алгоритм из доступных в AlgoEnum")

    check_group_deviation(df_sales_np, group_a, group_b, train_slice, valid_slice)
    draw_groups_sales_plot(df_sales, group_a, group_b, train_end)

    # Запуск тестов ошибок 1 и 2 рода
    start_aa_test(df_sales_np, group_a, group_b, valid_slice, ALPHA)
    start_ab_test(df_sales_np, group_a, group_b, valid_slice, ALPHA, EFFECT_SIZE)

    # Сохранение в Excel
    with pd.ExcelWriter("AB_groups_output.xlsx") as writer:
        df.iloc[group_a].to_excel(writer, sheet_name="Group_A", index=False)
        df.iloc[group_b].to_excel(writer, sheet_name="Group_B", index=False)


if __name__ == "__main__":
    main()
