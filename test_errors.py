from numpy import ndarray
from scipy.stats import ttest_ind


def start_aa_test(
        df_sales_np: ndarray, group_a: list[int], group_b: list[int], df_slice: slice, alpha: float
):
    # H0: утверждаем, что разницы нет
    false_positives = 0
    a_sliced = df_sales_np[group_a][:, df_slice]
    b_sliced = df_sales_np[group_b][:, df_slice]

    _, ps = ttest_ind(a_sliced, b_sliced)
    for p in ps:
        # Если данные очень нетипичны для такой гипотезы
        if p < alpha:
            false_positives += 1
        else:
            continue
    print(f'Шанс ошибки 1 рода: {false_positives / len(ps):.3f}')


def start_ab_test(
        df_sales_np: ndarray, group_a: list[int], group_b: list[int], df_slice: slice, alpha: float, effect_size: float
):
    # H1: утверждаем, что разница есть
    a_sliced = df_sales_np[group_a][:, df_slice]
    b_sliced = df_sales_np[group_b][:, df_slice]

    # Вносим эффект в группу B
    b_sliced_effect = b_sliced * (1 + effect_size)

    _, p = ttest_ind(a_sliced.sum(axis=1), b_sliced_effect.sum(axis=1))
    print(f'A/B Тест p-value: {p}')
    if p >= alpha:
        print('Ошибка 2 рода')
    else:
        print("Ок")
