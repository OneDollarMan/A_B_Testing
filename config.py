from enum import Enum

TRAIN_DEPTH_WEEK = 40  # Сколько недель используем для подбора
VALID_DEPTH_WEEK = 12  # Сколько недель используем для оценки
GROUP_SIZE = 200
FILENAME = 'static/stores.xlsx'
ID_COLUMN_NAME = 'Store_ID'
ALPHA = 0.05 # Уровень значимости
EFFECT_SIZE = 0.03 # Размер эффекта для A/B теста


# Алгоритмы выбора групп
class AlgoEnum(str, Enum):
    GREEDY = 'greedy'
    PAIRWISE = 'pairwise'


# Выбранный алгоритм
ALGO = AlgoEnum.PAIRWISE
