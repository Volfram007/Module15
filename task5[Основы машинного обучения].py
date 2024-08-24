import os
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib
from matplotlib.pyplot import figure
from nltk.metrics import edit_distance

log_dir = "logs"
# Создаем папку, если она не существует
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_filename = os.path.join(log_dir, datetime.now().strftime("log_%Y-%m-%d_%H-%M-%S.log"))
# Настройка логирования
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    filemode='w',  # Используйте 'w' для записи в новый файл
    format=f'%(asctime)s | %(levelname)s:\n%(message)s\n',
    encoding='utf-8'
)


def print_log(message, log_on: bool = True, print_on: bool = True):
    if log_on:
        logging.info(
            f'=====================================================\n{message}\n'
            f'-----------------------------------------------------')
    if print_on:
        print(message)


# Применяем стиль ggplot для всех графиков, создаваемых с помощью matplotlib
plt.style.use('ggplot')

# Используется в Jupyter Notebook для отображения графиков внутри тетрадей
# %matplotlib inline
# Устанавливаем размер графиков по умолчанию (12x8 дюймов)
matplotlib.rcParams['figure.figsize'] = (12, 8)

# Отключаем предупреждения о возможных копиях DataFrame (SettingWithCopyWarning)
pd.options.mode.chained_assignment = None

# Чтение данных из
df = pd.read_csv('sberbank.csv')

"""Выводим статистику по таблице"""
# Выводим форму данных (количество строк, колонок) и типы данных в каждом столбце
print_log(f'Количество строк, колонок: {df.shape}', print_on=False)
# print_log(df.dtypes)

# Просмотра статистической сводки каждого столбца
# Этот метод показывает нам количество строк в столбце - count, среднее значение столбца - mean,
# столбец стандартное отклонение - std, минимальные (min) и максимальные (max) значения,
# а также границу каждого квартиля - 25%, 50% и 75%. Любые значения NaN автоматически пропускаются.
# Для категориальных признаков этот метод показывает: - Сколько уникальных значений в наборе данных - unique;
# top значения; частота появления значений - freg.
print_log(f"Сводка:\n{df.describe()}", print_on=False)
print_log(f'[life_sq]\n{df['life_sq'].describe()}', print_on=False)

"""Вычисляем процент и количество пропущенных данных по каждой колонке, сортируем по убыванию"""
# isnull()
# notnull()
missing_percent = df.isnull().mean() * 100  # Процент пропущенных данных
missing_percent = missing_percent[missing_percent > 0]
missing_percent = missing_percent.sort_values(ascending=False)
missing_count = df.isnull().sum()  # Количество пропущенных данных
missing_count = missing_count[missing_count > 0]
missing_count = missing_count.sort_values(ascending=False)

# Определяем максимальную длину названия колонки для выравнивания
max_len = max(len(col) for col in missing_percent.index)
tmp = ''
for col_name, pct, count in zip(missing_percent.index, missing_percent, missing_count):
    tmp += f"{col_name.ljust(max_len)}: {pct:.2f}% ({count})\n"
print_log(f'Вычисляем процент и количество пропущенных данных по каждой колонке [{tmp.count('\n') - 1}]\n{tmp}',
          print_on=False)

# values = df['cafe_sum_500_max_price_avg'].head(15)
# print(values)

"""Создание тепловой карты пропусков"""
# # Берем первые 30 колонок данных для дальнейшего анализа
# cols = df.columns[:30]
# # Определяем цвета для тепловой карты
# # зеленый цвет будет обозначать пропущенные данные, синий - данные, которые присутствуют
# colours = ['#000099', '#1aff12']
# # Выводим тепловую карту пропущенных значений по выбранным колонкам
# sns.heatmap(df[cols].isnull(), cmap=sns.color_palette(colours), yticklabels=False)
# # Вызов отображения графика
# plt.show()

"""Создание индикаторов пропущенных значений"""
tmp = ''
# Создание индикаторов пропущенных значений
for col in df.columns:
    missing = df[col].isnull()
    num_missing = np.sum(missing)

    if num_missing > 0:
        # print_log('создан недостающий индикатор для: {}'.format(col), print_on=False)
        tmp += f"Создан недостающий индикатор для: {format(col)}\n"
        df['{}_ismissing'.format(col)] = missing
print_log(tmp, print_on=False)

# Построение гистограммы количества пропущенных значений
ismissing_cols = [col for col in df.columns if 'ismissing' in col]
df['num_missing'] = df[ismissing_cols].sum(axis=1)

# Создание DataFrame для гистограммы
num_missing_counts = df['num_missing'].value_counts().reset_index()
num_missing_counts.columns = ['num_missing_values', 'count']
num_missing_counts = num_missing_counts.sort_values(by='num_missing_values')

# Вывод таблицы с пропущенными значениями
missing_table = df.isnull().sum().reset_index()
missing_table.columns = ['column', 'num_missing']
missing_table = missing_table[missing_table['num_missing'] > 0]
print_log("Таблица с количеством пропущенных значений в каждом столбце:", print_on=False)
print_log(missing_table, print_on=False)

# Построение гистограммы
plt.figure(figsize=(10, 6))
plt.bar(num_missing_counts['num_missing_values'], num_missing_counts['count'], color='skyblue')
plt.xlabel('Количество пропущенных значений в строке')
plt.ylabel('Количество строк')
plt.title('Гистограмма пропущенных значений в строке')
plt.xticks(num_missing_counts['num_missing_values'])  # Установить метки по оси X для всех значений
# plt.show()

"""Отбрасывание записей"""
# Это решение подходит только в том случае, если недостающие данные не являются информативными.
# Например, из гистограммы, построенной в предыдущем разделе, мы узнали, что лишь небольшое количество строк содержат
# более 35 пропусков
ind_missing = df[df['num_missing'] > 35].index
df_less_missing_rows = df.drop(ind_missing, axis=0)

"""Отбрасывание признаков"""
# В процентном списке, построенном ранее, мы увидели, что признак hospital_beds_raion имеет высокий процент недостающих
# значений. Мы можем полностью отказаться от этого признака
cols_to_drop = ['hospital_beds_raion']
df_less_hos_beds_raion = df.drop(cols_to_drop, axis=1)

"""Внесение недостающих значений"""
# med = df['life_sq'].median()
# print_log(med)
# df['life_sq'] = df['life_sq'].fillna(med)
#
# print_log(df['life_sq'])

# А теперь примените ту же стратегию заполнения сразу для всех числовых признаков
tmp = ''
df_numeric = df.select_dtypes(include=[np.number])
numeric_cols = df_numeric.columns.values
for col in numeric_cols:
    missing = df[col].isnull()
    num_missing = np.sum(missing)

    if num_missing > 0:  # вычисление только для тех столбцов, в которых отсутствуют значения.
        # print_log('вычисление пропущенных значений для: {}'.format(col), print_on=False)
        tmp += f"Вычисление пропущенных значений для: {format(col)}\n"
        df['{}_ismissing'.format(col)] = missing
        med = df[col].median()
        df[col] = df[col].fillna(med)
print_log(tmp, print_on=False)

# и для категориальных признаков
tmp = ''
df_non_numeric = df.select_dtypes(exclude=[np.number])
non_numeric_cols = df_non_numeric.columns.values

for col in non_numeric_cols:
    missing = df[col].isnull()
    num_missing = np.sum(missing)

    if num_missing > 0:  # вычисление только для тех столбцов, в которых отсутствуют значения.
        # print_log('вычисление пропущенных значений для: {}'.format(col))
        tmp += f"Вычисление пропущенных значений для: {format(col)}\n"
        df['{}_ismissing'.format(col)] = missing

        top = df[col].describe()['top']  # вменяйте в значение наиболее часто встречающееся значение.
        df[col] = df[col].fillna(top)
print_log(tmp, print_on=False)

"""Замена недостающих значений"""
# категориальные признаки
df['sub_area'] = df['sub_area'].fillna('_MISSING_')

# численные признаки
df['life_sq'] = df['life_sq'].fillna(-999)

"""Нетипичные данные (выбросы)"""
# # Гистограмма/коробчатая диаграмма
# df['life_sq'].hist(bins=100)
# # Чтобы изучить особенность поближе, построим еще одну диаграмму
# df.boxplot(column=['life_sq'])

"""Описательная статистика"""
print_log(f'[life_sq]\n{df['life_sq'].describe()}', print_on=False)

"""Столбчатая диаграмма"""
# df['ecology'].value_counts().plot.bar()

"""Неинформативные признаки"""
tmp = ''
num_rows = len(df.index)
low_information_cols = []

for col in df.columns:
    cnts = df[col].value_counts(dropna=False)
    top_pct = (cnts / num_rows).iloc[0]

    if top_pct > 0.95:
        low_information_cols.append(col)
        tmp += f"{col}: {top_pct * 100}\n"
        tmp += f"{cnts}\n"
        tmp += f"\n"
print_log(tmp, print_on=False)

"""Дубликаты записей"""
# отбрасываем неуникальные строки
df_dedupped = df.drop('id', axis=1).drop_duplicates()

# сравниваем формы старого и нового наборов
print_log(df.shape, print_on=False)
print_log(df_dedupped.shape, print_on=False)

# Найдем в нашем наборе дубликаты по группе критических признаков – full_sq,life_sq, floor, build_year, num_room
key = ['timestamp', 'full_sq', 'life_sq', 'floor', 'build_year', 'num_room']
tmp = df.fillna(-999).groupby(key)['id'].count().sort_values(ascending=False).head(20)
print_log(tmp, print_on=False)

key = ['timestamp', 'full_sq', 'life_sq', 'floor', 'build_year', 'num_room']
df_dedupped2 = df.drop_duplicates(subset=key)

print_log(df.shape, print_on=False)
print_log(df_dedupped2.shape, print_on=False)

"""Разные регистры символов"""
tmp = df['sub_area'].value_counts(dropna=False)
print_log(f'\nРазные регистры символов:\n{tmp}', print_on=False)

# Если в какой-то записи вместо Poselenie Sosenskoe окажется poselenie sosenskoe, они будут расценены как два разных
# значения. Нас это не устраивает.
sub_area_lower = df['sub_area'].str.lower().rename('sub_area_lower')
df = pd.concat([df, sub_area_lower], axis=1)
# print(f'{df['sub_area_lower'].value_counts(dropna=False)}')

"""Разные форматы данных"""
# # Признак timestamp представляет собой object, хотя является датой
print(df['timestamp'])

# # Чтобы было проще анализировать транзакции по годам и месяцам, значения признака timestamp следует преобразовать в
# # удобный формат:
# # Преобразование столбца 'timestamp' в формат datetime
timestamp_dt = pd.to_datetime(df['timestamp'], format='%Y-%m-%d').rename('timestamp_dt')
df = pd.concat([df, timestamp_dt], axis=1)
df['year'] = df['timestamp_dt'].dt.year
df['month'] = df['timestamp_dt'].dt.month
df['weekday'] = df['timestamp_dt'].dt.weekday

# Теперь DataFrame не будет фрагментирован
print_log(df['year'].value_counts(dropna=False), print_on=True)
print()
print_log(df['month'].value_counts(dropna=False), print_on=True)

"""Опечатки"""
df_city_ex = pd.DataFrame(
    data={'city': ['torontoo', 'toronto', 'tronto', 'vancouver', 'vancover', 'vancouvr', 'montreal', 'calgary']})
df_city_ex['city_distance_toronto'] = df_city_ex['city'].map(lambda x: edit_distance(x, 'toronto'))
df_city_ex['city_distance_vancouver'] = df_city_ex['city'].map(lambda x: edit_distance(x, 'vancouver'))
print_log(df_city_ex, log_on=False, print_on=False)

# Мы можем установить критерии для преобразования этих опечаток в правильные значения.
# Например, если расстояние некоторого значения от слова toronto не превышает 2 буквы,
# мы преобразуем это значение в правильное – toronto
msk = df_city_ex['city_distance_toronto'] <= 2
df_city_ex.loc[msk, 'city'] = 'toronto'
msk = df_city_ex['city_distance_vancouver'] <= 2
df_city_ex.loc[msk, 'city'] = 'vancouver'
print_log(df_city_ex, print_on=False)

"""Адреса"""
df_add_ex = pd.DataFrame(['123 MAIN St Apartment 15', '123 Main Street Apt 12   ', '543 FirSt Av', '  876 FIRst Ave.'],
                         columns=['address'])
print_log(df_add_ex, log_on=False, print_on=False)
'''
Ваш код должен выглядеть следующим образом:
df_add_ex['address_std'] = df_add_ex['address'].str.lower()
чтобы удалить лишние пробелы в начале и конце каждой строки в столбце 'address_std'.
Примените метод str.strip()
чтобы заменить все точки в строках столбца 'address_std' на пустые строки, то есть удалите точки из адресов.
Примените метод str.replace('\\.', '')
для замены всех слов "street" (с учетом границ слова) в адресах на сокращенное обозначение "st".
Примените метод str.replace('\\bstreet\\b', 'st')
для замены слова "apartment" (с учетом границ слова) на сокращенную версию "apt".
Примените метод str.replace('\\bapartment\\b', 'apt')
заменить слово "av" (с учетом границ слова) на "ave", что является стандартнымм сокращением "avenue".
Примените метод str.replace('\\bav\\b', 'ave')
'''
df_add_ex['address_std'] = df_add_ex['address'].str.lower()
df_add_ex['address_std'] = df_add_ex['address_std'].str.strip()
df_add_ex['address_std'] = df_add_ex['address_std'].str.replace('\\.', '')
df_add_ex['address_std'] = df_add_ex['address_std'].str.replace('\\bstreet\\b', 'st')
df_add_ex['address_std'] = df_add_ex['address_std'].str.replace('\\bapartment\\b', 'apt')
df_add_ex['address_std'] = df_add_ex['address_std'].str.replace('\\bav\\b', 'ave')
print_log(df_add_ex, print_on=False)
