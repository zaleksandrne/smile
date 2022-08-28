# SMILE

Мы написали класс SmileModel для классификации кликстрима.
Его так же можно переобучить, при необходимости и применить на новых данных для получения вероятностей.

На всякий случай мы добавили докер конфиг, и настроили контейнер.
Если лень настраивать окружение и тд - можно скачать контейнер, запустить его и получить вероятность через него. 

## Использование docker контейнера

Мы рекомендуем использовать docker hub. Для скачивания введите следующую команду в терминал: . 

### Собрать контейнер в папке проекта 

Если вы все же решили собрать докер из папки, то введите следующую команду:

build: `docker build . -t smile` 

### Запуск контейнера

Для запуска контейнера есть вот такая вот чудесная команда:

`docker run -v ~/dev/data:/code/data -e INPUT_FOLDER='data/input_data' -e OUTPUT_FOLDER='data/output_data' -it smile`

Что тут есть:
- `-v ~/dev/data:/code/data` - share директория, в ней должны быть `input_folder` и `output_folder`
- `-e INPUT_FOLDER='data/input_data'` - input_data директория, содержит `test.csv`. По умолчанию всегда берется первый файл
- `-e OUTPUT_FOLDER='data/output_data'` - output_data директория, куда сохранятся вероятности
- `-it smile` - название образа

## Использование разработанного класса

Создание класса выглядет вот так:
`smile = SmileModel(mode='time', verbose=True)`

- mode - параметр `mode` определяет мод, по которому будет произведена предпоготовка данных. Изначально планировалось 4 мода:
    - `time` - генерирует признаки заложенные изначально, после анализа данных для обучения
    - `score` - добавляет векторизацию токенов
    - `very_score` - добавляет векторизацию хэшей (безконтекстная модель)
    - `super_score` - добавляет использование контекстной модели для векторизации хэшей
- verbose - печаталь ли в консоль сообщения в процессе предобработки данных

Важно отметить, что моды: `[score, very_score, super_score]` в данный момент не работают. Так как они не дали никакого прироста к результату, 
мы решили не тратить на них время и не разрабатывать.

### Методы

- `fit(X, y)` - X это сырые данные типа dask.datasets.DataFrame. y принимает np.ndarray или pd.Series
- `with_model(model_path: str)` - model_path - путь к моделе в pickle. Лежит в репозитории `src/models/xgb_model.pkl`. 
- `predict_proba(X)` - X это сырые данные типа dask.datasets.DataFrame
- `score(X, y_true)` - X это сырые данные типа dask.datasets.DataFrame. y это np.ndarray или pd.Series

### Пример запуска предсказания вероятностей

```python
from dask import dataframe as dd
from src import SmileModel

train_path = '../data/train'  # it is .csv file, with sep='\t'

# to create an instace
smile = SmileModel(mode='time', verbose=True)

# to set a model (already fitted)
smile.with_model('src/models/xgb_model.pkl')

df = dd.read_csv(train_path, sep='\t')  # read data using dask
y = None

if 'DEF' in df.columns:
    X, y = df.drop('DEF', axis=1), df['DEF'].compute().values

if y is None:
    proba = smile.predict_proba(X)
else:
    smile.score(X, y)  # it will return roc_auc metric
```