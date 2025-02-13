# Запускаем ML-модели с помощью Docker и Nvidia Triton Server

## Введение

Запускать и сопровождать модели машинного обучения в рамках веб-сервисов - это нетривиальная задача. Вам придется решать кучу проблем от взаимодействия различных частей вашего приложения с моделью до мониторинга ее производительности. А если модель не одна и вам грозит перспектива добавления новых моделей, то сложность задачи кратно увеличивается. Попытки самостоятельной разработки решений всех упомянутых задач могут обернуться большими трудностями и даже сильной головной болью. Именно поэтому мне стало интересно рассмотреть готовые решения для упрощения инференса и сопровождения моделей машинного обучения. Для меня было важно, чтобы выбранная технология обладала следующими характеристиками:
- Позволяла единообразно запускать модели машинного обучения. Это бы решило вопрос с добавлением новых моделей;
- Позволяла обрабатывать HTTP-запросы;
- Имела возможность подсчета и выгрузки значений важных метрик качества работы модели (например, `latency`);
- Имела docker-образ для контейнеризации итогового решения;
- Имела поддержку Python;

Одно из готовых решений, которое удовлетворяет всем перечисленным выше требованиям - это [`Nvidia Triton Server`](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/introduction/index.html). Более того, `Nvidia Triton Server` предлагает средства для управления количеством инстансов одной модели, средства для распределения ресурсов и балансировки нагрузки между этими инстансами, также тритон сервер обладает поддержкой популярных ML-фреймворков, и это далеко не полный список предлагаемых возможностей. В общем, именно поэтому мне бы хотелось познакомить вас с данным инструментом и продемонстрировать возможности его использования в ваших приложениях.

В этом тексте я постарался собрать полезную информацию о запуске моделей машинного обучения с помощью `Nvidia Triton Server` в docker-контейнерах. Этот текст предназначен в первую очередь для людей, которые только начинают работать с данной технологией, а не для опытных пользователей. Текст содержит инструкцию, которой не хватало мне самому в тот момент, когда я только начинал работать с тритоном. Ну и последнее замечание, весь код из данного текста можно найти в моем GitHub [репозитории](https://github.com/EvgrafovMichail/triton-example-habr).

## Модель

Выше упоминалось, что тритон сервер умеет работать со многими популярными фреймворками типа `TensorFlow` или `PyTorch`. Согласно терминологии тритона, функциональность, позволяющая работать с конкретным фреймворком, называется *бекендом*. В рамках этого текста я буду описывать некоторые принципы работы с Python-бекендом. Выбор данного бекенда позволит нам рассмотреть работу со сторонними зависимостями и работу с дополнительными данными в контейнере с моделью. Фактически, в рамках Python-бекенда мы не будем использовать никакой фреймворк, но будем описывать некоторую обертку вокруг существующей модели машинного обучения. Модель машинного обучения вместе с ее оберткой назовем *тритон моделью*. Вне учебных примеров Python-бекенд может быть полезен для запуска моделей с [HuggingFace](https://huggingface.co/), а все описанные принципы легко могут быть адаптированы для решения реальных задач.

Прежде чем начинать, нам необходимо определиться с моделью. В качестве модели мы будем использовать простенькую самописную заглушку, которая в ответ на любой запрос пользователя будет возвращать одну и ту же картинку, заранее загруженную в оперативную память компьютера. Это, на первый взгляд сомнительное, решение было принято, чтобы не перегружать материал деталями о конкретных моделях. Наша заглушка будет выглядеть так:

```python
from pathlib import Path
from random import uniform
from time import sleep

import cv2 as cv
import numpy as np


class DumbStub:
    _image: np.ndarray
    _sleep_time_max: float = 4
    _sleep_time_min: float = 1

    def __init__(self, path_to_image: Path) -> None:
        self._image = cv.imread(filename=path_to_image)
        self._image = cv.cvtColor(self._image, code=cv.COLOR_BGR2RGB)

    def generate_image(self, _: str) -> np.ndarray:
        sleep_time = uniform(self._sleep_time_min, self._sleep_time_max)
        sleep(sleep_time)

        return self._image
```

В методе `__init__` происходит чтение изображения. Фактически, этот шаг соответствует загрузке весов модели. Для чтения изображения используется библиотека [`OpenCV`](https://docs.opencv.org/4.x/), что создает для нас необходимость в работе со сторонними зависимостями при контейнеризации нашей модели. Т.е. готовым докер-образом нам уже не обойтись. Помимо этого у нас есть сторонние данные в виде картинки, которые также придется передавать в контейнер.

Метод `generate_image` имитирует процесс генерации изображения по текстовому описанию. На вход принимается некоторая строка, т.е. условный текстовый промпт. Но этот промпт никак не используется. В самом методе происходит выбор из диапазона $[1, 4]$ случайного числа `sleep_time`, которое соответствует длительности задержки между получением запроса и возвращением ответа. Далее происходит ожидание `sleep_time` секунд, после чего функция возвращает загруженное ранее изображение. Да, это немного, но для учебных целей вполне подойдет.

## Репозиторий

Чтобы запустить тритон сервер с моделью машинного обучения на борту, нам потребуется создать специальный конфигурационный файл тритон модели и файл с кодом обертки. Эти файлы необходимо будет расположить определенным образом в определенных директориях. Путь до директории со всеми созданными каталогами и файлами, которая называется *репозиторием*, мы сообщим серверу перед запуском. Подробную информацию о репозиториях можно найти в соответствующем [разделе](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_repository.html) официальной документации. Мы же с вами создадим базовый варианта репозитория. Исходя из личного опыта, могу сказать, что, вероятнее всего, именно подобную структуру вы будете использовать для решения большинства практических задач. Структура базового репозитория выглядит следующим образом:

```console
├─── ...
└───<model_repo>
    └───<model_name>
        ├───config.pbtxt
        └───<version>
            └───model.py
```

Кратко опишем элементы этой структуры:
- `model_repo` - это и есть репозиторий. Путь до этой директории необходимо будет сообщить тритон серверу перед его запуском. В нашем примере в роли репозитория будет выступать папка `models`, которую мы создадим в корне нашего проекта.
- `model_name` - директория с описанием тритон модели, которая будет функционировать в рамках данного репозитория. В общем случае в рамках одного тритон сервера одновременно может функционировать несколько тритон моделей, т.е. один репозиторий может содержать несколько таких директорий. Однако в рассматриваемом примере мы ограничимся одной тритон моделью, описание которой будет храниться в папке `dumb_stub`.
- `config.pbtxt` - это файл с дефолтной конфигурацией для конкретной тритон модели, запущенной на сервере. О нем мы поговорим позже.
- `version` - папка, в которой содержится конкретная версия кода данной тритон модели. Любая модель должна иметь хотя бы одну версию, поэтому в директории `model_name` должна содержаться хотя бы одна папка `version`. Имя папки с конкретной версией тритон модели должно быть валидным натуральным числом (считаем, что $0$ натуральным числом не является). В нашем примере имя этой папки - `1`.
- `model.py` - файл с конкретной реализацией обертки для модели машинного обучения. В нашем случае этот файл будет содержать обертку вокруг объекта `DumbStub`. Этот файл мы также подробнее рассмотрим ниже.

Таким образом в первом приближении структура нашего проекта будет выглядеть так: 

```console
├─── ...
└───models
    └───dumb_stub
        ├───config.pbtxt
        └───1
            └───model.py
```

Теперь, когда структура нам ясна, можно приступать к ее наполнению.

## config.pbtxt

Как было сказано ранее, файл `config.pbtxt` - это конфигурационный файл для конкретной тритон модели, запущенной на сервере. В файле `config.pbtxt` хранится основная информацию о том, как сервер должен запускать вашу модель, а также в каких форматах тритон модель будет принимать и возвращать данные. 

Сам по себе формат `.pbtxt` является человекочитаемым форматом и частично напоминает формат `.json`. В формате `.pbtxt` значение какой либо настройки можно записать в виде key-value пары: `key: value`. Значение настроек можно группировать в структуры:
```pbtxt
my_struct: {
    key1: val1
    key2: val2
}
```

А структуры можно помещать в списки:
```pbtxt
struct_list: [
    {
        key1: val1
        key2: val2
    },
    {
        key1: val3
        key2: val4
    }
]
```

### backend

Каждый файл `config.pbtxt` обязан содержать информацию о бекенде, которая записывается в качестве значения ключа `backend`. В нашем случае этому ключу будет соответствовать значение `python`: `backend: "python"`. Значения бекенда записывается в формате строки, а потому заключено в двойные кавычки. Стоит отметить, что при работе с некоторыми фреймворками вместо поля `backend` или вместе с ним требуется описание поля `platfrom`. Однако для Python-бекенда эта опция не требуется, поэтому в нашем конфиге ее не будет.

### name

В рамках одного тритон сервера может быть запущено несколько тритон моделей. Чтобы понимать, какую модель использовать для обработки конкретного запроса, сервер присваивает каждой тритон модели уникальный идентификатор. При отправке запроса пользователь должен использовать этот идентификатор вместе с версией модели, чтобы сообщить серверу информацию об обработчике запроса.

Этот уникальный идентификатор называется *именем модели*. По умолчанию в качестве имени модели тритон сервер использует имя директории, в которой расположен файл `config.pbtxt`. В нашем случае дефолтным именем модели будет `dumb_stub`. Если нас не удовлетворяет выбранное имя модели, мы можем изменить его, добавив в конфиг значение для ключа `name`. Поскольку нас вполне устраивает дефолтное название модели, в нашем случае ключ `name` не будет включен в конфиг. 

### max_batch_size

Следующий ключ в нашем конфиге - это ключ `max_batch_size`. Его значение - это целое неотрицательное число. Данный ключ позволяет задать максимальный размер батча данных, в случае, если используемая модель машинного обучения поддерживает обработку данных батчами. Если модель поддерживает работу с батчами и мы хотим воспользоваться этой функциональностью, то связанное с ключом значение должно быть натуральным числом, соответствующим максимальному размеру батча. Если же модель не поддерживает обработку данных батчами, или нам просто не требуется осуществлять такую обработку, значение, связанное с данным ключом, должно быть равно нулю. В нашем случае модель машинного обучения не поддерживает обработку данных батчами, поскольку в качестве параметра `DumbStub` используется одна единственная строка. Поэтому в качестве значения, связываемого с ключом `max_batch_size`, мы будем использовать $0$: `max_batch_size: 0`.

### input и output

Теперь нам необходимо описать данные, которые будут подаваться на вход тритон модели, и данные, которые будут возвращаться ею в результате обработки пользовательских запросов. В качестве входных и выходных данных тритон сервер использует тензоры. Под тензорами понимаются N-мерные массивы. Тензоры описываются следующими структурами:
```pbtxt
{
    name: "tensor_name"
    data_type: TYPE
    dims: [ -1 ]
}
```

Поле `name` - это уникальный идентификатор данного тензора. Используя идентификатор, мы сможем обращаться к конкретному тензору, связанному с ним, в коде. В целом, если вы не [работаете](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html#special-conventions-for-pytorch-backend) с `PyTorch`, то особых ограничений на имена тензоров нет.

Следующее поле - поле `data_type`, которое описывает тип данных элементов тензора. Разные бекенды поддерживают разные наборы типов данных. Официальная [документация](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html#datatypes) содержит очень удобную таблицу для определения типов данных, поддерживаемых в рамках конкретных бекендов. В приведенной таблице нас интересует первая колонка `Model Config` - это наименование типов данных, которые нужно использовать для заполнения `config.pbtxt`, и последняя колонка `NumPy` - это типы данных, доступные для Python-бекенда.

Последнее поле в структуре тензора - это `dims`, - описание количества элементов для каждой размерности тензора. Значение `dims` представляет собой список целых чисел. Значения всех чисел в списке должны быть или положительными, или равными $-1$. Если элемент в позиции `i` - это положительное число, значит `i`-ая размерность тензора должна содержать количество элементов, строго равное указанному числу. Если же в позиции `i` находится $-1$, тогда `i`-ая размерность может содержать любое число элементов, большее $0$.

Чтобы лучше понять это правило, рассмотрим пару примеров. Предположим у нас есть некоторый тензор, полю `dims` которого соответствует следующее значение: `[ 2, 3 ]`. В этом случае корректными значениями тензора будут являться только матрицы, размером `2 x 3`, т.е. матрицы, состоящие из двух строк и трех столбцов. Теперь рассмотрим второй тензор, поле `dims` которого описывается значением: `[ 2, -1 ]`. В этом случае корректным значением будет любая матрица, состоящая из двух строк. Т.е. такая `2 x 3`, такая `2 x 10` и даже такая `2 x 1` матрица будет корректным значением данного тензора.

Теперь, понимая правила определения тензоров, мы готовы описать входные и выходные данные используемой тритон модели. За описание входных данных в `config.pbtxt` отвечает поле `input`. Значение этого поля - список описаний входных тензоров. В нашем примере полю `input` будет соответствовать список из одного тензора. Назовем наш входной тензор `prompt`. Поскольку единственный аргумент используемой модели-заглушки `DumbModel` - это строка, входной тензор должен содержать данные строкового типа данных. Из этих соображений в качестве значения поля `data_type` указываем `TYPE_STRING`. Полю `dims` присвоим значение `[ 1 ]`. Т.е. в качестве входа тритон модели будет выступать тензор, состоящий из одного элемента - текстовой строки, и доступный в коде под именем `prompt`. Поле `input` будет выглядеть так:
```pbtxt
input: [
    {
        name: "prompt"
        data_type: TYPE_STRING
        dims: [ 1 ]
    }
]
```

За описание выходных данных тритон модели отвечает поле `output`. Значение поля `output` также является списком описаний тензоров. Так же как и `input` список `output` в нашем примере будет состоять из одного тензора, который мы назовем `image`. Результатом использования нашей модели-заглушки `DumbModel` будет трехканальное изображение, которое является трехмерным тензором. Элементы этого тензора - беззнаковые восьмибитные целые числа, поэтому в качестве значения поля `data_type` мы будем использовать `TYPE_UINT8`, а в качестве значения поля `dims` - `[ -1, -1, 3 ]`. Таким образом выход тритон модели - это трехканальное изображение с любой шириной и высотой. Поскольку мы используем одну и ту же картинку в качестве ответов модели, можно было бы зафиксировать и ширину с высотой. Но мы не станем этого делать, чтобы при желании иметь возможность заменить изображение без дополнительных изменений в конфиге. Итак, поле `output` будет выглядеть так:
```pbtxt
output: [
    {
        name: "image"
        data_type: TYPE_UINT8
        dims: [ -1, -1, 3 ]
    }
]
```

### Итоговый конфиг

Для базового варианта конфига перечисленных полей вполне достаточно. Если вам требуется более детальная настройка тритон модели с описанием конкретной видеокарты для инференса и настройкой кеширования ответов, ознакомьтесь с соответствующим разделом [документации](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html#model-configurations). В нашем же случае конфиг будет выглядеть так:

```pbtxt
backend: "python"
max_batch_size: 0
input: [
    {
        name: "prompt"
        data_type: TYPE_STRING
        dims: [ 1 ]
    }
]
output: [
    {
        name: "image"
        data_type: TYPE_UINT8
        dims: [ -1, -1, 3 ]
    }
]
```

Также в документации есть [метариал](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html#custom-model-configuration) о порядке работы с несколькими конфигурационными файлами. Создание нескольких файлов конфигурации для одной и той же тритон модели может быть полезно, например, если модель требуется запускать на разных серверах с разными аппаратными ресурсами. Однако ознакомление с этими возможностями выходит за рамки данного текста.

## model.py

В рамках Python-бекенда тритон по умолчанию будет искать весь код, связанный с тритон моделью, в файле `model.py`. Вы можете поместить код в файл с другим именем, но тогда вам придется явным образом указать имя этого файла в конфиге, описав поле `default_model_filename`. Мы будем использовать дефолтное имя.

В файл `model.py` мы с вами сразу поместим наш класс `DumbStub`. К сожалению, мы не сможем хранить этот код в отдельном Python-модуле, поскольку после запуска сервера тритон будет работать исключительно с файлом `model.py`. Импорт содержимого из любого модуля, не установленного в виртуальное окружение сервера, будет приводить к возникновению исключения `ImportError`.

Файл `model.py` обязательно должен содержать класс `TritonPythonModel` - тритон обертку для работы с моделью машинного обучения. Именно этот класс и будет использоваться для выполнения всей полезной работы сервером. Мы будем использовать этот класс для инициализации модели-заглушки `DumbStub` и обработки пользовательских запросов с ее помощью. Интерфейс класса `TritonPythonModel` состоит из 4 методов:
- `auto_complete_config`. Этот метод является необязательным для реализации. Если метод был реализован, то он будет выполнен до загрузки тритон модели в отдельном экземпляре интерпретатора Python. `auto_complete_config` предназначен для анализа и дополнения конфигурационного файла `config.pbtxt`. Например, в этом методе можно проверить, содержит ли конфиг поле `max_batch_size`. Если такое поле в конфиге не найдено, данный метод позволяет его динамически добавить. Поскольку мы описали все необходимые характеристики модели в конфиге на предыдущем шаге, у нас нет необходимости в реализации этого метода.
- `initialize`. Данный метод используется для инициализации тритон модели. В этом методе можно описать все подготовительные действия, которые необходимо выполнить, прежде чем начинать обрабатывать запросы пользователей. Например, в данный метод можно поместить код загрузки весов и инициализации модели машинного обучения. Метод выполняется всего один раз и тоже является необязательным для реализации.
- `execute`. Этот метод является обязательным. В нем содержится код, который будет использоваться для обработки пользовательских запросов.
- `finalize`. Этот необязательный метод можно рассматривать как деструктор. Он выполняется в момент прекращения работы сервера для того, чтобы освободить все захваченные ресурсы.

Мы будем реализовывать два метода: `initialize` и `execute`.  

### initialize

В методе `initialize` нам необходимо создать экземпляр класса `DumbStub` и сохранить его в качестве атрибута экземпляра класса `TritonPythonModel`, чтобы иметь возможность в дальнейшем обращаться к его методам. Для создания экземпляра класса `DumbStub` в его метод `__init__` необходимо передать путь до изображения. Мы будем использовать следующий путь до изображения `/assets/image.jpg`. Теперь при запуске нашего сервера в докере нам нужно не забыть разместить картинку по данному адресу файловой системы контейнера. Итак, метод `initialize` будет выглядеть так:

```python
...

import triton_python_backend_utils as pb_utils

...

class TritonPythonModel:
    _model: DumbStub

    def initialize(self, _: dict) -> None:
        path_to_image = Path("/assets/image.jpg")

        if not path_to_image.exists():
            raise FileNotFoundError(f"there is file: {path_to_image}")

        pb_utils.Logger.log_info(
            f"try to load model weights: {path_to_image}"
        )
        self._model = DumbStub(path_to_image=path_to_image)
        pb_utils.Logger.log_info("model weights successfuly loaded")
```

Разберем подробнее приведенный листинг кода. Начнем с импорта `triton_python_backend_utils`. Данной библиотеки нет в PyPI, так что у вас не получится скачать ее с помощью pip. Однако, в случае необходимости, вы можете скачать исходники из репозитория проекта и собрать эту библиотеку самостоятельно. В случае если вы будете запускать сервер с помощью докера и вас не напрягает отсутствие подсказок от IDE при работе с `triton_python_backend_utils`, библиотеку можно не собирать. `triton_python_backend_utils` уже будет собрана и установлена в докер-контейнере на базе официального образа от Nvidia.

Со строки `N` начинается определение метода `initialize`. Метод `initialize` обладает одним аргументом - словарем с конфигурацией модели и дополнительной метаинформацией, типа версии модели. Поскольку ничего из этого мы использовать не планируем, этот аргумент был назван `_`.

В строках `N1` - `N2` происходит инициализация пути до изображения и проверка существования файла по данному пути. Если файла не существует, мы возбудим исключение типа `FileNotFoundError` в строке `N`.

В строках `N1` - `N2` кода мы создаем экземпляр класса `DumbStub`, который будет работать с изображением `/assets/image.jpg`. Созданный экземпляр сохраняется в приватный (ха-ха) атрибут `_model` экземпляра класса `TritonPythonModel` для дальнейшего использования в `execute`.

До и после создания экземпляра мы вызываем метод `log_info` объекта `pb_utils.Logger`. Дело в том, что печать в стандартный поток вывода с помощью других методов (`print` или `logging`) не будет иметь никакого эффекта. Чтобы писать в поток вывода логи, мы должны использовать нативный логгер тритон сервера - `pb_utils.Logger`. Подробнее ознакомиться с возможностями данного логгера можно [тут](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/python_backend/README.html#logging).

### execute

В методе `execute` нам необходимо извлечь присланную пользователем строку из входного тензора, передать ее в качестве аргумента в метод `generate_image` экземпляра класса `DumbStub`, дождаться, когда он "сгенерирует" изображение, и вернуть это изображение в качестве результата. Таким образом метод `execute` будет выглядеть так:

```python
...

import triton_python_backend_utils as pb_utils

...

class TritonPythonModel:
    ...
    def execute(self, requests: list[Any]) -> list[Any]:
        responses = []

        for request in requests:
            prompt: bytes = pb_utils.get_input_tensor_by_name(
                request, "prompt"
            ).as_numpy()[0]
            prompt = prompt.decode()
            pb_utils.Logger.log_info(f"got next prompt for generation: {prompt}")

            image = self._model.generate_image(prompt)
           
            response_tensor = pb_utils.Tensor("image", image)
            response = pb_utils.InferenceResponse(output_tensors=[response_tensor])
            responses.append(response)

        return responses
```

Рассмотрим и этот листинг, потому что в нем явно происходит больше, чем было описано в предыдущем абзаце. Начнем с сигнатуры метода. Единственный аргумент метода `execute` - `requests`, список объектов `pb_utils.InferenceRequest`. Именно в этих объектах содержатся те самые тензоры, с которыми нам предстоит работать. Возвращать метод `execute` должен список объектов `pb_utils.InferenceResponse`, которые будут содержать тензоры с результатами вычислений. Именно поэтому в строке `N` мы создаем пустой список `responses`, в который в дальнейшем поместим результаты обработки запросов.

Далее в теле цикла мы "генерируем" изображения для каждого запроса из списка `requests`. В строках `N1` - `N2` мы с помощью функции `pb_utils.get_input_tensor_by_name` извлекаем из текущего запроса тензор с именем `"prompt"`. Т.е. мы извлекаем тот самый тензор, содержащий строковое описание картинки, который мы определили в `config.pbtxt`. Извлеченный тензор мы переводим в `np.ndarray` с помощью метода `as_numpy`, после чего читаем его первый элемент. В результате всех этих манипуляций переменная `prompt` будет указывать на объект типа `bytes`, поэтому в строке `N` мы вызываем метод `decode` и связываем указатель `prompt` с декодированной строкой. В строке `N` мы логируем декодированный промпт, потому что логирование входных данных - всегда полезно.

В строке `N` мы "генерируем" изображение с помощью созданного ранее экземпляра класса `DumbStub`.

В строках `N1` - `N2` мы сохраняем "сгенерированное" изображение в виде `pb_utils.Tensor` и записываем созданный тензор в `pb_utils.InferenceResponse`. Полученный объект `pb_utils.InferenceResponse` добавляется в конец списка `responses`, который возвращается в качестве результата выполнения метода в строке `N`. 

Этого кода вполне хватит для наших учебных целей. Однако, если вы хотите узнать больше, например, как формировать ответ сервера в случае возникновения исключения при выполнении запроса, можете ознакомиться с [документацией](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/python_backend/README.html#python-backend) по Python-бекенду.

С итоговым содержимым файла `model.py` вы можете ознакомиться [тут](https://github.com/EvgrafovMichail/triton-example-habr/blob/triton-server-code/models/dumb_stub/1/model.py).

## Контейнеризация

Репозиторий создан. Конфигурация описана, код для обработки запросов реализован. Теперь нам остается запустить наш сервер в докер-контейнере. Прежде чем монтировать образ и запускать контейнер, нам потребуется добавить несколько файлов в наш проект. Мы добавим файл `requirements.txt`, `Dockerfile` и директорию `assets/`, в которую поместим следующее изображение:

![image.jpg](./assets/image.jpg)

Таким образом итоговый проект будет иметь следующую структуру:

```console
├─── ...
├───assets
│   └───image.jpg
├───models
│   └───dumb_stub
│       ├───config.pbtxt
│       └───1
│           └───model.py
├───Dockerfile
└───requiremets.txt
```

`assets/` будет использована для передачи нашего изображения в файловую систему контейнера. 

### requirements.txt

`requirements.txt` необходим, поскольку объект `DumbStub`, который мы используем для обработки запросов, зависит от библиотеки `OpenCV`. В виртуальном окружении тритон сервера нет этой библиотеки, поэтому нам придется ее каким-то образом туда установить. На мой взгляд, самый простой способ это сделать - установить ее с помощью pip в виртуальное окружение контейнера до запуска самого сервера. Именно поэтому мы и добавляем файлик `requirements.txt` со всеми зависимостями, который будет скопирован в контейнер и использован там. Содержимое файла будет выглядеть так:

```txt
opencv-python-headless==4.10.0.82
```

При указании версий библиотек стоит быть осторожным, поскольку по умолчанию в контейнере с тритон сервером будет установлен интерпретатор Python версии 3.10. Если вы не намерены кастомизировать версию интерпретатора, то, возможно, самые актуальные версии той или иной библиотеки будут для вас недоступны.

### Dockerfile

Теперь опишем `Dockerfile`, который понадобится для запуска нашего проекта. `Dockerfile` будет выглядеть следующим образом:

```Dockerfile
FROM nvcr.io/nvidia/tritonserver:23.12-py3

WORKDIR /app

EXPOSE 8000
EXPOSE 8002

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt --no-cache-dir

CMD [ "tritonserver", "--model-repository=/models" ]
```

В строке `1` мы наследуемся от официального образа. В данном примере я использую образ `nvcr.io/nvidia/tritonserver:23.12-py3`. Однако, если этот образ не подходит для ваших нужд, вы можете ознакомиться со [списком доступных образов](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver) и выбрать более подходящий.

В строках `5` - `6` мы документируем намерение опубликовать порты `8000` и `8002`. Тритон сервер использует порт `8000` для работы с инференс-запросами в формате HTTP. Порт `8002` используется для предоставление метрик. Еще есть порт `8001`, который используется для коммуникации по `gRPC`. Но в этом примере мы хотим, чтобы наш сервер обрабатывал только HTTP-запросы, поэтому порт `8001` в `Dockerfile` не упоминается.

В строках `8` - `9` мы копируем файл `requirements.txt` и устанавливаем необходимые зависимости. В строке `11` мы запускаем тритон сервер.

## Запускаем сервер

Чтобы запустить сервер, нам понадобится смонтировать образ. В корне проекта выполним команду:
```console
docker build -t dumb-stub .
```

После того как образ смонтирован, создадим и запустим контейнер с тритон сервером, выполнив команду:
```
docker run --name dumb-stub -d -p 8000:8000 -p 8002:8002 --volume .\assets:/assets --volume .\models:/models dumb-stub
```

**Примечание**: При создании биндингов директорий я использую `\`, поскольку работаю на `Windows`. Если у вас UNIX-подобная ОС, используйте `/`.

Данная команда запустит контейнер `dumb-stub` с тритон сервером. При этом модели в рамках сервера будут запущены на CPU. Если вы хотите запустить ваши модели на другом вычислительном устройстве, ознакомьтесь с этой [инструкцией](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/getting_started/quickstart.html#run-on-system-with-gpus).

В случае успешного запуска в самом контейнере вы должны найти следующие логи:
```console
...
... model.py:36] try to load model weights: /assets/image.jpg
... model.py:40] model weights successfuly loaded
... model_lifecycle.cc:818] successfully loaded 'dumb_stub'
...
+-----------+---------+--------+
| Model     | Version | Status |
+-----------+---------+--------+
| dumb_stub | 1       | READY  |
+-----------+---------+--------+
...
... grpc_server.cc:2495] Started GRPCInferenceService at 0.0.0.0:8001
... http_server.cc:4619] Started HTTPService at 0.0.0.0:8000
... http_server.cc:282] Started Metrics Service at 0.0.0.0:8002
```

Сервер запущен и готов к работе.

## Отправляем запросы

Тема взаимодействия с сервером немного выходит за рамки этого текста, поэтому изложу основные положения, не вдаваясь в детали. После запуска тритон сервера все HTTP-запросы на инференс должны отправляться по адресу `http://127.0.0.1:8000`. Для коммуникации тритон использует протокол [kserve](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md). Однако, вам не нужно составлять тело запросов и парсить ответы сервера вручную. Гораздо удобнее воспользоваться официальным HTTP-клиентом для работы с тритон сервером - [`tritonclient`](https://pypi.org/project/tritonclient/). Вы можете установить его при помощи pip:
```console
pip install tritonclient[http]
```

Протестировать работоспособность нашего сервера можно с помощью следующего кода:

```python
import matplotlib.pyplot as plt
import numpy as np
import tritonclient.http as triton_http


def get_image(
    triton_client: triton_http.InferenceServerClient,
    model_name: str,
    model_version: str,
) -> np.ndarray | None:
    if not triton_client.is_model_ready(
        model_name=model_name,
        model_version=model_version,
    ):
        return None
        
    prompt = np.array(["beautiful picture"], dtype=np.object_)
    prompt_tensor = triton_http.InferInput(
        name="prompt",
        shape=list(prompt.shape),
        datatype=triton_http.np_to_triton_dtype(prompt.dtype),
    )
    prompt_tensor.set_data_from_numpy(prompt)

    response = triton_client.infer(
        model_name=model_name,
        model_version=model_version,
        inputs=[prompt_tensor],
    )
    return response.as_numpy(name="image")


def main() -> None:
    triton_client = triton_http.InferenceServerClient(
        url="127.0.0.1:8000"
    )
    model_name = "dumb_stub"
    model_version = "1"
    print("Send test request")
    print("Wait for response...")

    if (
        image := get_image(triton_client, model_name, model_version)
    ) is None:
        print("Model is not ready... Check your server")
        return
    
    print("Got image from server")
    plt.imshow(image)
    plt.show()

    
if __name__ == "__main__":
    main()
```

Для выполнения этого кода, помимо `tritonclient`, вам потребуется установить `Matplotlib` для визуализации полученного изображения. Отдельная установка `NumPy` не требуется, поскольку он будет установлен в качестве зависимости для `Matplotlib`. В результате выполнения этого кода у вас на экране должно появиться изображение, которое сохранено в файле `assets/image.jpg`. 

## Заключение

В данном тексте мы познакомились с `Nvidia Triton Server` - технологией, которая значительно упрощает запуск моделей машинного обучения и их использование в веб-приложениях. Мы рассмотрели основы данной технологии и научились запускать докер-контейнер с тритон сервером.

Разумеется, данный текст - не исчерпывающее руководство, а скорее гайд для тех, кто только начинает работать с `Nvidia Triton Server`. И, разумеется, как и все люди, автор не застрахован от ошибок и заблуждений. Так что, если вы нашли какую-либо неточность или какой-то аспект не был освещен достаточно полно, обязательно дайте знать в комментариях.

Также приглашаю вас в свой [канал](https://t.me/misha_porgommist), где я пишу небольшие заметки про Python и разработку.
