# Пример использования Nvidia Triton Server

В этом репозитории хранится весь код, написанный для статьи "Запускаем ML-модели с помощью Nvidia Triton Server" на Habr. Чтобы лучше понять содержимое репозитория, ознакомьтесь с оригинальной [статьей](https://yandex.ru/video/preview/10510689359700548639).

## Запуск сервера

- Смонтируйте образ, используя [Dockerfile](./Dockerfile). Для этого выполните команду:

    ```console
    docker build -t dumb-stub .
    ```

- Создайте и запустите контейнер с тритон сервером с помощью команды:

    *Windows*:
    ```console
    docker run --name dumb-stub -d -p 8000:8000 --volume .\assets:/assets --volume .\models:/models dumb-stub
    ```

    *Linux/MacOS*:
    ```console
    docker run --name dumb-stub -d -p 8000:8000 --volume ./assets:/assets --volume ./models:/models dumb-stub
    ```

## Проверка сервера

- Создайте виртуальное окружение с помощью команды:

    ```console
    python -m venv venv
    ```

- Активируйте виртуальное окружение с помощью команды:

    *Windows*:
    ```console
    .\venv\Scripts\activate
    ```

    *Linux/MacOS*:
    ```console
    source venv/bin/activate
    ```

- Установите зависимости с помощью команды:

    ```console
    pip install -r requirements-test.txt
    ```

- Выполните скрипт [test_server](./test_server.py):

    ```console
    python test_server.py
    ```

Если вы все сделали правильно, на вашем экране должно появиться следующее изображение:

![test-image](./assets/image.jpg)
