# FaceVerifLabs

## Установка

1. Создайте виртуальную среду и активируйте ее

    ```shell
    conda create -n faceid=3.7 -y
    conda activate faceid
    ```
2. Установите необходимые для работы пакеты

    ```shell
    pip install -r requirements.txt
    ```
    
## Подготовка тренировочной выборки
1. Прежде чем начать обрабатывать видео и изображения, необходимо подготовить датасет для личностей, которые в последствие будут идентифицироваться.
Для этого создаем папку со следующей структурой:
    ```
    .
    ├── First Person Name
    │   └── nonaligned
    │       ├── img_name_first.jpg
    │       ├── ...
    │       └── img_name_last.png
    ├── ...
    │
    └── Last person Name
        └── nonaligned
            ├── img_name_first.jpeg
            ├── ...
            └── img_name_last.jpg
    ```
2. Далее запускаем код для извлечения скрытых векторов для собранных изобрежний.
    ```shell
    python3 FaceID.py --prep_features --dataset_dir путь_к_собранному_датасету
    ```
