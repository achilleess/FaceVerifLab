# FaceVerifLabs

## Установка

1. Создайте виртуальную среду и активируйте ее
    ```shell
    conda create -n faceid=3.7 -y
    conda activate faceid
    ```
2. Подгрузите текущий репозиторий
    ```shell
    git clone https://github.com/achilleess/FaceVerifLabs.git
    ```

3. Установите необходимые для работы пакеты

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
    В последствии в папке с нашим датасетом для каждой личности должна появится подпапка "aligned", которая содержит выпрямленные и обрезанные изображения найденных лиц. Структура папки с датасетом после выполнения команды:
    ```
    .
    ├── First Person Name
    │   ├── nonaligned
    │   │   ├── img_name_first.jpg
    │   │   ├── ...
    │   │   └── img_name_last.png
    │   ├── emmbedings.pkl
    │   └── aligned
    │       ├── img_name_first.jpg
    │       ├── ...
    │       └── img_name_last.png
    ├── ...
    │
    └── Last person Name
        ├── aligned
        │   ├── img_name_first.jpg
        │   ├── ...
        │   └── img_name_last.png
        ├── emmbedings.pkl
        └── nonaligned
            ├── img_name_first.jpg
            ├── ...
            └── img_name_last.png
    ```
3. На исходных изображения помимо целевых личностей могут появляться другие люди, а так же возможна неточность детектора. Нам необходимо вручную расчистить автоматически сформировавашеся "aligned" папки. После удаления ненужных лиц из "aligned" папок, выборку можно считать готовой.
 
## Обработка изобрежний и видеопотоков.
1. Для обработки изобрежнния введите следующую команду:
    ```shell
    python3 FaceID.py --dataset_dir путь_к_собранному_датасету --img_path путь_к_изобрежиняю
    ```
2. Для обработки сохраненного видо введите следующую команду:
    ```shell
    python3 FaceID.py --dataset_dir путь_к_собранному_датасету --video_path путь_к_видео
    ```
3. Для обработки видеопотока с подключенной веб-камеры введите следующую команду:
    ```shell
    python3 FaceID.py --dataset_dir путь_к_собранному_датасету --webcam_id новер_вебкамеры
    ```
