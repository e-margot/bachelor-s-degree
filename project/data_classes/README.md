<h2>Файл my_coco.py</h2>
В данном файле определен класс 'MyCOCO', с помощью которого считывается и в дальнейшем визуализируется аннотация к набору данных 'FLIR ADAS Thermal Dataset'.
<h2>Файл augmentate_coco.py</h2>
С помощью класса 'AugMyCocoDetection' считывается набор данных 'FLIR ADAS Thermal Dataset', после чего этот набор данных аугментируется.
<h2>Файл my_coco_collection.py</h2>
С помощью класса 'MyCocoDetection' считывается набор данных 'FLIR ADAS Thermal Dataset' без дальнейших аугментаций (для снимков тепловизионного спектра).
<h2>Файл device_data_loader.py</h2>
Вспомогательный класс 'DeviceDataLoader' используется для перемещения нашей модели и данных в GPU по мере необходимости.
