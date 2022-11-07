<h2>Файл resnet.py</h2>
в классах 'ResNet' и 'BasicBlock', определяются слои нейронной сети Resnet.
<h2>Файл flir_resnet.py</h2>
В классе 'FLIR80Resnet', наследующим класс 'ImageClassificationBase', происходит инициализация нейронной сети Resnet.
<br>С помощью функции можно изменить количество слоев, таким образом провести эксперимент с разной размерностью Resnet.
<h2>Файл image_classificaton_base.py</h2>
В данном файле определены функции 'accuracy' (точность распознавания), 'confusion_matrix' (определение confusion matrix), 'plot_confusion' (построение heatmap для confusion matrix).
 Класс 'ImageClassificationBase' рассчитывает функцию потерь, а также точность предсказания для валидационной выборки.

