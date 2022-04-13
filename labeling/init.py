import os
import sys
from PyQt5 import uic
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.QtGui import QPixmap

import db

data_path = './photos.csv'

# Database
data_db = db.PhotoTable(data_path)

# pass
def get_class(object_id, photo_num):
    return data_db.get_class(object_id, photo_num)

# pass
def set_class(object_id, photo_num, class_num):
    class_name = get_class_name(class_num)
    data_db.set_class(object_id, photo_num, class_name)

# pass
def remove_class(object_id, photo_num):
    data_db.remove_class(object_id, photo_num)

# pass
def get_class_name(class_num):
    class_name = None
    if class_num == 1:
        class_name = 'bedroom'
    if class_num == 2:
        class_name = 'kitchen'
    if class_num == 3:
        class_name = 'bathroom'
    if class_num == 4:
        class_name = 'livingroom'
    if class_num == 5:
        class_name = 'hallway'
    return class_name

# pass
def get_last_data():
    return data_db.get_last_data()


class MainWindow(QMainWindow):

    # State
    count = 0
    amount = 0
    object_id = None
    photo_num = None
    class_name = None
    image_path = None
    objects = []  # список объектов
    photos = {}  # словарь фото объектов

    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi('design.ui', self)

        self.load_photos()
        self.set_last_data()
        self.set_counter()
        self.configure_ui()
        self.bind()
        self.update_ui_elements()

    # Загружаем список объектов и картинок
    def load_photos(self):
        self.objects = list(sorted(map(lambda y: int(y), filter(lambda x: x.isnumeric(), os.listdir('./photos')))))
        self.photos = dict([(obj, sorted(map(lambda y: int(y), filter(lambda x: x.isnumeric(), map(lambda x: x[:-5], os.listdir('./photos/' + obj.__str__())))))) for obj in self.objects])

    # Устанавливаем начальную картинку
    def set_last_data(self):
        self.object_id, self.photo_num = get_last_data()
        if self.object_id is None:
            self.object_id = self.objects[0]
            self.photo_num = self.photos[self.object_id][0]
        self.image_path = './photos/{}/{}.jpeg'.format(self.object_id, self.photo_num)

    # Высчитываем номер начальной картинки
    def set_counter(self):
        for obj in self.objects:
            if obj == self.object_id:
                self.count = self.amount + self.photos[obj].index(self.photo_num) + 1
            self.amount += len(self.photos[obj])
    
    # Конфигурируем UI
    def configure_ui(self):
        self.imageView.setMinimumSize(1, 1)
        self.imageView.setScaledContents(True)

    # Метод связывающий ui-события с методами. Вызывается единожды (при инициализации).
    def bind(self):
        self.button1.clicked.connect(
            lambda: self.handle_tap_on_class_button(1))
        self.button2.clicked.connect(
            lambda: self.handle_tap_on_class_button(2))
        self.button3.clicked.connect(
            lambda: self.handle_tap_on_class_button(3))
        self.button4.clicked.connect(
            lambda: self.handle_tap_on_class_button(4))
        self.button5.clicked.connect(
            lambda: self.handle_tap_on_class_button(5))
        self.leftButton.clicked.connect(lambda: self.hande_tap_on_arrows(True))
        self.rightButton.clicked.connect(
            lambda: self.hande_tap_on_arrows(False))

    # Handle actions
    # Обрабатывает нажатия на кнопки с классами
    def handle_tap_on_class_button(self, class_num):
        set_class(self.object_id, self.photo_num, class_num)
        self.set_next_image()

    # Обрабатывает нажатия на кнопки с стрелочками
    def hande_tap_on_arrows(self, direction):
        self.set_prev_image() if direction else self.set_next_image()

    # Обрабатывает нажатия на клавиши
    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_0:
            self.set_next_image()
        elif e.key() == Qt.Key.Key_9:
            self.set_prev_image()
        elif e.key() == Qt.Key.Key_1:
            set_class(self.object_id, self.photo_num, 1)
            self.set_next_image()
        elif e.key() == Qt.Key.Key_2:
            set_class(self.object_id, self.photo_num, 2)
            self.set_next_image()
        elif e.key() == Qt.Key.Key_3:
            set_class(self.object_id, self.photo_num, 3)
            self.set_next_image()
        elif e.key() == Qt.Key.Key_4:
            set_class(self.object_id, self.photo_num, 4)
            self.set_next_image()
        elif e.key() == Qt.Key.Key_5:
            set_class(self.object_id, self.photo_num, 5)
            self.set_next_image()
        elif e.key() == Qt.Key.Key_Delete:
            remove_class(self.object_id, self.photo_num)
            self.class_name = None
            self.update_ui_elements()

    # Handle mutations
    # Переключает на следующее фото
    def set_next_image(self):
        result = self.set_image(self.try_get_next_image, self.object_id, self.photo_num)
        self.class_name = get_class(self.object_id, self.photo_num) # перед if, потому что у последней фотки не обновляется класс
        if not result:
            return
        self.count += 1
        self.update_ui_elements()

    # Переключает на предыдущее фото
    def set_prev_image(self):
        result = self.set_image(self.try_get_prev_image, self.object_id, self.photo_num)
        if not result:
            return
        self.class_name = get_class(self.object_id, self.photo_num)
        self.count -= 1
        self.update_ui_elements()

    # Устанваливет картинку, проученную из data_getter, получаем класс картинки и обновляем интерфейс
    def set_image(self, data_getter, object_id, photo_num):
        if object_id is None or photo_num is None:
            return False
        object_id, photo_num, image_path = data_getter(object_id, photo_num)
        if image_path is None:
            return False
        self.object_id = object_id
        self.photo_num = photo_num
        self.image_path = image_path
        return True

    def try_get_next_image(self, object_id, photo_num):
        object_photos = self.photos[object_id]
        # Если это последнее фото у объекта
        if object_photos.index(photo_num) == len(object_photos) - 1:
            next_object_idx = self.objects.index(object_id) + 1
            if next_object_idx == len(self.objects):
                return 0, 0, None  # Это было последнее фото в последнем объекте
            object_id = self.objects[next_object_idx]
            object_photos = self.photos[object_id]
            photo_num = object_photos[0]  # надеюсь у нас нас пустых папок
        else:
            next_photo_idx = object_photos.index(photo_num) + 1
            photo_num = object_photos[next_photo_idx]
        next_image = self.get_photo(object_id, photo_num)
        return object_id, photo_num, next_image

    def try_get_prev_image(self, object_id, photo_num):
        object_photos = self.photos[object_id]
        if object_photos.index(photo_num) == 0:  # Если это первое фото у объекта
            prev_object_idx = self.objects.index(object_id) - 1
            if prev_object_idx < 0:
                return 0, 0, None  # Это было последнее фото в последнем объекте
            object_id = self.objects[prev_object_idx]
            object_photos = self.photos[object_id]
            # надеюсь у нас нас пустых папок
            photo_num = object_photos[len(object_photos) - 1]
        else:
            prev_photo_idx = object_photos.index(photo_num) - 1
            photo_num = object_photos[prev_photo_idx]
        prev_image = self.get_photo(object_id, photo_num)
        return object_id, photo_num, prev_image

    def get_photo(self, object_id, photo_num):
        return "./photos/{}/{}.jpeg".format(object_id, photo_num)

    # Update UI
    # Мастер-функция обновления ui (в общем случае следует вызывать ее)
    def update_ui_elements(self):
        self.update_info_label()
        self.update_image_view()

    def update_info_label(self):
        if self.object_id is None or self.photo_num is None:
            return
        class_name = self.class_name
        if class_name is None:
            class_name = "----"
        self.infoLabel.setText('id: {}, num: {}, class: {} ({}/{})'.format(self.object_id, self.photo_num, class_name, self.count, self.amount))

    def update_image_view(self):
        if self.image_path == None:
            return
        pixmap = QPixmap(self.image_path)
        self.imageView.setPixmap(pixmap)


app = QApplication(sys.argv)
mw = MainWindow()
mw.show()
sys.exit(app.exec_())
