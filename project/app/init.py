import os
import sys
import asyncio
from asyncqt import QEventLoop
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QSizePolicy, QWidget
from PyQt5.QtWidgets import  QHBoxLayout, QVBoxLayout, QGridLayout, QFormLayout
from PyQt5.QtWidgets import QListView, QPushButton, QProgressBar, QTreeView, QFileDialog
from PyQt5.QtGui import QStandardItemModel, QStandardItem, QIcon

from networks.classifier.ImageClassifier import ImageClassifier
from networks.siamese.ImageSimilarity import ImageSimilarity

from app.group_images import group_images


app = QApplication(sys.argv)

def start_app():
    img_c=ImageClassifier()
    img_sim=ImageSimilarity()
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)
    with loop:
        window = AppMainWindow()
        window.img_c = img_c
        window.img_sim = img_sim
        window.show()
        loop.run_forever()

async def add_success_callback(fut, callback):
    result = await fut
    await callback(result)

class AppMainWindow(QMainWindow):


    # State
    images = []
    is_loading = False
    progress = 0
    grouped_images = {}

    # UI
    container = QHBoxLayout()
    leftContainer = QVBoxLayout()
    leftList = QListView()
    selectImagesButton = QPushButton()
    centralContainer = QVBoxLayout()
    processButton = QPushButton()
    progressBar = QProgressBar()
    rightTreeView= QTreeView()

    def __init__(self):
        super().__init__()

        self.create_ui()
        self.configure_layout()
        self.configure_ui()
        self.bind()

        self.update_ui()

    def create_ui(self):
        widget = QWidget()
        widget.setLayout(self.container)
        self.setCentralWidget(widget)
        
        self.container.addLayout(self.leftContainer)
        self.container.addLayout(self.centralContainer)
        self.container.addWidget(self.rightTreeView)

        self.leftContainer.addWidget(self.leftList)
        self.leftContainer.addWidget(self.selectImagesButton)

        self.centralContainer.addWidget(self.processButton)
        self.centralContainer.addWidget(self.progressBar)

    def configure_ui(self):
        self.setWindowTitle('Демонстрационное приложение')

        self.leftList.setModel(QStandardItemModel())
        self.leftList.setIconSize(QSize(100, 100))
        self.selectImagesButton.setText('Выберите фото...')

        self.processButton.setText('=>')

        self.progressBar.setTextVisible(False)

        self.rightTreeView.setModel(QStandardItemModel())
        self.rightTreeView.setIconSize(QSize(100, 100))
        self.rightTreeView.setHeaderHidden(True)
    
    def configure_layout(self):
        self.setGeometry(100, 100, 800, 600)
        self.setSizePolicy(QSizePolicy())

        self.processButton.setFixedSize(100, 25)
        self.progressBar.setFixedSize(100, 25)
    
    def bind(self):
        self.selectImagesButton.clicked.connect(self.handle_click_on_selectImageButton)
        self.processButton.clicked.connect(self.handle_click_on_processButton)

    # Handlers
    def handle_click_on_selectImageButton(self):
        options = QFileDialog.Options()
        fileNames, _ = QFileDialog.getOpenFileNames(self, 'Выберите фото...', '', 'Images (*.png *.jpeg *.jpg)', options=options)
        self.set_images(fileNames)
    
    def handle_click_on_processButton(self):
        if len(self.images) == 0:
            return
        
        self.set_is_loading(True)

        loop = asyncio.get_running_loop()
        loop.create_task(group_images(image_paths=self.images, img_c=self.img_c, img_sim=self.img_sim, success_callback=self.set_grouped_images, progress_callback=self.set_progress))
        
    # Mutations
    def set_images(self, value):
        self.images = value
        self.update_ui()
    
    def set_is_loading(self, value):
        self.is_loading = value
        self.update_ui()
    
    def set_progress(self, value):
        self.progress = min(100, max(0, int(value * 100)))
        self.update_ui()

    def set_grouped_images(self, value):
        self.grouped_images = value
        self.is_loading = False
        self.update_ui()
    
    # UIEngine
    def update_ui(self):
        self.update_left_list()
        self.update_select_button()
        self.update_progress_bar()
        self.update_progress_bar_value()
        self.update_tree_view()

    
    def update_left_list(self):
        model = self.leftList.model()
        model.clear()
        for image in self.images:
            item = self._crate_item_for_image(image)
            model.appendRow(item)
    
    def update_select_button(self):
        self.selectImagesButton.setEnabled(not self.is_loading)

    def update_progress_bar(self):
        self.processButton.setVisible(not self.is_loading)
        self.progressBar.setVisible(self.is_loading)
    
    def update_progress_bar_value(self):
        self.progressBar.setValue(self.progress)
    
    def update_tree_view(self):
        model = self.rightTreeView.model()
        model.clear()
        for class_name, clusters in self.grouped_images.items():
            class_item = self._create_item_for_string(class_name)
            for i, (_, images) in enumerate(clusters.items()):
                cluster_item = self._create_item_for_string(str(i + 1))
                for image in images:
                    image_item = self._crate_item_for_image(image)
                    cluster_item.appendRow(image_item)
                class_item.appendRow(cluster_item)
            model.appendRow(class_item)
        self.rightTreeView.expandAll()
    
    def _crate_item_for_image(self, image):
        icon = QIcon(image)
        title = os.path.basename(image)
        item = QStandardItem(icon, title)
        item.setEditable(False)
        return item
    
    def _create_item_for_string(self, title):
        item = QStandardItem(title)
        item.setEditable(False)
        return item
