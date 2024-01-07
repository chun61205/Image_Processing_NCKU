import sys
import os
import cv2
import random
import torch

import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from PIL import Image
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog, QGraphicsPixmapItem, QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QGraphicsScene, QGraphicsView, QGroupBox
from PyQt5.QtGui import QPixmap, QImageReader, QPainterPath, QPen, QImage, QPainter
from torchsummary import summary
from torchvision.transforms import Lambda

class CustomScene(QGraphicsScene):
    def __init__(self, parent=None):
        super(CustomScene, self).__init__(parent)
        self.path = QPainterPath()
        self.pen = QPen(Qt.white, 20)
        self.pen.setCapStyle(Qt.RoundCap)
        self.drawingEnabled = True

    def enableDrawing(self, enable):
        self.drawingEnabled = enable

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawingEnabled:
            self.path.moveTo(event.scenePos())

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton and self.drawingEnabled:
            self.path.lineTo(event.scenePos())
            self.addPath(self.path, self.pen)
            self.path = QPainterPath(event.scenePos())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawingEnabled:
            self.path.lineTo(event.scenePos())
            self.addPath(self.path, self.pen)
            self.path = QPainterPath()

class App(QWidget):
    def __init__(self):
        super().__init__()

        # Initializing image labels
        self.imageLabel1 = QLabel("No image loaded")

        # Initializing buttoms
        # Load image buttom
        self.loadImageButton1 = QPushButton("Load Image 1")
        self.loadImageButton1.clicked.connect(lambda: self.loadImage(1))
        # Buttom 1
        self.button1_1 = QPushButton("1.1 Draw Contour")
        self.button1_1.clicked.connect(lambda: self.drawContour())
        self.button1_2 = QPushButton("1.2 Count Coins")
        self.button1_2.clicked.connect(lambda: self.countCoins())
        self.label1 = QLabel("There are _ coins in the image.")
        # Buttom 2
        self.button2 = QPushButton("2 Histogram Equalization")
        self.button2.clicked.connect(lambda: self.histogramEqualization())
        # Buttom 3
        self.button3_1 = QPushButton("3.1 Closing")
        self.button3_1.clicked.connect(lambda: self.closing())
        self.button3_2 = QPushButton("3.2 Opening")
        self.button3_2.clicked.connect(lambda: self.opening())
        # Buttom 4
        self.button4_1 = QPushButton("4.1 Show Model Structure")
        self.button4_1.clicked.connect(lambda: self.showModelStructure(0))
        self.button4_2 = QPushButton("4.2 Show Accuracy and Loss")
        self.button4_2.clicked.connect(lambda: self.showAccAndLoss())
        self.button4_3 = QPushButton("4.3 Predict")
        self.button4_3.clicked.connect(lambda: self.predict())
        self.button4_4 = QPushButton("4.4 Reset")
        self.button4_4.clicked.connect(lambda: self.reset())
        self.label4 = QLabel("")

        self.scene4 = CustomScene()
        self.scene4.setSceneRect(0, 0, 600, 325)
        self.view4 = QGraphicsView(self.scene4)
        self.view4.setFixedHeight(325)
        self.view4.setFixedWidth(600)
        self.view4.setBackgroundBrush(Qt.black)
        self.view4.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view4.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Buttom 5
        self.loadImageButton5 = QPushButton("Load Image")
        self.loadImageButton5.clicked.connect(lambda: self.loadImage(5))
        self.button5_1 = QPushButton("5.1 Show Images")
        self.button5_1.clicked.connect(lambda: self.showImage())
        self.button5_2 = QPushButton("5.2 Show Model Structure")
        self.button5_2.clicked.connect(lambda: self.showModelStructure(1))
        self.button5_3 = QPushButton("5.3 Show Comparison")
        self.button5_3.clicked.connect(lambda: self.showComparison())
        self.button5_4 = QPushButton("5.4 Inference")
        self.button5_4.clicked.connect(lambda: self.inference())
        self.predictionLabel = QLabel("Prediction: ")

        self.scene5 = QGraphicsScene()
        self.view5 = QGraphicsView(self.scene5)
        self.view5.setFixedHeight(224)
        self.view5.setFixedWidth(224)

        # Initializing images
        self.image1 = None
        self.image5 = None

        # Initializing coin counts
        self.coinCounts = None

        self.initUI()

    def initUI(self):
        #1
        vbox1 = QVBoxLayout()
        vbox1.addWidget(self.loadImageButton1)
        vbox1.addWidget(self.imageLabel1)
        #2
        vbox2 = QVBoxLayout()
        #2-1
        group1 = QGroupBox()
        vboxGroup1 = QVBoxLayout()
        vboxGroup1.addWidget(QLabel("1. Hough Circle Transform"))
        vboxGroup1.addWidget(self.button1_1)
        vboxGroup1.addWidget(self.button1_2)
        vboxGroup1.addWidget(self.label1)
        group1.setLayout(vboxGroup1)
        vbox2.addWidget(group1)
        #2-2
        group2 = QGroupBox()
        vboxGroup2 = QVBoxLayout()
        vboxGroup2.addWidget(QLabel("2. Histogram Equalization"))
        vboxGroup2.addWidget(self.button2)
        group2.setLayout(vboxGroup2)
        vbox2.addWidget(group2)
        #2-3
        group3 = QGroupBox()
        vboxGroup3 = QVBoxLayout()
        vboxGroup3.addWidget(QLabel("3. Morphology Operation"))
        vboxGroup3.addWidget(self.button3_1)
        vboxGroup3.addWidget(self.button3_2)
        group3.setLayout(vboxGroup3)
        vbox2.addWidget(group3)
        #3
        vbox3 = QVBoxLayout()
        #3-1
        group4 = QGroupBox()
        vboxGroup4 = QVBoxLayout()
        hboxGroup4 = QHBoxLayout()

        vboxGroup4.addWidget(QLabel("4. MNIST Classifier Using VGG19"))
        vboxGroup4.addWidget(self.button4_1)
        vboxGroup4.addWidget(self.button4_2)
        vboxGroup4.addWidget(self.button4_3)
        vboxGroup4.addWidget(self.button4_4)
        vboxGroup4.addWidget(self.label4)

        hboxGroup4.addLayout(vboxGroup4)
        hboxGroup4.addWidget(self.view4)

        group4.setLayout(hboxGroup4)
        vbox3.addWidget(group4)
        #3-2
        group5 = QGroupBox()
        vboxGroup51 = QVBoxLayout()
        vboxGroup52 = QVBoxLayout()
        hboxGroup5 = QHBoxLayout()

        vboxGroup51.addWidget(QLabel("5. ResNet50"))
        vboxGroup51.addWidget(self.loadImageButton5)
        vboxGroup51.addWidget(self.button5_1)
        vboxGroup51.addWidget(self.button5_2)
        vboxGroup51.addWidget(self.button5_3)
        vboxGroup51.addWidget(self.button5_4)

        
        vboxGroup52.addWidget(self.view5)
        vboxGroup52.addWidget(self.predictionLabel)

        hboxGroup5.addLayout(vboxGroup51)
        hboxGroup5.addLayout(vboxGroup52)

        group5.setLayout(hboxGroup5)
        vbox3.addWidget(group5)

        hbox = QHBoxLayout()
        hbox.addLayout(vbox1)
        hbox.addLayout(vbox2)
        hbox.addLayout(vbox3)

        self.setLayout(hbox)

        self.setWindowTitle('Hw1')
        self.show()
    
    def loadImage(self, n):
        filePath, _ = QFileDialog.getOpenFileName()
        if n == 1:
            self.image1 = cv2.imread(filePath)
            self.updataImageLabel(os.path.basename(filePath), n)
        elif n == 5:
            self.image5 = cv2.imread(filePath)
            self.image5 = cv2.resize(self.image5, (224, 224), interpolation=cv2.INTER_LINEAR)
            self.image5 = cv2.cvtColor(self.image5, cv2.COLOR_BGR2RGB)
            height, width, channel = self.image5.shape
            bytesPerLine = 3 * width
            qImage = QImage(self.image5.data, width, height, bytesPerLine, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qImage)
            pixmapItem = QGraphicsPixmapItem(pixmap)
            self.scene5.addItem(pixmapItem)
            self.view5.fitInView(pixmapItem, Qt.KeepAspectRatio)

    def updataImageLabel(self, text, n):
        if n == 1:
            self.imageLabel1.setText(text)

    def drawContour(self):
        image = self.image1.copy()
        circleCenters = np.zeros_like(image)

        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        blurredImage = cv2.GaussianBlur(grayImage, (5, 5), 0)

        circles = cv2.HoughCircles(blurredImage, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20, param1=200, param2=40)

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            self.coinCounts = len(circles)
            for (x, y, r) in circles:
                cv2.circle(image, (x, y), r, (0, 255, 0), 4)
                cv2.rectangle(image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
        
                cv2.rectangle(circleCenters, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

        cv2.imshow("Original Image", self.image1)
        cv2.imshow("Processed Image", image)
        cv2.imshow("Circle Center Image ", circleCenters)
    
    def countCoins(self):
        self.label1.setText(f"There are {self.coinCounts} coins in the image.")
    
    def histogramEqualization(self):
        def equalization(img):
            hist, bins = np.histogram(img.ravel(), 256, [0, 255])
            cdf = hist.cumsum()
            cdfNormalized = np.round(((cdf * 255) / img.size)).astype(np.uint8)
            
            precessedImg = np.interp(img.ravel(), bins[:-1], cdfNormalized)
            precessedImg = precessedImg.reshape(img.shape).astype(np.uint8)
            
            return precessedImg

        originalImage = self.image1
        grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
        equalizedImageOpenCV = cv2.equalizeHist(grayImage)
        equalizedImageManually = equalization(originalImage)

        originalHistogram = cv2.calcHist([originalImage], [0], None, [256], [0, 256])
        equalizedHistogramOpenCV = cv2.calcHist([equalizedImageOpenCV], [0], None, [256], [0, 256])
        equalizedHistogramManually = cv2.calcHist([equalizedImageManually], [0], None, [256], [0, 256])

        plt.figure(figsize=(10, 8))
        plt.subplot(2, 3, 1)
        plt.imshow(originalImage, cmap='gray')
        plt.title('Original Image')

        plt.subplot(2, 3, 2)
        plt.imshow(equalizedImageOpenCV, cmap='gray')
        plt.title('Equalized with OpenCV')

        plt.subplot(2, 3, 3)
        plt.imshow(equalizedImageOpenCV, cmap='gray')
        plt.title('Equalized Manually')

        plt.subplot(2, 3, 4)
        plt.bar(range(256), originalHistogram.ravel(), color='black')
        plt.title('Histogram of Original')

        plt.subplot(2, 3, 5)
        plt.bar(range(256), equalizedHistogramOpenCV.ravel(), color='black')
        plt.title('Histogram of Equalized (OpenCV)')

        plt.subplot(2, 3, 6)
        plt.bar(range(256), equalizedHistogramManually.ravel(), color='black')
        plt.title('Histogram of Equalized (OpenCV)')

        plt.tight_layout()
        plt.show()

    def closing(self):
        originalImage = self.image1
        grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
        binaryImage = (grayImage > 127).astype(np.uint8) * 255

        # Structuring element
        kernel = np.ones((3, 3), np.uint8)

        # Perform opening
        erodedImage = self.dilation(binaryImage, kernel)
        openedImage = self.erosion(erodedImage, kernel)

        # Display the image
        plt.imshow(openedImage, cmap='gray')
        plt.title("Opened Image")
        plt.show()

    def opening(self):
        originalImage = self.image1
        grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
        binaryImage = (grayImage > 127).astype(np.uint8) * 255

        # Structuring element
        kernel = np.ones((3, 3), np.uint8)

        # Perform opening
        erodedImage = self.erosion(binaryImage, kernel)
        openedImage = self.dilation(erodedImage, kernel)
        
        # Display the image
        plt.imshow(openedImage, cmap='gray')
        plt.title("Opened Image")
        plt.show()

    def dilation(self, image, kernel):
        paddedImage = np.pad(image, pad_width=1, mode='constant', constant_values=0)
        dilatedImage = np.zeros_like(image)

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if np.sum(paddedImage[i:i+3, j:j+3] * kernel) >= 255:
                    dilatedImage[i, j] = 255

        return dilatedImage
    
    def erosion(self, image, kernel):
        paddedImage = np.pad(image, pad_width=1, mode='constant', constant_values=0)
        erodedImage = np.zeros_like(image)

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if np.sum(paddedImage[i:i+3, j:j+3] * kernel) == 255 * 9:
                    erodedImage[i, j] = 255

        return erodedImage
    
    def showAugmentedImage(self):
        dirPath = QFileDialog.getExistingDirectory()
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.Resize((320, 320), transforms.InterpolationMode.NEAREST)
        ])
        fileNames = [f for f in os.listdir(dirPath)]
        images = []
        for fileName in fileNames[:9]:
            filePath = os.path.join(dirPath, fileName)
            image = Image.open(filePath)
            image = transform(image)
            images.append(image)

        height, width = images[0].size
        
        margin = 10
        canvasHeight = height * 3 + margin * 2
        canvasWidth = width * 3 + margin * 2
        canvas = np.zeros((canvasHeight, canvasWidth, 3), dtype=np.uint8)
        for i in range(3):
            for j in range(3):
                y = i * height + i * margin
                x = j * width + j * margin

                canvas[y: y + height, x: x + width] = images[i * 3 + j]
        
        cv2.imshow('Augmentation Images', canvas)

    def showModelStructure(self, id):
        if id == 0:
            filePath = './model/vgg19_bn.pth'
            self.model = models.vgg19_bn(pretrained=False)
            num_features = self.model.classifier[6].in_features
            self.model.classifier[6] = torch.nn.Linear(num_features, 10)
            self.model.load_state_dict(torch.load(filePath, map_location=torch.device('cpu')))
            summary(self.model, (3, 32, 32))
        elif id == 1:
            filePath = './model/resnet50.pth'
            self.model = models.resnet50(pretrained=False)
            numFeatures = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Linear(numFeatures, 1),
                nn.Sigmoid()
            )
            self.model.load_state_dict(torch.load(filePath, map_location=torch.device('cpu')))
            summary(self.model, (3, 224, 224))
            
    def showAccAndLoss(self):
        filePath = './trainVGG19.jpg'
        reader = QImageReader(filePath)
        reader.setAutoTransform(True)
        image = reader.read()
        pixmap = QPixmap.fromImage(image).scaled(600, 400, Qt.KeepAspectRatio, Qt.FastTransformation)
        self.scene4.clear()
        pixmapItem = QGraphicsPixmapItem(pixmap)
        self.scene4.addItem(pixmapItem)
        self.scene4.enableDrawing(False)
        self.scene4.update()

    def predict(self):
        # Getting the image
        rect = self.scene4.sceneRect()

        image = QImage(rect.size().toSize(), QImage.Format_ARGB32)
        image.fill(Qt.black)

        painter = QPainter(image)
        self.scene4.render(painter)
        painter.end()

        ptr = image.bits()
        ptr.setsize(image.byteCount())
        arr = np.array(ptr).reshape(image.height(), image.width(), 4)

        image = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)

        # Prediction
        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        filePath = './model/vgg19_bn.pth'
        model = models.vgg19_bn(pretrained=False)
        num_features = model.classifier[6].in_features
        model.classifier[6] = torch.nn.Linear(num_features, 10)
        model.load_state_dict(torch.load(filePath, map_location=torch.device('cpu')))
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.Resize((32, 32)),
        ])
        image = transform(image)
        image = image.unsqueeze(0)
        with torch.no_grad():
            model.eval()
            output = model(image)
            output = torch.nn.functional.softmax(output)

        plt.figure(figsize=(10, 6))
        plt.bar(classes, output[0].numpy(), color='skyblue')
        plt.xticks(classes)
        plt.xlabel('Class')
        plt.ylabel('Probability')
        plt.title('Class Probabilities')
        plt.show()

        _, output = output.max(1)
        output = output.item()
        self.label4.setText(f'{output}')   

    def reset(self):
        self.scene4.clear()
        self.view4.setBackgroundBrush(Qt.black)
        self.scene4.enableDrawing(True)

    def showImage(self):
        filePath = './inference_dataset'

        catFolderPath = os.path.join(filePath, 'Cat')
        dogFolderPath = os.path.join(filePath, 'Dog')

        catImageName = random.choice(os.listdir(catFolderPath))
        dogImageName = random.choice(os.listdir(dogFolderPath))

        catImage = Image.open(os.path.join(catFolderPath, catImageName)).resize((224, 224))
        dogImage = Image.open(os.path.join(dogFolderPath, dogImageName)).resize((224, 224))

        _, ax = plt.subplots(1, 2, figsize=(10, 5))

        ax[0].imshow(catImage)
        ax[0].set_title('Cat')
        ax[0].axis('off')

        ax[1].imshow(dogImage)
        ax[1].set_title('Dog')
        ax[1].axis('off')

        plt.show()
    
    def showComparison(self):
        filePath = './Accuracy Comparison.jpg'
        image = mpimg.imread(filePath)
        plt.imshow(image)
        plt.axis('off')  # 不显示坐标轴
        plt.show()
    
    def inference(self):
        filePath = './model/resnet50.pth'
        self.model = models.resnet50(pretrained=False)
        numFeatures = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(numFeatures, 1),
            nn.Sigmoid()
        )
        self.model.load_state_dict(torch.load(filePath, map_location=torch.device('cpu')))

        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        image = transform(self.image5)
        image = image.unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(image)
        prediction = "Dog" if outputs.squeeze() >= 0.5 else "Cat"
        self.predictionLabel.setText(f"Prediction: {prediction}")
        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())