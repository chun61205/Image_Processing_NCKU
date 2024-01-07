import sys
import os
import cv2
import torch

import numpy as np
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt

from PIL import Image
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog, QGraphicsPixmapItem, QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit, QGraphicsScene, QGraphicsView, QGroupBox
from PyQt5.QtGui import QPixmap, QImageReader
from torchsummary import summary

class App(QWidget):
    def __init__(self):
        super().__init__()
        # Initializing the inference image scene.
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setFixedHeight(130)
        self.view.setFixedWidth(130)
        textItem = self.scene.addText("Inference Image")
        textItem.setPos(10, 65)

        # Initializing image labels
        self.imageLabel1 = QLabel("No image loaded")
        self.imageLabel2 = QLabel("No image loaded")

        # Initializing buttoms
        # Load image buttom
        self.loadImageButton1 = QPushButton("Load Image 1")
        self.loadImageButton1.clicked.connect(lambda: self.loadImage(1))
        self.loadImageButton2 = QPushButton("Load Image 2")
        self.loadImageButton2.clicked.connect(lambda: self.loadImage(2))
        # Buttom 1
        self.button1_1 = QPushButton("1.1 Color Separation")
        self.button1_1.clicked.connect(lambda: self.colorSeperation())
        self.button1_2 = QPushButton("1.2 Color Transformation")
        self.button1_2.clicked.connect(lambda: self.colorTransformation())
        self.button1_3 = QPushButton("1.3 Color Extraction")
        self.button1_3.clicked.connect(lambda: self.colorExtraction())
        # Buttom 2
        self.button2_1 = QPushButton("2.1 Gaussian blur")
        self.button2_1.clicked.connect(lambda: self.gaussianBlur())
        self.button2_2 = QPushButton("2.2 Bilateral filter")
        self.button2_2.clicked.connect(lambda: self.bilateralFilter())
        self.button2_3 = QPushButton("2.3 Median filter")
        self.button2_3.clicked.connect(lambda: self.medianFilter())
        # Buttom 3
        self.button3_1 = QPushButton("3.1 Sobel X")
        self.button3_1.clicked.connect(lambda: self.edgeDetection('Sobel X'))
        self.button3_2 = QPushButton("3.2 Sobel Y")
        self.button3_2.clicked.connect(lambda: self.edgeDetection('Sobel Y'))
        self.button3_3 = QPushButton("3.3 Combination and Threshold")
        self.button3_3.clicked.connect(lambda: self.combinationAndThreshold())
        self.button3_4 = QPushButton("3.4 Gradient Angle")
        self.button3_4.clicked.connect(lambda: self.gradientAngle())
        # Buttom 4
        self.rotationValue = QLineEdit()
        self.scalingValue = QLineEdit()
        self.tXValue = QLineEdit()
        self.tYValue = QLineEdit()
        self.button4 = QPushButton("4. Transforms")
        self.button4.clicked.connect(lambda: self.transform())
        # Buttom 5
        self.loadImageButton5 = QPushButton("Load Image")
        self.loadImageButton5.clicked.connect(lambda: self.loadImage(5))
        self.button5_1 = QPushButton("5.1 Show Augmented Images")
        self.button5_1.clicked.connect(lambda: self.showAugmentedImage())
        self.button5_2 = QPushButton("5.2 Show Model Structure")
        self.button5_2.clicked.connect(lambda: self.showModelStructure())
        self.button5_3 = QPushButton("5.3 Show Acc and Loss")
        self.button5_3.clicked.connect(lambda: self.showAccAndLoss())
        self.button5_4 = QPushButton("5.4 Inference")
        self.button5_4.clicked.connect(lambda: self.inference())
        self.predictionLabel = QLabel("Predict = ")

        # Initializing images
        self.image1 = None
        self.image2 = None
        self.image5 = None

        # Initializing Model
        self.model = None

        self.initUI()

    def initUI(self):
        vbox1 = QVBoxLayout()
        vbox1.addWidget(self.loadImageButton1)
        vbox1.addWidget(self.imageLabel1)
        vbox1.addWidget(self.loadImageButton2)
        vbox1.addWidget(self.imageLabel2)

        vbox2 = QVBoxLayout()
        group1 = QGroupBox()
        vboxGroup1 = QVBoxLayout()
        vboxGroup1.addWidget(QLabel("1. Image Processing"))
        vboxGroup1.addWidget(self.button1_1)
        vboxGroup1.addWidget(self.button1_2)
        vboxGroup1.addWidget(self.button1_3)
        group1.setLayout(vboxGroup1)
        vbox2.addWidget(group1)

        group2 = QGroupBox()
        vboxGroup2 = QVBoxLayout()
        vboxGroup2.addWidget(QLabel("2. Image Smoothing"))
        vboxGroup2.addWidget(self.button2_1)
        vboxGroup2.addWidget(self.button2_2)
        vboxGroup2.addWidget(self.button2_3)
        group2.setLayout(vboxGroup2)
        vbox2.addWidget(group2)

        group3 = QGroupBox()
        vboxGroup3 = QVBoxLayout()
        vboxGroup3.addWidget(QLabel("3. Edge Detection"))
        vboxGroup3.addWidget(self.button3_1)
        vboxGroup3.addWidget(self.button3_2)
        vboxGroup3.addWidget(self.button3_3)
        vboxGroup3.addWidget(self.button3_4)
        group3.setLayout(vboxGroup3)
        vbox2.addWidget(group3)

        vbox3 = QVBoxLayout()
        group4 = QGroupBox()
        vboxGroup4 = QVBoxLayout()
        vboxGroup4.addWidget(QLabel("4. Transforms"))
        vboxGroup4.addWidget(QLabel("Rotation:"))
        vboxGroup4.addWidget(self.rotationValue)
        vboxGroup4.addWidget(QLabel("Scaling:"))
        vboxGroup4.addWidget(self.scalingValue)
        vboxGroup4.addWidget(QLabel("Tx:"))
        vboxGroup4.addWidget(self.tXValue)
        vboxGroup4.addWidget(QLabel("Ty:"))
        vboxGroup4.addWidget(self.tYValue)
        vboxGroup4.addWidget(self.button4)
        group4.setLayout(vboxGroup4)
        vbox3.addWidget(group4)

        group5 = QGroupBox()
        vboxGroup5 = QVBoxLayout()
        vboxGroup5.addWidget(QLabel("5. VGG19"))
        vboxGroup5.addWidget(self.loadImageButton5)
        vboxGroup5.addWidget(self.button5_1)
        vboxGroup5.addWidget(self.button5_2)
        vboxGroup5.addWidget(self.button5_3)
        vboxGroup5.addWidget(self.button5_4)
        vboxGroup5.addWidget(self.predictionLabel)
        vboxGroup5.addWidget(self.view)
        group5.setLayout(vboxGroup5)
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
        elif n == 2:
            self.image2 = cv2.imread(filePath)
            self.updataImageLabel(os.path.basename(filePath), n)
        elif n == 5:
            self.image5 = cv2.imread(filePath)
            self.updateInferenceImage(filePath)

    def updataImageLabel(self, text, n):
        if n == 1:
            self.imageLabel1.setText(text)
        elif n == 2:
            self.imageLabel2.setText(text)

    def colorSeperation(self):
        channels = cv2.split(self.image1)

        # Initializing a black channel with the same size as one of the image channels
        black = np.zeros_like(channels[0])

        # Creating R G B images repectively
        blueImage = cv2.merge([channels[2], black, black])
        greenImage = cv2.merge([black, channels[1], black])
        redImage = cv2.merge([black, black, channels[0]])

        cv2.imshow("R", redImage)
        cv2.imshow("G", greenImage)
        cv2.imshow("B", blueImage)
    
    def colorTransformation(self):
        # Exploiting cvtColor
        grayScaleImage1 = cv2.cvtColor(self.image1, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Gray Scale 1", grayScaleImage1)

        # Merge R, G, and B images
        b, g, r = cv2.split(self.image1)
        grayScaleImage2 = ((b + g + r) / 3).astype(np.uint8)
        cv2.imshow("Gray Scale 2", grayScaleImage2)
    
    def colorExtraction(self):
        hsvImg = cv2.cvtColor(self.image1, cv2.COLOR_BGR2HSV)
    
        lowerBound = np.array([18, 25, 25])
        upperBound = np.array([85, 255, 255])
        
        yellowGreenMask = cv2.inRange(hsvImg, lowerBound, upperBound)
        
        image1 = cv2.cvtColor(yellowGreenMask, cv2.COLOR_GRAY2BGR)
        cv2.imshow("Image 1", image1)
        
        image2 = cv2.bitwise_not(image1, self.image1, mask=yellowGreenMask)
        cv2.imshow("Image 2", image2)
    
    def gaussianBlur(self):
        def update(val):
            m = cv2.getTrackbarPos('Kernel', 'Gaussian Blur') * 2 + 1
            if m < 3:
                kernelSize = 3
            else:
                kernelSize = m
            blurredImage = cv2.GaussianBlur(self.image1, (kernelSize, kernelSize), 0)
            cv2.imshow('Gaussian Blur', blurredImage)

        cv2.namedWindow('Gaussian Blur')
        cv2.createTrackbar('Kernel', 'Gaussian Blur', 1, 5, update)
        update(1)
    
    def bilateralFilter(self):
        def update(val):
            m = cv2.getTrackbarPos('Kernel', 'Bilateral Filter') * 2 + 1
            if m < 3:
                kernelSize = 3
            else:
                kernelSize = m
            blurredImage = cv2.bilateralFilter(self.image1, kernelSize, sigmaColor=90, sigmaSpace=90)
            cv2.imshow('Bilateral Filter', blurredImage)

        cv2.namedWindow('Bilateral Filter')
        cv2.createTrackbar('Kernel', 'Bilateral Filter', 1, 5, update)
        update(1)
    
    def medianFilter(self):
        def update(val):
            m = cv2.getTrackbarPos('Kernel', 'Median Filter') * 2 + 1
            if m < 3:
                kernelSize = 3
            else:
                kernelSize = m
            blurredImage = cv2.medianBlur(self.image1, kernelSize)
            cv2.imshow('Median Filter', blurredImage)

        cv2.namedWindow('Median Filter')
        cv2.createTrackbar('Kernel', 'Median Filter', 1, 5, update)
        update(1)
    
    def edgeDetection(self, operatorName):
        # Preprocessing the image
        image = cv2.cvtColor(self.image1, cv2.COLOR_BGR2GRAY)
        image = cv2.GaussianBlur(image, (3, 3), 0)

        # Operator preparation
        operators = {
            'Sobel X': np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
            'Sobel Y': np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        }
        operator = operators[operatorName]

        # Convolution
        imageHeight, imageWidge = image.shape
        operatorHeight, operatorWidth = operator.shape

        padHeight = operatorHeight // 2
        padWidth = operatorWidth // 2

        paddedImg = np.pad(image, ((padHeight, padHeight), (padWidth, padWidth)), mode='constant')

        result = np.zeros_like(image)

        for i in range(imageHeight):
            for j in range(imageWidge):
                tmp = paddedImg[i: i + operatorHeight, j: j + operatorWidth]
                result[i, j] = np.sum(tmp * operator) / 9

        cv2.imshow(operatorName, result)
    
    def combinationAndThreshold(self):
        # Preprocessing the image
        image = cv2.cvtColor(self.image1, cv2.COLOR_BGR2GRAY)
        image = cv2.GaussianBlur(image, (3, 3), 0)

        # Operator preparation
        operators = {
            'Sobel X': np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
            'Sobel Y': np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        }

        results = []

        for _, operator in operators.items():
            # Convolution
            imageHeight, imageWidge = image.shape
            operatorHeight, operatorWidth = operator.shape

            padHeight = operatorHeight // 2
            padWidth = operatorWidth // 2

            paddedImg = np.pad(image, ((padHeight, padHeight), (padWidth, padWidth)), mode='constant')

            result = np.zeros_like(image)

            for i in range(imageHeight):
                for j in range(imageWidge):
                    tmp = paddedImg[i: i + operatorHeight, j: j + operatorWidth]
                    result[i, j] = np.sum(tmp * operator) / 9
            
            results.append(result)

        combination = np.sqrt(np.square(results[0]) + np.square(results[1])).astype(np.float32)
        combination = cv2.normalize(combination, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        _, combination = cv2.threshold(combination, 128, 255, cv2.THRESH_BINARY)
        cv2.imshow('Combination and Threshold', result)
    
    def gradientAngle(self):
        # Preprocessing the image
        image = cv2.cvtColor(self.image1, cv2.COLOR_BGR2GRAY)
        image = cv2.GaussianBlur(image, (3, 3), 0)

        # Operator preparation
        operators = {
            'Sobel X': np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
            'Sobel Y': np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        }

        gradX = None
        gradY = None

        for name, operator in operators.items():
            # Convolution
            imageHeight, imageWidge = image.shape
            operatorHeight, operatorWidth = operator.shape

            padHeight = operatorHeight // 2
            padWidth = operatorWidth // 2

            paddedImg = np.pad(image, ((padHeight, padHeight), (padWidth, padWidth)), mode='constant')
            result = np.zeros_like(image, dtype=np.float32)

            for i in range(imageHeight):
                for j in range(imageWidge):
                    tmp = paddedImg[i: i + operatorHeight, j: j + operatorWidth]
                    result[i, j] = np.sum(tmp * operator)

            if name == 'Sobel X':
                gradX = result
            elif name == 'Sobel Y':
                gradY = result
        

        # Calculating the gradient angle
        angle = np.arctan2(gradY, gradX) * (180 / np.pi) % 360
        
        # Generating masks based on the angle ranges
        mask1 = np.where((angle >= 120) & (angle <= 180), 255, 0).astype(np.uint8)
        mask2 = np.where((angle >= 210) & (angle <= 330), 255, 0).astype(np.uint8)

        combination = np.sqrt(np.square(gradX) + np.square(gradY)).astype(np.uint8)
        # Applying masks to the combination
        result1 = cv2.bitwise_and(combination, mask1)
        result2 = cv2.bitwise_and(combination, mask2)

        # Showing the results using cv2.imshow
        cv2.imshow('120-180 Degrees', result1)
        cv2.imshow('210-330 Degrees', result2)
    
    def transform(self):
        # Initializing transformation attributes
        textValue = self.rotationValue.text()
        try:
            rotationValue = float(textValue)
        except ValueError:
            rotationValue = 0

        textValue = self.scalingValue.text()
        try:
            scalingValue = float(textValue)
        except ValueError:
            scalingValue = 0

        textValue = self.tXValue.text()
        try:
            tXValue = int(textValue)
        except ValueError:
            tXValue = 0
        
        textValue = self.tYValue.text()
        try:
            tYValue = int(textValue)
        except ValueError:
            tYValue = 0

        # Getting the shape of the image
        height, width = self.image1.shape[:2]

        # Defining the center of the image (where we want to apply the rotation)
        center = (240, 200)

        # Generating the rotation matrix
        matrix = cv2.getRotationMatrix2D(center, rotationValue, scalingValue)
        
        # Adjusting the transformation matrix to include translation
        matrix[0, 2] += tXValue
        matrix[1, 2] += tYValue

        # Applying the affine transform
        result = cv2.warpAffine(self.image1, matrix, (width, height))

        # Showing the image
        cv2.imshow('Transformed Image', result)
    
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

    def showModelStructure(self):
        filePath, _ = QFileDialog.getOpenFileName()
        self.model = models.vgg19_bn(pretrained=False)
        num_features = self.model.classifier[6].in_features
        self.model.classifier[6] = torch.nn.Linear(num_features, 10)
        self.model.load_state_dict(torch.load(filePath, map_location=torch.device('cpu')))
        summary(self.model, (3, 32, 32))
            
    def showAccAndLoss(self):
        filePath, _ = QFileDialog.getOpenFileName()
        image = cv2.imread(filePath)
        cv2.imshow('Accuracy and Loss', image)

    def inference(self):
        classes = {
            0: "airplane",
            1: "automobile",
            2: "bird",
            3: "cat",
            4: "deer",
            5: "dog",
            6: "frog",
            7: "horse",
            8: "ship",
            9: "truck"
        }
        filePath, _ = QFileDialog.getOpenFileName()
        self.model = models.vgg19_bn(pretrained=False)
        num_features = self.model.classifier[6].in_features
        self.model.classifier[6] = torch.nn.Linear(num_features, 10)
        self.model.load_state_dict(torch.load(filePath, map_location=torch.device('cpu')))
        
        image = cv2.cvtColor(self.image5, cv2.COLOR_BGR2RGB)
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image = transform(image)
        image = image.unsqueeze(0)
        with torch.no_grad():
            self.model.eval()
            output = self.model(image)
            output = torch.nn.functional.softmax(output)
        
        classLabels = [classes[i] for i in range(10)]

        plt.figure(figsize=(10, 6))
        plt.bar(classLabels, output[0].numpy(), color='skyblue')
        plt.xlabel('Class')
        plt.ylabel('Probability')
        plt.title('Class Probabilities')
        plt.show()

        _, output = output.max(1)
        output = output.item()
        self.predictionLabel.setText('Predict = ' + classes[output])

    def updateInferenceImage(self, filePath):
        reader = QImageReader(filePath)
        reader.setAutoTransform(True)
        image = reader.read()
        pixmap = QPixmap.fromImage(image).scaled(128, 128, Qt.KeepAspectRatio, Qt.FastTransformation)
        self.scene.clear()
        pixmapItem = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(pixmapItem)
        self.scene.update()
        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())