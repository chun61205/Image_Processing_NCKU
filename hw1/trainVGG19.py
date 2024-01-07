import torch

import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from torchsummary import summary
from torch.utils.data import DataLoader

if __name__ == '__main__':

    epochs = 40
    modelPath = 'vgg19_bn.pth'

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    vgg19 = models.vgg19_bn(pretrained=True)
    num_features = vgg19.classifier[6].in_features
    vgg19.classifier[6] = torch.nn.Linear(num_features, 10)

    trainDataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainLoader = DataLoader(trainDataset, batch_size=32, shuffle=True)
    valDataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    valLoader = DataLoader(valDataset, batch_size=32, shuffle=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("CUDA not available!")
        device = torch.device("cpu")
    weight_dtype = torch.float32
    vgg19.to(device, dtype=weight_dtype)
    summary(vgg19, (3, 32, 32))

    optimizerParams = {
        'params': vgg19.parameters(),
        'lr': 0.0001,
        'betas': (0.9, 0.999),
        'eps': 1e-8,
        'weight_decay': 0
    }
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(**optimizerParams)

    trainLosses = []
    valLosses = []
    trainAccuracies = []
    valAccuracies = []

    bestAccuracy = 0.0
    bestEpoch = 0
    for epoch in range(epochs):
        vgg19.train()
        trainRunningLoss = 0.0
        trainTotal = 0
        trainCorrect = 0
        for inputs, labels in trainLoader:
            optimizer.zero_grad()

            outputs = vgg19(inputs.to(device, dtype=weight_dtype))
            loss = criterion(outputs, labels.to(device))

            loss.backward()
            optimizer.step()

            _, prediction = torch.max(outputs.data, 1)
            trainTotal += labels.size(0)
            trainCorrect += (prediction == labels.to(device)).sum().item()

            trainRunningLoss += loss.item()
        
        trainLoss = trainRunningLoss / len(trainLoader)
        trainLosses.append(trainLoss)
        trainAccuracy = trainCorrect / trainTotal
        trainAccuracies.append(trainAccuracy)
        

        vgg19.eval()
        valRunningLoss = 0.0
        valTotal = 0
        valCorrect = 0
        with torch.no_grad():
            for inputs, labels in valLoader:
                outputs = vgg19(inputs.to(device))
                loss = criterion(outputs, labels.to(device))
                valRunningLoss += loss.item()

                _, prediction = torch.max(outputs.data, 1)
                valTotal += labels.size(0)
                valCorrect += (prediction == labels.to(device)).sum().item()

        valLoss = valRunningLoss / len(valLoader)
        valLosses.append(valLoss)
        valAccuracy = valCorrect / valTotal
        valAccuracies.append(valAccuracy)

        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {trainLoss}')
        print(f'Epoch {epoch+1}/{epochs}, Train Accuracy: {trainAccuracy}')
        print(f'Epoch {epoch+1}/{epochs}, Validation Loss: {valLoss}')
        print(f'Epoch {epoch+1}/{epochs}, Validation Accuracy: {valAccuracy}')
        

        if valAccuracy > bestAccuracy:
            bestAccuracy = valAccuracy
            bestEpoch = epoch
            torch.save(vgg19.state_dict(), modelPath)
            print(f'Improved Model Saved with Accuracy: {valAccuracy}')
    
    print('Finished Training')
    torch.save(vgg19.state_dict(), modelPath)

    with open('training_log.txt', 'w') as f:
        for epoch in range(epochs):
            f.write(f"{epoch+1}\t{trainLosses[epoch]}\t{trainAccuracies[epoch]}\t{valLosses[epoch]}\t{valAccuracies[epoch]}\n")

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs+1), trainLosses, label='Train Loss')
    plt.plot(range(1, epochs+1), valLosses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs+1), trainAccuracies, label='Train Accuracy')
    plt.plot(range(1, epochs+1), valAccuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.savefig('Loss and Accuracy.jpg', bbox_inches='tight')