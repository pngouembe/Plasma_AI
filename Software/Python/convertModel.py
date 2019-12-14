import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

precision = 16

factor = 2**precision

numImage = int(input("Entrez le numero de l'image :"))

model_state = torch.load('model.ckpt')
layer1Weights = model_state['layer1.0.weight']
layer2Weights = model_state['layer2.0.weight']
layer3Weights = model_state['fc.weight']
layer1bias = model_state['layer1.0.bias']
layer2bias = model_state['layer2.0.bias']
layer3bias = model_state['fc.bias']

lib = open("../C/model.h", 'w')
lib.write("#ifndef MODEL_H\n#define MODEL_H\n")
lib.write('#include "Matrice.h"\n')

nbImg = int(input("entrez le nombre d'images de test : "))
lib.write('#define NB_IMG   {}\n'.format(nbImg))

lib.write('int FIXED_POINT_FACTOR = {};\n'.format(factor))
lib.write('int FIXED_POINT_FRACTIONNAL_BITS = {};\n'.format(precision))


lib.write("int weight1[{}][{}]={{".format(len(layer1Weights), 25))
for k in range(len(layer1Weights)):
    lib.write("\n\t\t\t\t\t\t")
    for l in range(len(layer1Weights[0])):
        for j in range(len(layer1Weights[0][0])):
            if j == 0:
                lib.write("{")
            for i in range(len(layer1Weights[0][0][0])):
                lib.write(str(int(layer1Weights[k][l][j][i]*factor)))
                #lib.write(str(round(float(layer1Weights[k][l][j][i]),precision)))
                if j * i != (len(layer1Weights[0][0])-1)*(len(layer1Weights[0][0][0])-1):
                    lib.write(",")
                if i == len(layer1Weights[0][0][0])-1:
                    lib.write("\n\t\t\t\t\t\t")
    if l != len(layer1Weights[0]-1):
        lib.write("}")
    if k != len(layer1Weights-1):
        lib.write(",")
lib.write("};\n\n")


lib.write("int weight2[{}][{}][{}]={{".format(len(layer2Weights),16, 25))
for k in range(len(layer1Weights)):
    lib.write("\n\t\t\t\t\t\t ")
    for l in range(len(layer2Weights[0])):
        if l == 0:
            lib.write("{")
        for j in range(len(layer2Weights[0][0])):
            if j == 0:
                lib.write("{")
            for i in range(len(layer2Weights[0][0][0])):
                lib.write(str(int(layer2Weights[k][l][j][i]*factor)))
                #lib.write(str(round(float(layer2Weights[k][l][j][i]),precision)))
                if j * i != (len(layer2Weights[0][0])-1)*(len(layer2Weights[0][0][0])-1):
                    lib.write(",")
                if i == len(layer2Weights[0][0][0])-1:
                    lib.write("\n\t\t\t\t\t\t ")
        if j != len(layer2Weights[0][0]-1):
            lib.write("}")
            if j * l != (len(layer2Weights[0][0])-1)*(len(layer2Weights[0])-1):
                    lib.write(",")
                    lib.write("\n\t\t\t\t\t\t ")
    if l != len(layer2Weights[0]-1):
        lib.write("}")
        if l * k != (len(layer2Weights[0])-1)*(len(layer2Weights)-1):
            lib.write(",")

lib.write("};\n\n")



lib.write("int weight3[{}][{}]={{".format(len(layer3Weights), len(layer3Weights[0])))
for k in range(len(layer3Weights)):
    lib.write("\n\t\t\t\t\t\t")
    lib.write("{")
    for i in range(len(layer3Weights[0])):
        lib.write(str(int(layer3Weights[k][i]*factor)))
        #lib.write(str(round(float(layer3Weights[k][i]),precision)))
        if i != len(layer3Weights[0])-1:
            lib.write(",")
        else:
            lib.write("\n\t\t\t\t\t\t")
    lib.write("}")

    if k != len(layer3Weights)-1:
        lib.write(",")
lib.write("};\n\n")

lib.write("int bias1[{}]={{".format(len(layer1bias)))
for k in range(len(layer1bias)):
    lib.write(str(int(layer1bias[k]*factor)))
    #lib.write(str(round(float(layer1bias[k]),precision)))
    if k != len(layer1bias)-1:
        lib.write(",")
lib.write("};\n\n")

lib.write("int bias2[{}]={{".format(len(layer2bias)))
for k in range(len(layer2bias)):
    lib.write(str(int(layer2bias[k]*factor)))
    #lib.write(str(round(float(layer2bias[k]),precision)))
    if k != len(layer2bias)-1:
        lib.write(",")
lib.write("};\n\n")

lib.write("int bias3[{}]={{".format(len(layer3bias)))
for k in range(len(layer3bias)):
    lib.write(str(int(layer3bias[k]*factor)))
    #lib.write(str(round(float(layer3bias[k]),precision)))
    if k != len(layer3bias)-1:
        lib.write(",")
lib.write("};\n\n")

test_dataset = torchvision.datasets.MNIST(root='../data/MNIST/', 
                                          train=False, 
                                          transform=transforms.ToTensor(),
                                          download=True)
testImage = test_dataset[numImage][0]
testLabel = []

lib.write("int testValues[{}][{}] = {{".format(nbImg, len(testImage[0])*len(testImage[0][0])))
for i in range(nbImg):
    testImage = test_dataset[numImage + i][0]
    testLabel.append(test_dataset[numImage + i][1])
    if i != 0:
        lib.write("\n,{")
    else:
        lib.write("{")
    for k in range(len(testImage)):
        for l in range(len(testImage[k])):
            for m in range(len(testImage[k][l])):
                lib.write(str(int(testImage[k][l][m]*factor)))
                if l*m != (len(testImage[k])-1)*(len(testImage[k][l])-1):
                    lib.write(",")
    lib.write("}")
    if i % 100 == 0 :
        print("image #{} ajoutée à la librairie".format(i))
lib.write("};\n\n")

lib.write("int testLabel [{}]= {{".format(nbImg))
for j in range(len(testLabel)):
    if j == 0:
        lib.write("{}".format(testLabel[j]))
    else:
        lib.write(",{}".format(testLabel[j]))
    
lib.write("};\n")


lib.write("#endif\n")
'''
plt.figure(1)
image, label = test_dataset[numImage]
plt.imshow(transforms.ToPILImage()(image))
plt.title('True label {}'.format(label))
plt.pause(0.1)
'''
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            #nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            #nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        '''
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        '''
        return out

model = CNN(10)
model.load_state_dict(model_state)
model.eval()
testImage = testImage.unsqueeze(0)

out = model(testImage)
#print(out[0][0])
#print(testLabel)
