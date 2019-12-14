# Source https://github.com/yunjey/pytorch-tutorial/
from __future__ import division
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, transform

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def validation(test_loader, model):
    # Test the model
    model.eval()
    # In test phase, we don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return (correct, total)


#%% Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'

#%% Hyper-parameters 
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

#%% MNIST dataset (images and labels)
train_dataset = torchvision.datasets.MNIST(root='../data/MNIST/', 
                                           train=True, 
                                           transform=transforms.ToTensor(),
                                           download=True)
num_training_data = len(train_dataset)

# Show 4 pairs of data
plt.figure(1)
for i in range(4):
    image, label = train_dataset[i]
    plt.subplot('14{}'.format(i))
    plt.imshow(transforms.ToPILImage()(image))
    plt.title('True label {}'.format(label))
plt.pause(0.1)

test_dataset = torchvision.datasets.MNIST(root='../data/MNIST/', 
                                          train=False, 
                                          transform=transforms.ToTensor(),
                                          download=True)

#%% Data loader (input pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)


#%% Fully connected neural network with one hidden layer
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            #nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            #nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

model = CNN(num_classes).to(device)
print('Number of parameters = {}'.format(count_parameters(model)))



#%% Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

#%% Train the model
num_batch = len(train_loader) #600 batches each containing 100 images = 60000 images
training_loss_v = []
valid_acc_v = []
for epoch in range(num_epochs):
    loss_tot = 0
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad() #set gradients of all parameters to zero
        loss.backward()
        optimizer.step()
        
        loss_tot += loss.item()
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Batch Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, num_batch, loss.item()/len(labels)))
            
    (correct, total) = validation(test_loader, model)
    print ('Epoch [{}/{}], Training Loss: {:.4f}, Valid Acc: {} %'
           .format(epoch+1, num_epochs, loss_tot/num_training_data, 100 * correct / total))
    training_loss_v.append(loss_tot/num_training_data)
    valid_acc_v.append(correct / total)


#%% Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')

#%% plot results
plt.figure(2)
plt.clf()
plt.plot(np.array(training_loss_v),'r',label='Training loss')
plt.legend()

plt.figure(3)
plt.clf()
plt.plot(np.array(valid_acc_v),'g',label='Validation accuracy')
plt.legend()

#%% plot filters
W = model.layer1[0].weight.data.cpu().numpy()
fig = plt.figure(5)
plt.clf()
for i in range(8):
    for j in range(2):
        W_cur = W[i+(8*j),0,:,:]
        plt.subplot(2,8,1+i+(8*j))
        plt.imshow(W_cur)
fig.suptitle('Filters first layer')
plt.pause(0.1)

(images, labels) = iter(test_loader).next()
outputs = model(images.to(device))
_, predicted = torch.max(outputs.data, 1)
plt.figure(4)
plt.clf()
for i in range(7):
    for j in range(3):
        image = images[i+(7*j),:]
        plt.subplot(3,7,1+i+(7*j))
        plt.imshow(transforms.ToPILImage()(image))
        plt.title('True {} / Pred {}'.format(labels[i+(7*j)], predicted[i+(7*j)]))

input("press any key to close")


# %%
