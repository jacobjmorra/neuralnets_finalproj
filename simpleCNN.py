import torch
import torchvision
import torchvision.transforms as transforms

#get dataset
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



#plot some data
import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
#imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))




#define CNN
import torch.nn as nn
import torch.nn.functional as F


# here we can modify the CNN to look like the HCNN in the paper
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #32 x 32 image

        #define our 3 convolution operations
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=5, out_channels=7, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=7, out_channels=12, kernel_size=3)

        #define our max pooling operations
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        #define our global average pooling operation
        self.gap = nn.AvgPool2d(kernel_size=2, padding=1)

        #readout layer is the class outputs (10 labels)
        #self.fc1 = nn.Linear(in_features = 3 * 3 * 12, out_features = 12)
        #self.fc2 = nn.Linear(in_features= 12, out_features= 10)

        self.fc = nn.Linear(in_features=3*3*12, out_features=10)
    def forward(self, x):

        #first round of convolution and max pooling
        x = F.relu(self.conv1(x)) # convolution
        #print(x.shape)
        x = self.pool(x) #max pooling
        #print(x.shape)

        #second round of convolution and max pooling
        x = F.relu(self.conv2(x)) #convolution
        #print(x.shape)
        x = self.pool(x) #max pooling
        #print(x.shape)

        #third round of convolution
        x = F.relu(self.conv3(x))
        #print(x.shape)

        x = self.gap(x).flatten()
        #print(x.shape)

        #fcnn to classifier, fcnn to output labels
        x = x.view(-1, 3 * 3 * 12) #the "-1" is fill in the blank row size, col size is 4*4*12
        #x = self.fc1(x)
        #x = self.fc2(x)

        x = self.fc(x)
        return x

net = Net()




#set up GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)

net.to(device)




#define optim/loss
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)






#train
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

#save model
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)




#test
dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

#load model
net = Net()
net.load_state_dict(torch.load(PATH))

#test model
outputs = net(images)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))


#next, train on some numerosity images and see if there is a preference..

#to do..


