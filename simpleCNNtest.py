import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# functions to show an image

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

#define CNN
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

PATH = './cifar_net.pth'
















#load model
net = Net()
net.load_state_dict(torch.load(PATH))


















#my images
image_dir = '../neuralnets_finalproj'

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

dataset = datasets.ImageFolder(image_dir, transform=transform)
#print(dataset)

newloader = torch.utils.data.DataLoader(dataset, batch_size=100,
                                         shuffle=False, num_workers=2)


# get some random training images
dataiter = iter(newloader)
images, labels = dataiter.next()

# show images
#imshow(torchvision.utils.make_grid(images))
# print labels

#outputs = net(images)
#print(images)

#_, predicted = torch.max(outputs, 1)

#classes = ('plane', 'car', 'bird', 'cat',
#           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
#                              for j in range(10)))




















class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
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
        #print("YAYAYAYA")
        #x = self.fc1(x)
        #x = self.fc2(x)

        #x = self.fc(x)
        return x

net2 = Net2()
net2.load_state_dict(torch.load(PATH))





outputs = net2(images)
#print(outputs)
print("Testing on " , len(outputs), " samples..")

_, predicted = torch.max(outputs, 1)

print(outputs)
#print(predicted)

#classes = ('plane', 'car', 'bird', 'cat',
#           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
#                              for j in range(10)))







# next things to do.. pick 20 images for each numerosity, put into their own folder
# next, test on those images and see if the final layer activation (outputs) changes


#Question: What would a higher/lower activation correspond to???