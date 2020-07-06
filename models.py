## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        #output size = (W-F)/S +1 = (224-5)/1 +1 = 220
        #the output Tensor for one image, will have the dimensions: (32, 220, 220)
        self.conv1 = nn.Conv2d(1, 32, 5)
        
        # maxpool layer
        # pool with kernel_size=2, stride=2
        #output of maxpooling1 => (W-F)/s +1=(220-2)/2 +1=110
        self.pool = nn.MaxPool2d(2, 2) #output=(32, 110, 110)
        
        
        # second conv layer: 32 inputs, 64 outputs, 5x5 conv
        ## output size = (W-F)/S +1 = (110-5)/1 +1 = 106
        # the output tensor will have dimensions: (64, 106, 106)
        # after another pool layer this becomes (64, 53, 53)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # third conv layer: 64 inputs, 128 outputs, 3x3 conv
        ## output size = (W-F)/S +1 = (53-3)/1 +1 = 51
        # the output tensor will have dimensions: (128, 51, 51)
        # after another pool layer this becomes (128, 25, 25); 25.5 is rounded down
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # 128 outputs * the 25*25 filtered/pooled map size
        self.fc1 = nn.Linear(128*25*25, 800)
        
        # dropout with p=0.4
        self.fc1_drop = nn.Dropout(p=0.4)
        
        # finally, create 13610 output channels (for the 136 classes)
        self.fc2 = nn.Linear(800, 136)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = self.fc2(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x