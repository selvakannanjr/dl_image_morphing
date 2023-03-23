import cv2
import numpy as np
import torch 
from torchvision import transforms, utils
from PIL import Image

import torch.nn as nn
import torch.nn.functional as F

class CustomFacePointsModel(nn.Module):
    def __init__(self):
        super(CustomFacePointsModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.fc1 = nn.Linear(512*7*7, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 68*2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc3(x)
        return x


#defining transformations using transforms.Compose([])
data_transform =   transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean = [0.5, 0.5, 0.5],std = [0.5, 0.5, 0.5])])

model=torch.load('modelfinal.pth')
model.eval()


img=Image.open('face-morphing-multiple-images//aligned_images//bradpitt.png')
img = data_transform(img).unsqueeze(0)
x=model(img).detach().numpy().reshape(68,2)

# #image=cv2.imread('face-morphing-multiple-images//aligned_images//bradpitt.png')
# image = img.squeeze(0).detach().numpy()
# image=np.transpose(image,(1,2,0))
# print(image.shape)
# print(x)
# for i in x:
#     image = cv2.circle(np.float32(image), (int(i[0]),int(i[1])), radius=2, color=(0, 0, 255), thickness=-1)

# cv2.imwrite('plot.png',image)

img1=Image.open('face-morphing-multiple-images//aligned_images//bradpitt.png')
img_t=transforms.Resize((224,224))(img1)
img_t=np.asarray(img_t)
print(img_t.shape)
for i in x:
    img_t = cv2.circle(img_t, (int(i[0])-25,int(i[1])+7), radius=1, color=(255, 0, 0), thickness=-1)
cv2.imwrite('plot.png',cv2.cvtColor(img_t, cv2.COLOR_BGR2RGB))
