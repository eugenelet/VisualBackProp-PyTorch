import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torchvision import transforms

import numpy as np
#from net.utility import tools
import cv2

class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        logit = x
        prob = F.sigmoid(logit)
        return logit, prob


def alexnet(**kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(**kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
    return model


if __name__=="__main__":

    image = cv2.imread('/Users/Eugene/Documents/Git/pytorch/image/cat1.jpg')
    image = cv2.resize(image,(227,227))
    loader = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                 ])
    image = loader(image)
    image = image.contiguous().view(1,3,227,227)
    pretrained_file = "/Users/Eugene/Documents/Git/pytorch/VisualBackProp/snap/alexnet-owt-4df8aa71.pth"
    model = alexnet()

    tools.load_valid(model,pretrained_file,None)
    logit, prob = model.forward(Variable(image))
    prob = prob.view(-1)
    value, indices = prob.max(0)
    print("Max Value: ", value)
    print("Index: ",indices)
