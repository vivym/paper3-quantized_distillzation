import torch.nn as nn
import numpy as np
from .functions import ReverseLayerF
# from resnet import ResNet18
from .model_resnet import ResidualNet

class SmallCNNModel(nn.Module):

    def __init__(self,num_classes=3):
        super(SmallCNNModel, self).__init__()
        # self.feature = nn.Sequential()
        # self.feature.add_module('f_conv1', nn.Conv2d(1, 8, kernel_size=3))
        # self.feature.add_module('f_bn1', nn.BatchNorm2d(8))
        # self.feature.add_module('f_pool1', nn.MaxPool2d(2))
        # self.feature.add_module('f_relu1', nn.ReLU(True))
        # self.feature.add_module('f_conv2', nn.Conv2d(8, 16, kernel_size=3))
        # self.feature.add_module('f_bn2', nn.BatchNorm2d(16))
        # self.feature.add_module('f_drop1', nn.Dropout2d())
        # self.feature.add_module('f_pool2', nn.MaxPool2d(2))
        # self.feature.add_module('f_relu2', nn.ReLU(True))
        self.sharedNet = ResidualNet("small", "small", 3, None)
        self.bottleneck = nn.Linear(3, 256)
        self.source_fc = nn.Linear(256, num_classes)
        self.softmax = nn.Softmax(dim=1)

        #self.class_classifier = nn.Sequential()
        # self.class_classifier.add_module('c_fc1', nn.Linear(50 * 4 * 4, 100))
        # self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))

        # self.class_classifier.add_module('c_fc1', nn.Linear(16*6*6, 1024))
        # self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(1024))
        # self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        # self.class_classifier.add_module('c_drop1', nn.Dropout())
        # # self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        # # self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        # self.class_classifier.add_module('c_fc2', nn.Linear(1024, 1024))
        # self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(1024))
        # self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        # # self.class_classifier.add_module('c_fc3', nn.Linear(100, 10))
        # self.class_classifier.add_module('c_fc3', nn.Linear(1024, 3))
        # self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))

        # self.class_classifier.add_module('c_fc1', nn.Linear(3, 256))
        # #self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(256))
        # self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        # self.class_classifier.add_module('c_drop1', nn.Dropout())
        # self.class_classifier.add_module('c_fc1', nn.Linear(256, num_classes))
        # self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))

        self.domain_classifier = nn.Sequential()
        # self.domain_classifier.add_module('d_fc1', nn.Linear(50 * 4 * 4, 100))
        # self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        # self.domain_classifier.add_module('d_fc1', nn.Linear(16*6*6, 1024))
        self.domain_classifier.add_module('d_fc1', nn.Linear(3, 1024))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(1024))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        # self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_fc2', nn.Linear(1024, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data, alpha = None):
        if alpha is None:
            alpha = 2. / (1. + np.exp(-10 * 1.)) - 1

        # input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
        input_data = input_data.expand(input_data.data.shape[0], 1, 32, 32)
        feature = self.sharedNet(input_data)
        source_share = self.bottleneck(feature)
        source = self.source_fc(source_share)
        p_source = self.softmax(source)

        # print(feature.shape)
        #feature = feature.view(-1, 16*6*6)
        # print(feature.shape)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        #class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return p_source, domain_output
