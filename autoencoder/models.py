import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim

## RESNET18 model #########################

# Encoder 
class ResNet18_Encoder(nn.Module):
    def __init__(self):
        super(ResNet18_Encoder, self).__init__()
        # Do not load pre-trained weights
        self.resnet = models.resnet18(weights=None)  
        # Eliminate the classification head
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])
        # output is 512 feature maps of size 7x7

    def forward(self, x):
        x = self.resnet(x)
        return x
    

# Decoder
class ResNet18_Decoder(nn.Module):
    def __init__(self):
        super(ResNet18_Decoder, self).__init__()
        
        self.upconv1 = nn.ConvTranspose2d(512, 256, 2, stride=2)  # Output size: 14x14
        self.conv1 = nn.Conv2d(256, 256, 3, padding=1)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)  # Output size: 28x28
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.upconv3 = nn.ConvTranspose2d(128, 64, 2, stride=2)   # Output size: 56x56
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.upconv4 = nn.ConvTranspose2d(64, 64, 2, stride=2)    # Output size: 112x112
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.upconv5 = nn.ConvTranspose2d(64, 32, 2, stride=2)    # Output size: 224x224
        self.conv5 = nn.Conv2d(32, 32, 3, padding=1)
        self.final = nn.Conv2d(32, 3, 3, padding=1)               # Output size: 224x224

    def forward(self, x):
        x = self.upconv1(x)
        x = nn.ReLU()(self.conv1(x))
        x = self.upconv2(x)
        x = nn.ReLU()(self.conv2(x))
        x = self.upconv3(x)
        x = nn.ReLU()(self.conv3(x))
        x = self.upconv4(x)
        x = nn.ReLU()(self.conv4(x))
        x = self.upconv5(x)
        x = nn.ReLU()(self.conv5(x))
        x = torch.sigmoid(self.final(x)) 
        return x
    

## RESNET50 model #########################
class ResNet50_Encoder(nn.Module):
    def __init__(self):
        super(ResNet50_Encoder, self).__init__()
        # Load a not pre-trained ResNet
        self.resnet = models.resnet50(weights=None)  
        # Load the weights
        # self.resnet.load_state_dict(torch.load('models/resnet50_weights.pth'))
        
        # Remove the fully connected layers (classification head)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])

    def forward(self, x):
        x = self.resnet(x)
        return x
    

    
class ResNet50_Decoder(nn.Module):
    def __init__(self):
        super(ResNet50_Decoder, self).__init__()
        
        self.upconv1 = nn.ConvTranspose2d(2048, 1024, 2, stride=2) # 2048 to 1024 feature maps
        self.conv1 = nn.Sequential(
            nn.Conv2d(1024, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 1024, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )

        self.upconv2 = nn.ConvTranspose2d(1024, 512, 2, stride=2) # 1024 to 512 feature maps
        self.conv2 = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2) # 512 to 256 feature maps
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.upconv4 = nn.ConvTranspose2d(256, 128, 2, stride=2) # 256 to 128 feature maps
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.upconv5 = nn.ConvTranspose2d(128, 64, 2, stride=2) # 128 to 64 feature maps
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.final = nn.Conv2d(64, 3, 3, padding=1) # Final output layer

    def forward(self, x):
        x = self.upconv1(x)
        x = self.conv1(x)
        x = self.upconv2(x)
        x = self.conv2(x)
        x = self.upconv3(x)
        x = self.conv3(x)
        x = self.upconv4(x)
        x = self.conv4(x)
        x = self.upconv5(x)
        x = self.conv5(x)
        x = torch.sigmoid(self.final(x))
        return x
    
########## DenseNet #############################
class DenseNet_Encoder(nn.Module):
    def __init__(self):
        super(DenseNet_Encoder, self).__init__()
        self.densenet = models.densenet121(weights=None)  
        # Load the weights
        self.densenet.load_state_dict(torch.load('models/densenet121-a639ec97(1).pth'), strict=False )
        
        # Remove the fully connected layer
        self.densenet = nn.Sequential(*list(self.densenet.children())[:-2])

    def forward(self, x):
        x = self.densenet(x)
        return x
    
class DenseNet_Decoder(nn.Module):
    def __init__(self, num_init_features=1024, growth_rate=32, block_config=(16, 24, 12, 6)):
        super(DenseNet_Decoder, self).__init__()

        # Start with the number of features coming from the end of the encoder
        num_features = num_init_features

        # Blocks of the decoder
        self.features = nn.Sequential()

        # Iterate over the reversed block configuration
        for i, num_layers in enumerate(block_config):
            block = self._make_dense_block(num_features, growth_rate, num_layers)
            self.features.add_module(f'decode_denseblock{i+1}', block)
            num_features += num_layers * growth_rate

            if i != len(block_config) - 1:  # No transition up after the last block
                num_features //= 2  # Reduce the number of features by 2
                trans_up = self._make_transition_up(num_features)
                self.features.add_module(f'transition_up{i+1}', trans_up)

    def _make_dense_block(self, in_features, growth_rate, num_layers):
        layers = []
        for _ in range(num_layers):
            layers.append(nn.BatchNorm2d(in_features))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.ConvTranspose2d(in_features, growth_rate, kernel_size=3, stride=1, padding=1))
            in_features += growth_rate
        return nn.Sequential(*layers)

    def _make_transition_up(self, in_features):
        # Transposed convolution to double the feature map size
        return nn.Sequential(
            nn.ConvTranspose2d(in_features, in_features, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.features(x)

