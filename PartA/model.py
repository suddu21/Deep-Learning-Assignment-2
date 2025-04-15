import torch
import torch.nn as nn

class FlexibleCNN(nn.Module):
    def __init__(self, 
                 num_classes=10,
                 conv_filters=[32, 32, 32, 32, 32],  # Number of filters for each conv layer
                 kernel_size=3,
                 activation='relu',
                 dense_neurons=512,
                 dropout_rate=0.0,
                 use_batch_norm=False):
        super(FlexibleCNN, self).__init__()
        
        # Activation function selection
        if activation.lower() == 'relu':
            self.activation = nn.ReLU()
        elif activation.lower() == 'gelu':
            self.activation = nn.GELU()
        elif activation.lower() == 'silu':
            self.activation = nn.SiLU()
        elif activation.lower() == 'mish':
            self.activation = nn.Mish()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
        
        # Convolutional layers
        self.conv_blocks = nn.ModuleList()
        in_channels = 3  # RGB input
        
        for num_filters in conv_filters:
            conv_block = []
            # Conv layer
            conv_block.append(nn.Conv2d(in_channels, num_filters, kernel_size, padding=kernel_size//2))
            
            # Batch normalization if requested
            if use_batch_norm:
                conv_block.append(nn.BatchNorm2d(num_filters))
            
            # Activation
            conv_block.append(self.activation)
            
            # Max pooling
            conv_block.append(nn.MaxPool2d(2))
            
            # Add dropout if rate > 0
            if dropout_rate > 0:
                conv_block.append(nn.Dropout2d(dropout_rate))
            
            self.conv_blocks.append(nn.Sequential(*conv_block))
            in_channels = num_filters
        
        # Calculate the size of flattened features
        self.flatten = nn.Flatten()
        
        # Dense layers
        self.dense = nn.Sequential(
            nn.Linear(in_channels * 8 * 8, dense_neurons),  # 8x8 due to 5 max pooling layers
            self.activation,
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Linear(dense_neurons, num_classes)
        )
    
    def forward(self, x):
        # Apply convolutional blocks
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        
        # Flatten and apply dense layers
        x = self.flatten(x)
        x = self.dense(x)
        return x 