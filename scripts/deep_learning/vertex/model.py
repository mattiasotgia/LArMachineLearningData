# model.py

import torch.nn as nn
import torch


def maxpool():
    """Return a max pooling layer.
    
        The maxpooling layer has a kernel size of 2, a stride of 2 and no padding.

        Returns:
            The max pooling layer
    """
    return nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)


def dropout(prob):
    """Return a dropout layer.

        Args:
            prob: The probability that drop out will be applied.

        Returns:
            The dropout layer
    """
    return nn.Dropout(prob)


def reinit_layer(layer, leak = 0.0, use_kaiming_normal=True):
    """Reinitialises convolutional layer weights.
    
        The default Kaiming initialisation in PyTorch is not optimal, this method
        reinitialises the layers using better parameters

        Args:
            seq_block: The layer to be reinitialised.
            leak: The leakiness of ReLU (default: 0.0)
            use_kaiming_normal: Use Kaiming normal if True, Kaiming uniform otherwise (default: True)
    """
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
        if use_kaiming_normal:
            nn.init.kaiming_normal_(layer.weight, a = leak)
        else:
            nn.init.kaiming_uniform_(layer.weight, a = leak)
            layer.bias.data.zero_()


class ConvBlock(nn.Module):
    """A convolution block
    """
    
    # Sigmoid activation suitable for binary cross-entropy
    def __init__(self, c_in, c_out, k_size = 3, k_pad = 1):
        """Constructor.

            Args:
                c_in: The number of input channels
                c_out: The number of output channels
                k_size: The size of the convolution filter
                k_pad: The amount of padding around the images
        """
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size = k_size, padding = k_pad, stride = 1)
        self.norm1 = nn.GroupNorm(8, c_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size = k_size, padding = k_pad, stride = 1)
        self.norm2 = nn.GroupNorm(8, c_out)
        self.identity = nn.Conv2d(c_in, c_out, kernel_size = 1, padding = 0, stride = 1)
        reinit_layer(self.conv1)
        reinit_layer(self.conv2)

    def forward(self, x):
        """Forward pass.
        
            Args:
                x: The input to the layer
                
            Returns:
                The output from the layer
        """
        identity = self.identity(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return self.relu(x + identity)


class TransposeConvBlock(nn.Module):
    """A tranpose convolution block
    """
    
    def __init__(self, c_in, c_out, k_size = 3, k_pad = 1):
        """Constructor.

            Args:
                c_in: The number of input channels
                c_out: The number of output channels
                k_size: The size of the convolution filter
                k_pad: The amount of padding around the images
        """
        super(TransposeConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(c_in, c_out, kernel_size = k_size, padding = k_pad, output_padding = 1, stride = 2),
            nn.GroupNorm(8, c_out),
            nn.ReLU(inplace=True))
        reinit_layer(self.block[0])

    def forward(self, x):
        """Forward pass.
        
            Args:
                x: The input to the layer
                
            Returns:
                The output from the layer
        """
        return self.block(x)

class Sigmoid(nn.Module):
    """A sigmoid activation function that supports categorical cross-entropy
    """
    
    def __init__(self, out_range = None):
        """Constructor.

            Args:
                out_range: A tuple covering the minimum and maximum values to map to
        """
        super(Sigmoid, self).__init__()
        if out_range is not None:
            self.low, self.high = out_range
            self.range = self.high - self.low
        else:
            self.low = None
            self.high = None
            self.range = None
    
    def forward(self, x):
        """Applies the sigmoid function.
        
            Rescales to the specified range if provided during construction
        
            Args:
                x: The input to the layer
                
            Returns:
                The (potentially scaled) sigmoid of the input
        """
        if self.low is not None:
            return torch.sigmoid(x) * (self.range) + self.low
        else:
            return torch.sigmoid(x)

class UNet(nn.Module):
    """A U-Net for semantic segmentation.
    """
    
    def __init__(self, in_dim, n_classes, depth = 4, n_filters = 16, drop_prob = 0.1, y_range = None):
        """Constructor.

            Args:
                in_dim: The number of input channels
                n_classes: The number of classes
                depth: The number of convolution blocks in the downsampling and upsampling arms of the U (default: 4)
                n_filters: The number of filters in the first layer (doubles for each downsample) (default: 16)
                drop_prob: The dropout probability for each layer (default: 0.1)
                y_range: The range of values (low, high) to map to in the output (default: None)
        """
        super(UNet, self).__init__()
        # Contracting Path
        self.ds_conv_1 = ConvBlock(in_dim, n_filters)
        self.ds_conv_2 = ConvBlock(n_filters, 2 * n_filters)
        self.ds_conv_3 = ConvBlock(2 * n_filters, 4 * n_filters)
        self.ds_conv_4 = ConvBlock(4 * n_filters, 8 * n_filters)

        self.ds_maxpool_1 = maxpool()
        self.ds_maxpool_2 = maxpool()
        self.ds_maxpool_3 = maxpool()
        self.ds_maxpool_4 = maxpool()
        
        self.ds_dropout_1 = dropout(drop_prob)
        self.ds_dropout_2 = dropout(drop_prob)
        self.ds_dropout_3 = dropout(drop_prob)
        self.ds_dropout_4 = dropout(drop_prob)
        
        self.bridge = ConvBlock(8 * n_filters, 16 * n_filters)
        
        # Expansive Path
        self.us_tconv_4 = TransposeConvBlock(16 * n_filters, 8 * n_filters)
        self.us_tconv_3 = TransposeConvBlock(8 * n_filters, 4 * n_filters)
        self.us_tconv_2 = TransposeConvBlock(4 * n_filters, 2 * n_filters)
        self.us_tconv_1 = TransposeConvBlock(2 * n_filters, n_filters)

        self.us_conv_4 = ConvBlock(16 * n_filters, 8 * n_filters)
        self.us_conv_3 = ConvBlock(8 * n_filters, 4 * n_filters)
        self.us_conv_2 = ConvBlock(4 * n_filters, 2 * n_filters)
        self.us_conv_1 = ConvBlock(2 * n_filters, 1 * n_filters)

        self.us_dropout_4 = dropout(drop_prob)
        self.us_dropout_3 = dropout(drop_prob)
        self.us_dropout_2 = dropout(drop_prob)
        self.us_dropout_1 = dropout(drop_prob)

        self.output = nn.Sequential(nn.Conv2d(n_filters, n_classes, 1), Sigmoid(y_range))

    def forward(self, x):
        """Forward pass.
        
            Args:
                x: The input to the layer
                
            Returns:
                The output from the layer
        """
        res = x

        # Downsample
        res = self.ds_conv_1(res); conv_stack_1 = res.clone()
        res = self.ds_maxpool_1(res)
        res = self.ds_dropout_1(res)
        
        res = self.ds_conv_2(res); conv_stack_2 = res.clone()
        res = self.ds_maxpool_2(res)
        res = self.ds_dropout_2(res)
        
        res = self.ds_conv_3(res); conv_stack_3 = res.clone()
        res = self.ds_maxpool_3(res)
        res = self.ds_dropout_3(res)
        
        res = self.ds_conv_4(res); conv_stack_4 = res.clone()
        res = self.ds_maxpool_4(res)
        res = self.ds_dropout_4(res)
        
        # Bridge
        res = self.bridge(res)
        
        # Upsample
        res = self.us_tconv_4(res)
        res = torch.cat([res, conv_stack_4], dim=1)
        res = self.us_dropout_4(res)
        res = self.us_conv_4(res)

        res = self.us_tconv_3(res)
        res = torch.cat([res, conv_stack_3], dim=1)
        res = self.us_dropout_3(res)
        res = self.us_conv_3(res)
        
        res = self.us_tconv_2(res)
        res = torch.cat([res, conv_stack_2], dim=1)
        res = self.us_dropout_2(res)
        res = self.us_conv_2(res)
        
        res = self.us_tconv_1(res)
        res = torch.cat([res, conv_stack_1], dim=1)
        res = self.us_dropout_1(res)
        res = self.us_conv_1(res)
        
        output = self.output(res)

        return output