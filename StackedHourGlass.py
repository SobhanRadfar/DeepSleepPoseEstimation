import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F




class BatchNormReluConv(nn.Module):
    """
    Implements a sequence of Batch Normalization, ReLU activation, 
    and a Convolutional layer.
    """
    def __init__(self, input_channels, output_channels, kernel_size=1, stride=1, padding=0):
        super(BatchNormReluConv, self).__init__()
        
        # Creating a sequential layer comprising of BatchNorm, ReLU and Conv2D
        self.seq_layers = nn.Sequential(
            nn.BatchNorm2d(input_channels),
            nn.ReLU(),
            nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding)
        )

    def forward(self, tensor):
        return self.seq_layers(tensor)


class TripleConvBlock(nn.Module):
    """
    Represents a block of 3 BatchNormReluConv layers commonly used in deep neural networks.
    It's structured as: small filter -> larger filter -> small filter.
    """
    def __init__(self, input_channels, output_channels):
        super(TripleConvBlock, self).__init__()
        
        middle_channels = output_channels // 2
        
        # Creating a sequential layer of 3 BatchNormReluConv layers
        self.seq_layers = nn.Sequential(
            BatchNormReluConv(input_channels, middle_channels, 1),
            BatchNormReluConv(middle_channels, middle_channels, 3, padding=1),
            BatchNormReluConv(middle_channels, output_channels, 1)
        )

    def forward(self, tensor):
        return self.seq_layers(tensor)


class ShortcutLayer(nn.Module):
    """
    A layer that provides a shortcut in residual connections.
    It involves a potential 1x1 convolution if the input and output channels differ.
    """
    def __init__(self, input_channels, output_channels):
        super(ShortcutLayer, self).__init__()
        
        # If input and output channels are different, use a 1x1 convolution for matching dimensions.
        self.shortcut_conv = None if input_channels == output_channels else nn.Conv2d(input_channels, output_channels, 1)

    def forward(self, tensor):
        return tensor if self.shortcut_conv is None else self.shortcut_conv(tensor)


class ResidualBlock(nn.Module):
    """
    Implements a residual block combining a TripleConvBlock and a ShortcutLayer.
    This helps in learning identity functions which aids deeper networks to 
    be trained more effectively.
    """
    def __init__(self, input_channels, output_channels):
        super(ResidualBlock, self).__init__()
        
        self.triple_conv_block = TripleConvBlock(input_channels, output_channels)
        self.shortcut = ShortcutLayer(input_channels, output_channels)

    def forward(self, tensor):
        return self.triple_conv_block(tensor) + self.shortcut(tensor)


class CustomUpsample(nn.Module):
    """
    Custom upsampling layer which doubles the height and width of input tensor.
    """
    def __init__(self):
        super(CustomUpsample, self).__init__()

    def forward(self, tensor):
        return tensor[:, :, :, None, :, None].expand(-1, -1, -1, 2, -1, 2).reshape(tensor.size(0), tensor.size(1), tensor.size(2)*2, tensor.size(3)*2)



class Hourglass(nn.Module):
    """
    Hourglass module, a foundational piece of the stacked hourglass architecture.
    It captures and consolidates multi-resolution features.
    """
    def __init__(self, channel_count=256, depth=4, blocks_count=2, pooling_kernel=(2,2), pooling_stride=(2,2), upsampling_kernel=2):
        super(Hourglass, self).__init__()

        # Define the configuration parameters
        self.depth = depth
        self.blocks_count = blocks_count
        self.channel_count = channel_count
        self.pooling_kernel = pooling_kernel
        self.pooling_stride = pooling_stride
        self.upsampling_kernel = upsampling_kernel

        # Skip connections for hourglass
        skip_layers = [ResidualBlock(input_channels=self.channel_count, output_channels=self.channel_count) for _ in range(self.blocks_count)]
        self.skip_connection = nn.Sequential(*skip_layers)

        # Pooling layers followed by residuals after pooling
        self.max_pooling = nn.MaxPool2d(self.pooling_kernel, self.pooling_stride)
        after_pool_layers = [ResidualBlock(input_channels=self.channel_count, output_channels=self.channel_count) for _ in range(self.blocks_count)]
        self.after_pooling = nn.Sequential(*after_pool_layers)

        # Recursive hourglass, or sequence of residuals if this is the final reduction
        if depth > 1:
            self.sub_hourglass = Hourglass(channel_count=self.channel_count, depth=self.depth-1, blocks_count=self.blocks_count, pooling_kernel=self.pooling_kernel, pooling_stride=self.pooling_stride)
        else:
            base_res_layers = [ResidualBlock(input_channels=self.channel_count, output_channels=self.channel_count) for _ in range(self.blocks_count)]
            self.base_residuals = nn.Sequential(*base_res_layers)

        # Residual layers and upsample
        low_res_layers = [ResidualBlock(input_channels=self.channel_count, output_channels=self.channel_count) for _ in range(self.blocks_count)]
        self.low_resolution = nn.Sequential(*low_res_layers)
        self.upsample_layer = CustomUpsample()

    def forward(self, input_tensor):
        # Propagate through the hourglass
        skip_out = self.skip_connection(input_tensor)
        pooled_out = self.after_pooling(self.max_pooling(input_tensor))
        if self.depth > 1:
            hg_out = self.sub_hourglass(pooled_out)
        else:
            hg_out = self.base_residuals(pooled_out)
        
        upsampled_out = self.upsample_layer(self.low_resolution(hg_out))

        return upsampled_out + skip_out

class StackedHourGlass(nn.Module):
    """
    Stacked Hourglass network, a series of hourglass modules stacked together.
    Each hourglass module captures multi-resolution features.
    """
    def __init__(self, channel_count=256, stack_count=2, module_count=2, reduction_count=4, joint_count=14, multi_loss=True, input_channels=3):
        super(StackedHourGlass, self).__init__()
        self.channel_count = channel_count
        self.stack_count = stack_count
        self.module_count = module_count
        self.reduction_count = reduction_count
        self.joint_count = joint_count
        self.multi_loss = multi_loss

        # Initial preprocessing layers
        self.initial_layer = BatchNormReluConv(input_channels=input_channels, output_channels=64, kernel_size=7, stride=2, padding=3)
        self.first_residual = ResidualBlock(input_channels=64, output_channels=128)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.second_residual = ResidualBlock(input_channels=128, output_channels=128)
        self.third_residual = ResidualBlock(input_channels=128, output_channels=self.channel_count)

        # Lists for hourglasses and other components
        hg_layers, residual_layers, lin1_layers, channel_to_joint_layers, lin2_layers, joints_to_channel_layers = [], [], [], [], [], []

        for _ in range(stack_count):
            hg_layers.append(Hourglass(channel_count=channel_count, depth=reduction_count, blocks_count=module_count))
            res_layers = [ResidualBlock(input_channels=channel_count, output_channels=channel_count) for _ in range(module_count)]
            residual_layers.append(nn.Sequential(*res_layers))
            lin1_layers.append(BatchNormReluConv(input_channels=channel_count, output_channels=channel_count))
            channel_to_joint_layers.append(nn.Conv2d(channel_count, joint_count, 1))
            lin2_layers.append(nn.Conv2d(channel_count, channel_count, 1))
            joints_to_channel_layers.append(nn.Conv2d(joint_count, channel_count, 1))

        # Convert lists to ModuleLists for proper handling in PyTorch
        self.hg_modules = nn.ModuleList(hg_layers)
        self.residual_modules = nn.ModuleList(residual_layers)
        self.lin1_modules = nn.ModuleList(lin1_layers)
        self.channel_to_joint_modules = nn.ModuleList(channel_to_joint_layers)
        self.lin2_modules = nn.ModuleList(lin2_layers)
        self.joints_to_channel_modules = nn.ModuleList(joints_to_channel_layers)

    def forward(self, input_tensor):
        input_tensor = self.initial_layer(input_tensor)
        input_tensor = self.first_residual(input_tensor)
        input_tensor = self.max_pool(input_tensor)
        input_tensor = self.second_residual(input_tensor)
        input_tensor = self.third_residual(input_tensor)
        outputs = []

        for i in range(len(self.hg_modules)):
            tensor = self.lin1_modules[i](self.residual_modules[i](self.hg_modules[i](input_tensor)))
            outputs.append(self.channel_to_joint_modules[i](tensor))
            tensor = self.lin2_modules[i](tensor)
            input_tensor = input_tensor + tensor + self.joints_to_channel_modules[i](outputs[i])

        return outputs if self.multi_loss else outputs[-1]



# class Hourglass(nn.Module):
#     """
#     Hourglass module, a foundational piece of the stacked hourglass architecture.
#     It captures and consolidates multi-resolution features.
#     """
#     def __init__(self, nChannels=256, numReductions=4, nModules=2, poolKernel=(2,2), poolStride=(2,2), upSampleKernel=2):
#         super(Hourglass, self).__init__()

#         # Define the configuration parameters
#         self.numReductions = numReductions
#         self.nModules = nModules
#         self.nChannels = nChannels
#         self.poolKernel = poolKernel
#         self.poolStride = poolStride
#         self.upSampleKernel = upSampleKernel

#         # Skip connections for hourglass
#         _skip = [Residual(self.nChannels, self.nChannels) for _ in range(self.nModules)]
#         self.skip = nn.Sequential(*_skip)

#         # Pooling layers followed by residuals after pooling
#         self.mp = nn.MaxPool2d(self.poolKernel, self.poolStride)
#         _afterpool = [Residual(self.nChannels, self.nChannels) for _ in range(self.nModules)]
#         self.afterpool = nn.Sequential(*_afterpool)

#         # Recursive hourglass, or sequence of residuals if this is the final reduction
#         if numReductions > 1:
#             self.hg = Hourglass(self.nChannels, self.numReductions-1, self.nModules, self.poolKernel, self.poolStride)
#         else:
#             _num1res = [Residual(self.nChannels,self.nChannels) for _ in range(self.nModules)]
#             self.num1res = nn.Sequential(*_num1res)

#         # Residual layers and upsample
#         _lowres = [Residual(self.nChannels,self.nChannels) for _ in range(self.nModules)]
#         self.lowres = nn.Sequential(*_lowres)
#         self.up = myUpsample()

#     def forward(self, x):
#         # Propagate through the hourglass
#         out1 = self.skip(x)
#         out2 = self.afterpool(self.mp(x))
#         out2 = self.hg(out2) if self.numReductions > 1 else self.num1res(out2)
#         out2 = self.up(self.lowres(out2))

#         return out2 + out1

# class StackedHourGlass(nn.Module):
#     """
#     Stacked Hourglass network, a series of hourglass modules stacked together.
#     Each hourglass module captures multi-resolution features.
#     """
#     def __init__(self, nChannels=256, nStack=2, nModules=2, numReductions=4, nJoints=14, mu_loss=True, in_ch=3):
#         super(StackedHourGlass, self).__init__()
#         self.nChannels = nChannels
#         self.nStack = nStack
#         self.nModules = nModules
#         self.numReductions = numReductions
#         self.nJoints = nJoints
#         self.mu_loss = mu_loss

#         # Initial preprocessing layers
#         self.start = BnReluConv(in_ch, 64, kernelSize=7, stride=2, padding=3)
#         self.res1 = Residual(64, 128)
#         self.mp = nn.MaxPool2d(2, 2)
#         self.res2 = Residual(128, 128)
#         self.res3 = Residual(128, self.nChannels)

#         # Lists for hourglasses and other components
#         _hourglass, _Residual, _lin1, _chantojoints, _lin2, _jointstochan = [], [], [], [], [], []

#         for _ in range(nStack):
#             _hourglass.append(Hourglass(nChannels, numReductions, nModules))
#             _ResidualModules = [Residual(nChannels, nChannels) for _ in range(nModules)]
#             _Residual.append(nn.Sequential(*_ResidualModules))
#             _lin1.append(BnReluConv(nChannels, nChannels))
#             _chantojoints.append(nn.Conv2d(nChannels, nJoints,1))
#             _lin2.append(nn.Conv2d(nChannels, nChannels,1))
#             _jointstochan.append(nn.Conv2d(nJoints,nChannels,1))

#         # Convert lists to ModuleLists for proper handling in PyTorch
#         self.hourglass = nn.ModuleList(_hourglass)
#         self.Residual = nn.ModuleList(_Residual)
#         self.lin1 = nn.ModuleList(_lin1)
#         self.chantojoints = nn.ModuleList(_chantojoints)
#         self.lin2 = nn.ModuleList(_lin2)
#         self.jointstochan = nn.ModuleList(_jointstochan)
#         self.mu_loss = mu_loss

#     def forward(self, x):
#         x = self.start(x)
#         x = self.res1(x)
#         x = self.mp(x)
#         x = self.res2(x)
#         x = self.res3(x)
#         out = []

#         for i in range(len(self.hourglass)):
#             x1 = self.lin1[i](self.Residual[i](self.hourglass[i](x)))
#             out.append(self.chantojoints[i](x1))
#             x1 = self.lin2[i](x1)
#             x = x + x1 + self.jointstochan[i](out[i])

#         return out if self.mu_loss else out[-1]


# def get_pose_net(nChannels=256, nStack=2, nModules=2, numReductions=4, out_ch=14, in_ch=1, mu_loss=True):
#     """
#     Utility function to create the StackedHourGlass model with the given configuration.
#     """
#     model = StackedHourGlass(nChannels=nChannels, nStack=nStack, nModules=nModules, numReductions=numReductions, nJoints=out_ch, in_ch=in_ch, mu_loss=mu_loss)
#     return model


# class BnReluConv(nn.Module):
#     """
#     Implements a sequence of Batch Normalization, ReLU activation, 
#     and a Convolutional layer.
#     """
#     def __init__(self, in_channels, out_channels, kernelSize=1, stride=1, padding=0):
#         super(BnReluConv, self).__init__()
        
#         # Creating a sequential layer comprising of BatchNorm, ReLU and Conv2D
#         self.layers = nn.Sequential(
#             nn.BatchNorm2d(in_channels),  # Normalize input features
#             nn.ReLU(),                    # Activation function
#             nn.Conv2d(in_channels, out_channels, kernelSize, stride, padding)  # Convolutional layer
#         )

#     def forward(self, x):
#         return self.layers(x)


# class ConvBlock(nn.Module):
#     """
#     Represents a block of 3 BnReluConv layers commonly used in deep neural networks.
#     It's structured as: small filter -> larger filter -> small filter.
#     """
#     def __init__(self, in_channels, out_channels):
#         super(ConvBlock, self).__init__()
        
#         mid_channels = out_channels // 2
        
#         # Creating a sequential layer of 3 BnReluConv layers
#         self.layers = nn.Sequential(
#             BnReluConv(in_channels, mid_channels, 1),             # 1x1 Convolution
#             BnReluConv(mid_channels, mid_channels, 3, padding=1), # 3x3 Convolution
#             BnReluConv(mid_channels, out_channels, 1)             # 1x1 Convolution
#         )

#     def forward(self, x):
#         return self.layers(x)


# class SkipLayer(nn.Module):
#     """
#     A layer that provides a shortcut in residual connections.
#     It involves a potential 1x1 convolution if the input and output channels differ.
#     """
#     def __init__(self, in_channels, out_channels):
#         super(SkipLayer, self).__init__()
        
#         # If input and output channels are different, use a 1x1 convolution for matching dimensions.
#         self.conv = None if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

#     def forward(self, x):
#         return x if self.conv is None else self.conv(x)


# class Residual(nn.Module):
#     """
#     Implements a residual block combining a ConvBlock and a SkipLayer.
#     This helps in learning identity functions which aids deeper networks to 
#     be trained more effectively.
#     """
#     def __init__(self, in_channels, out_channels):
#         super(Residual, self).__init__()
        
#         self.conv_block = ConvBlock(in_channels, out_channels)
#         self.skip_layer = SkipLayer(in_channels, out_channels)

#     def forward(self, x):
#         # Sum of ConvBlock output and SkipLayer output
#         return self.conv_block(x) + self.skip_layer(x)


# class myUpsample(nn.Module):
#     """
#     Custom upsampling layer which doubles the height and width of input tensor.
#     """
#     def __init__(self):
#         super(myUpsample, self).__init__()

#     def forward(self, x):
#         return x[:, :, :, None, :, None].expand(-1, -1, -1, 2, -1, 2).reshape(x.size(0), x.size(1), x.size(2)*2, x.size(3)*2)



