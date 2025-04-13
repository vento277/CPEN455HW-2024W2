import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pdb
import os

# Assuming layers.py provides these functions
from layers import gated_resnet, down_shifted_conv2d, down_right_shifted_conv2d, down_shifted_deconv2d, down_right_shifted_deconv2d, nin, concat_elu, discretized_mix_logistic_loss

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        self.position_encoding = nn.Parameter(torch.randn(1, in_channels, 32, 32))  # Assuming max size 32x32


    def forward(self, x):
        batch_size, C, H, W = x.size()
        # Add position encoding scaled to current feature map size
        pos_encoding = F.interpolate(self.position_encoding, size=(H, W), mode='bilinear', align_corners=False)
        x = x + 0.1 * pos_encoding
        
        # Multi-head attention mechanism
        query = self.query(x).view(batch_size, -1, H * W).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, H * W)
        
        # Temperature scaling for sharper attention
        attention = self.softmax(torch.bmm(query, key) / (C ** 0.5))
        value = self.value(x).view(batch_size, -1, H * W)
        
        out = torch.bmm(value, attention.permute(0, 2, 1)).view(batch_size, C, H, W)
        return self.gamma * out + x  # Residual connection

class ConditionalNorm(nn.Module):
    def __init__(self, num_features, embedding_dim):
        super(ConditionalNorm, self).__init__()
        self.instance_norm = nn.InstanceNorm2d(num_features, affine=False)
        self.gamma = nn.Linear(embedding_dim, num_features)
        self.beta = nn.Linear(embedding_dim, num_features)
        # Don't hardcode layer norm dimensions
        self.use_layer_norm = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, embedding):
        # Create layer norm on the fly based on current input dimensions
        if x.size(2) <= 16 and x.size(3) <= 16:
            # Create layer norm with the exact dimensions of the input
            layer_norm = nn.LayerNorm([x.size(1), x.size(2), x.size(3)], device=x.device)
            x_norm = self.instance_norm(x)
            layer_norm_output = layer_norm(x)
            x_norm = self.use_layer_norm * layer_norm_output + (1 - self.use_layer_norm) * x_norm
        else:
            x_norm = self.instance_norm(x)
            
        gamma = self.gamma(embedding).view(-1, x.size(1), 1, 1)
        beta = self.beta(embedding).view(-1, x.size(1), 1, 1)
        return x_norm * (1 + gamma) + beta

class PixelCNNLayer_up(nn.Module):
    def __init__(self, nr_resnet, nr_filters, resnet_nonlinearity):
        super(PixelCNNLayer_up, self).__init__()
        self.nr_resnet = nr_resnet
        self.u_stream = nn.ModuleList([gated_resnet(nr_filters, down_shifted_conv2d,
                                        resnet_nonlinearity, skip_connection=0)
                                            for _ in range(nr_resnet)])
        self.ul_stream = nn.ModuleList([gated_resnet(nr_filters, down_right_shifted_conv2d,
                                        resnet_nonlinearity, skip_connection=1)
                                        for _ in range(nr_resnet)])

    def forward(self, u, ul):
        u_list, ul_list = [], []
        for i in range(self.nr_resnet):
            u = self.u_stream[i](u)
            ul = self.ul_stream[i](ul, a=u)
            u_list += [u]
            ul_list += [ul]
        return u_list, ul_list

class PixelCNNLayer_down(nn.Module):
    def __init__(self, nr_resnet, nr_filters, resnet_nonlinearity):
        # Keep existing initialization code
        super(PixelCNNLayer_down, self).__init__()
        self.nr_resnet = nr_resnet
        self.u_stream = nn.ModuleList([gated_resnet(nr_filters, down_shifted_conv2d,
                                        resnet_nonlinearity, skip_connection=1)
                                            for _ in range(nr_resnet)])
        self.ul_stream = nn.ModuleList([gated_resnet(nr_filters, down_right_shifted_conv2d,
                                        resnet_nonlinearity, skip_connection=2)
                                        for _ in range(nr_resnet)])
        self.res_scales = nn.ParameterList([nn.Parameter(torch.ones(1) * 0.1) for _ in range(nr_resnet)])

    def forward(self, u, ul, u_list, ul_list):
        for i in range(self.nr_resnet):
            u_skip = u_list.pop()
            u = self.u_stream[i](u, a=u_skip)
            u = u + self.res_scales[i] * u_skip
            
            ul_skip = ul_list.pop()
            
            # Check and resize u to match ul_skip's spatial dimensions if needed
            if u.size(2) != ul_skip.size(2) or u.size(3) != ul_skip.size(3):
                u_resized = F.interpolate(u, size=(ul_skip.size(2), ul_skip.size(3)), 
                                          mode='bilinear', align_corners=False)
            else:
                u_resized = u
                
            ul = self.ul_stream[i](ul, a=torch.cat((u_resized, ul_skip), 1))
            ul = ul + self.res_scales[i] * ul_skip
            
        return u, ul

class PixelCNN(nn.Module):
    def __init__(self, nr_resnet=5, nr_filters=80, nr_logistic_mix=10,
                 resnet_nonlinearity='concat_elu', input_channels=3, num_classes=4):
        super(PixelCNN, self).__init__()
        if resnet_nonlinearity == 'concat_elu':
            self.resnet_nonlinearity = lambda x: concat_elu(x)
        else:
            raise Exception('Only concat_elu is supported as resnet nonlinearity.')

        self.nr_filters = nr_filters
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.nr_logistic_mix = nr_logistic_mix
        self.right_shift_pad = nn.ZeroPad2d((1, 0, 0, 0))
        self.down_shift_pad = nn.ZeroPad2d((0, 0, 1, 0))

        down_nr_resnet = [nr_resnet] + [nr_resnet + 2] * 2
        self.down_layers = nn.ModuleList([PixelCNNLayer_down(down_nr_resnet[i], nr_filters,
                                            self.resnet_nonlinearity) for i in range(3)])
        self.up_layers = nn.ModuleList([PixelCNNLayer_up(nr_resnet, nr_filters,
                                            self.resnet_nonlinearity) for _ in range(3)])
        self.downsize_u_stream = nn.ModuleList([down_shifted_conv2d(nr_filters, nr_filters,
                                                stride=(2,2)) for _ in range(2)])
        self.downsize_ul_stream = nn.ModuleList([down_right_shifted_conv2d(nr_filters,
                                                nr_filters, stride=(2,2)) for _ in range(2)])
        self.upsize_u_stream = nn.ModuleList([down_shifted_deconv2d(nr_filters, nr_filters,
                                                stride=(2,2)) for _ in range(2)])
        self.upsize_ul_stream = nn.ModuleList([down_right_shifted_deconv2d(nr_filters,
                                                nr_filters, stride=(2,2)) for _ in range(2)])

        self.embedding_dim = 256
        self.class_embedding = nn.Embedding(num_classes, self.embedding_dim)
        # More sophisticated class embedding projection
        self.condition_projection = nn.Sequential(
            nn.Linear(self.embedding_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),  # Slightly increased dropout
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, nr_filters)
        )
        # Improved normalization layer
        self.cond_norm = ConditionalNorm(nr_filters, self.embedding_dim)
        
        # Improved attention module
        self.attention = SelfAttention(nr_filters)
        
        # Initial convolutions remain the same
        self.input_norm = nn.LayerNorm([input_channels, 32, 32])
        self.u_init = down_shifted_conv2d(input_channels, nr_filters, filter_size=(2,3),
                                         shift_output_down=True)
        self.ul_init = nn.ModuleList([
            down_shifted_conv2d(input_channels, nr_filters, filter_size=(1,3),
                               shift_output_down=True),
            down_right_shifted_conv2d(input_channels, nr_filters, filter_size=(2,1),
                                    shift_output_right=True)
        ])

        # Output layer
        num_mix = 3 if self.input_channels == 1 else 10
        self.nin_out = nin(nr_filters, num_mix * nr_logistic_mix)
        
        # Add a final refinement layer
        self.refine = nn.Sequential(
            nn.Conv2d(nr_filters, nr_filters, kernel_size=1),
            nn.ELU(),
            nn.Conv2d(nr_filters, nr_filters, kernel_size=1)
        )
        
        self.init_padding = None

    def forward(self, x, labels, sample=False):
        labels = labels.to(x.device)
        label_embeddings = self.class_embedding(labels)
        channel_cond = self.condition_projection(label_embeddings)

        # Fixed: Removed sample padding to avoid adding extra channel
        if self.init_padding is not sample:
            xs = [int(y) for y in x.size()]
            self.init_padding = None  # No padding added

        x = self.input_norm(x)
        u_list = [self.u_init(x)]
        ul_list = [self.ul_init[0](x) + self.ul_init[1](x)]
        for i in range(3):
            u_out, ul_out = self.up_layers[i](u_list[-1], ul_list[-1])
            u_list += u_out
            ul_list += ul_out
            if i != 2:
                u_list += [self.downsize_u_stream[i](u_list[-1])]
                ul_list += [self.downsize_ul_stream[i](ul_list[-1])]
            u_list[-1] = self.cond_norm(u_list[-1], label_embeddings)
            ul_list[-1] = self.cond_norm(ul_list[-1], label_embeddings)
            if i == 1:
                u_list[-1] = self.attention(u_list[-1])
                ul_list[-1] = self.attention(ul_list[-1])

        u = u_list.pop()
        ul = ul_list.pop()
        for i in range(3):
            u, ul = self.down_layers[i](u, ul, u_list, ul_list)
            if i != 2:
                u = self.upsize_u_stream[i](u)
                ul = self.upsize_ul_stream[i](ul)
            u = self.cond_norm(u, label_embeddings)
            ul = self.cond_norm(ul, label_embeddings)

        ul = ul + channel_cond.unsqueeze(2).unsqueeze(3)
        ul = self.refine(ul) + ul  # Residual refinement

        x_out = self.nin_out(F.elu(ul))

        assert len(u_list) == len(ul_list) == 0, pdb.set_trace()

        return x_out

    def infer_img(self, x, device):
        B, _, _, _ = x.size()
        inferred_loss = torch.zeros((self.num_classes, B)).to(device)

        # Class-specific temperature scaling for more accurate inference
        temp_scale = 0.8 + 0.4 * torch.sigmoid(torch.arange(self.num_classes).float() / self.num_classes).to(device)

        for i in range(self.num_classes):
            inferred_label = (torch.ones(B, dtype=torch.int64) * i).to(device)
            model_output = self(x, inferred_label)
            
            # Apply temperature scaling for more confident predictions
            class_loss = discretized_mix_logistic_loss(x, model_output, True)
            inferred_loss[i] = class_loss / temp_scale[i]
            
        losses, labels = torch.min(inferred_loss, dim=0)
        return losses, labels, inferred_loss

class random_classifier(nn.Module):
    def __init__(self, NUM_CLASSES):
        super(random_classifier, self).__init__()
        self.NUM_CLASSES = NUM_CLASSES
        self.fc = nn.Linear(3, NUM_CLASSES)
        print("Random classifier initialized")
        if os.path.join(os.path.dirname(__file__), 'models') not in os.listdir():
            os.mkdir(os.path.join(os.path.dirname(__file__), 'models'))
        torch.save(self.state_dict(), os.path.join(os.path.dirname(__file__), 'models/conditional_pixelcnn.pth'))

    def forward(self, x, device):
        return torch.randint(0, self.NUM_CLASSES, (x.shape[0],)).to(device)