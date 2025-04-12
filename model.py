import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *

# Add Self-Attention Layer for long-range dependencies
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, H, W = x.size()
        query = self.query(x).view(batch_size, -1, H * W).permute(0, 2, 1)  # [B, HW, C//8]
        key = self.key(x).view(batch_size, -1, H * W)  # [B, C//8, HW]
        attention = self.softmax(torch.bmm(query, key))  # [B, HW, HW]
        value = self.value(x).view(batch_size, -1, H * W)  # [B, C, HW]
        out = torch.bmm(value, attention.permute(0, 2, 1)).view(batch_size, C, H, W)
        return self.gamma * out + x

# Conditional Normalization for better class conditioning
class ConditionalNorm(nn.Module):
    def __init__(self, num_features, embedding_dim):
        super(ConditionalNorm, self).__init__()
        self.norm = nn.LayerNorm(num_features, elementwise_affine=False)
        self.gamma = nn.Linear(embedding_dim, num_features)
        self.beta = nn.Linear(embedding_dim, num_features)

    def forward(self, x, embedding):
        # x: [B, C, H, W], embedding: [B, embedding_dim]
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x = self.norm(x)
        gamma = self.gamma(embedding).view(-1, 1, 1, x.size(-1))  # [B, 1, 1, C]
        beta = self.beta(embedding).view(-1, 1, 1, x.size(-1))  # [B, 1, 1, C]
        x = x * (1 + gamma) + beta
        return x.permute(0, 3, 1, 2)  # [B, C, H, W]

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
        super(PixelCNNLayer_down, self).__init__()
        self.nr_resnet = nr_resnet
        self.u_stream = nn.ModuleList([gated_resnet(nr_filters, down_shifted_conv2d,
                                        resnet_nonlinearity, skip_connection=1)
                                            for _ in range(nr_resnet)])
        self.ul_stream = nn.ModuleList([gated_resnet(nr_filters, down_right_shifted_conv2d,
                                        resnet_nonlinearity, skip_connection=2)
                                            for _ in range(nr_resnet)])

    def forward(self, u, ul, u_list, ul_list):
        for i in range(self.nr_resnet):
            u = self.u_stream[i](u, a=u_list.pop())
            ul = self.ul_stream[i](ul, a=torch.cat((u, ul_list.pop()), 1))
        return u, ul

class PixelCNN(nn.Module):
    def __init__(self, nr_resnet=5, nr_filters=80, nr_logistic_mix=10,
                 resnet_nonlinearity='concat_elu', input_channels=3, num_classes=4):
        super(PixelCNN, self).__init__()
        if resnet_nonlinearity == 'concat_elu':
            self.resnet_nonlinearity = lambda x: concat_elu(x)
        else:
            raise Exception('right now only concat elu is supported as resnet nonlinearity.')

        self.nr_filters = nr_filters
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.nr_logistic_mix = nr_logistic_mix
        self.right_shift_pad = nn.ZeroPad2d((1, 0, 0, 0))
        self.down_shift_pad  = nn.ZeroPad2d((0, 0, 1, 0))

        down_nr_resnet = [self.nr_resnet] + [self.nr_resnet + 1] * 2
        self.down_layers = nn.ModuleList([PixelCNNLayer_down(down_nr_resnet[i], self.nr_filters,
                                                self.resnet_nonlinearity) for i in range(3)])
        self.up_layers = nn.ModuleList([PixelCNNLayer_up(self.nr_resnet, self.nr_filters,
                                                self.resnet_nonlinearity) for _ in range(3)])

        self.downsize_u_stream = nn.ModuleList([down_shifted_conv2d(self.nr_filters, self.nr_filters,
                                                    stride=(2,2)) for _ in range(2)])
        self.downsize_ul_stream = nn.ModuleList([down_right_shifted_conv2d(self.nr_filters,
                                                    self.nr_filters, stride=(2,2)) for _ in range(2)])
        self.upsize_u_stream = nn.ModuleList([down_shifted_deconv2d(self.nr_filters, self.nr_filters,
                                                    stride=(2,2)) for _ in range(2)])
        self.upsize_ul_stream = nn.ModuleList([down_right_shifted_deconv2d(self.nr_filters,
                                                    self.nr_filters, stride=(2,2)) for _ in range(2)])

        # Enhanced embedding dimension
        self.embedding_dim = 256  # Increased from 128
        self.class_embedding = nn.Embedding(num_classes, self.embedding_dim)

        # Improved conditioning network with dropout
        self.condition_projection = nn.Sequential(
            nn.Linear(self.embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, self.nr_filters)
        )

        # Add conditional normalization
        self.cond_norm = ConditionalNorm(self.nr_filters, self.embedding_dim)

        # Add self-attention layer
        self.attention = SelfAttention(self.nr_filters)

        self.u_init = down_shifted_conv2d(input_channels + 1, self.nr_filters, filter_size=(2,3),
                                          shift_output_down=True)
        self.ul_init = nn.ModuleList([down_shifted_conv2d(input_channels + 1, self.nr_filters,
                                            filter_size=(1,3), shift_output_down=True),
                                      down_right_shifted_conv2d(input_channels + 1, self.nr_filters,
                                            filter_size=(2,1), shift_output_right=True)])

        num_mix = 3 if self.input_channels == 1 else 10
        self.nin_out = nin(nr_filters, num_mix * nr_logistic_mix)
        self.init_padding = None

        # Enhanced spatial conditioning
        self.spatial_mapper = nn.Sequential(
            nn.Linear(self.embedding_dim, 64*64),  # Increased resolution
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64*64, 32*32)
        )

    def forward(self, x, labels, sample=False):
        labels = labels.to(x.device)
        batch_size = x.size(0)

        # Get class embeddings
        label_embeddings = self.class_embedding(labels)
        channel_cond = self.condition_projection(label_embeddings)
        
        # Create spatial conditioning map
        spatial_cond = self.spatial_mapper(label_embeddings)
        spatial_cond = spatial_cond.view(batch_size, 1, 32, 32)
        x = x + spatial_cond.expand_as(x)

        # Initialize padding
        if self.init_padding is None or not sample:
            xs = [int(y) for y in x.size()]
            padding = torch.ones(xs[0], 1, xs[2], xs[3], device=x.device, requires_grad=False)
            self.init_padding = padding

        if sample:
            xs = [int(y) for y in x.size()]
            padding = torch.ones(xs[0], 1, xs[2], xs[3], device=x.device, requires_grad=False)
            x = torch.cat((x, padding), 1)
        else:
            x = torch.cat((x, self.init_padding), 1)

        ### UP PASS ###
        u_list = [self.u_init(x)]
        ul_list = [self.ul_init[0](x) + self.ul_init[1](x)]
        for i in range(3):
            u_out, ul_out = self.up_layers[i](u_list[-1], ul_list[-1])
            u_list += u_out
            ul_list += ul_out
            if i != 2:
                u_list += [self.downsize_u_stream[i](u_list[-1])]
                ul_list += [self.downsize_ul_stream[i](ul_list[-1])]
            # Apply conditional normalization and attention
            u_list[-1] = self.cond_norm(u_list[-1], label_embeddings)
            ul_list[-1] = self.cond_norm(ul_list[-1], label_embeddings)
            if i == 1:  # Apply attention at middle scale
                u_list[-1] = self.attention(u_list[-1])
                ul_list[-1] = self.attention(ul_list[-1])

        ### DOWN PASS ###
        u = u_list.pop()
        ul = ul_list.pop()
        for i in range(3):
            u, ul = self.down_layers[i](u, ul, u_list, ul_list)
            if i != 2:
                u = self.upsize_u_stream[i](u)
                ul = self.upsize_ul_stream[i](ul)
            # Apply conditional normalization
            u = self.cond_norm(u, label_embeddings)
            ul = self.cond_norm(ul, label_embeddings)

        # Apply final conditioning
        ul = ul + channel_cond.unsqueeze(2).unsqueeze(3)
        x_out = self.nin_out(F.elu(ul))

        assert len(u_list) == len(ul_list) == 0, "Lists not empty"
        return x_out

    def infer_img(self, x, device):
        B, _, _, _ = x.size()
        inferred_loss = torch.zeros((self.num_classes, B)).to(device)

        # Get the loss for each class
        for i in range(self.num_classes):
            inferred_label = (torch.ones(B, dtype=torch.int64) * i).to(device)
            model_output = self(x, inferred_label)
            inferred_loss[i] = discretized_mix_logistic_loss(x, model_output, True)

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