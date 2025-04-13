import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pdb
import os

# Assuming layers.py provides these functions
from layers import gated_resnet, down_shifted_conv2d, down_right_shifted_conv2d, down_shifted_deconv2d, down_right_shifted_deconv2d, nin, concat_elu, discretized_mix_logistic_loss

class ConditionalEmbedding(nn.Module):
    """
    Module that creates embeddings for label conditioning in PixelCNN.
    """
    def __init__(self, num_classes, embedding_dim):
        super(ConditionalEmbedding, self).__init__()
        # Simple embedding layer to convert class labels to feature vectors
        self.embedding = nn.Embedding(num_classes, embedding_dim)
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
    def forward(self, labels):
        # Get embeddings and project them for better representation
        embeddings = self.embedding(labels)
        return self.projection(embeddings)

class PixelCNNLayer_up(nn.Module):
    def __init__(self, nr_resnet, nr_filters, resnet_nonlinearity):
        super(PixelCNNLayer_up, self).__init__()
        self.nr_resnet = nr_resnet
        # stream from pixels above
        self.u_stream = nn.ModuleList([gated_resnet(nr_filters, down_shifted_conv2d,
                                        resnet_nonlinearity, skip_connection=0)
                                            for _ in range(nr_resnet)])

        # stream from pixels above and to the left
        self.ul_stream = nn.ModuleList([gated_resnet(nr_filters, down_right_shifted_conv2d,
                                        resnet_nonlinearity, skip_connection=1)
                                            for _ in range(nr_resnet)])

    def forward(self, u, ul):
        u_list, ul_list = [], []

        for i in range(self.nr_resnet):
            u  = self.u_stream[i](u)
            ul = self.ul_stream[i](ul, a=u)
            u_list  += [u]
            ul_list += [ul]

        return u_list, ul_list


class PixelCNNLayer_down(nn.Module):
    def __init__(self, nr_resnet, nr_filters, resnet_nonlinearity):
        super(PixelCNNLayer_down, self).__init__()
        self.nr_resnet = nr_resnet
        # stream from pixels above
        self.u_stream  = nn.ModuleList([gated_resnet(nr_filters, down_shifted_conv2d,
                                        resnet_nonlinearity, skip_connection=1)
                                            for _ in range(nr_resnet)])

        # stream from pixels above and to the left
        self.ul_stream = nn.ModuleList([gated_resnet(nr_filters, down_right_shifted_conv2d,
                                        resnet_nonlinearity, skip_connection=2)
                                            for _ in range(nr_resnet)])

    def forward(self, u, ul, u_list, ul_list):
        for i in range(self.nr_resnet):
            u  = self.u_stream[i](u, a=u_list.pop())
            ul = self.ul_stream[i](ul, a=torch.cat((u, ul_list.pop()), 1))

        return u, ul


class PixelCNN(nn.Module):
    def __init__(self, nr_resnet=5, nr_filters=80, nr_logistic_mix=10,
                    resnet_nonlinearity='concat_elu', input_channels=3, num_classes=10):
        super(PixelCNN, self).__init__()
        if resnet_nonlinearity == 'concat_elu':
            self.resnet_nonlinearity = lambda x : concat_elu(x)
        else:
            raise Exception('right now only concat elu is supported as resnet nonlinearity.')

        # Store model parameters
        self.nr_filters = nr_filters
        self.input_channels = input_channels
        self.nr_logistic_mix = nr_logistic_mix
        self.num_classes = num_classes
        self.right_shift_pad = nn.ZeroPad2d((1, 0, 0, 0))
        self.down_shift_pad = nn.ZeroPad2d((0, 0, 1, 0))

        # Define conditional embedding with appropriate size for projection
        self.embedding_dim = 128  # Dimension for label embeddings
        self.conditional_embedding = ConditionalEmbedding(num_classes, self.embedding_dim)
        
        # Create condition projector for feature map modulation
        self.cond_to_spatial = nn.Sequential(
            nn.Linear(self.embedding_dim, nr_filters),
            nn.ReLU()
        )
        
        # Define number of residual blocks in each layer
        down_nr_resnet = [nr_resnet] + [nr_resnet + 1] * 2
        
        # Create conditional PixelCNN layers
        self.down_layers = nn.ModuleList([PixelCNNLayer_down(down_nr_resnet[i], nr_filters,
                                                self.resnet_nonlinearity) for i in range(3)])

        self.up_layers = nn.ModuleList([PixelCNNLayer_up(nr_resnet, nr_filters,
                                                self.resnet_nonlinearity) for _ in range(3)])

        # Downsampling and upsampling layers
        self.downsize_u_stream  = nn.ModuleList([down_shifted_conv2d(nr_filters, nr_filters,
                                                    stride=(2,2)) for _ in range(2)])

        self.downsize_ul_stream = nn.ModuleList([down_right_shifted_conv2d(nr_filters,
                                                    nr_filters, stride=(2,2)) for _ in range(2)])

        self.upsize_u_stream  = nn.ModuleList([down_shifted_deconv2d(nr_filters, nr_filters,
                                                    stride=(2,2)) for _ in range(2)])

        self.upsize_ul_stream = nn.ModuleList([down_right_shifted_deconv2d(nr_filters,
                                                    nr_filters, stride=(2,2)) for _ in range(2)])

        # Initial convolution layers
        # Note: We keep the +1 dimension for consistency with original code as it adds a dummy channel
        self.u_init = down_shifted_conv2d(input_channels + 1, nr_filters, filter_size=(2,3),
                        shift_output_down=True)

        self.ul_init = nn.ModuleList([down_shifted_conv2d(input_channels + 1, nr_filters,
                                            filter_size=(1,3), shift_output_down=True),
                                       down_right_shifted_conv2d(input_channels + 1, nr_filters,
                                            filter_size=(2,1), shift_output_right=True)])
                                            
        # Output layer
        num_mix = 3 if self.input_channels == 1 else 10
        self.nin_out = nin(nr_filters, num_mix * nr_logistic_mix)
        self.init_padding = None

    def forward(self, x, label, sample=False):
        # Process conditioning information
        label_embedding = self.conditional_embedding(label)
        h_condition = self.cond_to_spatial(label_embedding).unsqueeze(2).unsqueeze(3)  # Convert to spatial dimension [B, C, 1, 1]
        
        # Similar to the original implementation for padding
        if self.init_padding is not sample:
            xs = [int(y) for y in x.size()]
            padding = Variable(torch.ones(xs[0], 1, xs[2], xs[3]), requires_grad=False)
            self.init_padding = padding.to(x.device)

        if sample:
            xs = [int(y) for y in x.size()]
            padding = Variable(torch.ones(xs[0], 1, xs[2], xs[3]), requires_grad=False)
            padding = padding.to(x.device)
            x = torch.cat((x, padding), 1)

        ###      UP PASS    ###
        x = x if sample else torch.cat((x, self.init_padding), 1)
        u_list = [self.u_init(x)]
        ul_list = [self.ul_init[0](x) + self.ul_init[1](x)]
        
        for i in range(3):
            # Apply conditional information at each layer of the up pass
            u_list[-1] = u_list[-1] + h_condition
            ul_list[-1] = ul_list[-1] + h_condition
            
            # Resnet block
            u_out, ul_out = self.up_layers[i](u_list[-1], ul_list[-1])
            u_list += u_out
            ul_list += ul_out

            if i != 2:
                # Downscale (only twice)
                u_list += [self.downsize_u_stream[i](u_list[-1])]
                ul_list += [self.downsize_ul_stream[i](ul_list[-1])]
                
                # Apply conditioning at each scale
                u_list[-1] = u_list[-1] + h_condition
                ul_list[-1] = ul_list[-1] + h_condition

        ###    DOWN PASS    ###
        u = u_list.pop()
        ul = ul_list.pop()

        for i in range(3):
            # Apply conditional information at each layer of the down pass
            u = u + h_condition
            ul = ul + h_condition
            
            # Resnet block
            u, ul = self.down_layers[i](u, ul, u_list, ul_list)

            # Upscale (only twice)
            if i != 2:
                u = self.upsize_u_stream[i](u)
                ul = self.upsize_ul_stream[i](ul)
                
                # Apply conditioning at each scale
                u = u + h_condition
                ul = ul + h_condition

        # Final layer with additional conditioning
        ul = ul + h_condition  # Apply conditioning before final output
        x_out = self.nin_out(F.elu(ul))

        assert len(u_list) == len(ul_list) == 0, pdb.set_trace()

        return x_out
        
    def infer_img(self, x, device):
        """
        Infer the class of an image by computing the likelihood under each class
        """
        B, _, _, _ = x.size()
        inferred_loss = torch.zeros((self.num_classes, B)).to(device)
        
        # Compute the likelihood of the image under each class
        for i in range(self.num_classes):
            # Create labels for this class
            inferred_label = (torch.ones(B, dtype=torch.int64) * i).to(device)
            # Get model output for this class
            model_output = self(x, inferred_label)
            # Compute discretized logistic loss (negative log-likelihood)
            class_loss = discretized_mix_logistic_loss(x, model_output, True)
            inferred_loss[i] = class_loss
            
        # Return the class with minimum loss (maximum likelihood)
        losses, labels = torch.min(inferred_loss, dim=0)
        return losses, labels, inferred_loss


class random_classifier(nn.Module):
    def __init__(self, NUM_CLASSES):
        super(random_classifier, self).__init__()
        self.NUM_CLASSES = NUM_CLASSES
        self.fc = nn.Linear(3, NUM_CLASSES)
        print("Random classifier initialized")
        # create a folder
        if 'models' not in os.listdir(os.path.dirname(__file__)):
            os.mkdir(os.path.join(os.path.dirname(__file__), 'models'))
        torch.save(self.state_dict(), os.path.join(os.path.dirname(__file__), 'models/conditional_pixelcnn.pth'))
        
    def forward(self, x, device):
        return torch.randint(0, self.NUM_CLASSES, (x.shape[0],)).to(device)