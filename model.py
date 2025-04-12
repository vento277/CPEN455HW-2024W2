import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import os
from layers import *


class PixelCNNLayer_up(nn.Module):
    def __init__(self, nr_resnet, nr_filters, resnet_nonlinearity):
        super(PixelCNNLayer_up, self).__init__()
        self.nr_resnet = nr_resnet
        # stream from pixels above
        self.u_stream = nn.ModuleList([gated_resnet(nr_filters, down_shifted_conv2d,
                                        resnet_nonlinearity, skip_connection=0)
                                            for _ in range(nr_resnet)])

        # stream from pixels above and to thes left
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

        # stream from pixels above and to thes left
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
                    resnet_nonlinearity='concat_elu', input_channels=3, num_classes=4):
        super(PixelCNN, self).__init__()
        if resnet_nonlinearity == 'concat_elu' :
            self.resnet_nonlinearity = lambda x : concat_elu(x)
        else :
            raise Exception('right now only concat elu is supported as resnet nonlinearity.')

        self.nr_filters = nr_filters
        self.input_channels = input_channels
        self.nr_logistic_mix = nr_logistic_mix
        self.right_shift_pad = nn.ZeroPad2d((1, 0, 0, 0))
        self.down_shift_pad  = nn.ZeroPad2d((0, 0, 1, 0))
        # Added to track number of classes for conditioning
        self.num_classes = num_classes

        down_nr_resnet = [nr_resnet] + [nr_resnet + 1] * 2
        self.down_layers = nn.ModuleList([PixelCNNLayer_down(down_nr_resnet[i], nr_filters,
                                                self.resnet_nonlinearity) for i in range(3)])

        self.up_layers   = nn.ModuleList([PixelCNNLayer_up(nr_resnet, nr_filters,
                                                self.resnet_nonlinearity) for _ in range(3)])

        self.downsize_u_stream  = nn.ModuleList([down_shifted_conv2d(nr_filters, nr_filters,
                                                    stride=(2,2)) for _ in range(2)])

        self.downsize_ul_stream = nn.ModuleList([down_right_shifted_conv2d(nr_filters,
                                                    nr_filters, stride=(2,2)) for _ in range(2)])

        self.upsize_u_stream  = nn.ModuleList([down_shifted_deconv2d(nr_filters, nr_filters,
                                                    stride=(2,2)) for _ in range(2)])

        self.upsize_ul_stream = nn.ModuleList([down_right_shifted_deconv2d(nr_filters,
                                                    nr_filters, stride=(2,2)) for _ in range(2)])

        # Modified to account for conditional input (input_channels + 1) where +1 is for the padding
        self.u_init = down_shifted_conv2d(input_channels + 1, nr_filters, filter_size=(2,3),
                        shift_output_down=True)

        self.ul_init = nn.ModuleList([down_shifted_conv2d(input_channels + 1, nr_filters,
                                            filter_size=(1,3), shift_output_down=True),
                                       down_right_shifted_conv2d(input_channels + 1, nr_filters,
                                            filter_size=(2,1), shift_output_right=True)])

        num_mix = 3 if self.input_channels == 1 else 10
        self.nin_out = nin(nr_filters, num_mix * nr_logistic_mix)
        self.init_padding = None

        # Modified: Class conditioning approach - using embeddings
        # Using embedding size that matches input dimensions for easy addition
        embedding_dim = input_channels * 32 * 32  # Assuming 32x32 images
        self.class_embedding = nn.Embedding(num_classes, embedding_dim)
        
        # Added: Condition network to process label embeddings
        self.condition_network = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.Linear(512, input_channels * 32 * 32)
        )


    def forward(self, x, labels, sample=False):
        # Modified: Enhanced conditional mechanism
        # Generate label embeddings
        labels = labels.to(x.device)
        batch_size = x.shape[0]
        
        # Get embeddings for each class and process them
        label_embeddings = self.class_embedding(labels)
        # Process embeddings through condition network
        condition_features = self.condition_network(label_embeddings)
        # Reshape to match input dimensions
        condition_features = condition_features.view(batch_size, self.input_channels, 32, 32)
        
        # Apply conditioning by adding to input (feature-wise conditioning)
        x = x + condition_features
    
        # similar as done in the tf repo :
        if self.init_padding is not sample:
            xs = [int(y) for y in x.size()]
            padding = Variable(torch.ones(xs[0], 1, xs[2], xs[3]), requires_grad=False)
            self.init_padding = padding.cuda() if x.is_cuda else padding

        if sample:
            xs = [int(y) for y in x.size()]
            padding = Variable(torch.ones(xs[0], 1, xs[2], xs[3]), requires_grad=False)
            padding = padding.cuda() if x.is_cuda else padding
            x = torch.cat((x, padding), 1)

        ###      UP PASS    ###
        x = x if sample else torch.cat((x, self.init_padding), 1)
        u_list  = [self.u_init(x)]
        ul_list = [self.ul_init[0](x) + self.ul_init[1](x)]
        for i in range(3):
            # resnet block
            u_out, ul_out = self.up_layers[i](u_list[-1], ul_list[-1])
            u_list  += u_out
            ul_list += ul_out

            if i != 2:
                # downscale (only twice)
                u_list  += [self.downsize_u_stream[i](u_list[-1])]
                ul_list += [self.downsize_ul_stream[i](ul_list[-1])]

        ###    DOWN PASS    ###
        u  = u_list.pop()
        ul = ul_list.pop()

        for i in range(3):
            # resnet block
            u, ul = self.down_layers[i](u, ul, u_list, ul_list)

            # upscale (only twice)
            if i != 2:
                u  = self.upsize_u_stream[i](u)
                ul = self.upsize_ul_stream[i](ul)

        x_out = self.nin_out(F.elu(ul))

        assert len(u_list) == len(ul_list) == 0, pdb.set_trace()

        return x_out
    
    # Modified: Enhanced inference method for classification
    def infer_img(self, x, device):
        """
        Perform inference to find the most likely class for an image
        by computing the loss under each possible class condition.
        """
        B, _, _, _ = x.size()
        inferred_loss = torch.zeros((self.num_classes, B)).to(device)

        # Get the loss for each class
        for i in range(self.num_classes):
            # Run the model with each inferred label to get the loss
            inferred_label = (torch.ones(B, dtype=torch.int64) * i).to(device)
            model_output = self(x, inferred_label)
            inferred_loss[i] = discretized_mix_logistic_loss(x, model_output, reduction=True)

        # Get the minimum loss and the corresponding label
        losses, labels = torch.min(inferred_loss, dim=0)
        return losses, labels, inferred_loss
    
    # Added: Sampling method for conditional generation
    def sample(self, condition_labels, device, img_size=(3, 32, 32)):
        """
        Sample images from the model conditioned on class labels
        """
        # Start with random noise or zeros
        batch_size = condition_labels.shape[0]
        x = torch.zeros(batch_size, *img_size).to(device)
        
        # Iteratively sample each pixel (this is slow but conceptually simple)
        for h in range(img_size[1]):
            for w in range(img_size[2]):
                for c in range(img_size[0]):
                    # Get the model's prediction
                    out = self(x, condition_labels, sample=True)
                    # Sample from the predicted distribution for this pixel
                    x_sample = sample_from_discretized_mix_logistic(out, self.nr_logistic_mix)
                    # Update just this pixel
                    x[:, c, h, w] = x_sample[:, c, h, w]
        
        return x
    

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