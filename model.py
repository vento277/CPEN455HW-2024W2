import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
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
                    resnet_nonlinearity='concat_elu', input_channels=3, num_classes=4):
        super(PixelCNN, self).__init__()
        if resnet_nonlinearity == 'concat_elu':
            self.resnet_nonlinearity = lambda x : concat_elu(x)
        elif resnet_nonlinearity == 'elu':  # Added support for ELU nonlinearity
            self.resnet_nonlinearity = lambda x : F.elu(x)
        elif resnet_nonlinearity == 'relu':  # Added support for ReLU nonlinearity
            self.resnet_nonlinearity = lambda x : F.relu(x)
        else:
            raise Exception('Currently supported nonlinearities: concat_elu, elu, relu')

        self.num_classes = num_classes
        self.nr_filters = nr_filters
        self.input_channels = input_channels
        self.nr_logistic_mix = nr_logistic_mix
        self.right_shift_pad = nn.ZeroPad2d((1, 0, 0, 0))
        self.down_shift_pad  = nn.ZeroPad2d((0, 0, 1, 0))

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

        self.u_init = down_shifted_conv2d(input_channels + 1, nr_filters, filter_size=(2,3),
                        shift_output_down=True)

        self.ul_init = nn.ModuleList([down_shifted_conv2d(input_channels + 1, nr_filters,
                                            filter_size=(1,3), shift_output_down=True),
                                       down_right_shifted_conv2d(input_channels + 1, nr_filters,
                                            filter_size=(2,1), shift_output_right=True)])

        # Added condition to properly handle grayscale vs RGB images
        self.num_mix = 3 if self.input_channels == 1 else 10
        self.nin_out = nin(nr_filters, self.num_mix * nr_logistic_mix)
        self.init_padding = None

        # Add the embedding layer with a more descriptive name
        self.class_embedding = nn.Embedding(num_classes, input_channels * 32 * 32)
        
        # Added dropout for regularization
        self.dropout = nn.Dropout(0.1)

    # Add the embeddings to the input
    def addPositionalEmbedding(self, x, labels, img_height, img_width):
        """
        Adds class embedding to the input tensor.
        
        Args:
            x: Input image tensor
            labels: Class labels
            img_height: Height of input image
            img_width: Width of input image
            
        Returns:
            Image tensor with added embeddings
        """
        embs = self.class_embedding(labels).view(-1, self.input_channels, img_height, img_width)
        return x + embs

    def forward(self, x, labels, sample=False):
        # Get dimensions from input tensor
        B, C, H, W = x.size()
        labels = labels.to(x.device)
        
        # Apply embedding
        x = self.addPositionalEmbedding(x, labels, H, W)

        # similar as done in the tf repo :
        # FIXED: Corrected the padding initialization to avoid dimension mismatch
        if self.init_padding is None or self.init_padding.size(0) != B:
            # Create padding with correct batch size
            padding = Variable(torch.ones(B, 1, H, W), requires_grad=False)
            padding = padding.cuda() if x.is_cuda else padding
            if not sample:
                self.init_padding = padding

        ###      UP PASS    ###
        # FIXED: Use proper padding tensor with correct dimensions
        if sample:
            # Create new padding for each sampling step
            padding = Variable(torch.ones(B, 1, H, W), requires_grad=False)
            padding = padding.cuda() if x.is_cuda else padding
            x = torch.cat((x, padding), 1)
        else:
            # Use cached padding
            x = torch.cat((x, self.init_padding), 1)
            
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

        # Added dropout before final layer for regularization
        ul = self.dropout(ul)
        x_out = self.nin_out(F.elu(ul))

        # Added assertion to catch errors during development
        assert len(u_list) == len(ul_list) == 0, "Unmatched elements in u_list or ul_list"

        return x_out
    
    # Run model inference with improved documentation and error handling
    def infer_img(self, x, device):
        """
        Perform inference to determine the most likely class for each input image.
        
        Args:
            x: Input image tensor of shape [B, C, H, W]
            device: Device to perform computation on
            
        Returns:
            tuple: (losses, predicted_labels, all_class_losses)
        """
        B, _, H, W = x.size()
        
        # Validate input
        if B <= 0:
            raise ValueError("Batch size must be positive")
        
        try:
            # Run the model once for all classes in parallel (this is the key optimization)
            # Create a batch with B*num_classes samples
            x_expanded = x.repeat(self.num_classes, 1, 1, 1)
            
            # Create labels for all classes for each sample in the batch
            all_labels = torch.arange(self.num_classes, device=device).repeat_interleave(B)
            
            # Run the model once with the expanded batch
            model_output = self(x_expanded, all_labels)
            
            # Calculate losses for all classes in one forward pass
            all_losses = discretized_mix_logistic_loss(x.repeat(self.num_classes, 1, 1, 1), 
                                                    model_output, True)
            
            # Reshape losses to [num_classes, B]
            inferred_loss = all_losses.view(self.num_classes, B)
            
            # Get the minimum loss and the corresponding label
            losses, labels = torch.min(inferred_loss, dim=0)
            return losses, labels, inferred_loss
            
        except RuntimeError as e:
            if 'out of memory' in str(e):
                # Fallback to per-class sequential processing when OOM occurs
                print("Warning: Out of memory. Falling back to sequential processing.")
                losses = torch.zeros(B, device=device)
                labels = torch.zeros(B, dtype=torch.long, device=device)
                all_losses = torch.zeros(self.num_classes, B, device=device)
                
                # Process each class sequentially
                for c in range(self.num_classes):
                    class_labels = torch.full((B,), c, dtype=torch.long, device=device)
                    output = self(x, class_labels)
                    class_loss = discretized_mix_logistic_loss(x, output, True)
                    all_losses[c] = class_loss
                
                # Get minimum loss and corresponding label
                for i in range(B):
                    class_losses = all_losses[:, i]
                    min_loss, min_idx = torch.min(class_losses, dim=0)
                    losses[i] = min_loss
                    labels[i] = min_idx
                
                return losses, labels, all_losses
            else:
                # Re-raise other errors
                raise
    
class random_classifier(nn.Module):
    def __init__(self, NUM_CLASSES):
        super(random_classifier, self).__init__()
        self.NUM_CLASSES = NUM_CLASSES
        self.fc = nn.Linear(3, NUM_CLASSES)
        print("Random classifier initialized")
        
        # Create models directory if it doesn't exist
        if not os.path.exists('models'):
            os.makedirs('models')
            
        torch.save(self.state_dict(), 'models/conditional_pixelcnn.pth')
        
    def forward(self, x, device):
        # Added batch handling to ensure consistent behavior
        batch_size = x.shape[0]
        return torch.randint(0, self.NUM_CLASSES, (batch_size,), device=device)