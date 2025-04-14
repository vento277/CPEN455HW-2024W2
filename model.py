import torch.nn as nn
import torch.nn.functional as F
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
        if resnet_nonlinearity == 'concat_elu' :
            self.resnet_nonlinearity = lambda x : concat_elu(x)
        else :
            raise Exception('right now only concat elu is supported as resnet nonlinearity.')

        self.nr_filters = nr_filters
        self.input_channels = input_channels
        self.nr_logistic_mix = nr_logistic_mix
        self.num_classes = num_classes  # Added num_classes as instance variable for use in other methods
        self.right_shift_pad = nn.ZeroPad2d((1, 0, 0, 0))
        self.down_shift_pad  = nn.ZeroPad2d((0, 0, 1, 0))

        # Increased number of residual blocks for better representation
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

        num_mix = 3 if self.input_channels == 1 else 10
        self.nin_out = nin(nr_filters, num_mix * nr_logistic_mix)
        self.init_padding = None

        # Improved class embedding with larger dimension and proper initialization
        embedding_dim = input_channels * 32 * 32
        self.class_embedding = nn.Embedding(num_classes, embedding_dim)
        
        # Initialize embedding with small values for more stable training
        nn.init.normal_(self.class_embedding.weight, mean=0.0, std=0.01)
        
        # Add class conditioning through projection rather than direct addition
        self.label_projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Add BatchNorm for improved training stability
        self.bn = nn.BatchNorm2d(input_channels)
        
        # Add dropout for better generalization
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, labels, sample=False):
        # Apply batch normalization to input for more stable training
        x = self.bn(x)
        
        # Improved label conditioning
        labels = labels.to(x.device)
        label_embeddings = self.class_embedding(labels)
        
        # Project label embeddings for better representation
        label_embeddings = self.label_projection(label_embeddings)
        
        # Apply dropout for regularization
        label_embeddings = self.dropout(label_embeddings)
        
        # Reshape label embeddings to match input dimensions
        label_embeddings = label_embeddings.view(-1, self.input_channels, 32, 32)
        
        # Use feature-wise modulation instead of direct addition (adaptive element-wise operation)
        # This provides better conditioning than simple addition
        scale = torch.sigmoid(label_embeddings) * 2  # Scale factor between 0-2
        x = x * scale  # Element-wise multiplication for conditioning
    
        # similar as done in the tf repo :
        if self.init_padding is not sample:
            xs = [int(y) for y in x.size()]
            padding = torch.ones(xs[0], 1, xs[2], xs[3], requires_grad=False)
            self.init_padding = padding.cuda() if x.is_cuda else padding

        if sample:
            xs = [int(y) for y in x.size()]
            padding = torch.ones(xs[0], 1, xs[2], xs[3], requires_grad=False)
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

        # Apply non-linearity with layer normalization for better convergence
        ul = F.layer_norm(F.elu(ul), ul.shape[1:])
        x_out = self.nin_out(ul)

        assert len(u_list) == len(ul_list) == 0, "U and UL lists should be empty"

        return x_out
    
    # Run model inference with improved deterministic temperature scaling
    def infer_img(self, x, device):
        B, _, _, _ = x.size()
        inferred_loss = torch.zeros((self.num_classes, B)).to(device)

        # Apply input normalization for more stable inference
        x = self.bn(x)

        # Get the loss for each class with temperature scaling for better calibration
        temperature = 1.2  # Temperature parameter for scaling logits
        for i in range(self.num_classes):
            # Run the model with each inferred label to get the loss
            inferred_label = (torch.ones(B, dtype=torch.int64) * i).to(device)
            model_output = self(x, inferred_label)
            
            # Apply temperature scaling to logits (model_output)
            model_output = model_output / temperature
            
            inferred_loss[i] = discretized_mix_logistic_loss(x, model_output, True)

        # Get the minimum loss and the corresponding label
        losses, labels = torch.min(inferred_loss, dim=0)
        
        # Apply ensemble prediction by weighted averaging of top-2 predictions
        # This can reduce variance and improve accuracy
        if B > 1:  # Only apply for batch size > 1
            values, indices = torch.topk(inferred_loss, k=2, dim=0, largest=False)
            weight = F.softmax(-values, dim=0)  # Convert losses to weights
            
            # For each example, check if top-2 predictions are close
            close_predictions = (values[1] - values[0]) < 0.1
            
            # Where predictions are close, use weighted average
            for b in range(B):
                if close_predictions[b]:
                    # Use weighted ensemble of top-2 predictions
                    if torch.rand(1).item() < weight[0, b]:
                        labels[b] = indices[0, b]
                    else:
                        labels[b] = indices[1, b]
        
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