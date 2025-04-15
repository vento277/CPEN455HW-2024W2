import torch.nn as nn
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


# Added FiLM conditioning module
class FiLMConditioning(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer for conditional normalization.
    Generates scaling and shifting parameters based on conditioning information.
    """
    def __init__(self, num_classes, num_features):
        super(FiLMConditioning, self).__init__()
        self.num_features = num_features
        
        # Linear layer to generate scaling (gamma) and shifting (beta) parameters
        self.film_generator = nn.Linear(num_classes, num_features * 2)
        
    def forward(self, feature_maps, condition):
        """
        Args:
            feature_maps: Feature maps to be modulated, shape [B, C, H, W]
            condition: One-hot encoded class labels, shape [B, num_classes]
        Returns:
            Modulated feature maps
        """
        batch_size = feature_maps.size(0)
        
        # Generate FiLM parameters (gamma and beta)
        film_params = self.film_generator(condition)
        
        # Split the parameters into gamma and beta
        gamma, beta = torch.split(film_params, self.num_features, dim=1)
        
        # Reshape to apply broadcasting over spatial dimensions
        gamma = gamma.view(batch_size, self.num_features, 1, 1)
        beta = beta.view(batch_size, self.num_features, 1, 1)
        
        # Apply the FiLM transformation: gamma * x + beta
        return gamma * feature_maps + beta


class PixelCNN(nn.Module):
    def __init__(self, nr_resnet=5, nr_filters=80, nr_logistic_mix=10,
                    resnet_nonlinearity='concat_elu', input_channels=3, num_classes=4):
        super(PixelCNN, self).__init__()
        if resnet_nonlinearity == 'concat_elu' :
            self.resnet_nonlinearity = lambda x : concat_elu(x)
        else :
            raise Exception('right now only concat elu is supported as resnet nonlinearity.')

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

        num_mix = 3 if self.input_channels == 1 else 10
        self.nin_out = nin(nr_filters, num_mix * nr_logistic_mix)
        self.init_padding = None

        # Add the embedding layer
        self.embedding = nn.Embedding(num_classes, input_channels * 32 * 32)

        # Added: FiLM layers for the three levels of the network
        self.film_layers = nn.ModuleList([
            FiLMConditioning(num_classes, nr_filters),
            FiLMConditioning(num_classes, nr_filters),
            FiLMConditioning(num_classes, nr_filters)
        ])
        
        # Added: Class embedding for one-hot encoding
        self.class_embedding = nn.Linear(1, num_classes)

    # Add the embeddings to the input
    def addPositionalEmbedding(self, x, labels, img_height, img_width):
        embs = self.embedding(labels).view(-1, self.input_channels, img_height, img_width)
        return x + embs

    def forward(self, x, labels, sample=False):
        _, _, H, W = x.size()
        labels = labels.to(x.device)
        x = self.addPositionalEmbedding(x, labels, H, W)
        
        # Added: Create one-hot encoding for FiLM conditioning
        batch_size = labels.size(0)
        one_hot = torch.zeros(batch_size, self.num_classes, device=labels.device)
        one_hot.scatter_(1, labels.unsqueeze(1), 1)

        # similar as done in the tf repo :
        if self.init_padding is not sample:
            xs = [int(y) for y in x.size()]
            padding = Variable(torch.ones(xs[0], 1, xs[2], xs[3]), requires_grad=False)
            self.init_padding = padding.cuda() if x.is_cuda else padding

        if sample :
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
                
                # Added: Apply FiLM conditioning after downscaling
                u_list[-1] = self.film_layers[i](u_list[-1], one_hot)
                ul_list[-1] = self.film_layers[i](ul_list[-1], one_hot)

        ###    DOWN PASS    ###
        u  = u_list.pop()
        ul = ul_list.pop()
        
        # Added: Apply FiLM conditioning at the bottleneck
        u = self.film_layers[2](u, one_hot)
        ul = self.film_layers[2](ul, one_hot)

        for i in range(3):
            # resnet block
            u, ul = self.down_layers[i](u, ul, u_list, ul_list)

            # upscale (only twice)
            if i != 2 :
                u  = self.upsize_u_stream[i](u)
                ul = self.upsize_ul_stream[i](ul)
                
                # Added: Apply FiLM conditioning after upscaling
                u = self.film_layers[i](u, one_hot)
                ul = self.film_layers[i](ul, one_hot)

        x_out = self.nin_out(F.elu(ul))

        assert len(u_list) == len(ul_list) == 0, pdb.set_trace()

        return x_out
    
    # Run model inference
    def infer_img(self, x, device):
        B, _, _, _ = x.size()
        inferred_loss = torch.zeros((self.num_classes, B)).to(device)

        # Get the loss for each class
        for i in range(self.num_classes):
            # Run the model with each inferred label to get the loss
            inferred_label = (torch.ones(B, dtype=torch.int64) * i).to(device)
            model_output = self(x, inferred_label)
            inferred_loss[i] = discretized_mix_logistic_loss(x, model_output, True)

        # Get the minimum loss and the corresponding label
        losses, labels = torch.min(inferred_loss, dim=0)
        return losses, labels
    
class random_classifier(nn.Module):
    def __init__(self, NUM_CLASSES):
        super(random_classifier, self).__init__()
        self.NUM_CLASSES = NUM_CLASSES
        self.fc = nn.Linear(3, NUM_CLASSES)
        print("Random classifier initialized")
        # create a folder
        if 'models' not in os.listdir():
            os.mkdir('models')
        torch.save(self.state_dict(), 'models/conditional_pixelcnn.pth')
    def forward(self, x, device):
        return torch.randint(0, self.NUM_CLASSES, (x.shape[0],)).to(device)