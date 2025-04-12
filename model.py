import torch.nn as nn
from layers import *


# class PixelCNNLayer_up(nn.Module):
#     def __init__(self, nr_resnet, nr_filters, resnet_nonlinearity):
#         super(PixelCNNLayer_up, self).__init__()
#         self.nr_resnet = nr_resnet
#         # stream from pixels above
#         self.u_stream = nn.ModuleList([gated_resnet(nr_filters, down_shifted_conv2d,
#                                         resnet_nonlinearity, skip_connection=0)
#                                             for _ in range(nr_resnet)])

#         # stream from pixels above and to thes left
#         self.ul_stream = nn.ModuleList([gated_resnet(nr_filters, down_right_shifted_conv2d,
#                                         resnet_nonlinearity, skip_connection=1)
#                                             for _ in range(nr_resnet)])

#     def forward(self, u, ul):
#         u_list, ul_list = [], []

#         for i in range(self.nr_resnet):
#             u  = self.u_stream[i](u)
#             ul = self.ul_stream[i](ul, a=u)
#             u_list  += [u]
#             ul_list += [ul]

#         return u_list, ul_list

class PixelCNNLayer_up(nn.Module):
    def __init__(self, nr_resnet, nr_filters, resnet_nonlinearity, diffusion_steps=10, noise_strength=0.1):
        super(PixelCNNLayer_up, self).__init__()
        self.nr_resnet = nr_resnet
        self.diffusion_steps = diffusion_steps
        self.noise_strength = noise_strength

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

            # Modification 1: Injecting noise into the feature maps
            # This is a simplified form of diffusion where we add Gaussian noise
            # to the feature maps during the forward pass. The amount of noise
            # could be scheduled or fixed. Here, it's a fixed strength.
            if self.training:
                noise_u = torch.randn_like(u) * self.noise_strength
                u = u + noise_u
                noise_ul = torch.randn_like(ul) * self.noise_strength
                ul = ul + noise_ul
            # End of Modification 1

        return u_list, ul_list
    
# class PixelCNNLayer_down(nn.Module):
#     def __init__(self, nr_resnet, nr_filters, resnet_nonlinearity):
#         super(PixelCNNLayer_down, self).__init__()
#         self.nr_resnet = nr_resnet
#         # stream from pixels above
#         self.u_stream  = nn.ModuleList([gated_resnet(nr_filters, down_shifted_conv2d,
#                                         resnet_nonlinearity, skip_connection=1)
#                                             for _ in range(nr_resnet)])

#         # stream from pixels above and to thes left
#         self.ul_stream = nn.ModuleList([gated_resnet(nr_filters, down_right_shifted_conv2d,
#                                         resnet_nonlinearity, skip_connection=2)
#                                             for _ in range(nr_resnet)])

#     def forward(self, u, ul, u_list, ul_list):
#         for i in range(self.nr_resnet):
#             u  = self.u_stream[i](u, a=u_list.pop())
#             ul = self.ul_stream[i](ul, a=torch.cat((u, ul_list.pop()), 1))

#         return u, ul

# Modified PixelCNNLayer_down to handle the potentially noisy feature maps
class PixelCNNLayer_down(nn.Module):
    def __init__(self, nr_resnet, nr_filters, resnet_nonlinearity, diffusion_steps=10, noise_strength=0.1):
        super(PixelCNNLayer_down, self).__init__()
        self.nr_resnet = nr_resnet
        self.diffusion_steps = diffusion_steps
        self.noise_strength = noise_strength

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

            # Modification 2: Injecting noise into the feature maps
            # Similar to the up layer, we inject noise here as well.
            if self.training:
                noise_u = torch.randn_like(u) * self.noise_strength
                u = u + noise_u
                noise_ul = torch.randn_like(ul) * self.noise_strength
                ul = ul + noise_ul
            # End of Modification 2

        return u, ul

# class PixelCNN(nn.Module):
#     def __init__(self, nr_resnet=5, nr_filters=80, nr_logistic_mix=10,
#                     resnet_nonlinearity='concat_elu', input_channels=3, num_classes=4):
#         super(PixelCNN, self).__init__()
#         if resnet_nonlinearity == 'concat_elu' :
#             self.resnet_nonlinearity = lambda x : concat_elu(x)
#         else :
#             raise Exception('right now only concat elu is supported as resnet nonlinearity.')

#         self.nr_filters = nr_filters
#         self.input_channels = input_channels
#         self.nr_logistic_mix = nr_logistic_mix
#         self.right_shift_pad = nn.ZeroPad2d((1, 0, 0, 0))
#         self.down_shift_pad  = nn.ZeroPad2d((0, 0, 1, 0))

#         down_nr_resnet = [nr_resnet] + [nr_resnet + 1] * 2
#         self.down_layers = nn.ModuleList([PixelCNNLayer_down(down_nr_resnet[i], nr_filters,
#                                                 self.resnet_nonlinearity) for i in range(3)])

#         self.up_layers   = nn.ModuleList([PixelCNNLayer_up(nr_resnet, nr_filters,
#                                                 self.resnet_nonlinearity) for _ in range(3)])

#         self.downsize_u_stream  = nn.ModuleList([down_shifted_conv2d(nr_filters, nr_filters,
#                                                     stride=(2,2)) for _ in range(2)])

#         self.downsize_ul_stream = nn.ModuleList([down_right_shifted_conv2d(nr_filters,
#                                                     nr_filters, stride=(2,2)) for _ in range(2)])

#         self.upsize_u_stream  = nn.ModuleList([down_shifted_deconv2d(nr_filters, nr_filters,
#                                                     stride=(2,2)) for _ in range(2)])

#         self.upsize_ul_stream = nn.ModuleList([down_right_shifted_deconv2d(nr_filters,
#                                                     nr_filters, stride=(2,2)) for _ in range(2)])

#         self.u_init = down_shifted_conv2d(input_channels + 1, nr_filters, filter_size=(2,3),
#                         shift_output_down=True)

#         self.ul_init = nn.ModuleList([down_shifted_conv2d(input_channels + 1, nr_filters,
#                                             filter_size=(1,3), shift_output_down=True),
#                                        down_right_shifted_conv2d(input_channels + 1, nr_filters,
#                                             filter_size=(2,1), shift_output_right=True)])

#         num_mix = 3 if self.input_channels == 1 else 10
#         self.nin_out = nin(nr_filters, num_mix * nr_logistic_mix)
#         self.init_padding = None

#         # Class embeddings
#         self.class_embedding = nn.Embedding(num_classes, input_channels*32*32)


#     def forward(self, x, labels, sample=False):
#         # label embeddings are created then attached to the input
#         labels = labels.to(x.device)
#         label_embeddings = self.class_embedding(labels)
#         label_embeddings = label_embeddings.view(-1, self.input_channels, 32, 32)
#         # label_embeddings = label_embeddings.expand(-1, -1, x.size(2), x.size(3))
#         x = x + label_embeddings
    
#         # similar as done in the tf repo :
#         if self.init_padding is not sample:
#             xs = [int(y) for y in x.size()]
#             padding = Variable(torch.ones(xs[0], 1, xs[2], xs[3]), requires_grad=False)
#             self.init_padding = padding.cuda() if x.is_cuda else padding

#         if sample :
#             xs = [int(y) for y in x.size()]
#             padding = Variable(torch.ones(xs[0], 1, xs[2], xs[3]), requires_grad=False)
#             padding = padding.cuda() if x.is_cuda else padding
#             x = torch.cat((x, padding), 1)

#         ###      UP PASS    ###
#         x = x if sample else torch.cat((x, self.init_padding), 1)
#         u_list  = [self.u_init(x)]
#         ul_list = [self.ul_init[0](x) + self.ul_init[1](x)]
#         for i in range(3):
#             # resnet block
#             u_out, ul_out = self.up_layers[i](u_list[-1], ul_list[-1])
#             u_list  += u_out
#             ul_list += ul_out

#             if i != 2:
#                 # downscale (only twice)
#                 u_list  += [self.downsize_u_stream[i](u_list[-1])]
#                 ul_list += [self.downsize_ul_stream[i](ul_list[-1])]

#         ###    DOWN PASS    ###
#         u  = u_list.pop()
#         ul = ul_list.pop()

#         for i in range(3):
#             # resnet block
#             u, ul = self.down_layers[i](u, ul, u_list, ul_list)

#             # upscale (only twice)
#             if i != 2 :
#                 u  = self.upsize_u_stream[i](u)
#                 ul = self.upsize_ul_stream[i](ul)

#         x_out = self.nin_out(F.elu(ul))

#         assert len(u_list) == len(ul_list) == 0, pdb.set_trace()

#         return x_out
    
#         # Run model inference
#     def infer_img(self, x, device):
#         B, _, _, _ = x.size()
#         inferred_loss = torch.zeros((self.num_classes, B)).to(device)

#         # Get the loss for each class
#         for i in range(self.num_classes):
#             # Run the model with each inferred label to get the loss
#             inferred_label = (torch.ones(B, dtype=torch.int64) * i).to(device)
#             model_output = self(x, inferred_label)
#             inferred_loss[i] = discretized_mix_logistic_loss(x, model_output, True)

#         # Get the minimum loss and the corresponding label
#         losses, labels = torch.min(inferred_loss, dim=0)
#         return losses, labels, inferred_loss
    

# Modified PixelCNN to use the diffusion-aware layers
class PixelCNN(nn.Module):
    def __init__(self, nr_resnet=5, nr_filters=80, nr_logistic_mix=10,
                    resnet_nonlinearity='concat_elu', input_channels=3, num_classes=4,
                    diffusion_steps=10, noise_strength=0.1):
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
        self.diffusion_steps = diffusion_steps
        self.noise_strength = noise_strength

        down_nr_resnet = [nr_resnet] + [nr_resnet + 1] * 2
        # Modification 3: Using the diffusion-aware down layers
        self.down_layers = nn.ModuleList([PixelCNNLayer_down(down_nr_resnet[i], nr_filters,
                                                self.resnet_nonlinearity, self.diffusion_steps, self.noise_strength) for i in range(3)])

        # Modification 4: Using the diffusion-aware up layers
        self.up_layers   = nn.ModuleList([PixelCNNLayer_up(nr_resnet, nr_filters,
                                                self.resnet_nonlinearity, self.diffusion_steps, self.noise_strength) for _ in range(3)])

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

        # Class embeddings
        self.class_embedding = nn.Embedding(num_classes, input_channels*32*32)


    def forward(self, x, labels, sample=False):
        # label embeddings are created then attached to the input
        labels = labels.to(x.device)
        label_embeddings = self.class_embedding(labels)
        label_embeddings = label_embeddings.view(-1, self.input_channels, 32, 32)
        # label_embeddings = label_embeddings.expand(-1, -1, x.size(2), x.size(3))
        x = x + label_embeddings

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

        ###    DOWN PASS    ###
        u  = u_list.pop()
        ul = ul_list.pop()

        for i in range(3):
            # resnet block
            u, ul = self.down_layers[i](u, ul, u_list, ul_list)

            # upscale (only twice)
            if i != 2 :
                u  = self.upsize_u_stream[i](u)
                ul = self.upsize_ul_stream[i](ul)

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
        return losses, labels, inferred_loss

    
class random_classifier(nn.Module):
    def __init__(self, NUM_CLASSES):
        super(random_classifier, self).__init__()
        self.NUM_CLASSES = NUM_CLASSES
        self.fc = nn.Linear(3, NUM_CLASSES)
        print("Random classifier initialized")
        # create a folder
        if os.path.join(os.path.dirname(__file__), 'models') not in os.listdir():
            os.mkdir(os.path.join(os.path.dirname(__file__), 'models'))
        torch.save(self.state_dict(), os.path.join(os.path.dirname(__file__), 'models/conditional_pixelcnn.pth'))
    def forward(self, x, device):
        return torch.randint(0, self.NUM_CLASSES, (x.shape[0],)).to(device)
    
    