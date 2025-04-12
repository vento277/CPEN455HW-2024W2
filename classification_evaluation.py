'''
This code is used to evaluate the classification accuracy of the trained model.
You should at least guarantee this code can run without any error on validation set.
And whether this code can run is the most important factor for grading.
We provide the remaining code, all you should do are, and you can't modify other code:
1. Replace the random classifier with your trained model.(line 69-72)
2. modify the get_label function to get the predicted label.(line 23-29)(just like Leetcode solutions, the args of the function can't be changed)

REQUIREMENTS:
- You should save your model to the path 'models/conditional_pixelcnn.pth'
- You should Print the accuracy of the model on validation set, when we evaluate your code, we will use test set to evaluate the accuracy
'''
from torchvision import datasets, transforms
from utils import *
from model import * 
from dataset import *
from tqdm import tqdm
from pprint import pprint
import argparse
import csv
NUM_CLASSES = len(my_bidict)

#TODO: Begin of your code
# Write your code here
# And get the predicted label, which is a tensor of shape (batch_size,)
# Begin of your code
def get_label(model, model_input, device):
    num_classes = 4   
    batch_size = model_input.size(0)
    log_likelihood = torch.zeros(batch_size, num_classes, device=device)
    
    for c in range(num_classes):
        labels = torch.full((batch_size,), c, dtype=torch.long, device=device)
        model_output = model(model_input, labels)
        nll = discretized_mix_logistic_classify(model_input, model_output) #I modified loss to give output for each picture
        log_likelihood[:, c] = -nll  #negative log likelihood

    # I'm now selecting the class with the highest log likelihood==>(lowest nll)
    _, predicted_labels = log_likelihood.max(1)
    return predicted_labels
# End of your code

def classifier(model, data_loader, device):
    model.eval()
    acc_tracker = ratio_tracker()
    for batch_idx, item in enumerate(tqdm(data_loader)):
        model_input, categories = item
        model_input = model_input.to(device)
        # original_label = [my_bidict[item] for item in categories]
        original_label = [item for item in categories]
        original_label = torch.tensor(original_label, dtype=torch.int64).to(device)
        answer = get_label(model, model_input, device)
        correct_num = torch.sum(answer == original_label)
        acc_tracker.update(correct_num.item(), model_input.shape[0])
    
    return acc_tracker.get_ratio()
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-i', '--data_dir', type=str,
                        default='data', help='Location for the dataset')
    parser.add_argument('-b', '--batch_size', type=int,
                        default=32, help='Batch size for inference')
    parser.add_argument('-m', '--mode', type=str,
                        default='validation', help='Mode for the dataset')
    
    args = parser.parse_args()
    pprint(args.__dict__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {'num_workers':1, 'pin_memory':True, 'drop_last':True}

    ds_transforms = transforms.Compose([transforms.Resize((32, 32)), rescaling])
    dataloader = torch.utils.data.DataLoader(CPEN455Dataset(root_dir=args.data_dir, 
                                                            mode = args.mode, 
                                                            transform=ds_transforms), 
                                             batch_size=args.batch_size, 
                                             shuffle=True, 
                                             **kwargs)

    #Write your code here
    #You should replace the random classifier with your trained model
    #Begin of your code
    model = PixelCNN(nr_resnet=1, nr_filters=60, nr_logistic_mix=5, input_channels=3, num_classes=4)
    model.load_state_dict(torch.load('models/conditional_pixelcnn.pth'))
    #End of your code

    model = model.to(device)
    #Attention: the path of the model is fixed to 'models/conditional_pixelcnn.pth'
    #You should save your model to this path
    model.load_state_dict(torch.load('conditional_pixelcnn.pth'))
    model.eval()
    print('model parameters loaded')
    acc = classifier(model = model, data_loader = dataloader, device = device)
    print(f"Accuracy: {acc}")
        
        