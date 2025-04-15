'''
This code is used to evaluate the classification accuracy of the trained model.
You should at least guarantee this code can run without any error on validation set.
And whether this code can run is the most important factor for grading.
We provide the remaining code, all you should do are, and you can't modify the remaining code:
1. Replace the random classifier with your trained model.(line 64-68)
2. modify the get_label function to get the predicted label.(line 18-24)(just like Leetcode solutions)
'''
from torchvision import datasets, transforms
import numpy as np
from utils import *
from model import * 
from dataset import *
from tqdm import tqdm
from pprint import pprint
import argparse
import csv
NUM_CLASSES = len(my_bidict)

# Write your code here
# And get the predicted label, which is a tensor of shape (batch_size,)
# Begin of your code
def get_label(model, model_input, device):
    
    B, _, _, _ = model_input.shape

    # Log likelihood tensor move to device 
    loss_from_log_likelihood = torch.zeros((NUM_CLASSES, B))
    loss_from_log_likelihood = loss_from_log_likelihood.to(device)

    for possible_class in my_bidict.values():

        # Predicts based on the current label
        answer = model(model_input, torch.full((B, ), possible_class), device)

        # Here we use discretized_mix_logistic_loss in classification mode because we care about loss not summed over
        # the batch, we want to know which one would minimize loss so we can tell which class is the most likely
        loss_from_log_likelihood[possible_class, :] = discretized_mix_logistic_loss(model_input, answer, training=False)

    # Need to minimize loss along class dimension to get best class for each image in the batch
    return torch.argmin(loss_from_log_likelihood, dim=0), loss_from_log_likelihood
# End of your code

def classify(model, data_loader, device, csv_test_file, csv_output_file_name, fid):

    # Read all img names from ./data/test.csv
    img_names = []
    with open(csv_test_file, mode='r') as file:
        reader = csv.reader(file)
        
        # Iterate over each row in the CSV file
        for row in reader:
            # Should ignore the -1 dummy label and also drop the test/ prefix
            img_name = row[0].split(',')[0]
            img_name = img_name.replace('test/', '')

            img_names.append(img_name)

    logits = []

    model.eval()
    img_idx = 0

    for batch_idx, item in enumerate(tqdm(data_loader)):
        model_input, categories, _ = item
        model_input = model_input.to(device)
        answer, logit = get_label(model, model_input, device)
        logits.append(logit.T.cpu().detach().numpy())
        for label in answer.cpu().detach().numpy():
            img_names[img_idx] = [img_names[img_idx], str(label)]
            img_idx += 1

    # Prepare CSV for submission
    with open(csv_output_file_name, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write header
        writer.writerow(['id' , 'label'])
        
        # Write classes
        for row in img_names:
            writer.writerow(row)

        # Write fid score
        writer.writerow(['fid', fid])
    
    # Prepare npy file for logits
    logits_arr = np.concatenate(logits, axis=0)
    np.save('logits.npy', logits_arr)

    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-i', '--data_dir', type=str,
                        default='data', help='Location for the dataset')
    parser.add_argument('-b', '--batch_size', type=int,
                        default=32, help='Batch size for inference')
    parser.add_argument('-m', '--mode', type=str,
                        default='test', help='Mode for the dataset')
    
    args = parser.parse_args()
    pprint(args.__dict__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {'num_workers':0, 'pin_memory':True, 'drop_last':False}

    ds_transforms = transforms.Compose([transforms.Resize((32, 32)), rescaling])
    dataloader = torch.utils.data.DataLoader(CPEN455Dataset(root_dir=args.data_dir, 
                                                            mode = args.mode, 
                                                            transform=ds_transforms), 
                                             batch_size=args.batch_size, 
                                             shuffle=False, # Do not shuffle so as to be able to know name of image files
                                             **kwargs)

    #Write your code here
    #You should replace the random classifier with your trained model
    #Begin of your code
    model = PixelCNN(nr_resnet=1, nr_filters=40, input_channels=3, nr_logistic_mix=10)
    #End of your code
    
    model = model.to(device)
    #Attention: the path of the model is fixed to 'models/conditional_pixelcnn.pth'
    #You should save your model to this path
    model.load_state_dict(torch.load('models/conditional_pixelcnn.pth'))
    model.eval()
    print('model parameters loaded')
    classify(model = model, data_loader = dataloader, device = device, csv_test_file = './data/test.csv', csv_output_file_name = 'submission.csv', fid = 37.5056)
        
        