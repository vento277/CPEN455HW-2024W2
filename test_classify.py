'''
This code is used to evaluate the classification accuracy of the trained model.
You should at least guarantee this code can run without any error on validation set.
And whether this code can run is the most important factor for grading.
We provide the remaining code, all you should do are, and you can't modify the remaining code:
1. Replace the random classifier with your trained model.(line 64-68)
2. modify the get_label function to get the predicted label.(line 18-24)(just like Leetcode solutions)
'''
from torchvision import datasets, transforms
from utils import *
from model import * 
from dataset import *
from tqdm import tqdm
from pprint import pprint
import argparse
NUM_CLASSES = len(my_bidict)
import csv
import os


def get_label_logits(model, model_input, device):
    answer, logits = model.classify(model_input, len(my_bidict), logits=True)
    return answer, logits

def classify_and_submit(model, data_loader, device):
    rows = []
    path = 'data/test'
    full_answers = []
    full_logits = torch.Tensor().to(device)
    
    model.eval()
    for batch_idx, item in enumerate(tqdm(data_loader)):
        with torch.no_grad():
            # model works in batches of 32
            model_input, categories = item
            model_input = model_input.to(device)
            answer, logits = get_label_logits(model, model_input, device)

            full_logits = torch.cat((full_logits, logits), 0)

            answer = [pred.item() for pred in answer]
            answer = [int(i) for i in answer]
            full_answers.extend(answer)

    torch.save(full_logits, 'test_logits.pt')

    # Learned from https://www.geeksforgeeks.org/writing-csv-files-in-python/
    fields = ['id', 'label']
    fid = ['fid', '24.227917819822295']

    filename = "submission.csv"

    with open(filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)

        csvwriter.writerow(fields)

        for _, _, filenames in os.walk(path, topdown=True):
            # https://stackoverflow.com/questions/33159106/sort-filenames-in-directory-in-ascending-order
            filenames.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
            for i in range(0, len(filenames)):
                csvwriter.writerow([filenames[i], str(full_answers[i])])

        csvwriter.writerow(fid)
        

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
                                             shuffle=False, 
                                             **kwargs)

    model = PixelCNN(nr_resnet=2, nr_filters=40, 
            input_channels=3, nr_logistic_mix=5)
    
    model = model.to(device)
    #Attention: the path of the model is fixed to 'models/conditional_pixelcnn.pth'
    #You should save your model to this path
    model.load_state_dict(torch.load('models/conditional_pixelcnn.pth', map_location=device)) # MODIFICATION TO RUN ON CPU IF TRAINED ON GPU
    model.eval()
    print('model parameters loaded')
    classify_and_submit(model = model, data_loader = dataloader, device = device)
        
        