import csv
import os
from torchvision import datasets, transforms
from utils import *
from model import * 
from dataset import *
from tqdm import tqdm
from pprint import pprint
import argparse
import torch

NUM_CLASSES = len(my_bidict)

#TODO: Begin of your code
# classification function to convert the output of conditional PixelCNN++ to the prediction labels when given a new image
# Begin of your code
def get_label(model, model_input, device):
    _, labels, _ = model.infer_img(model_input, device)
    return labels
# End of your code

def classifier(model, data_loader, device):
    model.eval()
    acc_tracker = ratio_tracker()
    # List to store predictions
    all_preds = []
    
    for batch_idx, item in enumerate(tqdm(data_loader)):
        model_input, categories = item
        model_input = model_input.to(device)
        original_label = [my_bidict[item] for item in categories]
        original_label = torch.tensor(original_label, dtype=torch.int64).to(device)
        answer = get_label(model, model_input, device)
        correct_num = torch.sum(answer == original_label)
        acc_tracker.update(correct_num.item(), model_input.shape[0])
        
        # Store predictions
        all_preds.extend(answer.cpu().numpy().tolist())
    
    # Save to CSV
    csv_path = os.path.join(os.path.dirname(__file__), 'classifier_results.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write relative path and predicted label, no header
        for image_path, pred in zip(data_loader.dataset.samples, all_preds):
            # Get path relative to root_dir (e.g., test/0000021.jpg)
            img_path = os.path.relpath(image_path[0], start=data_loader.dataset.root_dir)
            csv_writer.writerow([img_path, pred])
    
    print(f"Results saved to {csv_path}")
    return acc_tracker.get_ratio()

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
                                                            mode=args.mode, 
                                                            transform=ds_transforms), 
                                             batch_size=args.batch_size, 
                                             shuffle=True, 
                                             **kwargs)
    
    dataset = CPEN455Dataset(root_dir=args.data_dir, mode=args.mode, transform=ds_transforms)
    #TODO:Begin of your code
    model = PixelCNN(nr_resnet=1, nr_filters=50, input_channels=3, nr_logistic_mix=5, num_classes=NUM_CLASSES)
    #End of your code

    model = model.to(device)
    #Attention: the path of the model is fixed to './models/conditional_pixelcnn.pth'
    #You should save your model to this path
    model_path = os.path.join(os.path.dirname(__file__), 'models/conditional_pixelcnn.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print('model parameters loaded')
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model.eval()
    
    acc = classifier(model = model, data_loader = dataloader, device = device)
    print(f"Accuracy: {acc}")