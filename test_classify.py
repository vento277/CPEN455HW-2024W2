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
def classify(model, model_input, device):
    batch_size = model_input.shape[0]

    # replicate input for number of classes
    model_input = model_input.repeat(NUM_CLASSES,1,1,1)

    # lookup tensor of potential class labels that can be guessed for each image in the batch
    batched_labels = torch.arange(NUM_CLASSES).repeat_interleave(batch_size)
    
    # generate output for each class label
    model_out = model(model_input, batched_labels)

    # choice of loss function was given from piazza/ TA office hours
    logits = discretized_mix_logistic_loss(model_input, model_out, sum_over_batch=False).view(NUM_CLASSES, batch_size).permute(1, 0)
    
    # minimize logistic loss
    losses, pred_labels = torch.min(logits, dim=1)

    return logits, losses, pred_labels

def get_label(model, model_input, device):
    logits, losses, pred_labels = classify(model, model_input, device)
    return pred_labels
# End of your code

def classifier(model, data_loader, device):
    model.eval()
    acc_tracker = ratio_tracker()
    # Lists to store predictions and image IDs
    all_preds = []
    all_image_ids = []
    
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
        # Assuming categories contains image paths like 'test/0000021.jpg'
        all_image_ids.extend(categories)  # Adjust if paths are elsewhere in item
    
    # Save to CSV
    csv_path = os.path.join(os.path.dirname(__file__), 'classifier_results.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write image ID and predicted label, no header
        for image_id, pred in zip(all_image_ids, all_preds):
            csv_writer.writerow([image_id, pred])
    
    print(f"Results saved to {csv_path}")
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
    kwargs = {'num_workers':0, 'pin_memory':True, 'drop_last':False}

    ds_transforms = transforms.Compose([transforms.Resize((32, 32)), rescaling])
    dataloader = torch.utils.data.DataLoader(CPEN455Dataset(root_dir=args.data_dir, 
                                                            mode=args.mode, 
                                                            transform=ds_transforms), 
                                             batch_size=args.batch_size, 
                                             shuffle=True, 
                                             **kwargs)

    #TODO:Begin of your code
    model = PixelCNN(nr_resnet=1, nr_filters=40, input_channels=3, nr_logistic_mix=5, num_classes=NUM_CLASSES)
    #End of your code

    model = model.to(device)
    model_path = os.path.join(os.path.dirname(__file__), 'models/conditional_pixelcnn.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print('model parameters loaded')
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model.eval()
    
    acc = classifier(model=model, data_loader=dataloader, device=device)
    print(f"Accuracy: {acc}")