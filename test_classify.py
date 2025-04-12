'''
This code is used to generate the final submission csv file and .npy logits output of the test set.
Most of this file iscopied over from classification_evaluation, but since we are not allowed to modify
the classification_evaluation file, we have to create a new file.
'''

from torchvision import transforms
from utils import *
from model import * 
from dataset import *
from tqdm import tqdm
from pprint import pprint
import argparse
import csv
NUM_CLASSES = len(my_bidict)

def get_label_and_losses(model, model_input, device):
    _, labels, losses = model.infer_img(model_input, device)
    return labels, losses

# Modified to include dataset to write to csv and npy
def classifier(model, data_loader, device, ds):
    model.eval()
    preds = []
    logits = []
    for _, item in enumerate(tqdm(data_loader)):
        model_input, _ = item
        model_input = model_input.to(device)
        pred, losses = get_label_and_losses(model, model_input, device)
        preds.append(pred)
        logits = np.append(logits, losses.cpu().detach().numpy())
    preds = torch.cat(preds, -1)

    # Reshape the logits array
    logits = logits.reshape(-1, NUM_CLASSES)
    print(logits.shape)
    print(logits)
    np.save('test_logits.npy', logits)

    # write the final result csv file
    with open("submission.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['id', 'label'])

        for path, label in zip(ds.samples, preds):
            name = os.path.basename(path[0])
            writer.writerow([name, label.item()])

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
    ds = CPEN455Dataset(root_dir=args.data_dir, mode=args.mode, transform=ds_transforms)
    dataloader = torch.utils.data.DataLoader(ds, 
                                             batch_size=args.batch_size, 
                                             shuffle=False, 
                                             **kwargs)

    model = PixelCNN(nr_resnet=1, nr_filters=40, input_channels=3, nr_logistic_mix=5, num_classes=NUM_CLASSES)
    model = model.to(device)
    model.load_state_dict(torch.load('models/conditional_pixelcnn.pth'))
    model.eval()
    print('model parameters loaded')
    classifier(model = model, data_loader = dataloader, device = device, ds=ds)