'''
Create all the files needed for submission on Hugging Face

'''
from torchvision import datasets, transforms
from utils import *
from model import * 
from dataset import *
from tqdm import tqdm
import csv
from pprint import pprint
import argparse
NUM_CLASSES = len(my_bidict)

# Write your code here
# And get the predicted label, which is a tensor of shape (batch_size,)
#TODO: Begin of your code
# classification function to convert the output of conditional PixelCNN++ to the prediction labels when given a new image
def classify(model, model_input, device):
    batch_size = model_input.shape[0]

    # replicate input for number of classes
    model_input = model_input.repeat(NUM_CLASSES,1,1,1)

    # lookup tensor of potential class labels that can be guessed for each image in the batch
    # class label is repeated to match a batch of data; for parallel processing
    # eg.) when batch_size=3, num_classes=4
    # shape: ([0, 0, 0, 1, 1, 1, 2, 2, 2])
    batched_labels = torch.arange(NUM_CLASSES).repeat_interleave(batch_size)
    
    # generate output for each class label
    model_out = model(model_input, batched_labels)

    # choice of loss function was given from piazza/ TA office hours
    logits = discretized_mix_logistic_loss(model_input, model_out, sum_over_batch=False).view(NUM_CLASSES, batch_size).permute(1, 0)
    
    # minimize logistic loss
    losses, pred_labels = torch.min(logits, dim=1)

    return logits, losses, pred_labels

def get_label(model, model_input, device):
    # changed to predicted
    logits, losses, pred_labels = classify(model, model_input, device)
    return pred_labels
# End of your code

def classifier(model, data_loader,dataset,device):
    model.eval()
    logits_all = []
    answers_all = []
    with torch.no_grad():
        for batch_idx, item in enumerate(tqdm(data_loader)):
            model_input, categories = item
            model_input = model_input.to(device)
            _, answer, logits = get_label(model, model_input, device)
            logits_all.append(logits.T.cpu())
            answers_all.append(answer.cpu())

    # ChatGPT prompt: "How to save python list to csv"
    save_logits = np.concatenate(logits_all, axis=0)
    save_answers = np.concatenate(answers_all, axis=0)
    np.save('test_logits.npy', np.concatenate(save_logits, axis=0))
    print("Shape of saved logits: " + str(save_logits.shape))
    with open("submission.csv", mode='w', newline='') as file:
        writer = csv.writer (file)
        writer.writerow(['id', 'label'])
        for image_path, answer in zip(dataset.samples, save_answers):
            img_name = os.path.basename(image_path[0])
            writer.writerow([img_name, answer])
        writer.writerow(['fid', 455])
        print("Done writing submission.csv")

            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-i', '--data_dir', type=str,
                        default='data', help='Location for the dataset')
    parser.add_argument('-b', '--batch_size', type=int,
                        default=32, help='Batch size for inference')
    parser.add_argument('-m', '--mode', type=str,
                        default='test', help='Mode for the dataset')
    parser.add_argument('-q', '--nr_resnet', type=int, default=5,
                        help='Number of residual blocks per stage of the model')
    parser.add_argument('-n', '--nr_filters', type=int, default=160,
                        help='Number of filters to use across the model. Higher = larger model.')
    parser.add_argument('-o', '--nr_logistic_mix', type=int, default=10,
                        help='Number of logistic components in the mixture. Higher = more flexible model')
    parser.add_argument('-u', '--model_name', type=str,
                        default='conditional_pixelcnn', help='Location for the dataset')
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
    dataset = CPEN455Dataset(root_dir=args.data_dir, mode=args.mode, transform=ds_transforms)

    #Write your code here
    #You should replace the random classifier with your trained model
    #Begin of your code
    model = PixelCNN(nr_resnet=1, nr_filters=40, input_channels=3, nr_logistic_mix=5, num_classes=NUM_CLASSES)
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
    
    acc = classifier(model = model, data_loader = dataloader, dataset = dataset, device = device)

    print('model test set stats saved')
        
        