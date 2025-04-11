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
# Begin of your code
def get_label(model, model_input, device):
    batch_size = model_input.shape[0]
    best_loss = [float('inf')]*batch_size
    best_ans = [0]*batch_size
    logits = torch.zeros((4,batch_size), device=device)
    for i in range(4):
        curr_loss = discretized_mix_logistic_loss(model_input, model(model_input,[i]*batch_size),sum_batch=False)
        logits[i] = curr_loss
        for j in range(batch_size):
            if curr_loss[j] < best_loss[j]:
                best_ans[j] = i
                best_loss[j] = curr_loss[j]
    column_sums = torch.sum(logits, dim=0)
    switched_logits = 1-logits/column_sums
    new_column_sums = torch.sum(switched_logits, dim=0)
    normalized_logits = switched_logits/new_column_sums
    return torch.tensor(best_loss, device=device).detach(), torch.tensor(best_ans, device=device).detach(), torch.tensor(normalized_logits, device=device).detach()
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
    model = PixelCNN(nr_resnet=args.nr_resnet, nr_filters=args.nr_filters, 
                input_channels=3, nr_logistic_mix=args.nr_logistic_mix)
    #End of your code
    
    model = model.to(device)
    model_name = 'models/' + args.model_name + '.pth'
    model.load_state_dict(torch.load(model_name,map_location=torch.device(device)))
    #Attention: the path of the model is fixed to 'models/conditional_pixelcnn.pth'
    #You should save your model to this path
    model.load_state_dict(torch.load(model_name,map_location=torch.device(device)))
    model.eval()
    classifier(model = model, data_loader = dataloader, dataset = dataset, device = device)

    print('model test set stats saved')
        
        