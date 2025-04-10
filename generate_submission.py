# file generates the csv file needed for hugging face with the prediction of the model

from torchvision import transforms
from utils import *
from model import * 
from dataset import *
from tqdm import tqdm
from pprint import pprint
import argparse
import csv
NUM_CLASSES = len(my_bidict)

# And get the predicted label, which is a tensor of shape (batch_size,)
def get_label_and_logits(model, model_input, device):
    _, answer, logits = model.classify_image(model_input, device)
    return answer, logits

def write_to_csv(model, data_loader, device, dataset):
    model.eval()
    all_answer = []
    test_logits = np.array([])
    for _, item in enumerate(tqdm(data_loader)):
        model_input, _ = item
        model_input = model_input.to(device)
        answer, logits = get_label_and_logits(model, model_input, device)
        _, B = logits.shape
        test_logits = np.append(test_logits, logits.view(NUM_CLASSES, B).permute(1, 0).detach().cpu().numpy())
        all_answer.append(answer)
    
    all_answer = torch.cat(all_answer, -1)
    test_logits = test_logits.reshape(-1, NUM_CLASSES)

    np.save("test_logits.npy", test_logits)

    # used chat gpt where the prompt was: How to write to csv in pytorch with a tensor
    with open("final_result.csv", mode='w', newline='') as file:
        writer = csv.writer (file)
        writer.writerow(['id', 'label'])

        # write each image label prediction
        for img_path, label in zip(dataset.samples, all_answer):
            img_name = os.path.basename(img_path[0])
            writer.writerow([img_name, label.item()])

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
    dataset = CPEN455Dataset(root_dir=args.data_dir, mode=args.mode, transform=ds_transforms)
    dataloader = torch.utils.data.DataLoader(dataset, 
                                             batch_size=args.batch_size, 
                                             shuffle=False, 
                                             **kwargs)

    model = PixelCNN(nr_resnet=1, nr_filters=40, input_channels=3, nr_logistic_mix=5, num_classes=NUM_CLASSES)
    
    model = model.to(device)
    model.load_state_dict(torch.load('models/conditional_pixelcnn.pth', map_location=device))
    model.eval()
    print('model parameters loaded')
    write_to_csv(model = model, data_loader = dataloader, device = device, dataset=dataset)
