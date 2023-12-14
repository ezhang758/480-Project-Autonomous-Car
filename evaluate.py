from argparse import ArgumentParser
from models.unet import UNet
import torch
from utils.utils import criterion

def evaluate_model(dev_loader, n_epochs):
    model.eval()
    
    for epoch in range(n_epochs):
        loss = 0
        
        with torch.no_grad():
            for img_batch, mask_batch, regr_batch in dev_loader:
                img_batch = img_batch.to(device)
                mask_batch = mask_batch.to(device)
                regr_batch = regr_batch.to(device)
                
                output = model(img_batch)
                
                loss += criterion(output, mask_batch, regr_batch, size_average=False).data
                
        loss /= len(dev_loader.dataset)
        print('Dev loss: {:.4f}'.format(loss))
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default="VGGNet-11")
    parser.add_argument('-n', '--n_epochs', type=int, default=5)
    args = parser.parse_args()
    
    # choose device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    n_epochs = args.n_epochs
    backbone_model = args.models
    n_classes = 8
    
    # Load model
    model = UNet(backbone_model, n_classes).to(device)
    checkpoint = torch.load(f'./UNet_{backbone_model}.pth', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    
    evaluate_model(devloader, 2)