import tqdm
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from models.unet import UNet
from utils.utils import criterion
from argparse import ArgumentParser
from utils.dataset import CarDataset
from torch.utils.data import DataLoader
import pickle
import pandas as pd

def train_model(model, optimizer, train_loader, n_epochs):
    model.train()
    
    for epoch in range(n_epochs):
        for batch_idx, (img_batch, mask_batch, regr_batch) in enumerate(tqdm(train_loader)):
            img_batch = img_batch.to(device)
            mask_batch = mask_batch.to(device)
            regr_batch = regr_batch.to(device)
            
            optimizer.zero_grad()
            output = model(img_batch)
            loss = criterion(output, mask_batch, regr_batch)
            loss.backward()
            
            optimizer.step()
            exp_lr_scheduler.step()
        
        print('Train Epoch: {} \tLR: {:.6f}\tLoss: {:.6f}'.format(
            epoch,
			optimizer.state_dict()['param_groups'][0]['lr'],
			loss.data))
        
    return model


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-path')
    parser.add_argument('-m', '--model', type=str, default="VGGNet-11")
    parser.add_argument('-n', '--n_epochs', type=int, default=5)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
    parser.add_argument('-b', '--batch_size', type=int, default=4)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    backbone_model = args.model
    n_epochs = args.n_epochs
    n_classes = 8 # 8 classes: x, y, z, yaw, pitch_sin, pitch_cos, roll
    lr = args.learning_rate
    
    train_images_dir = args.path + 'train_images/{}.jpg'
    df_train = pd.read_csv("train.csv")
    train_dataset = CarDataset(df_train, train_images_dir, training=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # Define model, optimizer, and learning rate scheduler
    model = UNet(backbone_model, n_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=max(n_epochs, 10) * len(train_loader) // 3, gamma=0.1)
    
    model = train_model(model, optimizer, train_loader, n_epochs)
    
    # save model
    torch.save({'state_dict' : model.state_dict(),'optimizer_state_dict' : optimizer.state_dict(),}, f'./UNet_{backbone_model}.pth')

