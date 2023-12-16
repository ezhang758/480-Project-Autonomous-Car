from argparse import ArgumentParser
from models.unet import UNet
import torch
from utils.utils import criterion
import pandas as pd
from preprocessing import CarDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from preprocessing import convert_3d_to_2d, optimize_xy, clear_duplicates, extract_coords, coords2str
from mAP import calculate_map_image, plot_map_statistis

def evaluate_model_loss(model, test_loader, device='cpu'):
    model.eval()
    
    loss = 0
    
    with torch.no_grad():
        for img_batch, mask_batch, regr_batch in tqdm(test_loader):
            img_batch = img_batch.to(device)
            mask_batch = mask_batch.to(device)
            regr_batch = regr_batch.to(device)
            
            output = model(img_batch)
            
            loss += criterion(output, mask_batch, regr_batch, size_average=False).data
            
    loss /= len(test_loader.dataset)
    print('Test loss: {:.4f}'.format(loss))

def generate_df_map(filename):
    names_true = ['id', 'yaw_true', 'pitch_true', 'roll_true', 'x_true', 'y_true', 'z_true']
    names_pred = ['yaw', 'pitch', 'roll', 'x', 'y', 'z', 'confidence']
    rot_thre_list = [50, 45, 40, 35, 30, 25, 20, 15, 10, 5]
    tran_thre_list = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    pred_df = pd.read_csv(filename)
    pred_df['NewPredictionString'] = pred_df['NewPredictionString'].fillna(" ")
    pred_df['map'] = [calculate_map_image(img_idx, pred_df, names_true, names_pred, rot_thre_list, tran_thre_list) for img_idx in range(len(pred_df))]
    plot_map_statistis(pred_df)
    return pred_df

def evaluate_model_prediction(model, test_loader, device, df_test, filename):
    predictions = []

    model.eval()

    for img, _, _ in tqdm(test_loader):
        with torch.no_grad():
            output = model(img.to(device))
        output = output.data.cpu().numpy()
        for out in output:
            coords = extract_coords(out)
            s = coords2str(coords)
            predictions.append(s)

    df_test["NewPredictionString"] = predictions
    df_test.to_csv(filename,index=False)
    generate_df_map(filename)

    return predictions

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-dp', '--data_path', type=str, default='./data/dataset/')
    parser.add_argument('-cp', '--checkpoints_path', type=str, default='./data/checkpoints/')
    parser.add_argument('-m', '--model', type=str, default="VGGNet-11")
    parser.add_argument('-b', '--batch_size', type=int, default=4)
    parser.add_argument('-d', '--device', type=str, default='cpu')
    parser.add_argument('-ls', '--calculate_map', type=bool, default=False)
    parser.add_argument('-f', '--filename', type=str, default="predictions.csv")
    args = parser.parse_args()
    
    # load data
    test_images_dir = args.data_path + '{}.jpg'
    df_test = pd.read_csv(args.data_path + "test.csv")
    test_dataset = CarDataset(df_test, test_images_dir, training=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # choose device
    device = 'cuda' if args.device == 'gpu' or args.device == 'cuda' else 'cpu'
    device = torch.device(device)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    backbone_model = args.model
    n_classes = 8
    
    # Load model
    model = UNet(backbone_model, n_classes).to(device)
    checkpoint = torch.load(args.checkpoints_path + f'UNet_{backbone_model}.pth', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    
    if (args.calculate_map == True):
        evaluate_model_prediction(model, test_loader, device, df_test, args.filename)
    else:
        evaluate_model_loss(model, test_loader, device)


