import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# same feature in different locations should get different results.
def get_mesh(batch_size, shape_x, shape_y):
	mg_x, mg_y = np.meshgrid(np.linspace(0, 1, shape_y), np.linspace(0, 1, shape_x))
	mg_x = np.tile(mg_x[None, None, :, :], [batch_size, 1, 1, 1]).astype('float32')
	mg_y = np.tile(mg_y[None, None, :, :], [batch_size, 1, 1, 1]).astype('float32')
	mesh = torch.cat([torch.tensor(mg_x).to(device), torch.tensor(mg_y).to(device)], 1)
	return mesh

def criterion(prediction, mask, regr, size_average=True):
    # 8 outputs, 1 for binary mask, 7 for regression variables, x, y, z, yaw, pitch_sin, pitch_cos, roll
    # Binary mask loss
    pred_mask = torch.sigmoid(prediction[:, 0])
    
    mask_loss = mask * torch.log(pred_mask + 1e-12) + (1 - mask) * torch.log(1 - pred_mask + 1e-12)
    mask_loss = -mask_loss.mean(0).sum()

    # Regression L1 loss
    pred_regr = prediction[:, 1:]
    regr_loss = (torch.abs(pred_regr - regr).sum(1) * mask).sum(1).sum(1) / mask.sum(1).sum(1)
    regr_loss = regr_loss.mean(0)

    # Sum
    loss = mask_loss + regr_loss
    if not size_average:
        loss *= prediction.shape[0]
    return loss