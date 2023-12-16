import numpy as np
import pandas as pd
import pickle
import torch
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
import cv2
from argparse import ArgumentParser
from torch.utils.data import Dataset

IMG_WIDTH = 1024
IMG_HEIGHT = IMG_WIDTH // 16 * 5
MODEL_SCALE = 8
BATCH_SIZE = 4
IMG_SHAPE = (2710, 3384, 3)

# processing model output
DISTANCE_THRESH_CLEAR = 2

camera_matrix = np.array([[2304.5479, 0,  1686.2379],
                          [0, 2305.8757, 1354.9849],
                          [0, 0, 1]], dtype=np.float32)

camera_matrix_inverse = np.linalg.inv(camera_matrix)

class CarDataset(Dataset):
    """Car dataset."""

    def __init__(self, dataframe, root_dir, training=True, transform=None):
        self.df = dataframe
        self.root_dir = root_dir
        self.transform = transform
        self.training = training
    
    def imread(self, img_path):
        img = cv2.imread(img_path)
        if img is not None and len(img.shape) == 3:
            img = np.array(img[:, :, ::-1])
        return img
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get image name
        idx, labels = self.df.values[idx]
        img_name = self.root_dir.format(idx)

        # Augmentation
        flip = False
        if self.training:
            flip = np.random.randint(10) == 1

        # Read image
        img0 = self.imread(img_name)
        img = preprocess_image(img0, flip=flip)
        img = np.rollaxis(img, 2, 0)

        # Get mask and regression maps
        mask, regr = get_mask_and_regr(img0, labels, flip=flip)
        regr = np.rollaxis(regr, 2, 0)

        return [img, mask, regr]
    


def rotate(alpha, angle):
    alpha = alpha + angle
    alpha = alpha - (alpha + np.pi) * 2 * np.pi // (2 * np.pi) 
    return alpha

def str2coords(string, names=['id', 'yaw', 'pitch', 'roll', 'x', 'y', 'z']):
    '''
    Input:
        string: PredictionString (e.g. from train dataframe)
        names: array of what to extract from the string
    Output:
        list of dicts with keys from `names`
    '''
    coords = []
    for var in np.array(string.split()).reshape([-1, 7]):
        coords.append(dict(zip(names, var.astype('float'))))
        if 'id' in coords[-1]:
            coords[-1]['id'] = int(coords[-1]['id'])
    return coords

def create_points_df(data):
    points_df = pd.DataFrame()
    for col in ['x', 'y', 'z', 'yaw', 'pitch', 'roll']:
        arr = []
        for ps in data['PredictionString']:
            coords = str2coords(ps)
            arr += [c[col] for c in coords]
        points_df[col] = arr
    return points_df

def generate_xz_y_lin(points_df, path):
    xz_y_lin = LinearRegression()
    X = points_df[['x', 'z']].to_numpy()
    y = points_df['y'].to_numpy()
    xz_y_lin.fit(X, y)
    pickle.dump(xz_y_lin, open(path + "xzy_slope.sav", 'wb'))


def generate_img_coords(string, camera_matrix=camera_matrix):
    '''
    Input: 
        string - string, PredictionString from training data csv
        camera_matrix - np.array, camera matrix to perform point cloud transformation
    Return: 
        img_x, img_y - Pixel coordinates of input vehicle string
    '''
    coords_3d_dict = str2coords(string)
    x_list = [coords_3d['x'] for coords_3d in coords_3d_dict]
    y_list = [coords_3d['y'] for coords_3d in coords_3d_dict]
    z_list = [coords_3d['z'] for coords_3d in coords_3d_dict]
    coords_3d = np.transpose(np.array(list(zip(x_list, y_list, z_list))))
    coords_before_perspective = np.transpose(np.dot(camera_matrix, coords_3d))
    coords_before_perspective[:, 0] /= coords_before_perspective[:, 2]
    coords_before_perspective[:, 1] /= coords_before_perspective[:, 2]
    
    return coords_before_perspective[:, 0], coords_before_perspective[:, 1]


# 2D visualization 
def rotation_matrix(yaw, pitch, roll):
    Yaw_rotate = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                  [0, 1, 0],
                  [-np.sin(yaw), 0, np.cos(yaw)]])
    Pitch_rotate = np.array([[1, 0, 0],
                  [0, np.cos(pitch), -np.sin(pitch)],
                  [0, np.sin(pitch), np.cos(pitch)]])
    Roll_rotate = np.array([[np.cos(roll), -np.sin(roll), 0],
                  [np.sin(roll), np.cos(roll), 0],
                  [0, 0, 1]])
    return np.dot(Yaw_rotate, np.dot(Pitch_rotate, Roll_rotate))

def draw_line(img, points):
    color = (255, 0, 255)
    cv2.line(img, tuple(points[0][:2]), tuple(points[3][:2]), color, 16)
    cv2.line(img, tuple(points[0][:2]), tuple(points[1][:2]), color, 16)
    cv2.line(img, tuple(points[1][:2]), tuple(points[2][:2]), color, 16)
    cv2.line(img, tuple(points[2][:2]), tuple(points[3][:2]), color, 16)
    return img

def draw_dot(img, points):
    for (point_x, point_y, point_z) in points:
        cv2.circle(img, (point_x, point_y), int(1000 / point_z), (0, 255, 0), -1)
    return img

def visualize(img, coords, camera_matrix=camera_matrix):
    x_l = 1.02
    y_l = -0.80
    z_l = 2.31
    
    img = img.copy()
    for point in coords:
        # Get values
        x, y, z = point['x'], point['y'], point['z']
        yaw, pitch, roll = -point['pitch'], -point['yaw'], -point['roll']
        # Math
        Rotation_translation_matrix = np.eye(4)
        Rotation_translation_matrix[:3, 3] = np.array([x, y, z])
        Rotation_translation_matrix[:3, :3] = np.transpose(rotation_matrix(yaw, pitch, roll))
        Rotation_translation_matrix = Rotation_translation_matrix[:3, :] # convert it to 3 x 4 
        Points_space_coord = np.transpose(np.array([[x_l, y_l, -z_l, 1],
                      [x_l, y_l, z_l, 1],
                      [-x_l, y_l, z_l, 1],
                      [-x_l, y_l, -z_l, 1],
                      [0, 0, 0, 1]]))
        Points_img_coord = np.transpose(np.dot(camera_matrix, np.dot(Rotation_translation_matrix, Points_space_coord)))
        # Points_img_coord = Points_img_coord.T
        # perspective devide
        Points_img_coord[:, 0] /= Points_img_coord[:, 2]
        Points_img_coord[:, 1] /= Points_img_coord[:, 2]
        Points_img_coord = Points_img_coord.astype(int)
        # Drawing
        img = draw_line(img, Points_img_coord)
        img = draw_dot(img, Points_img_coord[-1:])
    
    return img

# Image preprocessing

def regr_preprocess(regr_dict, flip=False):
    if flip:
        for k in ['x', 'pitch', 'roll']:
            regr_dict[k] = -regr_dict[k]
    for name in ['x', 'y', 'z']:
        regr_dict[name] = regr_dict[name] / 100
    regr_dict['roll'] = rotate(regr_dict['roll'], np.pi)
    regr_dict['pitch_sin'] = np.sin(regr_dict['pitch'])
    regr_dict['pitch_cos'] = np.cos(regr_dict['pitch'])
    regr_dict.pop('pitch')
    regr_dict.pop('id')
    return regr_dict

def regr_back(regr_dict):
    for name in ['x', 'y', 'z']:
        regr_dict[name] = regr_dict[name] * 100
    regr_dict['roll'] = rotate(regr_dict['roll'], -np.pi)
    
    pitch_sin = regr_dict['pitch_sin'] / np.sqrt(regr_dict['pitch_sin']**2 + regr_dict['pitch_cos']**2)
    pitch_cos = regr_dict['pitch_cos'] / np.sqrt(regr_dict['pitch_sin']**2 + regr_dict['pitch_cos']**2)
    regr_dict['pitch'] = np.arccos(pitch_cos) * np.sign(pitch_sin)
    return regr_dict

def preprocess_image(img, flip=False):
    img = img[img.shape[0] // 2:]
    bg = np.ones_like(img) * img.mean(1, keepdims=True).astype(img.dtype)
    bg = bg[:, :img.shape[1] // 6]
    img = np.concatenate([bg, img, bg], 1)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    if flip:
        img = img[:,::-1]
    return (img / 255).astype('float32')

def get_mask_and_regr(img, labels, flip=False):
    mask = np.zeros([IMG_HEIGHT // MODEL_SCALE, IMG_WIDTH // MODEL_SCALE], dtype='float32')
    regr = np.zeros([IMG_HEIGHT // MODEL_SCALE, IMG_WIDTH // MODEL_SCALE, 7], dtype='float32')
    coords = str2coords(labels)
    xs, ys = generate_img_coords(labels)
    for x, y, regr_dict in zip(xs, ys, coords):
        x, y = y, x
        x = (x - img.shape[0] // 2) * IMG_HEIGHT / (img.shape[0] // 2) / MODEL_SCALE
        x = np.round(x).astype('int')
        y = (y + img.shape[1] // 6) * IMG_WIDTH / (img.shape[1] * 4/3) / MODEL_SCALE
        y = np.round(y).astype('int')
        if x >= 0 and x < IMG_HEIGHT // MODEL_SCALE and y >= 0 and y < IMG_WIDTH // MODEL_SCALE:
            mask[x, y] = 1
            regr_dict = regr_preprocess(regr_dict, flip)
            regr[x, y] = [regr_dict[n] for n in sorted(regr_dict)]
    if flip:
        mask = np.array(mask[:,::-1])
        regr = np.array(regr[:,::-1])
    return mask, regr

def convert_3d_to_2d(x, y, z, fx = 2304.5479, fy = 2305.8757, cx = 1686.2379, cy = 1354.9849):
    # stolen from https://www.kaggle.com/theshockwaverider/eda-visualization-baseline
    return x * fx / z + cx, y * fy / z + cy

def optimize_xy(r, c, x0, y0, z0, xzy_slope, flipped=False):
    def distance_fn(xyz):
        x, y, z = xyz
        xx = -x if flipped else x
        slope_err = (xzy_slope.predict([[xx,z]])[0] - y)**2
        x, y = convert_3d_to_2d(x, y, z)
        y, x = x, y
        x = (x - IMG_SHAPE[0] // 2) * IMG_HEIGHT / (IMG_SHAPE[0] // 2) / MODEL_SCALE
        y = (y + IMG_SHAPE[1] // 6) * IMG_WIDTH / (IMG_SHAPE[1] * 4 / 3) / MODEL_SCALE
        return max(0.2, (x-r)**2 + (y-c)**2) + max(0.4, slope_err)
    
    res = minimize(distance_fn, [x0, y0, z0], method='Powell')
    x_new, y_new, z_new = res.x
    return x_new, y_new, z_new

def clear_duplicates(coords):
    for c1 in coords:
        xyz1 = np.array([c1['x'], c1['y'], c1['z']])
        for c2 in coords:
            xyz2 = np.array([c2['x'], c2['y'], c2['z']])
            distance = np.sqrt(((xyz1 - xyz2)**2).sum())
            if distance < DISTANCE_THRESH_CLEAR:
                if c1['confidence'] < c2['confidence']:
                    c1['confidence'] = -1
    return [c for c in coords if c['confidence'] > 0]

def extract_coords(prediction, flipped=False):
    logits = prediction[0]
    regr_output = prediction[1:]
    points = np.argwhere(logits > 0)
    col_names = sorted(['x', 'y', 'z', 'yaw', 'pitch_sin', 'pitch_cos', 'roll'])
    coords = []
    for r, c in points:
        regr_dict = dict(zip(col_names, regr_output[:, r, c]))
        coords.append(regr_back(regr_dict))
        coords[-1]['confidence'] = 1 / (1 + np.exp(-logits[r, c]))
        coords[-1]['x'], coords[-1]['y'], coords[-1]['z'] = \
                optimize_xy(r, c,
                            coords[-1]['x'],
                            coords[-1]['y'],
                            coords[-1]['z'], flipped)
    coords = clear_duplicates(coords)
    return coords

def coords2str(coords, names=['yaw', 'pitch', 'roll', 'x', 'y', 'z', 'confidence']):
    s = []
    for c in coords:
        for n in names:
            s.append(str(c.get(n, 0)))
    return ' '.join(s)

def generate_train_test_data(path, train_test_split_ratio, random_state):
    data = pd.read_csv(path + 'data.csv')
    df_train, df_test = train_test_split(data, test_size=train_test_split_ratio, random_state=random_state)
    df_train.to_csv(path + "train.csv", index=False)
    df_test.to_csv(path + "test.csv", index=False)

    # create and save linear reg model 
    points_df = create_points_df(data)
    generate_xz_y_lin(points_df, path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-path')
    parser.add_argument('-r', '--train_test_split_ratio', type=float, default=0.01)
    parser.add_argument('-ran', '--random_state', type=int, default=42)
    args = parser.parse_args()

    generate_train_test_data(args.path, args.train_test_split_ratio, args.random_state)
    
  