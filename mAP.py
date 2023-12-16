import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# define classes for translation & rotation distance
class Quaternion:
    # https://math.stackexchange.com/questions/2975109/how-to-convert-euler-angles-to-quaternions-and-get-the-same-euler-angles-back-fr
    def __init__(self, yaw, pitch, roll):
        self.qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        self.qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        self.qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        self.qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

    # https://www.mathworks.com/help/aeroblks/quaternionnormalize.html
    def normalized(self):
        denominator = np.sqrt(self.qx**2 + self.qy**2 + self.qz**2 + self.qw**2)
        self.qx = self.qx / denominator
        self.qy = self.qy / denominator
        self.qz = self.qz / denominator
        self.qw = self.qw / denominator
        return self

    # https://www.mathworks.com/help/aeroblks/quaternioninverse.html
    def inversed(self):
        denominator = self.qx**2 + self.qy**2 + self.qz**2 + self.qw**2
        self.qx = -self.qx/denominator
        self.qy = -self.qy/denominator
        self.qz = -self.qz/denominator
        self.qw = self.qw/denominator
        return self

    # https://personal.utdallas.edu/~sxb027100/dock/quaternion.html
    def multiply(self, q2):
        qx_m = self.qw * q2.qx + self.qx * q2.qw + self.qy * q2.qz - self.qz * q2.qy
        qy_m = self.qw * q2.qy - self.qx * q2.qz + self.qy * q2.qw + self.qz * q2.qx
        qz_m = self.qw * q2.qz + self.qx * q2.qy - self.qy * q2.qx + self.qz * q2.qw
        qw_m = self.qw * q2.qw - self.qx * q2.qx - self.qy * q2.qy - self.qz * q2.qz
        return [qx_m, qy_m, qz_m, qw_m]

class Object3D(Quaternion):
    def __init__(self, yaw, pitch, roll, x, y, z, confidence=None):
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll
        self.x = x
        self.y = y
        self.z = z
        self.confidence = confidence
        self.quaternion = Quaternion(self.yaw, self.pitch, self.roll)


    def translation_distance(self, object2):
        dx = self.x - object2.x
        dy = self.y - object2.y
        dz = self.z - object2.z
        return np.sqrt(dx * dx + dy * dy + dz * dz)

    def rotation_distance(self, object2):
        q1 = self.quaternion
        q2 = object2.quaternion
        diff = q1.multiply(q2)
        diff_w = np.clip(diff[3], -1.0, 1.0)
        rot_dist = np.rad2deg(np.arccos(diff_w))

        return rot_dist if rot_dist < 90 else 180-rot_dist

# create a list of true object for every image
def create_objects_image(img_idx, pred_df, names_true, names_pred):
    true_object_list = []
    pred_object_list = []

    coords = str2coords(pred_df["PredictionString"][img_idx], names=names_true)
    for coord in coords:
        object0 = Object3D(coord['yaw_true'], coord['pitch_true'], coord['roll_true'], coord['x_true'], coord['y_true'], coord['z_true'])
        true_object_list.append(object0)

    pred_coords = str2coords(pred_df["NewPredictionString"][img_idx], names=names_pred)
    for pred_coord in pred_coords:
        object1 = Object3D(pred_coord['yaw'], pred_coord['pitch'], pred_coord['roll'], pred_coord['x'], pred_coord['y'], pred_coord['z'], pred_coord['confidence'])
        pred_object_list.append(object1)
    return true_object_list, pred_object_list

def create_sort_df_image(true_object_list, pred_object_list):
    img_df = pd.DataFrame()
    true_idx_list = []
    pred_idx_list = []
    trans_dist_list = []
    rot_dist_list = []
    confidence_list = []
    for pred_idx, pred_object in enumerate(pred_object_list):
        for true_idx, true_object in enumerate(true_object_list):
            trans_dist = true_object.translation_distance(pred_object)
            rot_dist = true_object.rotation_distance(pred_object)
            true_idx_list.append(true_idx)
            pred_idx_list.append(pred_idx)
            trans_dist_list.append(trans_dist)
            rot_dist_list.append(rot_dist)
            confidence_list.append(pred_object.confidence)
    img_df['true_idx'] = true_idx_list
    img_df['pred_idx'] = pred_idx_list
    img_df['trans_dist'] = trans_dist_list
    img_df['rot_dist'] = rot_dist_list
    img_df['confidence'] = confidence_list
    img_df['combine_dist'] = (img_df['trans_dist'] + img_df['rot_dist'])/img_df['confidence']
    img_df = img_df.sort_values(by=['combine_dist', 'confidence'], ascending=[True, False])
    img_df = img_df.drop_duplicates(subset='pred_idx', keep='first')
    img_df = img_df.sort_values(by=['pred_idx'])
    return img_df

# Precision—Precision is the ratio of the number of true positives to the total number of positive predictions.
# For example, if the model detected 100 trees, and 90 were correct, the precision is 90 percent. Recall—Recall
# is the ratio of the number of true positives to the total number of actual (relevant) objects.
# https://towardsdatascience.com/understanding-confusion-matrix-a9ad42dcfd62
# rot_thre_list = [50, 45, 40, 35, 30, 25, 20, 15, 10, 5]
# tran_thre_list = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

def calculate_mean_average_precision_helper(img_df, rot_thre_list, tran_thre_list, true_object_list, pred_object_list):
    total_average_precision = 0
    total_no_of_ground_truth_positives = len(true_object_list)
    total_no_of_pred_truth_positives = len(pred_object_list)

    for rot_thre in rot_thre_list:
        for tran_thre in tran_thre_list:
            img_df["correct"] = np.where((img_df['trans_dist'] < tran_thre) & (img_df['rot_dist'] < rot_thre), True, False)
            img_df['correct_sum'] = img_df['correct'].cumsum()
            img_df['correct_sum'] = img_df['correct_sum'] / (img_df['pred_idx'] + 1) * img_df['correct']
            average_precision = img_df['correct_sum'].sum() / len(true_object_list)
            total_average_precision += average_precision
    return total_average_precision / (len(rot_thre_list) * len(tran_thre_list))

def calculate_map_image(img_idx, pred_df, names_true, names_pred, rot_thre_list, tran_thre_list):
    true_object_list, pred_object_list = create_objects_image(img_idx, pred_df, names_true, names_pred)
    img_df = create_sort_df_image(true_object_list, pred_object_list)
    map_value = calculate_mean_average_precision_helper(img_df, rot_thre_list, tran_thre_list, true_object_list, pred_object_list)
    return map_value

def plot_map_statistis(pred_df):
    print("map std: ", pred_df['map'].std())
    print("map mean: ", pred_df['map'].mean())
    print("map max: ", pred_df['map'].max())
    pred_df['map'].plot.hist(bins=10, alpha=0.5)
    plt.xlabel("Mean Average Precision using ResNet Backbone")
    plt.show()

def str2coords(s, names=['id', 'yaw', 'pitch', 'roll', 'x', 'y', 'z']):
    '''
    Input:
        s: PredictionString (e.g. from train dataframe)
        names: array of what to extract from the string
    Output:
        list of dicts with keys from `names`
    '''
    coords = []
    for l in np.array(s.split()).reshape([-1, 7]):
        coords.append(dict(zip(names, l.astype('float'))))
        if 'id' in coords[-1]:
            coords[-1]['id'] = int(coords[-1]['id'])
    return coords

