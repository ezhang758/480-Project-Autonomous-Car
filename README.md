# CPSC 480 Final Project: Autonomous Car Orientation Detection

### Setup Environment

Set up the environment with
```
conda env create -f car.yml
```

Activate environment with 
```
conda activate car
```

### Run
Run `setup.sh` to download the data and run and evaluate the saved model checkpoint. By default, `setup.sh` loads the pretrained CenterNet model with the VGGNet-11 backbone and evaluates the result.


##### Train
If desired, `train.py` can be run to train the CenterNet model.
The parameters are as follows:
- `path (str, optional)`: Path to directory containing the dataset. Defaults to `./data/dataset/`
- `model (str, optional)`: Name of backbone model. Defaults to VGGNet-11
- `n_epochs (int, optional)`: Number of epochs. Defaults to 5
- `learning_rate (int, optional)`: Learning rate. Defaults to 0.001
- `batch_size (int, optional)`: Batch size. Defaults to 4

##### Evaluate
If desired, `evaluate.py` can be run to evaluate the CenterNet model.
The parameters are as follows:
- `data_path (str, optional)`: Path to directory containing the dataset. Defaults to `./data/dataset/`
- `checkpoint_path (str, optional)`: Path to directory containing the model checkpoints. Defaults to `./data/checkpoints/`
- `model (str, optional)`: Name of backbone model. Defaults to VGGNet-11
- `batch_size (int, optional)`: Batch size. Defaults to 4
- `device (str, optional)`: Device to use. Defaults to cpu
- `calculate_map, (bool, optional)`: Whether or not to calculate mean average precision. Defaults to False
- `filename (str, optional)`: File to save predictions to. Defaults to `predictions.csv`

