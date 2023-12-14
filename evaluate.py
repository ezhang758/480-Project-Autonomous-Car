# Load model
model = MyUNet("VGGNet19", 8).to(device)
checkpoint = torch.load('./MyUNet_vggnet19_final_checkpoint.pth', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['state_dict'])

evaluate_model(2)

import matplotlib.pyplot as plt
history['train_loss'].iloc[100:].plot();
plt.xlabel("Epoch")
plt.ylabel("Training Loss") #remember to save the figure

series = history.dropna()['dev_loss']
plt.scatter(series.index, series);