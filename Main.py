"""
Author -- JessyFritz
Date -- 13.07.2022
"""

import os
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path
from matplotlib import pyplot as plt
import pickle

# ------------ DEFINE HYPERPARAMETERS ------------
PATH_TRAIN = "images"  # folder containing training images (all pixels known)
PATH_OUTPUT = "output"  # path to write output to
SAVED_MODEL = os.path.join(PATH_OUTPUT, "best_model.pt")  # path to trained model
SPACING = list(range(2, 7))  # spacing range
OFFSET = list(range(0, 9))  # offset range
BATCH_SIZE = 128
IMG_CHANNELS = 3  # number of image channels
IMG_SIZE = 100  # resize image to this size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------ PREPARE DATA ------------
# custom dataset, which masks images
class ImageLoader(torch.utils.data.Dataset):
    def __init__(self):
        path = Path(PATH_TRAIN)
        self.data = sorted(path.glob('*.jpg'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = cv2.imread(str(self.data[idx]))
        # resize image
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        # calculate random spacing & offset
        spacing = (np.random.choice(SPACING), np.random.choice(SPACING))
        offset = (np.random.choice(OFFSET), np.random.choice(OFFSET))
        
        # erase pixel information according to spacing & offset
        image_array = np.array(img)

        if image_array.ndim != 3:
            raise NotImplementedError("image_array is not a 3D array")

        if image_array.shape[2] != 3:
            raise NotImplementedError("size of the 3rd dimension in image_array is not equal to 3")

        try:
            for o in offset:
                int(o)
            for s in spacing:
                int(s)
        except ValueError:
            print("offset/spacing not convertible to int")

        if any(int(o) < 0 or int(o) > 32 for o in offset):
            raise ValueError("offset value out of range")

        if any(int(s) < 2 or int(s) > 8 for s in spacing):
            raise ValueError("spacing value out of range")

        m_space = 0
        for m in range(image_array.shape[0]):
            if m == int(offset[1]) + m_space * spacing[1]: # check if we keep information in this row
                m_space += 1
                n_space = 0
                for n in range(image_array.shape[1]): # set individual positions in row to zero
                    if n == int(offset[0]) + n_space * spacing[0]: # check for offset & spacing in row at index n
                        n_space += 1
                    else:
                        image_array[m][n] = 0
            else: # set whole row to zero
                for n in range(image_array.shape[1]):
                    image_array[m][n] = 0

        # transpose array
        masked_img = np.transpose(image_array, (2, 0, 1))
        
        target = np.transpose(img, (2, 0, 1))  # reshape to have channels at first position
        return masked_img, target


plotpath = os.path.join(PATH_OUTPUT, "plots")
os.makedirs(plotpath, exist_ok=True)
data = ImageLoader()

# split dataset into training and validation set
training = torch.utils.data.Subset(data, indices=np.arange(int(len(data) * (3 / 5))))
validation = torch.utils.data.Subset(data, indices=np.arange(int(len(data) * (3 / 5)), len(data)))

train_loader = torch.utils.data.DataLoader(training, batch_size=32, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(validation, batch_size=1, shuffle=True, num_workers=0)


# ------------ CREATE MODEL ------------
# Convolution Neural Net with 3 channels & hidden layers
class SimpleCNN(torch.nn.Module):
    def __init__(self, n_in_channels: int = 3, n_hidden_layers: int = 4, n_kernels: int = 32, kernel_size: int = 7):
        super().__init__()

        cnn = []
        for i in range(n_hidden_layers):
            cnn.append(torch.nn.Conv2d(
                in_channels=n_in_channels,
                out_channels=n_kernels,
                kernel_size=kernel_size,
                padding=int(kernel_size / 2)
            ))
            cnn.append(torch.nn.ReLU())
            n_in_channels = n_kernels
        self.hidden_layers = torch.nn.Sequential(*cnn)

        self.output_layer = torch.nn.Conv2d(
            in_channels=n_in_channels,
            out_channels=3,
            kernel_size=kernel_size,
            padding=int(kernel_size / 2)
        )

    def forward(self, x):
        cnn_out = self.hidden_layers(x)
        pred = self.output_layer(cnn_out)
        return pred


# ------------ TRAINING HYPERPARAMETERS ------------
lr = 1e-3
weight_decay = 1e-5
n_updates = 10000  # number of updates
print_stats_at = 100  # print status to tensorboard every x updates
plot_at = 1000  # plot every x updates
validate_at = 500  # evaluate model on validation set every x updates


# ------------ MODEL EVALUATION ------------
def evaluate(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn):
    model.eval()
    loss = 0
    with torch.no_grad():
        for data in tqdm(dataloader, desc="scoring", position=0):
            inputs, targets = data
            inputs = inputs.float().to(device)
            targets = targets.float().to(device)

            outputs = model(inputs)

            loss += loss_fn(outputs, targets).item()
    loss /= len(dataloader)
    model.train()
    return loss


# ------------ MODEL TRAINING ------------
import numpy as np
from matplotlib import pyplot as plt


def train(model=SimpleCNN(n_hidden_layers=5, n_in_channels=IMG_CHANNELS, n_kernels=64, kernel_size=7),
          dataloader=train_loader,
          loss_fn=torch.nn.MSELoss()):
    if os.path.exists(SAVED_MODEL):
        print("Loading model ...")
        model = torch.load(SAVED_MODEL)
    best_validation_loss = np.inf  # best validation loss so far
    writer = SummaryWriter(log_dir=os.path.join(PATH_OUTPUT, "tensorboard"))
    update_progress_bar = tqdm(total=n_updates, desc=f"loss: {np.nan:7.5f}", position=0)

    model.to(device=device)
    torch.save(model, SAVED_MODEL)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    update = 0
    while update < n_updates:
        for data in dataloader:

            inputs, targets = data
            inputs = inputs.float().to(device)
            targets = targets.float().to(device)

            optimizer.zero_grad()
            pred = model(inputs)

            loss = loss_fn(pred, targets)

            loss.backward()
            optimizer.step()

            if update % print_stats_at == 0:
                writer.add_scalar(tag="training/loss", scalar_value=loss.cpu(), global_step=update)

            if update % plot_at == 0:
                os.makedirs(plotpath, exist_ok=True)
                fig, axes = plt.subplots(ncols=3, figsize=(15, 5))

                for i in range(len(inputs)):
                    for ax, data, title in zip(axes, [inputs.detach().cpu().numpy(), targets.detach().cpu().numpy(),
                                                      pred.detach().cpu().numpy()], ["Input", "Target", "Prediction"]):
                        ax.clear()
                        ax.set_title(title)
                        image = np.transpose(data[i], (1, 2, 0))
                        ax.imshow(image.astype('uint8'))
                        ax.set_axis_off()
                    fig.savefig(os.path.join(plotpath, f"{update:07d}_{i:02d}.png"), dpi=100)
                plt.close(fig)

            # Evaluate model
            if update % validate_at == 0:
                val_loss = evaluate(model, dataloader=val_loader, loss_fn=loss_fn)
                print("Validation loss after ", update, "updates: ", val_loss)
                writer.add_scalar(tag="validation/loss", scalar_value=val_loss, global_step=update)
                for i, (name, param) in enumerate(model.named_parameters()):
                    writer.add_histogram(tag=f"validation/param_{i} ({name})", values=param.cpu(), global_step=update)
                    writer.add_histogram(tag=f"validation/gradients_{i} ({name})", values=param.grad.cpu(),
                                         global_step=update)
                if val_loss < best_validation_loss:
                    best_validation_loss = val_loss
                    torch.save(model, SAVED_MODEL)

            update_progress_bar.set_description(f"loss: {loss:7.5f}", refresh=True)
            update_progress_bar.update()

            update += 1
            if update >= n_updates:
                break

    update_progress_bar.close()
    writer.close()


train()