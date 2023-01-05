import os
import torch
from tqdm import tqdm
import argparse, os
from urllib.parse import urlparse

def basename (url):
    return os.path.basename( 
      (url).path)

device='cpu'

parser = argparse.ArgumentParser()

parser.add_argument("-b", "--basedir",  default='/workspace/')
parser.add_argument("-m1", "--model1", default='')
parser.add_argument("-m2", "--model2", default='')
parser.add_argument("-a", "--alpha", default=0.5)


app_args = parser.parse_args()
basedir = app_args.basedir

modelsfolder = os.path.join(basedir,'models')

model_0 = torch.load(os.path.join(modelsfolder,app_args.model1), map_location=device)
model_1 = torch.load(os.path.join(modelsfolder,app_args.model2), map_location=device)
theta_0 = model_0["state_dict"]
theta_1 = model_1["state_dict"]
alpha = float( app_args.alpha )






output_file = os.path.join(modelsfolder, f'mix-{str(alpha)[2:] + "0"}.ckpt')


for key in tqdm(theta_0.keys(), desc="Stage 1/2"):
    if "model" in key and key in theta_1:
        theta_0[key] = (1 - alpha) * theta_0[key] + alpha * theta_1[key]

for key in tqdm(theta_1.keys(), desc="Stage 2/2"):
    if "model" in key and key not in theta_0:
        theta_0[key] = theta_1[key]

print("Saving...")

torch.save({"state_dict": theta_0}, output_file)

print("Done!")
