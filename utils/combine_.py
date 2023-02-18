
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
import random
from PIL import Image
from pathlib import Path
from typing import Tuple, Dict, List
import pathlib
import pydicom
import glob
import sys
import os
from torch.utils.data import Dataset
import pandas as pd

# Setup path to data folder
data_path = Path(r"D:\IAAA_CMMD\manifest-1616439774456")
image_path = data_path / "CMMD"
image_path_list = list(image_path.glob("*/*/*/*.dcm"))
# Get the last name of the path in the last element of the list
last_path = image_path_list[-1].name

df =  pd.read_csv(r"D:\IAAA_CMMD\manifest-1616439774456\all_data.csv")

cols = ['ID1', 'path']
df['combined'] = df[cols].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)

df.to_csv(r"D:\IAAA_CMMD\manifest-1616439774456\all_data_combined.csv")
