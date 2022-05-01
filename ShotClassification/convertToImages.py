import os 
import sys
from glob import glob
import shutil
import time

import cv2
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

VIDEOS_DIR = "videos"
CAT_DIRS = glob(os.path.join(VIDEOS_DIR, "*"))
CATS = [x.split(os.path.sep)[-1] for x in CAT_DIRS]
NCATS = len(CATS)

IMW = 240
IMH = 135

msl = 0 # maximum sequence length

for i in range(NCATS):
    category = CATS[i]
    videos = glob(os.path.join(VIDEOS_DIR, category, "*"))
    for video in videos:
        cap = cv2.VideoCapture(video)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        msl = max(msl, num_frames)

if os.path.exists("images"):
    shutil.rmtree("images")
os.mkdir("images")

X = []
y = []

for i in range(NCATS):
    category = CATS[i]
    videos = glob(os.path.join(VIDEOS_DIR, category, "*"))
    for j in range(len(videos)):
        video = videos[j]
        count = 0
        cap = cv2.VideoCapture(video)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        count = 0
        frame_locations = []
        while count < msl:
            ret, frame = cap.read()
            count += 1
            if not ret:
                frame = np.zeros((IMH, IMW), dtype=float)
            filename = f"{category}_{j:05d}_{count:05d}_{time.time_ns()}.jpg"
            filepath = os.path.join("images", filename)
            cv2.imwrite(filepath, frame)
            frame_locations.append(filepath)
        X.append(frame_locations)
        y.append(category)

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

df = pd.DataFrame()
df['X_train'] = X_train
df['X_test'] = X_test
df['y_train'] = y_train
df['y_test'] = y_test

df.to_csv("data.csv")

