from fastai import *
from fastai.vision.all import *
from fastai.metrics import error_rate
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2


from pathlib import Path
x  = 'C:\\Users\\Avraham\\Desktop\\project\\p'
path = Path(x)
path.ls()


np.random.seed(40)
data = ImageDataLoaders.from_folder(path, train='.', valid_pct=0.2,
                                   item_tfms=Resize(300),
                                   batch_tfms=aug_transforms(),
                                   num_workers=4,
                                   normalize=imagenet_stats)
print("Train:", data.train_ds)
print("Valid:", data.valid_ds)

data.show_batch(nrows=3, figsize=(7,6))

print(data)
print(data.train_ds.vocab)
print(len(data.train_ds.vocab))

learn = vision_learner(data, models.resnet18, metrics=[accuracy], model_dir=Path('C:\\Users\\Avraham\\Desktop\\project\\p'), path=Path("."))
print(learn)

learn.lr_find()

lr1 = 1e-3
lr2 = 1e-1
learn.fit_one_cycle(40,slice(lr1,lr2))


learn.unfreeze()
learn.fit_one_cycle(20,slice(1e-4,1e-3))


loss, acc = learn.validate()
print(f'האובדן: {loss}, האחוז המדויקות: {acc}')

learn.recorder.plot_loss()

interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()

interp.plot_top_losses(6,figsize = (25,5))

img = PILImage.create(r'C:\Users\Avraham\Desktop\project\p\Shirts\ABC.jfif')
print(learn.predict(img)[0])

learn.export(fname=Path("C:\\Users\\Avraham\\Desktop\\project\\MODEL\\download.pkl"))
learn.model_dir = "C:\\Users\\Avraham\\Desktop\\project\\MODEL"
model_path = learn.save("stage-1")