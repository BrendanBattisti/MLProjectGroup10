import os
import plotly.express as px
import pandas as pd

if not os.path.exists("images"):
    os.mkdir("images")

# Precision
data = pd.read_csv("results_both.csv")
fig = px.line(data, x='epoch', y=['metrics/precision', 'metrics/precision_dropout'], title='Precision No Dropout vs. Dropout (0.5)', range_y=[0, 0.7])
fig.show()
fig.write_image("images/training_precision.png")

# Precision
data = pd.read_csv("results_both.csv")
fig = px.line(data, x='epoch', y=['metrics/recall', 'metrics/recall_dropout'], title='Recall No Dropout vs. Dropout (0.5)', range_y=[0, 0.7])
fig.show()
fig.write_image("images/training_precision.png")

# mAP-50-95
data = pd.read_csv("results_both.csv")
fig = px.line(data, x='epoch', y=['metrics/mAP_0.5:0.95', 'metrics/mAP_0.5:0.95_dropout'], title='mAP-50-95 No Dropout vs. Dropout (0.5)', range_y=[0, 0.5])
fig.show()
fig.write_image("images/training_mAP_0.5_0.95.png")

# mAP-50
data = pd.read_csv("results_both.csv")
fig = px.line(data, x='epoch', y=['metrics/mAP_0.5','metrics/mAP_0.5_dropout'], title='mAP-50 No Dropout vs. Dropout (0.5)', range_y=[0, 0.5])
fig.show()
fig.write_image("images/training_mAP_0.5.png")

# Box Loss
data = pd.read_csv("results_both.csv")
fig = px.line(data, x='epoch', y=['train/box_loss','train/box_loss_dropout',], title='Box Loss No Dropout vs. Dropout (0.5)', range_y=[0, 7])
fig.show()
fig.write_image("images/training_box_loss.png")

# CLS Loss
data = pd.read_csv("results_both.csv")
fig = px.line(data, x='epoch', y=['train/cls_loss', 'train/cls_loss_dropout'], title='CLS Loss No Dropout vs. Dropout (0.5)', range_y=[0, 7])
fig.show()
fig.write_image("images/training_cls_loss.png")

# DFL Loss
data = pd.read_csv("results_both.csv")
fig = px.line(data, x='epoch', y=['train/dfl_loss', 'train/dfl_loss_dropout'], title='DFL No Dropout vs. Dropout (0.5)', range_y=[0, 7])
fig.show()
fig.write_image("images/training_dfl_loss.png")