import os
import plotly.express as px
import pandas as pd

if not os.path.exists("images"):
    os.mkdir("images")

# def bar_graph(metric, title, csv_path):
#     data = pd.read_csv(csv_path)
#     fig = px.bar(data, x='Class', y=metric,  title=title, range_y=[0, 1.1])
#     fig.show()
#     save_path = f"images/{title}-{metric}.png"
#     fig.write_image(save_path)

file_paths = [("No Dropout", "results_no_dropout.csv"), ("Dropout", "results_dropout.csv")]
# metrics = ["P"]
metrics = ["metrics/precision","metrics/mAP_0.5", "metrics/mAP_0.5:0.95"]

# for path in file_paths:
#     for metric in metrics:
#         bar_graph(metric, path[0], path[1])

data = pd.read_csv(file_paths[1][1])
fig = px.line(data, x='epoch', y=['metrics/precision', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95'], title='Dropout', range_y=[0, 0.7])
fig.show()

data = pd.read_csv(file_paths[0][1])
fig = px.line(data, x='epoch', y=['metrics/precision', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95'], title='No Dropout', range_y=[0, 0.7])
fig.show()