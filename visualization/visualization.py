# import plotly.express as px
import os
import plotly.express as px
import pandas as pd


# Attempt heatmap with random data
# z = [[.1, .3, .5, .7, .9],
#      [1, .8, .6, .4, .2],
#      [.2, 0, .5, .7, .9],
#      [.9, .8, .4, .2, 0],
#      [.3, .4, .5, .7, 1]]

# fig = px.imshow(z,
#      labels=dict(title="Heatmap", x="Actual Class", y="Prediction", color="Precision"),
#      x=['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5'],
#      y=['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5'],
#      text_auto=True,
# )
# fig.update_xaxes(tickangle=90)
# fig.show()
# fig.write_image("images/heatmap.png")
# fig.write_image("images/heatmap.svg")


if not os.path.exists("images"):
    os.mkdir("images")

def bar_graph(metric_no_dropout, metric_with_dropout, title, csv_path):
    data = pd.read_csv(csv_path)
    fig = px.bar(data, x='Class', y=[metric_no_dropout, metric_with_dropout],  title=title, range_y=[0, 1.1])
    fig.update_layout(barmode='group') 
    fig.show()
    save_path = f"images/{title}-{metric}.png"
    fig.write_image(save_path)

file_paths = [("No Dropout vs. Dropout (0.5)", "raw_summary_data_both.csv"),]
# metrics = ["P"]
metrics = [("P", "P-Dropout"), ("R","R-Dropout"), ("mAP50-95", "mAP50-95-Dropout"), ("mAP50", "mAP50-Dropout")]

for path in file_paths:
    for metric in metrics:
        bar_graph(metric[0], metric[1], path[0], path[1])

