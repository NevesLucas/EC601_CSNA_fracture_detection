import json
import torch
import torch.nn as nn
import torchio as tio
import pandas as pd
from rsna_cropped import RSNACervicalSpineFracture
import plotly.graph_objects as go
from tqdm import tqdm
from sklearn.metrics import classification_report,roc_curve
with open('config.json', 'r') as f:
    paths = json.load(f)

if torch.cuda.is_available():
     print("GPU enabled")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RSNA_2022_PATH = paths["RSNA_2022_PATH"]
cachedir = paths["CACHE_DIR"]
classWeights = paths["classifier_weights"]
classModel = torch.load(classWeights, map_location=device)
classModel.eval()


pred_cols = [
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",
    "C7",
    "patient_overall"
]

root_dir="./"
def column(matrix, i):
    return [row[i] for row in matrix]

#trainSet = tio.datasets.RSNACervicalSpineFracture(RSNA_2022_PATH, add_segmentations=False)
trainSet = RSNACervicalSpineFracture(RSNA_2022_PATH, add_segmentations=False) # pre-cropped data
with torch.no_grad():
    predicted_logits = []
    actual = []

    for classifier_input, _ in zip(trainSet,range(0,10)):
        # get original dims first
        #classifier_input = preprocess(samples)
        logits = classModel(classifier_input.ct.data.unsqueeze(0).to(device)).cpu()[0]
        gt = [classifier_input[target_col] for target_col in pred_cols]
        sig = nn.Sigmoid()
        preds = sig(logits)
        overall = preds.numpy().squeeze()
        predicted_logits.append(overall)
        actual.append(gt)

    scatterPlots = []
    for i in range(0,len(pred_cols)):
        fpr, tpr, thresholds = roc_curve(column(actual, i), column(predicted_logits, i))
        scatterPlots.append(go.Scatter3d(
        x=fpr,
        y=tpr,
        z=thresholds,
        name=pred_cols[i],
        showlegend=True,
        marker=dict(
            size=5
        ),
        line=dict(
        width=2)
    ))
    fig = go.Figure(data=scatterPlots)
    fig.update_layout(scene=dict(
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        zaxis_title='Threshold'),
        width=1920,
        height=1080,
        margin=dict(r=20, b=10, l=10, t=10))

    fig.write_html("classifier_roc_plot.html")
    fig.show()
    print("choose threshold for report")
    threshold = float(input())
    predicted = [(element > threshold)*1 for element in predicted_logits]
    report = classification_report(predicted, actual, output_dict=True,
                                                      target_names=pred_cols)

    df = pd.DataFrame(report).transpose()
    df.to_csv("modelReport.csv")
