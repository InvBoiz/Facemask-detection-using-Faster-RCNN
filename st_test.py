import os
import torch
import cv2
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
import torchvision.transforms as T
from torch.utils.data import random_split, DataLoader
import torchvision
import time
import matplotlib.pyplot as plt
import numpy as np
import shutil
import streamlit as st


torch.serialization.add_safe_globals([np.dtype])

label_map = {
    1: "with_mask",
    2: "without_mask",
    3: "mask_weared_incorrect"
}

colors = {
    "with_mask": (0,255,0),
    "without_mask": (0,0,255),
    "mask_weared_incorrect": (0,255,255)
}

def prediction_to_anno(pred, label_map, score_threshold=0):
    boxes=pred['boxes'].cpu().numpy()
    labels=pred['labels'].cpu().numpy()
    scores=pred['scores'].cpu().numpy()
    anno=[]
    for box,label,score in zip(boxes,labels,scores):
        if score < score_threshold:
            continue
        xmin, ymin, xmax, ymax = box.astype(int)
        anno.append({
            "label": label_map[label],
            "bbox": [xmin, ymin, xmax, ymax]
        })
    
    return anno

def parse_anno(anno_path):
    tree=ET.parse(anno_path)
    root=tree.getroot()

    ls=[]
    for object in root.findall("object"):
        label=object.find("name").text
        bbox=object.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)
        ls.append({
            "label":label,
            "bbox":[xmin,ymin,xmax,ymax]
        })

    return ls

def resize(img_path,anno_path):
    anno=parse_anno(anno_path)
    img=cv2.imread(img_path)
    h,w=img.shape[:2]

    scale_x=500/w
    scale_y=500/h
    new_anno=[]
    for item in anno:
        xmin, ymin, xmax, ymax = item["bbox"]

        xmin = int(xmin * scale_x)
        xmax = int(xmax * scale_x)
        ymin = int(ymin * scale_y)
        ymax = int(ymax * scale_y)

        new_anno.append({"label":item["label"],"bbox":[xmin,ymin,xmax,ymax]})
    img=cv2.resize(img,(500,500))
    return img,new_anno

def draw(img,anno):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for stuff in anno:
        label=stuff["label"]
        xmin,ymin,xmax,ymax=stuff["bbox"]
        color=(0,255,0) if label=="with_mask" else (0,0,255)
        cv2.rectangle(img,(xmin,ymin),(xmax,ymax),color,2)
        cv2.putText(img, label, (xmin + 3, ymin - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    return img

num_classes=4
cwd=os.getcwd()
checkpoint_path=os.path.join(cwd,'checkpoint.pt')
best_model_path=os.path.join(cwd,'bestmodel.pt')

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    weights="DEFAULT"
)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features,num_classes)
optimizer=torch.optim.Adam(model.parameters(),lr=0.0005)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

torch.serialization.add_safe_globals([np.dtype])

checkpoint = torch.load(
    checkpoint_path,
    map_location=torch.device("cpu"),
    weights_only=False
)

model.load_state_dict(checkpoint["state_dict"])
optimizer.load_state_dict(checkpoint["optimizer"])

model.eval()   
st.title("Demo camera laptop")
run=st.checkbox("Bật camera")
frame_window = st.image([])
if "cam" not in st.session_state:
    st.session_state.cam = None

if run:
    if st.session_state.cam is None:
        st.session_state.cam = cv2.VideoCapture(0)

    cam = st.session_state.cam
    ret, frame = cam.read()

    if ret:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = T.ToTensor()(img).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(img_tensor)[0]
            anno = prediction_to_anno(pred, label_map)

        drawn = draw(img, anno)
        frame_window.image(drawn)
    else:
        st.write("Không mở được cam")
else:
    # Turn off camera when unchecked
    if st.session_state.cam is not None:
        st.session_state.cam.release()
        st.session_state.cam = None