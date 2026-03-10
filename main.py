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
    for stuff in anno:
        label=stuff["label"]
        xmin,ymin,xmax,ymax=stuff["bbox"]
        color=(0,255,0) if label=="with_mask" else (0,0,255)
        cv2.rectangle(img,(xmin,ymin),(xmax,ymax),color,2)
        cv2.putText(img, label, (xmin + 3, ymin - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.imshow("img",img)
    cv2.waitKey(0)

class processDataset(Dataset):
    def __init__(self,img_dir,anno_dir,transforms=None):
        self.img_dir=img_dir
        self.anno_dir=anno_dir
        self.transforms=transforms
        self.images=sorted(os.listdir(img_dir))
        self.annos=sorted(os.listdir(anno_dir))
        self.label2id = {
            "with_mask": 1,
            "without_mask": 2,
            "mask_weared_incorrect":3 
        }


    def __len__(self):
        return len(self.images)

    def __getitem__(self,index):
        img_name=self.images[index]
        anno_name=self.annos[index]
        img_path=os.path.join(self.img_dir,img_name)
        anno_path=os.path.join(self.anno_dir,anno_name)
        img=cv2.imread(img_path)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        h,w=img.shape[:2]
        anno=parse_anno(anno_path)
        boxes=[]
        labels=[]
        for obj in anno:
            boxes.append(obj["bbox"])
            labels.append(self.label2id[obj["label"]])
        boxes=torch.tensor(boxes,dtype=torch.float32)
        labels=torch.tensor(labels,dtype=torch.int64)
        img=T.ToTensor()(img)
        target = {
            "boxes": boxes,
            "labels": labels
        }

        return img, target

cwd=os.getcwd()
img_dir=os.path.join(cwd,"images")
anno_dir=os.path.join(cwd,"annotations")
dataset=processDataset(
    img_dir=img_dir,
    anno_dir=anno_dir
)

num_classes=4
data_len=len(dataset)
train=int(data_len*0.7)
test=data_len-train
train_data,test_data=random_split(dataset,[train,test])
train_dataset=DataLoader(train_data,batch_size=1,shuffle=True,collate_fn=lambda x:tuple(zip(*x)))
test_dataset=DataLoader(test_data,batch_size=1,shuffle=False,collate_fn=lambda x:tuple(zip(*x)))

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    weights="DEFAULT"
)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
optimizer=torch.optim.Adam(model.parameters(),lr=0.0005)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def save_checkpoint(state,is_best,check_path,best_path):
    torch.save(state,check_path)
    if is_best:
        shutil.copyfile(check_path, best_path)

def load_checkpoint(check_path,model,optimizer):
    checkpoint=torch.load(check_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    valid_loss_min = checkpoint['valid_loss_min']
    return model, optimizer, checkpoint['epoch'], valid_loss_min.item()

checkpoint_path=os.path.join(cwd,'checkpoint.pt')
best_model_path=os.path.join(cwd,'bestmodel.pt')

total_train_loss = []
train_loss_min = 99

for epoch in range(10):
    start_time=time.time()
    train_loss=[]
    model.train()
    for imgs,targets in train_dataset:
        imgs = list(img.to(device) for img in imgs)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict=model(imgs,targets)
        loss=sum(loss for loss in loss_dict.values())
        train_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    epoch_train_loss = np.mean(train_loss)
    total_train_loss.append(epoch_train_loss)
    print(f'Epoch train loss is {epoch_train_loss}')
    checkpoint = {
            'epoch': epoch + 1,
            'train_loss_min': epoch_train_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
    save_checkpoint(checkpoint, False, checkpoint_path, best_model_path)
    if epoch_train_loss <= train_loss_min:
            print('Train loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(train_loss_min,epoch_train_loss))
            # save checkpoint as best model
            save_checkpoint(checkpoint, True, checkpoint_path, best_model_path)
            train_loss_min = epoch_train_loss
    time_elapsed = time.time() - start_time
    print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

plt.title('Train Loss')
plt.plot(total_train_loss)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()