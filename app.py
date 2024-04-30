from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import timm
from torch import nn
from PIL import Image
from io import BytesIO
from torchvision import transforms
import os
import json

app = FastAPI()

model = timm.create_model('efficientnet_b3', pretrained=True, num_classes=3)
model.classifier = nn.Linear(model.classifier.in_features, 3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load('./checkpoint/efficientnet_b3-last.pth', map_location=device)

model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

DATA_DIR = './dataset'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép yêu cầu từ mọi nguồn (bạn có thể hạn chế điều này trong môi trường sản xuất)
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
label_lists = ['Benign', 'Normal', 'Malignant']

with open(os.path.join(DATA_DIR, 'split_info.json'), 'r') as fp:
    split_info = json.load(fp) 
# Endpoint to diagnose the lung CT image
@app.post("/diagnose/")
async def diagnose_lung_ct_image(file: UploadFile = File(...)):
    try:
        image = Image.open(BytesIO(await file.read())).convert('RGB')
        image = transform(image).unsqueeze(0)
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_name = label_lists[predicted.item()]
        return {"prediction": predicted_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

# Endpoint to return train/test image directories
@app.get("/image-directories/{split}")
async def get_image_directories(split: str):
    if split not in ['train', 'test']:
        raise HTTPException(status_code=404, detail="Split not found")
    dir_list = split_info[split]
    return dir_list

# Endpoint to return metric scores
@app.get("/metrics/")
async def get_metrics():
    metrics = {
        "F1 Score (Weighted)": 0.9669,
        "Precision (Weighted)": 0.9687,
        "Recall (Weighted)": 0.9683,
        "Accuracy": 0.9683
    }
    return metrics

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
