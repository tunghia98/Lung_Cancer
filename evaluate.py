import torch
from dataloader import load_data
import timm
from torch import nn
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import numpy as np

model = timm.create_model('efficientnet_b3', pretrained=True, num_classes=3)
model.classifier = nn.Linear(model.classifier.in_features, 3)

test_loader = load_data(data_dir='./dataset', batch_size=32, split='test')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load('./checkpoint/efficientnet_b3-last.pth', map_location=device)

model.load_state_dict(checkpoint['model_state_dict'])

model.to(device)

model.eval()

all_labels = []
all_predictions = []

# Evaluate the model
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

all_labels = np.array(all_labels)
all_predictions = np.array(all_predictions)

# Calculate metrics
f1 = f1_score(all_labels, all_predictions, average='weighted')
precision = precision_score(all_labels, all_predictions, average='weighted')
recall = recall_score(all_labels, all_predictions, average='weighted')
accuracy = accuracy_score(all_labels, all_predictions)

# Print metrics
print(f'F1 Score (Weighted): {f1:.4f}') # Điểm F1 trọng số, là một phép đo tổng hợp của độ chính xác và độ phủ của một mô hình.
print(f'Precision (Weighted): {precision:.4f}') # Tỉ lệ giữa số lượng các dự đoán đúng với tổng số các dự đoán dương tính.
print(f'Recall (Weighted): {recall:.4f}') #Tỉ lệ giữa số lượng các dự đoán đúng với tổng số các trường hợp thực sự là dương tính.
print(f'Accuracy: {accuracy:.4f}') # Tỉ lệ giữa số lượng các dự đoán đúng với tổng số lượng dự đoán.
