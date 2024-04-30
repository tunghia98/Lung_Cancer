import torch
from dataloader import load_data
import timm
from torch import nn, optim

# load data bao gồm dataloader và dataset
train_loader = load_data(data_dir = './dataset', batch_size=32, split='train')

model = timm.create_model('efficientnet_b3', pretrained=True, num_classes=3)

model.classifier = nn.Linear(model.classifier.in_features, 3)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(100): 
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f'i = {i+1}, Loss: {loss.item()}')
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')


checkpoint = {
    'epoch': epoch + 1,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss.item(),
}

torch.save(checkpoint, './checkpoint/efficientv4_last.pth')
print("Saved the last checkpoint.")