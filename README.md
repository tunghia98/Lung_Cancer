
# Download Dataset
Please download the dataset from the following link: the dataset from this [Lung Cancer Dataset]("https://www.kaggle.com/datasets/antonixx/the-iqothnccd-lung-cancer-dataset/data"), and put it into `dataset` folder

# Model Training
```
python train.py
```

# Model Evaluation
```
python evaluate.py
```

F1 Score (Weighted): 0.9669
Precision (Weighted): 0.9687
Recall (Weighted): 0.9683
Accuracy: 0.9683
# FastAPI
```
python -m uvicorn app:app --reload
```