
# Download Dataset
Please download the dataset from the following link: the dataset from this [Lung Cancer Dataset](https://www.kaggle.com/datasets/antonixx/the-iqothnccd-lung-cancer-dataset/data), and put it into `dataset` folder by this structure
```
dataset
├───Bengin cases
├───Malignant cases
├───Normal cases
└───split_info.json
```

# Model Training
```
python train.py
```

# Model Evaluation
```
python evaluate.py
```

# FastAPI
```
python -m uvicorn app:app --reload
```