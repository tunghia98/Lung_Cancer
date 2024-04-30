# Lung Cancer
Một ứng dụng chẩn đoán hình ảnh CT phổi bằng
trí tuệ nhân tạo. Ứng dụng sẽ sử dụng một mô hình deep learning để dự đoán loại bệnh
từ hình ảnh CT phổi, giúp giảm thời gian và công sức của các chuyên gia y tế trong quá
trình chẩn đoán
## Hướng dẫn sử dụng
* **Chọn ảnh CT cần chẩn đoán**
* **Nhấn nút chẩn đoán**
* **Xem thông tin chẩn đoán**

## Yêu cầu
* **torch**:
* **timm**:
* **fastapi**:
* **uvicorn[standard]**:
* **scikit-learn**:
* **python-multipart**:
## Cài đặt
### Tải bản zip hoặc tạo bản sao để bắt đầu làm việc với dự án
```
git clone https://github.com/tunghia98/Lung_Cancer.git
```
### Tải Dataset
Please download the dataset from the following link: the dataset from this [Lung Cancer Dataset](https://www.kaggle.com/datasets/antonixx/the-iqothnccd-lung-cancer-dataset/data), and put it into `dataset` folder by this structure
```
dataset
├───Bengin cases
├───Malignant cases
├───Normal cases
└───split_info.json
```
### Trỏ đường dẫn tới thư mục:
```
cd Detect_Lung_cancer
```
### Tạo và cài đặt môi trường python3
```
python3 -m venv env source venv/bin/activate
```
### Cài đặt thư viện
```
pip install -r requirements.txt
```
### Model Training
```
python train.py
```

### Model Evaluation
```
python evaluate.py
```

### FastAPI
```
python -m uvicorn app:app --reload
```

### Run Web
```
start index.html
```
## Chú ý

## Tác Giả
* **Tư Nghĩa**: (https://tunghia98.github.io/)
* **Quốc Anh**: ()
## Giấy phép
