# Báo Cáo Quy Trình Thực Hiện Pipeline Yoon Kim (2014) Cho Phân Tích Cảm Xúc

> **Thư mục kiểm nghiệm:** `Yoon Kim test/`  
> **Các tệp đã xây dựng:**  
> 1. [`preprocessor.py`](file:///F:/HK%20261/NLP/Train/Yoon%20Kim%20test/preprocessor.py) — Module cầu nối (Bridge/Proxy), giải quyết lỗi phân giải import trong IDE, liên kết trực tiếp với [`preprocessor.py` gốc](file:///F:/HK%20261/NLP/Train/preprocessor.py).  
> 2. [`deadpool.py`](file:///F:/HK%20261/NLP/Train/Yoon%20Kim%20test/deadpool.py) — Module tiền xử lý, gọi trực tiếp lớp [`TextPreprocessor`](file:///F:/HK%20261/NLP/Train/preprocessor.py) qua `CleanDataPool`, nạp tập Train, Dev và nạp độc lập tập Test ([`Test_clean.csv`](file:///F:/HK%20261/NLP/Train/Test/Test_clean.csv)).  
> 3. [`cnn.py`](file:///F:/HK%20261/NLP/Train/Yoon%20Kim%20test/cnn.py) — Mô hình TextCNN chuẩn Yoon Kim (2014), **tự động kiểm thử trên tập Test sau mỗi epoch**, xuất file kết quả với cột `True Label`, `Predict Label` và bảng thống kê TOP 20 nhãn sai nhiều nhất vào thư mục `result epoch/`.  
> 4. [`w.md`](file:///F:/HK%20261/NLP/Train/Yoon%20Kim%20test/w.md) — Tài liệu tổng hợp toàn diện chi tiết kỹ thuật và đối chiếu lý thuyết.

---

## 1. Bối Cảnh & Mục Tiêu

Trong khuôn khổ đề tài môn học NLP (`RULES.md`), nhóm cần thực nghiệm mô hình Deep Learning chuẩn mực để đối sánh với Baseline [train1.py](file:///F:/HK%20261/NLP/Train/train1.py) (TF-IDF + LinearSVC). Tài liệu tham khảo nền tảng được chọn là bài báo:
* **Yoon Kim (2014)**: *"Convolutional Neural Networks for Sentence Classification"*, Proceedings of EMNLP 2014.

Thư mục `Yoon Kim test/` được tạo lập nhằm cô lập và kiểm nghiệm độc lập toàn bộ pipeline từ nạp vector từ pre-trained, làm sạch câu văn bản theo [`Clean.md`](file:///F:/HK%20261/NLP/Train/Clean.md), huấn luyện mô hình tích chập 1D (TextCNN) và **đánh giá kiểm thử trực tiếp trên tập Test độc lập (`Test/Test_clean.csv`) sau từng epoch**.

---

## 2. Chi Tiết Hiện Thực Từng Module

### 2.1. Module Cầu Nối IDE: `preprocessor.py` (Local Bridge)
* Tái xuất khẩu (re-export) trực tiếp `TextPreprocessor`, `prepare_tfidf_dataset`, `parse_labels`, `EMO` từ module gốc để IDE (Pylance/VS Code) nhận diện 100% không báo lỗi gạch đỏ, hỗ trợ đầy đủ Go to Definition (F12).

---

### 2.2. Module Tiền Xử Lý Dữ Liệu: `deadpool.py`

1. **Gọi trực tiếp bộ làm sạch `TextPreprocessor` (`CleanDataPool`):**
   * Chạy qua 15 bước của [`Clean.md`](file:///F:/HK%20261/NLP/Train/Clean.md) cho văn bản thô.
2. **Xây dựng Từ điển Từ vựng (`Vocabulary`):**
   * Quản lý ánh xạ 2 chiều: `word2idx` và `idx2word` với `<pad>=0`, `<unk>=1`.
   * Chuẩn hóa độ dài câu về $n = 80$ hoặc $100$ từ.
3. **Cơ chế Nạp Pre-trained Word Vectors & Xử Lý OOV (Yoon Kim Section 4.3):**
   * Khởi tạo từ OOV theo phân phối đều có cùng phương sai với pre-trained: $U[-a, a]$ với $a = \sqrt{3}\sigma$.
4. **Nạp độc lập tập Kiểm Thử (`prepare_test_dataloader`):**
   * Đọc tệp [`Test_clean.csv`](file:///F:/HK%20261/NLP/Train/Test/Test_clean.csv) (2,224 mẫu kiểm thử).
   * Dùng chung từ điển `vocab` đã học từ tập Train để tránh rò rỉ dữ liệu.
   * Cố định thứ tự 31 nhãn mục tiêu khớp tuyệt đối với tập Train.

---

### 2.3. Module Mạng Tích Chập & Đánh Giá Test Mỗi Epoch: `cnn.py`

#### Kiến Trúc TextCNN (Section 2 & 3.1):
* **Embedding (k = 300):** Hỗ trợ 4 biến thể (`rand`, `static`, `non-static`, `multichannel`).
* **Conv2D đa kích thước cửa sổ:** $h \in \{3, 4, 5\}$, mỗi kích thước 100 filters, kích hoạt ReLU.
* **1-Max Pooling (Max-over-time):** Rút trích giá trị lớn nhất từ mỗi feature map dọc theo chiều câu.
* **Điều chuẩn:** Penultimate Dropout $p = 0.5$ và ràng buộc chuẩn $L_2$-norm $\|w\|_2 \le 3.0$ trên tầng FC.
* **Đầu ra đa nhãn:** `nn.Linear(300, 31)` kết hợp hàm mất mát `BCEWithLogitsLoss` và Sigmoid.

#### Đánh Giá Tự Động Trên Tập Test Sau Mỗi Epoch (`result epoch/`):
Tại cuối mỗi epoch huấn luyện, mô hình được đưa vào kiểm thử trực tiếp trên 2,224 mẫu tập Test và xuất 2 tệp báo cáo vào thư mục [`result epoch/`](file:///F:/HK%20261/NLP/Train/Yoon%20Kim%20test/result%20epoch):
1. **Tệp dự đoán chi tiết (`epoch_{epoch}_predictions.csv`):**
   * `index`: Số thứ tự mẫu.
   * `comment`: Văn bản bình luận kiểm thử.
   * `True Label`: Chuỗi các nhãn thực tế của câu (ví dụ: `FEATURES#Positive; GENERAL#Positive`).
   * `Predict Label`: Chuỗi các nhãn do mạng TextCNN dự đoán có xác suất $\ge 0.5$.
   * `Is_Exact_Match`: Giá trị True nếu dự đoán đúng 100% tất cả 31 nhãn.
2. **Tệp thống kê TOP 20 nhãn sai nhiều nhất (`epoch_{epoch}_top20_errors.csv`):**
   * Thống kê chi tiết: `Rank`, `Label`, `Total Errors` ($FP + FN$), `False Positives (FP)` (dự đoán thừa), `False Negatives (FN)` (bỏ sót), `Support (True Count)`, `Predicted Count`, `Error Rate (%)`, và `F1-Score` của từng nhãn.
   * In trực tiếp bảng thống kê Top 20 ra terminal sau mỗi epoch để người dùng tiện theo dõi.

---

## 3. Bảng Đối Chiếu Chi Tiết Với Bài Báo Yoon Kim (2014)

| Thành phần kỹ thuật | Đặc tả trong Paper Yoon Kim (2014) | Hiện thực trong code |
| :--- | :--- | :--- |
| **Độ dài cửa sổ tích chập ($h$)** | $h \in \{3, 4, 5\}$ (Section 3.1) | `filter_sizes = [3, 4, 5]` |
| **Số lượng Feature Maps** | 100 maps mỗi kích thước cửa sổ (Section 3.1) | `num_filters = 100` (Tổng 300 đặc trưng) |
| **Hàm kích hoạt phi tuyến** | Rectified Linear Units (ReLU) (Section 3.1) | `F.relu()` sau tầng tích chập |
| **Cơ chế Pooling** | 1-Max Pooling (Collobert et al., 2011) | `torch.max(c, dim=2)[0]` |
| **Khởi tạo OOV** | Phân phối đều có cùng phương sai: $U[-a, a]$ với $a = \sqrt{3}\sigma$ | Hàm `load_pretrained_vectors()` trong `deadpool.py` |
| **Tỷ lệ Dropout** | $p = 0.5$ ở tầng penultimate (Section 2.1 & 3.1) | `nn.Dropout(p=0.5)` |
| **Ràng buộc chuẩn trọng số** | Chuẩn $L_2 \le s$ với $s = 3$ (Section 2.1 & 3.1) | `model.clip_l2_norm(s=3.0)` |
| **Kích thước mini-batch** | Mini-batch size = 50 (Section 3.1) | `batch_size = 50` trong `DataLoader` |
| **Thuật toán tối ưu** | Adadelta update rule (Zeiler, 2012) | `torch.optim.Adadelta(lr=1.0, rho=0.95, eps=1e-6)` |
| **Đánh giá kiểm thử độc lập** | Đánh giá trên tập test chuẩn | Tự động chạy trên `Test_clean.csv` (2,224 mẫu) mỗi epoch |

---

## 4. Hướng Dẫn Chạy Thực Nghiệm

```bash
# Huấn luyện mô hình ngẫu nhiên (CNN-rand) và tự động test + xuất kết quả mỗi epoch
python "Yoon Kim test/cnn.py" --mode rand --epochs 5 --batch_size 50

# Huấn luyện mô hình có pre-trained Word2Vec (CNN-non-static)
python "Yoon Kim test/cnn.py" --mode non-static --pretrained_vec "duong/dan/word2vec.vec" --epochs 10
```
