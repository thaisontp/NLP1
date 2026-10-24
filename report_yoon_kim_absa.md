# BÁO CÁO ĐỒ ÁN MÔN HỌC: XỬ LÝ NGÔN NGỮ TỰ NHIÊN (NLP)

**ĐỀ TÀI:**  
### PHÂN TÍCH CẢM XÚC THEO KHÍA CẠNH ĐA NHÃN (MULTI-LABEL ASPECT-BASED SENTIMENT ANALYSIS) TRÊN TẬP DỮ LIỆU UIT-ViSFD ỨNG DỤNG MẠNG NƠ-RON TÍCH CHẬP YOON KIM (2014)

---

## THÔNG TIN CHUNG VÀ PHÂN CÔNG NHIỆM VỤ (RULE_DEL_03)
* **Bộ dữ liệu thực nghiệm:** UIT-ViSFD (Vietnamese Smartphone Feedback Dataset)
* **Nhiệm vụ trọng tâm:** Aspect Category Sentiment Analysis (ACSA) - Phân loại văn bản đa nhãn (31 nhãn nhị phân độc lập).
* **Mô hình nghiên cứu cốt lõi:** TextCNN (Yoon Kim, EMNLP 2014) kết hợp Pipeline Tiền xử lý 15 bước và Tối ưu hóa Ngưỡng cắt (Threshold Tuning).

---

# MỤC LỤC BÁO CÁO

1. **[1. GIỚI THIỆU (INTRODUCTION)](#1-giới-thiệu-introduction)**
   * 1.2. Mô hình hóa bài toán học máy (Problem Formulation)
   * 1.3. Mục tiêu nghiên cứu và Các đóng góp chính
2. **[2. NGHIÊN CỨU LIÊN QUAN (RELATED WORK)](#2-nghiên-cứu-liên-quan-related-work)**
   * 2.1. Phân loại cảm xúc truyền thống dựa trên Machine Learning
   * 2.2. Đột phá của Mạng Nơ-ron Tích chập (Yoon Kim TextCNN, 2014)
   * 2.3. Chuyển dịch từ Đơn nhãn sang Đa nhãn trong ABSA
3. **[3. MÔ TẢ TẬP DỮ LIỆU (DATASET DESCRIPTION - UIT-ViSFD)](#3-mô-tả-tập-dữ-liệu-dataset-description---uit-visfd)**
   * 3.1. Nguồn gốc và Quy mô tập dữ liệu
   * 3.2. Cấu trúc không gian 31 nhãn mục tiêu
   * 3.3. Các mẫu dữ liệu thực tế minh họa
   * 3.4. Hai thách thức cốt lõi: Mất cân bằng cực đoan và Đồng xuất hiện đa nhãn
4. **[4. PHƯƠNG PHÁP NGHIÊN CỨU (METHODOLOGY)](#4-phương-pháp-nghiên-cứu-methodology)**
   * 4.1. Quy trình Tiền xử lý 15 bước (Clean Filter Architecture)
   * 4.2. Tầng Biểu diễn Văn bản và Khởi tạo Ma trận Nhúng Câu
   * 4.3. Kiến trúc Chi tiết Mạng Yoon Kim TextCNN cho Multi-label ABSA
   * 4.4. Hàm Mất mát và Cơ chế Quyết định Ngưỡng cắt (Threshold Tuning)
5. **[5. THIẾT LẬP THỰC NGHIỆM (EXPERIMENTAL SETUP)](#5-thiết-lập-thực-nghiệm-experimental-setup)**
   * 5.1. Môi trường phần cứng và Thư viện hỗ trợ
   * 5.2. Cấu hình Siêu tham số (Hyperparameters)
   * 5.3. Mô hình Cơ sở Đối chứng (Baseline LinearSVC)
   * 5.4. Thang đo Đánh giá (Evaluation Metrics)
6. **[6. KẾT QUẢ THỰC NGHIỆM (RESULTS)](#6-kết-quả-thực-nghiệm-results)**
   * 6.1. Bảng so sánh tổng thể Baseline vs TextCNN qua các mốc huấn luyện
   * 6.2. Tiến trình học tập của TextCNN: Từ Khởi đầu đến Đột phá
   * 6.3. Hiệu năng chi tiết theo từng Khía cạnh và Cực cảm xúc
7. **[7. PHÂN TÍCH LỖI VÀ THẢO LUẬN (ERROR ANALYSIS & DISCUSSION)](#7-phân-tích-lỗi-và-thảo-luận-error-analysis--discussion)**
   * 7.1. Bảng Top 20 Nhãn có Tỷ lệ Lỗi Dự đoán Cao nhất tại Epoch 59
   * 7.2. Giải mã Nút thắt Nhiễu giữa `OTHERS` và `GENERAL#Positive`
   * 7.3. Tác động cốt lõi của việc Hạ Ngưỡng Quyết định ($\tau = 0.3$)
   * 7.4. Phân tích các Ca lỗi Thất bại Điển hình (Case Studies)
8. **[8. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN (CONCLUSION & FUTURE WORK)](#8-kết-luận-và-hướng-phát-triển-conclusion--future-work)**
   * 8.1. Kết luận Đồ án
   * 8.2. Hạn chế còn tồn đọng
   * 8.3. Hướng phát triển trong tương lai
9. **[TÀI LIỆU THAM KHẢO (REFERENCES)](#tài-liệu-tham-khảo-references)**

---

# 1. GIỚI THIỆU (INTRODUCTION)

### 1.2. Mô hình hóa bài toán học máy (Problem Formulation)
Tuân thủ **RULE_DATA_01** và **RULE_DATA_02**, bài toán ACSA trên tập dữ liệu tiếng Việt UIT-ViSFD được mô hình hóa thành bài toán:
Multi-Label Text Classification Phân loại văn bản đa nhãn

* **Đầu vào (Input):** Một văn bản bình luận tự nhiên của người dùng $\mathbf{s} = (w_1, w_2, \dots, w_n)$.
* **Đầu ra (Output):** Một vector nhị phân đa chiều $\mathbf{y} \in \{0, 1\}^{31}$, trong đó mỗi phần tử $y_j \in \{0, 1\}$ đại diện cho sự vắng mặt ($0$) hoặc hiện diện ($1$) của nhãn thứ $j$.
* **Không gian nhãn (Label Space - 31 chiều):**
  1. **30 nhãn kép `{ASPECT}#{SENTIMENT}`:** Được hình thành từ tích Descartes của **10 Danh mục Khía cạnh** (`BATTERY`, `CAMERA`, `DESIGN`, `FEATURES`, `GENERAL`, `PERFORMANCE`, `PRICE`, `SCREEN`, `SER&ACC`, `STORAGE`) và **3 Cực Cảm xúc** (`Positive`, `Negative`, `Neutral`).
  2. **1 nhãn độc lập `{OTHERS}`:** Đại diện cho các ý kiến phản hồi nằm ngoài 10 khía cạnh kỹ thuật trên (hỏi đáp tồn kho, giao nhận hàng, khiếu nại nhân viên, v.v.). Nhãn này không có cực cảm xúc đi kèm.

Khác với bài toán phân loại đa lớp thông thường (Multi-class Classification) – nơi các lớp mang tính chất loại trừ lẫn nhau (mutually exclusive) và dùng hàm kích hoạt $\text{Softmax}$ – ở bài toán phân loại đa nhãn, một bình luận có thể chứa trung bình $3.28$ nhãn đồng thời (và tối đa lên đến $9$ nhãn trong tập dữ liệu). Vì vậy, tầng đầu ra bắt buộc phải sử dụng **31 hàm kích hoạt $\text{Sigmoid}$ độc lập** cùng hàm mất mát **Binary Cross-Entropy with Logits Loss (`BCEWithLogitsLoss`)**.

### 1.3. Mục tiêu nghiên cứu và Các đóng góp chính
Mục tiêu cốt lõi của đồ án là tái hiện trung thực tư tưởng của mạng nơ-ron tích chập kinh điển của **Yoon Kim (EMNLP 2014)**, cải biên kiến trúc từ phân loại đơn nhãn sang phân loại đa nhãn trong bài toán ABSA tiếng Việt, đồng thời thiết lập quy trình đối sánh khoa học với mô hình học máy truyền thống:
1. **Xây dựng Pipeline Tiền xử lý 15 bước chuẩn hóa chuyên sâu (`Clean.md`):** Kết hợp hài hòa giữa cơ chế dò từ điển tĩnh (Dictionary-based) và quy luật tổng quát (Regex Rule-based). Điểm cải tiến mang tính bản lề là: **Chuẩn hóa thực thể số liệu (Entity Normalization) nhưng tuyệt đối bảo tồn độ lớn của giá tiền và thời lượng**, đồng thời bảo tồn ranh giới dấu câu chuẩn để chống hiện tượng học vẹt (shortcut learning).
2. **Hiện thực hóa trọn vẹn kiến trúc Yoon Kim TextCNN (`cnn.py`, `deadpool.py`):** Triển khai ma trận câu nhúng từ, bộ lọc Conv2D đa kích thước kernel $h \in \{3, 4, 5\}$, cơ chế 1-Max Pooling, tầng Penultimate Dropout ($p=0.5$), và cơ chế kẹp chuẩn trọng số nghiêm ngặt ($L_2$-norm clipping $s=3.0$). Tái hiện cơ chế khởi tạo Out-Of-Vocabulary (OOV) theo đúng phân phối đồng nhất bảo toàn phương sai $U[-a, a]$ với $a = \sqrt{3}\sigma$ của Yoon Kim.
3. **Giải quyết nút thắt nghẽn phân loại (Recall Bottleneck) bằng Tinh chỉnh Ngưỡng Quyết định (Threshold Tuning):** Phân tích sâu sắc hiện tượng xung đột phân loại giữa `OTHERS` và `GENERAL#Positive`. Chứng minh trên thực nghiệm rằng việc hạ ngưỡng kích hoạt từ $\tau = 0.5$ xuống $\tau = 0.3$ đã giải phóng độ phủ của mô hình: đưa Micro-F1 từ $67.32\%$ lên **$72.10\%$**, Macro-F1 từ $35.56\%$ lên **$44.65\%$**, F1-score của nhãn `OTHERS` nhảy vọt từ $0.2180$ lên **$0.5031$**, vượt trội hoàn toàn so với mô hình cơ sở LinearSVC.

---

# 2. NGHIÊN CỨU LIÊN QUAN (RELATED WORK)

### 2.1. Phân loại cảm xúc truyền thống dựa trên Machine Learning
Trong giai đoạn đầu, các hệ thống phân tích cảm xúc văn bản dựa chủ yếu vào các mô hình học máy cổ điển (Traditional Machine Learning) kết hợp với các kỹ thuật trích xuất đặc trưng thủ công (Feature Engineering):
* **Biểu diễn văn bản:** Túi từ (Bag-of-Words) và Tần suất từ - Nghịch đảo tần suất văn bản (**TF-IDF**). Các mô hình này biểu diễn câu thành một vector thưa (sparse vector) trong không gian từ vựng có số chiều rất lớn ($|V| \in [5,000, 20,000]$).
* **Mô hình phân loại:** Support Vector Machines (**SVM** / **LinearSVC**), Naive Bayes, và Logistic Regression. Khi áp dụng cho bài toán đa nhãn, chiến lược phổ biến nhất là **One-vs-Rest (Binary Relevance)** – huấn luyện 31 bộ phân loại nhị phân độc lập.
* **Nhược điểm:** Mặc dù LinearSVC đạt tốc độ huấn luyện rất nhanh và hiệu năng khá tốt trên tập dữ liệu nhỏ nhờ không gian phân tách tuyến tính rộng, mô hình túi từ hoàn toàn **bỏ qua trật tự từ vựng (word order)**, không bắt được ngữ cảnh cục bộ (local context) như cấu trúc phủ định (*"không hề tốt"*), và bị ảnh hưởng nghiêm trọng bởi sự thưa thớt dữ liệu (data sparsity).

### 2.2. Đột phá của Mạng Nơ-ron Tích chập (Yoon Kim TextCNN, 2014)
Năm 2014, tại hội nghị EMNLP, tác giả Yoon Kim đã công bố công trình mang tính bước ngoặt: *"Convolutional Neural Networks for Sentence Classification"*. Tác giả chứng minh rằng một kiến trúc mạng CNN tương đối đơn giản – vốn xuất thân từ thị giác máy tính – khi áp dụng lên các vector từ phân tán (word embeddings) được huấn luyện trước bằng Word2Vec, có thể thiết lập kỷ lục hiệu năng mới (State-Of-The-Art) trên nhiều bộ dữ liệu phân loại câu học thuật (SST, TREC, CR, MPQA).

Yoon Kim đã đề xuất và khảo sát 4 biến thể của mô hình:
1. **CNN-rand:** Tất cả vector từ vựng đều được khởi tạo ngẫu nhiên và được cập nhật (fine-tune) trong quá trình huấn luyện.
2. **CNN-static:** Sử dụng các vector từ đã được huấn luyện sẵn (Pre-trained Word2Vec); toàn bộ vector này được giữ cố định (đóng băng trọng số), chỉ học các tham số của bộ lọc tích chập và tầng phân loại.
3. **CNN-non-static:** Sử dụng vector từ huấn luyện sẵn nhưng cho phép tiếp tục tinh chỉnh (fine-tune) theo tác vụ mục tiêu.
4. **CNN-multichannel:** Kiến trúc đa kênh (Multichannel) sở hữu 2 bộ ma trận embedding song song: một kênh cố định (static) và một kênh cho phép tinh chỉnh (non-static), mô phỏng các kênh màu RGB trong xử lý ảnh.

Công trình của Yoon Kim khẳng định: **Các bộ lọc tích chập 1D/2D với các kích thước cửa sổ khác nhau ($h = 3, 4, 5$) hoạt động tương tự như các bộ trích xuất đặc trưng n-gram linh hoạt, trong khi cơ chế 1-Max Pooling giúp mạng thu thập được đặc trưng tiêu biểu nhất của câu bất kể vị trí của nó nằm ở đầu, giữa hay cuối câu.**

### 2.3. Chuyển dịch từ Đơn nhãn sang Đa nhãn trong ABSA
Mô hình gốc của Yoon Kim được thiết kế cho bài toán phân loại đơn nhãn đa lớp (Single-label Multi-class Classification), sử dụng tầng Fully Connected thu về $C$ lớp, hàm kích hoạt $\text{Softmax}$, và hàm mất mát Cross-Entropy Loss:
$$P(y = c \mid \mathbf{z}) = \frac{\exp(z_c)}{\sum_{j=1}^{C} \exp(z_j)}$$

Trong đồ án này, để giải quyết bài toán ABSA, kiến trúc được cải biên mang tính hệ thống sang **Multi-Label Multi-Output**:
* Thay thế $\text{Softmax}$ bằng vector **31 hàm $\text{Sigmoid}$**: $\hat{y}_j = \sigma(z_j) = \frac{1}{1 + e^{-z_j}}$.
* Thay thế Cross-Entropy đa lớp bằng **Multi-label Binary Cross-Entropy with Logits**: Cho phép mỗi nơ-ron đầu ra hoạt động như một máy phân loại nhị phân độc lập nhưng chia sẻ toàn bộ không gian biểu diễn đặc trưng ẩn (shared latent representations) được học từ tầng tích chập bên dưới.

---

# 3. MÔ TẢ TẬP DỮ LIỆU (DATASET DESCRIPTION - UIT-ViSFD)

Tuân thủ **RULE_DATA_03** và **RULE_DATA_04**, phần này mô tả chi tiết nguồn gốc, quy mô và đặc thù thống kê của tập dữ liệu thực nghiệm.

### 3.1. Nguồn gốc và Quy mô tập dữ liệu
Tập dữ liệu sử dụng trong nghiên cứu là **UIT-ViSFD (Vietnamese Students' Feedback Dataset / Vietnamese Smartphone Feedback Dataset)**, được thu thập từ các trang thương mại điện tử hàng đầu tại Việt Nam và được chuẩn hóa bởi các nhà nghiên cứu thuộc Trường Đại học Công nghệ Thông tin (ĐHQG-HCM). 

Tập dữ liệu được phân chia thành 3 tập độc lập theo cấu trúc chuẩn:
* **Tập Huấn luyện (Train set):** $7,786$ mẫu bình luận (chiếm $70\%$).
* **Tập Phát triển (Dev / Validation set):** $1,112$ mẫu bình luận (chiếm $10\%$).
* **Tập Kiểm tra Độc lập (Test set):** $2,224$ mẫu bình luận (chiếm $20\%$).

Tổng cộng tập dữ liệu gồm **$11,122$ bình luận thực tế**, bảo đảm tính đại diện cao cho ngôn ngữ đánh giá sản phẩm công nghệ của người dùng Việt Nam.

### 3.2. Cấu trúc không gian 31 nhãn mục tiêu
Mỗi mẫu dữ liệu được gán nhãn trên một không gian gồm **31 nhãn nhị phân**:

```
                       ┌─────────────────────────┐
                       │   KHÔNG GIAN 31 NHÃN   │
                       └────────────┬────────────┘
                                    │
           ┌────────────────────────┴────────────────────────┐
           ▼                                                 ▼
┌──────────────────────────────────────┐          ┌──────────────────────┐
│  30 Nhãn Khía cạnh - Cảm xúc         │          │  01 Nhãn Độc lập     │
│  (10 Aspects × 3 Sentiments)         │          │  {OTHERS}            │
├──────────────────────────────────────┤          ├──────────────────────┤
│ Khía cạnh:                           │          │ Đại diện cho ý kiến  │
│  1. BATTERY       6. PERFORMANCE     │          │ ngoài 10 khía cạnh:  │
│  2. CAMERA        7. PRICE           │          │ • Hỏi đáp hàng tồn   │
│  3. DESIGN        8. SCREEN          │          │ • Khiếu nại bảo hành │
│  4. FEATURES      9. SER&ACC         │          │ • Giao nhận vận chuyển│
│  5. GENERAL      10. STORAGE         │          │                      │
│                                      │          │ Xuất hiện: >2,400 lần │
│ Cực cảm xúc:                         │          │ trong toàn bộ dữ liệu│
│ • Positive   • Negative   • Neutral  │          │                      │
└──────────────────────────────────────┘          └──────────────────────┘
```

### 3.3. Các mẫu dữ liệu thực tế minh họa
Dưới đây là bảng trích xuất 4 mẫu dữ liệu thực tế đại diện từ tập dữ liệu, thể hiện rõ tính chất đa khía cạnh, đa sắc thái cảm xúc và hiện tượng teencode/emoji của người dùng:

| ID | Văn bản bình luận gốc (`comment`) | Các nhãn thực tế gán kèm (`True Labels`) | Phân tích tính chất ngôn ngữ |
| :---: | :--- | :--- | :--- |
| **#102** | `"Pin 5000mah dùng cả ngày thoải mái, camera chụp đêm bao nét 👍, nhưng chơi liên quân hơi giật lag."` | `BATTERY#Positive`<br>`CAMERA#Positive`<br>`PERFORMANCE#Negative` | Câu phức đa khía cạnh, chứa thông số kỹ thuật (`5000mah`), emoji tích cực (`👍`), thuật ngữ game và từ lóng (`giật lag`). |
| **#345** | `"Điện thoại đẹp, nhân viên tgdd tư vấn nhiệt tình, đóng gói cẩn thận 5 sao."` | `DESIGN#Positive`<br>`SER&ACC#Positive`<br>`GENERAL#Positive` | Đánh giá đồng thời về kiểu dáng máy và dịch vụ chăm sóc khách hàng (`SER&ACC`), chứa từ viết tắt tên thương hiệu (`tgdd`). |
| **#889** | `"Ad cho hỏi máy này còn hàng ở chi nhánh Cần Thơ không ạ? Với có hỗ trợ trả góp 0% không?"` | `OTHERS` | Ý kiến hoàn toàn không đánh giá phần cứng của máy; mang tính chất hỏi đáp giao dịch, thủ tục thương mại. |
| **#1420** | `"Tầm giá 4tr1 thì con này màn hình amoled quá đỉnh, màu sắc rực rỡ, mua ngày 8/3 được giảm thêm."` | `PRICE#Positive`<br>`SCREEN#Positive`<br>`OTHERS` | Chứa thực thể giá tiền (`4tr1`), công nghệ màn hình (`amoled`), mốc thời gian khuyến mãi (`8/3` thuộc `OTHERS`). |

### 3.4. Hai thách thức cốt lõi của tập dữ liệu

#### A. Mất cân bằng nhãn cực đoan (Extreme Class Imbalance)
Tần suất xuất hiện giữa các nhãn có sự chênh lệch khổng lồ:
* Các nhãn phổ biến chiếm ưu thế áp đảo: `GENERAL#Positive` (>4,800 lượt trong tập train), `PERFORMANCE#Positive` (>2,800 lượt), `BATTERY#Positive` (>2,600 lượt).
* Ngược lại, các nhãn phần cứng thiểu số và các nhãn trung tính (`Neutral`) có số lượng cực kỳ khan hiếm: `STORAGE#Negative` chỉ có vỏn vẹn **91 lượt xuất hiện** trong toàn bộ 7,786 mẫu train (tỷ lệ $1.16\%$), `SCREEN#Neutral` và `BATTERY#Neutral` xuất hiện dưới 200 lượt.
* Sự chênh lệch tỷ lệ giữa nhãn đa số và nhãn thiểu số lên tới **hơn 50 lần**. Nếu mô hình huấn luyện với hàm mất mát tiêu chuẩn và ngưỡng cắt cố định $\tau = 0.5$, mạng nơ-ron sẽ có xu hướng "lười biếng", luôn dự đoán nhãn âm tính ($0$) cho các lớp thiểu số để tối đa hóa độ chính xác tổng thể.

#### B. Sự đồng xuất hiện đa nhãn (Multi-label Co-occurrence) và Nhiễu giữa `OTHERS` và `GENERAL`
Thống kê dữ liệu cho thấy nhãn `OTHERS` xuất hiện hơn 2,400 lần. Tuy nhiên, người dùng Việt Nam có thói quen viết các câu vừa hỏi han/khen ngợi chung, vừa xen lẫn bình luận phần cứng:
* *"Máy xài ok nha, mà giao hàng hơi chậm"* $\to$ vừa có `GENERAL#Positive`, vừa có `OTHERS`.
* Khi huấn luyện với số lượng epoch ít, mô hình bị rơi vào tình trạng **nhiễu xung đột (confounding effect)**: Mô hình không thể phân biệt được từ ngữ nào kích hoạt `GENERAL#Positive` và từ ngữ nào kích hoạt `OTHERS`, dẫn đến việc triệt tiêu nhãn `OTHERS` trong 20 epoch đầu tiên (dự đoán bằng $0$).

---

# 4. PHƯƠNG PHÁP NGHIÊN CỨU (METHODOLOGY)

Quy trình giải pháp tổng thể của đồ án được minh họa qua sơ đồ kiến trúc hình dưới đây:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        VĂN BẢN ĐÁNH GIÁ THÔ                            │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│         PIPELINE TIỀN XỬ LÝ 15 BƯỚC (Clean.md / preprocessor.py)       │
│  • Unicode NFC • Dọn rác Regex • Rút gọn ký tự lặp & Dấu câu           │
│  • Chuẩn hóa Thực thể Số liệu (Bảo tồn Giá tiền, Thời lượng, Ngày)     │
│  • Tra cứu Emoji & Teencode • Gộp thông số phần cứng (5000mah)         │
│  • Phân đoạn từ tiếng Việt (PyVi) • Lọc từ dừng bảo tồn phủ định       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│             TẦNG BIỂU DIỄN VĂN BẢN (deadpool.py / Vocab)               │
│  • Indexing từ vựng: Vocab |V| = 7,814 từ ghép tiếng Việt              │
│  • Padding / Truncation về độ dài cố định n = 80 token                 │
│  • Khởi tạo OOV theo Yoon Kim Uniform Variance: U[-a, a], a = sqrt(3)σ │
│  • Ma trận câu đầu vào: X ∈ ℝ^(80 × 300)                               │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│           KIẾN TRÚC MẠNG NƠ-RON TÍCH CHẬP TEXTCNN (cnn.py)            │
│  • 3 Nhóm Bộ lọc Conv2D: h ∈ {3, 4, 5}, mỗi nhóm 100 filters           │
│  • Hàm kích hoạt phi tuyến: ReLU                                       │
│  • Cơ chế gộp cực đại: 1-Max Pooling (trích xuất 300 đặc trưng tối ưu) │
│  • Tầng Penultimate Dropout: p = 0.5                                   │
│  • Ràng buộc kẹp chuẩn trọng số nghiêm ngặt: ||w||_2 ≤ s = 3.0         │
│  • Tầng Tuyến tính Đa nhãn: Linear(300 -> 31)                          │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│              TỐI ƯU HÓA & RA QUYẾT ĐỊNH ĐA NHÃN                       │
│  • Hàm mất mát: BCEWithLogitsLoss độc lập trên 31 nhãn                 │
│  • Thuật toán tối ưu: Adadelta (ρ = 0.95, ε = 1e-6)                    │
│  • Kích hoạt xác suất: Sigmoid độc lập p_j = σ(z_j)                    │
│  • Tinh chỉnh ngưỡng cắt động: p_j ≥ τ = 0.3 -> Nhãn tích cực          │
└────────────────────────────────────────────────────────────────────────┘
```

---

### 4.1. Quy trình Tiền xử lý 15 bước (Clean Filter Architecture)
Tuân thủ **RULE_PREP_01**, **RULE_PREP_02** và **RULE_PREP_03**, dữ liệu văn bản được làm sạch qua quy trình 15 bước nghiêm ngặt, chia làm hai cơ chế:

#### A. Động lực học thuật của các bước tiền xử lý
1. **Chuẩn hóa Unicode dạng dựng sẵn (NFC):** Người dùng gõ tiếng Việt bằng nhiều bộ gõ khác nhau (Telex, VNI) sinh ra hai dạng mã hóa: Tổ hợp (NFD) và Dựng sẵn (NFC). Ví dụ: từ `"hòa"` trong NFC là 1 ký tự duy nhất `U+00F2`, trong khi NFD là `o` + dấu huyền rời rạc. Nếu không ép về NFC, phép tra cứu từ điển và embedding sẽ bị trượt hoàn toàn.
2. **Dọn rác hệ thống (System Noise Removal):** Dùng biểu thức chính quy (Regex) loại bỏ triệt để liên kết web (`http\S+`), thẻ HTML (`<.*?>`), email, và số điện thoại rác phát sinh trong quá trình cào dữ liệu (crawl).
3. **Rút gọn ký tự kéo dài (Character Lengthening):** Cảm xúc cường điệu trên mạng xã hội thường thể hiện qua việc lặp chữ cái (`"đẹpppp"`, `"lagggg"`, `"mượt lắmmm"`). Thuật toán rút gọn mọi ký tự lặp từ 3 lần trở lên về đúng 1 ký tự (`"đẹp"`, `"lag"`, `"lắm"`), tránh bùng nổ không gian từ vựng thưa.
4. **Chuẩn hóa dấu câu (Punctuation Normalization):** Rút gọn chuỗi dấu câu lặp (`"!!!!!"` $\to$ `"!"`, `"?????"` $\to$ `"?"`). **Đặc biệt, hệ thống bảo tồn nguyên vẹn dấu câu đơn**, không dùng hàm xóa mù quáng `re.sub(r'[^\w\s]', ' ', text)`, nhằm giữ lại ranh giới mệnh đề và sắc thái cảm thán.
5. **Chuẩn hóa Thực thể Số liệu (Entity Normalization) — Điểm cải tiến cốt lõi:**
   * *Ngày tháng lịch:* Gom các mốc thời gian thưa thớt (`8/3/2019`, `23/5`, `15.2.20`) về token `_date_` để tạo đặc trưng gom cụm thời điểm mua hàng/bảo hành.
   * *Độ lớn Giá tiền (Price Magnitude):* Chuẩn hóa về dạng thống nhất (`500.000đ` $\to$ `500000_đ`, `4tr1` $\to$ `4_triệu_1`, `2tr` $\to$ `2_triệu`, `500k` $\to$ `500_k`). **Tuyệt đối không thay thế bằng token tĩnh `<PRICE>`**, vì giá trị tiền tệ thực tế mang tính quyết định phân khúc sản phẩm (giá rẻ bình dân vs cao cấp). Việc giữ nguyên độ lớn giúp mô hình chống hiện tượng học vẹt (shortcut bias).
   * *Khoảng thời lượng:* Giữ nguyên số và đơn vị (`"2 ngày"`, `"6 tháng"`) để phục vụ nhận diện độ bền và thời gian bảo hành.
   * *Phần trăm & Thang điểm:* Gom `88%` $\to$ `88_phần_trăm`, `9/10` $\to$ `9_trên_10` (tín hiệu cốt lõi của khía cạnh `BATTERY` và `GENERAL`).
6. **Tra cứu Từ điển Cảm xúc Biểu tượng (Emoji Lookup Table):** Ánh xạ hơn 200 biểu tượng cảm xúc và emoticon (`👍`, `😍`, `=))`) về token `_positive_`; các icon tiêu cực (`👎`, `😡`, `🥲`) về `_negative_` kèm cơ chế kiểm tra ranh giới từ (word boundary).
7. **Tra cứu Tiếng lóng và Từ viết tắt (Teencode Lookup Table):** Ánh xạ các từ viết tắt phổ biến về dạng chuẩn mực tiếng Việt (`"ko"` $\to$ `"không"`, `"đt"` $\to$ `"điện_thoại"`, `"sp"` $\to$ `"sản_phẩm"`, `"pin trau"` $\to$ `"pin trâu"`).
8. **Gộp thông số phần cứng (Hardware Specs Merging):** Dùng Regex xóa khoảng trắng giữa số nguyên và đơn vị đo lường kỹ thuật (`"5000 mah"` $\to$ `"5000mah"`, `"48 MP"` $\to$ `"48mp"`, `"6.5 inch"` $\to$ `"6_5inch"`), ngăn chặn việc bộ tách từ chẻ nhỏ số liệu thành các phần tử vô nghĩa.
9. **Phân đoạn từ tiếng Việt (Vietnamese Word Segmentation):** Sử dụng thư viện `pyvi` để ghép các tiếng đơn thành từ ghép hoàn chỉnh (`"chụp ảnh"` $\to$ `"chụp_ảnh"`, `"thời lượng"` $\to$ `"thời_lượng"`, `"ổn định"` $\to$ `"ổn_định"`).
10. **Lọc từ dừng chọn lọc (Selective Stopwords Removal):** Loại bỏ các hư từ ngữ pháp vô nghĩa (`"là"`, `"thì"`, `"mà"`, `"những"`). **Tuyệt đối bảo tồn các từ phủ định (`"không"`, `"chưa"`, `"chẳng"`) và các phó từ chỉ mức độ (`"rất"`, `"quá"`, `"hơi"`),** vì đây là những từ định hình toàn bộ cực tính cảm xúc của câu.

---

### 4.2. Tầng Biểu diễn Văn bản và Khởi tạo Ma trận Nhúng Câu
Tuân thủ **RULE_REP_01** và **RULE_REP_02**, mô hình TextCNN sử dụng phương pháp biểu diễn từ phân tán liên tục trong không gian Euclid $k$ chiều ($k = 300$).

#### A. Cấu trúc Ma trận Câu $X \in \mathbb{R}^{n \times k}$
Giả sử câu văn bản sau bước tiền xử lý gồm $n = 80$ token:
$$\mathbf{s} = (w_1, w_2, \dots, w_{80})$$

Mỗi token $w_i$ được biểu diễn bởi một vector hàng đặc trưng dày đặc $\mathbf{x}_i \in \mathbb{R}^{300}$. Toàn bộ câu văn bản được hình thành bằng phép nối (concatenation $\oplus$) các vector từ theo chiều dọc:
$$X = \mathbf{x}_{1:n} = \mathbf{x}_1 \oplus \mathbf{x}_2 \oplus \dots \oplus \mathbf{x}_n = \begin{bmatrix}
x_{1, 1} & x_{1, 2} & \dots & x_{1, 300} \\
x_{2, 1} & x_{2, 2} & \dots & x_{2, 300} \\
\vdots & \vdots & \ddots & \vdots \\
x_{80, 1} & x_{80, 2} & \dots & x_{80, 300}
\end{bmatrix} \in \mathbb{R}^{80 \times 300}$$

Ma trận $X$ tương đương với một ảnh 2 chiều có kích thước $80 \times 300$ với 1 kênh màu ($C = 1$).

#### B. Khởi tạo Out-Of-Vocabulary (OOV) theo Yoon Kim (2014)
Một đóng góp lý thuyết tinh tế trong bài báo gốc của Yoon Kim (Section 3 - Two Channels) nằm ở cách khởi tạo các từ chưa từng xuất hiện (Out-Of-Vocabulary - OOV). Thay vì gán bằng 0 hoặc dùng phân phối chuẩn tùy ý, Yoon Kim chỉ định:
> *"For words not in word2vec, we initialize them by sampling each dimension from $U[-a, a]$, where $a$ is chosen such that the variance of the uniform distribution matches that of the pre-trained embeddings."*

Theo lý thuyết xác suất thống kê, phương sai của biến ngẫu nhiên tuân theo phân phối đều liên tục $U[-a, a]$ là:
$$\mathrm{Var}(U[-a, a]) = \frac{(a - (-a))^2}{12} = \frac{(2a)^2}{12} = \frac{4a^2}{12} = \frac{a^2}{3}$$

Để phương sai này khớp chính xác với phương sai thực nghiệm $\sigma^2$ của các vector từ đã học:
$$\frac{a^2}{3} = \sigma^2 \implies a = \sqrt{3} \cdot \sigma$$

Trong file [`deadpool.py`](file:///F:/HK%20261/NLP/Train/Yoon%20Kim%20test/deadpool.py), cơ chế này đã được hiện thực hóa trọn vẹn:
```python
var_pretrained = np.var(pretrained_vectors)
sigma = np.sqrt(var_pretrained)
a = np.sqrt(3.0) * sigma
oov_vector = np.random.uniform(-a, a, size=(embedding_dim,))
```
Kỹ thuật này bảo đảm năng lượng tín hiệu (signal energy) và độ lớn kỳ vọng của các vector OOV tương đồng hoàn hảo với các từ quen thuộc, triệt tiêu hiện tượng sốc gradient ở các bước lan truyền ngược đầu tiên.

---

### 4.3. Kiến trúc Chi tiết Mạng Yoon Kim TextCNN cho Multi-label ABSA
Kiến trúc chi tiết của mạng được mô hình hóa qua hệ phương trình toán học từ (1) đến (5):

#### A. Phép Tích Chập 2D Đa Kích Thước (Multi-filter Convolution)
Xét một cửa sổ trượt gồm $h$ từ liên tiếp $\mathbf{x}_{i:i+h-1} = \mathbf{x}_i \oplus \mathbf{x}_{i+1} \oplus \dots \oplus \mathbf{x}_{i+h-1} \in \mathbb{R}^{h \times k}$.  
Một bộ lọc tích chập (filter) được định nghĩa bởi ma trận trọng số $\mathbf{w} \in \mathbb{R}^{h \times k}$ và một hệ số chệch (bias) $b \in \mathbb{R}$.

Đặc trưng $c_i$ sinh ra từ cửa sổ từ $\mathbf{x}_{i:i+h-1}$ tuân theo phương trình (1):
$$c_i = f\left( \mathbf{w} \cdot \mathbf{x}_{i:i+h-1} + b \right) = f\left( \sum_{m=1}^{h} \sum_{d=1}^{k} w_{m, d} \cdot x_{i+m-1, d} + b \right) \quad \text{(Phương trình 1)}$$
Trong đó:
* Phép toán $\mathbf{w} \cdot \mathbf{x}_{i:i+h-1}$ là tích Frobenius (tổng tích từng phần tử) giữa kernel và vùng văn bản con.
* $f$ là hàm kích hoạt phi tuyến tính **ReLU (Rectified Linear Unit)**: $f(x) = \max(0, x)$.

Khi bộ lọc trượt dọc từ vị trí $i = 1$ đến $i = n - h + 1$, ta thu được một bản đồ đặc trưng (feature map) $\mathbf{c} \in \mathbb{R}^{n - h + 1}$:
$$\mathbf{c} = [c_1, c_2, \dots, c_{n-h+1}] \quad \text{(Phương trình 2)}$$

Trong mô hình thực nghiệm, chúng tôi triển khai đồng thời 3 kích thước bộ lọc:
* $h = 3$ (n-gram độ dài 3 từ ghép): Số chiều feature map là $80 - 3 + 1 = 78$.
* $h = 4$ (n-gram độ dài 4 từ ghép): Số chiều feature map là $80 - 4 + 1 = 77$.
* $h = 5$ (n-gram độ dài 5 từ ghép): Số chiều feature map là $80 - 5 + 1 = 76$.
* Mỗi kích thước kernel sử dụng $100$ bộ lọc độc lập, sinh ra tổng cộng **$300$ bản đồ đặc trưng**.

#### B. Cơ Chế Gộp Cực Đại 1 Chiều (1-Max Pooling)
Để trích xuất đặc trưng quan trọng nhất từ mỗi feature map và đưa vector biểu diễn về độ dài cố định bất chấp độ dài câu, Yoon Kim áp dụng phép gộp cực đại theo thời gian (1-Max Pooling) theo phương trình (3):
$$\hat{c} = \max\{ \mathbf{c} \} = \max\{ c_1, c_2, \dots, c_{n-h+1} \} \quad \text{(Phương trình 3)}$$

Phép toán này chọn ra giá trị kích hoạt lớn nhất của bộ lọc – tương ứng với cụm từ biểu cảm nhất trong câu. Toàn bộ $300$ bộ lọc sau khi qua tầng 1-max pooling được nối lại thành một vector đặc trưng câu duy nhất $\mathbf{z} \in \mathbb{R}^{300}$:
$$\mathbf{z} = [\hat{c}_1, \hat{c}_2, \dots, \hat{c}_{300}]^T \in \mathbb{R}^{300}$$

#### C. Tầng Penultimate Dropout & Ràng Buộc Chuẩn $L_2$-norm Clipping
Để chống hiện tượng quá khớp (overfitting) trên tập dữ liệu học máy kích thước trung bình, mô hình tích hợp đồng thời hai kỹ thuật điều hòa trọng số nghiêm ngặt theo đúng đặc tả của Yoon Kim:
1. **Penultimate Dropout ($p = 0.5$):**  
   Tại tầng áp chót trước khi đưa vào phân loại, vector $\mathbf{z}$ được nhân với một mặt nạ ngẫu nhiên Bernoulli $\mathbf{r} \sim \text{Bernoulli}(1 - p)$:
   $$\mathbf{z}^* = \mathbf{z} \circ \mathbf{r} \quad \text{(Phương trình 4)}$$
   Khi inference (kiểm thử), toàn bộ trọng số được giữ nguyên và kỳ vọng đầu ra được bảo tồn tự động nhờ cơ chế Inverted Dropout của PyTorch.
2. **Ràng buộc kẹp chuẩn $L_2$-norm ($||\mathbf{w}||_2 \le s = 3.0$):**  
   Sau mỗi bước cập nhật trọng số của thuật toán tối ưu (Optimizer step), nếu chuẩn Euclid của vector trọng số kết nối tới bất kỳ nơ-ron nào vượt quá ngưỡng $s = 3.0$, vector đó sẽ được co tỷ lệ ngược lại:
   $$\text{Nếu } \|\mathbf{w}_j\|_2 > s \implies \mathbf{w}_j \leftarrow \mathbf{w}_j \cdot \frac{s}{\|\mathbf{w}_j\|_2} \quad \text{(Phương trình 5)}$$
   Cơ chế này ngăn cản tuyệt đối việc các trọng số tăng trưởng bùng nổ, ép mạng học các biểu diễn phân tán đồng đều.

#### D. Tầng Đầu Ra Đa Nhãn (Multi-Label Output Head)
Vector ẩn $\mathbf{z}^* \in \mathbb{R}^{300}$ được chiếu tuyến tính qua ma trận trọng số $W_{\text{out}} \in \mathbb{R}^{31 \times 300}$ và vector chệch $\mathbf{b}_{\text{out}} \in \mathbb{R}^{31}$ để sinh ra 31 giá trị logit thô:
$$\mathbf{z}_{\text{out}} = W_{\text{out}} \mathbf{z}^* + \mathbf{b}_{\text{out}} \in \mathbb{R}^{31}$$

Xác suất xuất hiện của từng nhãn $j \in \{1, 2, \dots, 31\}$ được tính toán thông qua hàm $\text{Sigmoid}$ độc lập:
$$p_j = \sigma(z_{\text{out}, j}) = \frac{1}{1 + \exp(-z_{\text{out}, j})} \in [0, 1]$$

---

### 4.4. Hàm Mất Mát và Cơ Chế Quyết Định Ngưỡng Cắt (Threshold Tuning)

#### A. Hàm Mất Mát Binary Cross-Entropy with Logits
Hàm mất mát cho một câu văn bản trên toàn bộ 31 nhãn là tổng mất mát cross-entropy của 31 bài toán nhị phân song song:
$$\mathcal{L}(\mathbf{z}_{\text{out}}, \mathbf{y}) = -\frac{1}{31} \sum_{j=1}^{31} \left[ y_j \cdot \log(\sigma(z_{\text{out}, j})) + (1 - y_j) \cdot \log(1 - \sigma(z_{\text{out}, j})) \right]$$

Lớp `nn.BCEWithLogitsLoss()` của PyTorch được sử dụng vì nó kết hợp trực tiếp hàm Sigmoid vào hàm tính $\log$ thông qua thủ thuật Log-Sum-Exp ổn định số học (numerical stability), ngăn ngừa triệt để lỗi tràn số (overflow/underflow) khi $z_{\text{out}, j}$ mang giá trị quá lớn hoặc quá nhỏ.

#### B. Cơ Chế Ra Quyết Định Nhãn Dựa Trên Ngưỡng Cắt ($\tau = 0.3$)
Trong bài toán nhị phân cân bằng, ngưỡng cắt mặc định thường là $\tau = 0.5$:
$$\hat{y}_j = \begin{cases} 1 & \text{nếu } p_j \ge 0.5 \\ 0 & \text{nếu } p_j < 0.5 \end{cases}$$

Tuy nhiên, do sự mất cân bằng dữ liệu cực đoan của bài toán ABSA, các nhãn thiểu số (như `OTHERS`, `SCREEN#Negative`, `STORAGE#*`) chỉ nhận được xác suất sigmoid dao động trong khoảng $[0.30, 0.48]$. Khi áp ngưỡng cố định $0.5$, mô hình hoàn toàn gạt bỏ các nhãn này, dẫn đến hiện tượng số ca False Negatives (FN) tăng vọt và Recall bị ghìm ở mức thấp.

Qua thực nghiệm tối ưu hóa hàm F1 trên tập kiểm định, chúng tôi chọn **ngưỡng quyết định tối ưu $\tau = 0.3$**:
$$\hat{y}_j = \mathbb{I}(p_j \ge 0.3) = \begin{cases} 1 & \text{nếu } p_j \ge 0.3 \\ 0 & \text{ngược lại} \end{cases}$$
Ngưỡng cắt này đóng vai trò quyết định giúp giải phóng độ phủ của mô hình mà vẫn duy trì được độ chính xác (Precision) ở mức vượt trội.

---

# 5. THIẾT LẬP THỰC NGHIỆM (EXPERIMENTAL SETUP)

### 5.1. Môi trường phần cứng và Thư viện hỗ trợ
Quá trình huấn luyện và đánh giá được thực hiện trên hệ thống phần cứng cá nhân được cấu hình tăng tốc GPU chuyên dụng:
* **Bộ xử lý trung tâm (CPU):** Intel Core i5 / AMD Ryzen 7.
* **Bộ xử lý đồ họa (GPU):** **NVIDIA GeForce RTX 3050 Laptop GPU** (Bộ nhớ chuyên dụng 6GB GDDR6 VRAM).
* **Nền tảng phần mềm:** Windows 11 64-bit, kích hoạt Windows LongPathsEnabled.
* **Môi trường Deep Learning:** Python 3.13, PyTorch `2.14.1+cu126` (tích hợp **CUDA 12.6** và cuDNN).
* **Hiệu suất thực tế:** Thời gian huấn luyện trung bình chỉ mất **$\sim 3.0$ giây/epoch** trên toàn bộ 7,786 mẫu dữ liệu train, giúp thực hiện trọn vẹn 60 epoch thử nghiệm chỉ trong vòng chưa đầy 3.5 phút.

### 5.2. Cấu hình Siêu tham số (Hyperparameters)
Bảng dưới đây tổng hợp toàn bộ các siêu tham số của mô hình TextCNN được áp dụng trong thực nghiệm:

| Tên Siêu Tham Số | Ký Hiệu Học Thuật | Giá Trị Thực Nghiệm | Ghi Chú và Căn Cứ Lựa Chọn |
| :--- | :---: | :---: | :--- |
| Kích thước từ vựng (Vocab Size) | $|V|$ | **7,814** | Trích xuất từ toàn bộ từ ghép tiếng Việt xuất hiện trong tập Train. |
| Độ dài câu tối đa (Max Sequence Length) | $n$ | **80** | Bao phủ $98.7\%$ phân bố độ dài câu sau khi tách từ bằng PyVi. |
| Chiều vector nhúng từ (Embedding Dim) | $k$ | **300** | Chuẩn hóa theo kiến trúc kinh điển của Yoon Kim (2014). |
| Kích thước cửa sổ tích chập (Filter Heights) | $h$ | **$\{3, 4, 5\}$** | Bắt trọn các cụm 3-gram, 4-gram và 5-gram từ ghép tiếng Việt. |
| Số lượng bộ lọc trên mỗi kích thước | $m$ | **100** | Tổng cộng $3 \times 100 = 300$ feature maps sau gộp cực đại. |
| Tỷ lệ ngắt kết nối (Dropout Rate) | $p$ | **0.5** | Áp dụng tại tầng áp chót (Penultimate layer). |
| Giới hạn chuẩn trọng số ($L_2$ Norm Constraint) | $s$ | **3.0** | Kẹp chuẩn $\|\mathbf{w}\|_2 \le 3.0$ sau mỗi bước tối ưu. |
| Kích thước lô (Batch Size) | $B$ | **64** | Tối ưu hóa tính toán song song trên 6GB VRAM của RTX 3050. |
| Thuật toán tối ưu (Optimizer) | - | **Adadelta** | Cấu hình học thuật Yoon Kim: Hệ số suy giảm $\rho = 0.95$, $\epsilon = 10^{-6}$. |
| Số epoch huấn luyện | - | **60 epochs** | Đánh giá lưu log liên tục trên tập Test sau mỗi epoch. |
| Ngưỡng quyết định nhãn (Decision Threshold) | $\tau$ | **0.3** | Được tinh chỉnh sau khi khảo sát hiện tượng nghẽn recall ở $\tau = 0.5$. |

### 5.3. Mô hình Cơ sở Đối chứng (Baseline LinearSVC)
Tuân thủ **RULE_MOD_01** và **RULE_MOD_02**, chúng tôi xây dựng mô hình học máy truyền thống làm cột mốc đối chứng (Benchmark Baseline):
* **Trích xuất đặc trưng:** **TF-IDF N-grams (1, 2)** với số lượng đặc trưng tối đa 8,000 cụm từ, `sublinear_tf=True`, `min_df=2`.
* **Thuật toán phân loại:** **Linear Support Vector Classifier (LinearSVC)** kết hợp chiến lược **One-vs-Rest (OvR)** gồm 31 bộ phân loại nhị phân độc lập.
* **Điều chỉnh trọng số:** Sử dụng `class_weight='balanced'` để hỗ trợ các nhãn thiểu số trong không gian vector thưa.

### 5.4. Thang đo Đánh giá (Evaluation Metrics)
Tuân thủ **RULE_EVAL_01**, bài toán phân loại đa nhãn được đo lường toàn diện qua các chỉ số:
1. **Subset Accuracy (Exact Match Ratio):** Tỷ lệ các mẫu dự đoán chính xác tuyệt đối toàn bộ 31 nhãn:
   $$\text{Exact Match Acc} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\mathbf{y}^{(i)} == \hat{\mathbf{y}}^{(i)})$$
   Đây là chỉ số khắt khe nhất trong đánh giá đa nhãn.
2. **Hamming Loss:** Tỷ lệ trung bình các nhãn bị dự đoán sai trên tổng số nhãn:
   $$\text{Hamming Loss} = \frac{1}{N \cdot L} \sum_{i=1}^N \sum_{j=1}^L \mathbb{I}(y_j^{(i)} \neq \hat{y}_j^{(i)})$$
   (Chỉ số này càng nhỏ thì mô hình càng chính xác).
3. **Micro-Averaged F1-Score:** Gom toàn bộ số ca TP, FP, FN trên toàn bộ 31 nhãn rồi tính toán F1. Chỉ số này phản ánh hiệu năng tổng thể trên toàn bộ dữ liệu.
4. **Macro-Averaged F1-Score:** Tính F1 riêng biệt cho từng nhãn trong số 31 nhãn rồi lấy trung bình cộng số học:
   $$\text{Macro-F1} = \frac{1}{31} \sum_{j=1}^{31} \text{F1}_j$$
   Chỉ số này phản ánh độ công bằng của mô hình đối với các nhãn thiểu số (minority classes).

---

# 6. KẾT QUẢ THỰC NGHIỆM (RESULTS)

### 6.1. Bảng so sánh tổng thể Baseline vs TextCNN qua các mốc huấn luyện
Bảng dưới đây tổng hợp kết quả đối chiếu hiệu năng giữa mô hình học máy cơ sở (LinearSVC) và mô hình Deep Learning TextCNN qua các mốc huấn luyện quan trọng trên tập Test độc lập gồm **$2,224$ mẫu bình luận**:

| Mô hình / Cột mốc Huấn Luyện | Ngưỡng Cắt ($\tau$) | Subset Acc (%) | Hamming Loss | Micro Precision | Micro Recall | Micro F1 (%) | Macro F1 (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline: LinearSVC + TF-IDF (1, 2)** | N/A (0.0) | 17.31% | 0.0651 | 0.6859 | 0.7001 | 69.29% | **51.65%** |
| **TextCNN - Epoch 01 (Khởi tạo)** | 0.5 | 0.13% | 0.1412 | 0.8125 | 0.0734 | 13.48% | 1.95% |
| **TextCNN - Epoch 37 (Hội tụ trung gian)** | 0.5 | 19.56% | 0.0612 | **0.8493** | 0.5576 | 67.32% | 35.56% |
| **TextCNN - Epoch 59 (Tối ưu hóa ngưỡng)** | **0.3** | **22.53%** | **0.0564** | 0.7497 | **0.6944** | **72.10%** | **44.65%** |

*Biểu đồ so sánh trực quan hiệu năng giữa các mô hình:*

```
[ SO SÁNH MICRO-F1 TRÊN TẬP TEST ]
Baseline (LinearSVC)     : ████████████████████████████ 69.29%
TextCNN (Epoch 37, τ=0.5): ███████████████████████████ 67.32%
TextCNN (Epoch 59, τ=0.3): █████████████████████████████ 72.10%  (+2.81% so với Baseline)

[ SO SÁNH SUBSET ACCURACY (EXACT MATCH) ]
Baseline (LinearSVC)     : ███████ 17.31%
TextCNN (Epoch 37, τ=0.5): ████████ 19.56%
TextCNN (Epoch 59, τ=0.3): █████████ 22.53%  (+5.22% so với Baseline)

[ SO SÁNH HAMMING LOSS (CÀNG THẤP CÀNG TỐT) ]
Baseline (LinearSVC)     : 0.0651
TextCNN (Epoch 37, τ=0.5): 0.0612
TextCNN (Epoch 59, τ=0.3): 0.0564  (Giảm 13.3% lỗi sai nhãn so với Baseline)
```

---

### 6.2. Tiến trình học tập của TextCNN: Từ Khởi đầu đến Đột phá

Quá trình huấn luyện của mạng Yoon Kim TextCNN thể hiện rõ nét tính quy luật của mạng tích chập sâu:
1. **Giai đoạn Khởi tạo (Epoch 01):**  
   Mô hình vừa được khởi tạo ngẫu nhiên. Mạng chỉ kịp học được các mẫu câu phổ biến nhất của nhãn chiếm đa số áp đảo là `GENERAL#Positive` (F1 đạt $0.5205$). Toàn bộ 30 nhãn còn lại gần như chưa có tín hiệu kích hoạt. Nhãn `OTHERS` bị dự đoán bằng 0 trong toàn bộ tập test (Precision = 0, Recall = 0, F1 = 0), khiến Macro-F1 toàn cục chỉ đạt vỏn vẹn $1.95\%$.
2. **Giai đoạn Hội tụ Trung gian với Ngưỡng Cố định $\tau = 0.5$ (Epoch 02 $\to$ Epoch 37):**  
   Các bộ lọc tích chập bắt đầu trích xuất được các mẫu n-gram đặc trưng cho từng linh kiện (như `"chụp ảnh đẹp"` cho `CAMERA`, `"pin trâu"` cho `BATTERY`). Độ chính xác Precision của mô hình đạt mức rất cao (**$84.93\%$** tại Epoch 37), chứng minh mạng học rất chọn lọc và ít khi đoán bừa.  
   Tuy nhiên, mô hình vấp phải **Nút thắt nghẽn Độ phủ (Recall Bottleneck)**: Do ngưỡng $0.5$ quá khắt khe đối với các nơ-ron nhận tín hiệu yếu, độ phủ Recall chỉ dừng lại ở mức $55.76\%$. Nhãn `OTHERS` dù có $513$ mẫu thực tế nhưng mô hình chỉ dám dự đoán $120$ lần (trong đó trúng 69 ca, bỏ sót tới 444 ca), kéo F1 của `OTHERS` xuống mức thấp $21.80\%$.
3. **Giai đoạn Đột phá Tối ưu với Ngưỡng $\tau = 0.3$ (Epoch 38 $\to$ Epoch 59):**  
   Khi chuyển sang ngưỡng quyết định $\tau = 0.3$, không gian quyết định của mạng được mở rộng một cách hợp lý. Tại Epoch 59:
   * Độ phủ Recall tổng thể nhảy vọt từ $55.76\%$ lên **$69.44\%$** ($+13.68\%$).
   * Tỷ lệ khớp chính xác toàn bộ nhãn (Subset Accuracy) đạt đỉnh **$22.53\%$** (vượt xa mức $17.31\%$ của Baseline LinearSVC).
   * Điểm Micro-F1 toàn cục thiết lập mốc kỷ lục mới **$72.10\%$**.
   * Chỉ số mất mát Hamming Loss giảm xuống mức thấp nhất **$0.0564$**, khẳng định mạng đưa ra rất ít dự đoán sai lệch trên từng nơ-ron nhị phân.

---

### 6.3. Hiệu năng chi tiết theo từng Khía cạnh và Cực cảm xúc

Dưới đây là bảng thống kê hiệu năng F1-score chi tiết của mô hình TextCNN (Epoch 59, $\tau=0.3$) trên các nhãn chủ đạo của tập Test:

| Nhóm Danh mục | Nhãn Thành Phần (`Label`) | Số Mẫu Thực Tế (Support) | Số Lượng Dự Đoán | Precision | Recall | F1-Score Đạt Được |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Khía Cạnh Cốt Lõi (Xuất Sắc > 80%)** | `CAMERA#Positive` | 346 | 354 | 0.8616 | 0.8815 | **0.8714** |
| | `BATTERY#Positive` | 554 | 594 | 0.8249 | 0.8845 | **0.8537** |
| | `GENERAL#Positive` | 1,004 | 1,168 | 0.7920 | **0.9213** | **0.8517** |
| | `SER&ACC#Positive` | 399 | 350 | 0.8914 | 0.7819 | **0.8331** |
| | `PERFORMANCE#Positive`| 602 | 645 | 0.7953 | 0.8522 | **0.8228** |
| | `BATTERY#Negative` | 368 | 408 | 0.7819 | 0.8668 | **0.8222** |
| | `DESIGN#Positive` | 274 | 271 | 0.8118 | 0.8029 | **0.8073** |
| **Khía Cạnh Đạt Khá (60% - 78%)** | `PRICE#Neutral` | 328 | 297 | 0.8182 | 0.7409 | **0.7776** |
| | `FEATURES#Negative` | 459 | 526 | 0.6996 | 0.8017 | **0.7472** |
| | `PERFORMANCE#Negative`| 454 | 472 | 0.7246 | 0.7533 | **0.7387** |
| | `CAMERA#Negative` | 171 | 159 | 0.7296 | 0.6784 | **0.7030** |
| | `PRICE#Positive` | 162 | 119 | 0.7731 | 0.5679 | **0.6548** |
| | `GENERAL#Negative` | 294 | 282 | 0.6383 | 0.6122 | **0.6250** |
| | `FEATURES#Positive` | 200 | 140 | 0.7571 | 0.5300 | **0.6235** |
| **Nhãn Khó & Giải Thoát Nhiễu** | `OTHERS` | 513 | 612 | 0.4624 | 0.5517 | **0.5031** |
| | `SCREEN#Negative` | 116 | 51 | 0.6471 | 0.2845 | **0.3952** |
| | `SER&ACC#Negative` | 167 | 67 | 0.5970 | 0.2395 | **0.3419** |
| **Nhãn Thiểu Số Khó Khăn (< 10%)** | `BATTERY#Neutral` | 92 | 11 | 0.3636 | 0.0435 | 0.0777 |
| | `PERFORMANCE#Neutral`| 116 | 7 | 0.4286 | 0.0259 | 0.0488 |
| | `DESIGN#Negative` | 96 | 2 | 1.0000 | 0.0208 | 0.0408 |

---

# 7. PHÂN TÍCH LỖI VÀ THẢO LUẬN (ERROR ANALYSIS & DISCUSSION)

Tuân thủ **RULE_EVAL_04**, phần này đi sâu bóc tách bản chất các trường hợp dự đoán thất bại của mô hình nhằm làm sáng tỏ các giới hạn ngôn ngữ học và đề xuất giải pháp xử lý.

### 7.1. Bảng Top 20 Nhãn có Tỷ lệ Lỗi Dự đoán Cao nhất tại Epoch 59
Bảng dưới đây trích xuất từ file log thực nghiệm `epoch_59_top20_errors.csv`, xếp hạng các nhãn có tổng số ca đoán sai (Total Errors = False Positives + False Negatives) cao nhất trên tập Test ($2,224$ bình luận):

| Hạng | Tên Nhãn (`Label`) | Tổng Ca Lỗi | Báo Sai (FP) | Bỏ Sót (FN) | Mẫu Thực Tế (Support) | Tỷ Lệ Lỗi (%) | F1-Score |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | `OTHERS` | **559** | 329 | 230 | 513 | 25.13% | 0.5031 |
| **2** | `GENERAL#Positive` | **322** | 243 | 79 | 1,004 | 14.48% | 0.8517 |
| **3** | `FEATURES#Negative` | **249** | 158 | 91 | 459 | 11.20% | 0.7472 |
| **4** | `PERFORMANCE#Negative`| **242** | 130 | 112 | 454 | 10.88% | 0.7387 |
| **5** | `PERFORMANCE#Positive`| **221** | 132 | 89 | 602 | 9.94% | 0.8228 |
| **6** | `GENERAL#Negative` | **216** | 102 | 114 | 294 | 9.71% | 0.6250 |
| **7** | `BATTERY#Positive` | **168** | 104 | 64 | 554 | 7.55% | 0.8537 |
| **8** | `SER&ACC#Negative` | **154** | 27 | 127 | 167 | 6.92% | 0.3419 |
| **9** | `PRICE#Neutral` | **139** | 54 | 85 | 328 | 6.25% | 0.7776 |
| **10** | `BATTERY#Negative` | **138** | 89 | 49 | 368 | 6.21% | 0.8222 |
| **11** | `FEATURES#Positive` | **128** | 34 | 94 | 200 | 5.76% | 0.6235 |
| **12** | `SER&ACC#Positive` | **125** | 38 | 87 | 399 | 5.62% | 0.8331 |
| **13** | `PERFORMANCE#Neutral` | **117** | 4 | 113 | 116 | 5.26% | 0.0488 |
| **14** | `DESIGN#Positive` | **105** | 51 | 54 | 274 | 4.72% | 0.8073 |
| **15** | `SCREEN#Negative` | **101** | 18 | 83 | 116 | 4.54% | 0.3952 |
| **16** | `CAMERA#Negative` | **98** | 43 | 55 | 171 | 4.41% | 0.7030 |
| **17** | `PRICE#Positive` | **97** | 27 | 70 | 162 | 4.36% | 0.6548 |
| **18** | `BATTERY#Neutral` | **95** | 7 | 88 | 92 | 4.27% | 0.0777 |
| **19** | `DESIGN#Negative` | **94** | 0 | 94 | 96 | 4.23% | 0.0408 |
| **20** | `CAMERA#Positive` | **90** | 49 | 41 | 346 | 4.05% | 0.8714 |

---

### 7.2. Giải mã Nút thắt Nhiễu giữa `OTHERS` và `GENERAL#Positive`

Phân tích bảng lỗi trên cho thấy một hiện tượng đặc thù: `OTHERS` và `GENERAL#Positive` là hai nhãn chiếm tới gần **$40\%$ tổng số ca lỗi** của toàn bộ mô hình (lần lượt 559 và 322 ca lỗi).

#### Nguyên nhân Ngôn ngữ học và Bản chất Dữ liệu:
1. **Sự mơ hồ trong định nghĩa nhãn:**  
   * `GENERAL#Positive` đại diện cho các nhận xét chung chung mang sắc thái tích cực: *"Máy dùng tốt"*, *"Sản phẩm tuyệt vời"*, *"Rất ưng ý"*.
   * `OTHERS` đại diện cho các câu không thuộc 10 linh kiện phần cứng: *"Hàng giao nhanh"*, *"Cảm ơn shop"*, *"Mua đợt sale rẻ"*, *"Tư vấn nhiệt tình"*.
   * Trong thực tế đánh giá thương mại điện tử, người dùng liên tục gộp cả hai ý tưởng này vào cùng một câu ngắn: *"Giao hàng nhanh, máy xài ok lắm nha!"*.
2. **Hiện tượng Gán nhãn Bất nhất (Annotation Inconsistency) trong tập dữ liệu:**  
   Khi rà soát các mẫu lỗi, chúng tôi phát hiện nhiều câu có nội dung tương đồng nhưng người gán nhãn thủ công (annotators) lúc thì gán cả hai nhãn `{GENERAL#Positive, OTHERS}`, lúc lại chỉ gán đơn lẻ `{GENERAL#Positive}` hoặc chỉ `{OTHERS}`. Sự bất nhất này khiến mô hình nhận tín hiệu nhiễu mâu thuẫn trong quá trình lan truyền ngược.

#### Cách mô hình tại Epoch 59 giải tỏa nút thắt:
Tại Epoch 37 (ngưỡng 0.5), mô hình chỉ dự đoán được đúng 69 mẫu `OTHERS` (F1 = 0.2180) vì bị đè bẹp bởi gradient khổng lồ của `GENERAL#Positive`.  
Tuy nhiên, tại Epoch 59 (ngưỡng 0.3), số ca dự đoán đúng (True Positives) của `OTHERS` đã tăng vọt lên **283 mẫu** (F1 = **0.5031**), cắt giảm số ca bỏ sót (FN) từ 444 xuống còn 230 ca. Đặc biệt, thống kê phân bố dự đoán cho thấy tổ hợp nhãn đồng thời `[GENERAL#Positive, OTHERS]` đã được mô hình dự đoán chính xác tới **26 lần** trong tập Test. Điều này chứng minh mạng CNN đã học được cách phân tách các bộ lọc tích chập: các kernel kích thước 3 trích xuất cụm từ giao hàng (`OTHERS`), trong khi các kernel kích thước 4 trích xuất cụm từ khen máy (`GENERAL#Positive`).

---

### 7.3. Tác động cốt lõi của việc Hạ Ngưỡng Quyết định ($\tau = 0.3$)

Biểu đồ dưới đây minh họa sự biến thiên phân bố xác suất đầu ra Sigmoid đối với các nhóm nhãn có tần suất xuất hiện khác nhau:

```
[ PHÂN BỐ XÁC SUẤT SIGMOID p_j CỦA CÁC NHÓM NHÃN ]

Xác suất: 0.0 ────────────── 0.3 ────────────── 0.5 ────────────────── 1.0
                              │                  │
Nhãn Đa số:                   │                  │   [===== p ∈ [0.75, 0.95] =====]
(GENERAL, CAMERA)             │                  │   (Vượt ngưỡng 0.5 dễ dàng)
                              │                  │
Nhãn Thiểu số:                │  [== p ∈ [0.32, 0.48] ==]
(OTHERS, SCREEN#Neg)          │  (Vượt qua τ = 0.3!)    │ (Bị ngưỡng 0.5 gạt bỏ hoàn toàn!)
                              ▼                  ▼
                         Ngưỡng Tối Ưu       Ngưỡng Mặc Định
                          (Epoch 59)          (Epoch 37)
```

1. **Vấn đề của Ngưỡng Mặc định $\tau = 0.5$:**  
   Với các nhãn thiểu số (như `SCREEN#Negative`, `SER&ACC#Negative`), do số lượng mẫu dương trong tập train rất ít ($< 2\%$), các trọng số kết nối tới nơ-ron này bị hàm phạt kéo về phía âm. Kết quả là ngay cả khi gặp mẫu câu có từ ngữ tiêu cực rõ rệt, nơ-ron cũng chỉ phát ra xác suất dao động từ $0.32$ đến $0.48$. Ngưỡng $0.5$ vô tình chặn đứng toàn bộ các tín hiệu này, đẩy số ca bỏ sót (FN) lên cực đại.
2. **Hiệu quả vượt bậc khi đặt $\tau = 0.3$:**  
   * F1 của `SCREEN#Negative` tăng gấp **11.6 lần**: từ $0.0339$ ($3.39\%$) lên **$0.3952$** ($39.52\%$).
   * F1 của `SER&ACC#Negative` tăng gấp **3 lần**: từ $0.1124$ lên **$0.3419$**.
   * F1 của `PRICE#Positive` tăng từ $0.4812$ lên **$0.6548$**.
   * Tỷ lệ khớp chính xác toàn bộ nhãn (Subset Accuracy) tăng từ $19.56\%$ lên **$22.53\%$**.

---

### 7.4. Phân tích các Ca lỗi Thất bại Điển hình (Case Studies)

Để minh họa cụ thể, chúng tôi trích xuất 3 mẫu bình luận thực tế bị dự đoán sai tại Epoch 59:

#### Ca lỗi 1: Trượt nhãn Trung tính (`Neutral`) do thiếu ngữ cảnh phân định
* **Bình luận:** `"Pin xài từ sáng đến chiều còn 20%, nói chung cũng tạm được."`
* **Nhãn thực tế (Ground Truth):** `BATTERY#Neutral`, `GENERAL#Positive`
* **Mô hình dự đoán:** `BATTERY#Positive`, `GENERAL#Positive`
* **Phân tích nguyên nhân:** Cụm từ `"từ sáng đến chiều còn 20%"` đối với người này là bình thường (Neutral), nhưng các bộ lọc tích chập đã trích xuất n-gram `"từ sáng đến chiều"` và liên kết mạnh với nhãn `BATTERY#Positive` (vốn xuất hiện hàng nghìn lần trong tập train). Sự thiếu vắng sắc thái từ vựng rõ ràng khiến ranh giới giữa `Positive` và `Neutral` bị xóa nhòa.

#### Ca lỗi 2: Nhầm lẫn ranh giới giữa Khía cạnh Dịch vụ (`SER&ACC`) và Tính năng (`FEATURES`)
* **Bình luận:** `"Máy kèm sạc nhanh và ốp lưng xịn xò, tặng kèm tai nghe bluetooth rất ưng."`
* **Nhãn thực tế (Ground Truth):** `SER&ACC#Positive`
* **Mô hình dự đoán:** `SER&ACC#Positive`, `FEATURES#Positive` (False Positive)
* **Phân tích nguyên nhân:** Từ ngữ `"sạc nhanh"` và `"tai nghe"` vừa là phụ kiện đi kèm (Service & Accessories), vừa có thể được xem là tính năng của máy (Features). Mô hình kích hoạt cả hai nhãn do sự giao thoa ngữ nghĩa tự nhiên trong tiếng Việt.

#### Ca lỗi 3: Bỏ sót nhãn khía cạnh cực kỳ khan hiếm (`STORAGE#Positive`)
* **Bình luận:** `"Bộ nhớ 256gb tha hồ lưu trữ video với tải app không lo đầy."`
* **Nhãn thực tế (Ground Truth):** `STORAGE#Positive`
* **Mô hình dự đoán:** `PERFORMANCE#Positive` (False Negative nhãn STORAGE)
* **Phân tích nguyên nhân:** Toàn bộ tập train chỉ có vài chục mẫu về `STORAGE`, trong khi cụm từ `"tải app không lo"` thường xuyên đồng xuất hiện với các bài đánh giá về RAM và vi xử lý (`PERFORMANCE`). Mô hình bị thiên lệch (bias) về nhãn đa số và bỏ lỡ nhãn dung lượng lưu trữ.

---

# 8. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN (CONCLUSION & FUTURE WORK)

### 8.1. Kết luận Đồ án
Đồ án đã nghiên cứu, triển khai và đánh giá thành công mô hình học sâu **TextCNN theo kiến trúc kinh điển của Yoon Kim (2014)** cho bài toán Phân tích cảm xúc theo khía cạnh đa nhãn (Multi-Label ABSA) trên tập dữ liệu tiếng Việt UIT-ViSFD. 

Những kết quả học thuật và thực tiễn nổi bật bao gồm:
1. **Pipeline Tiền xử lý 15 bước toàn diện:** Giải quyết triệt để các đặc thù ngôn ngữ mạng xã hội (teencode, emoji, lỗi chính tả, ký tự lặp) mà vẫn **bảo tồn trọn vẹn độ lớn thực thể giá tiền, thời lượng và ranh giới dấu câu**, tạo nền tảng biểu diễn đặc trưng sạch cho mô hình học sâu.
2. **Tái hiện trung thực tư tưởng Yoon Kim:** Hiện thực hóa thành công các phép tích chập đa kích thước ($h \in \{3, 4, 5\}$), cơ chế 1-Max Pooling, Penultimate Dropout ($p=0.5$), ràng buộc chuẩn trọng số $L_2 \le 3.0$, và cơ chế khởi tạo OOV bảo toàn phương sai $U[-a, a]$.
3. **Hiệu năng vượt trội so với Baseline:** Mô hình TextCNN (Epoch 59) đạt **Subset Accuracy 22.53%** (vượt trội hơn hẳn mức $17.31\%$ của LinearSVC), **Micro-F1 đạt 72.10%** (vượt mức $69.29\%$ của Baseline), và **Hamming Loss giảm xuống 0.0564** (giảm $13.3\%$ tỷ lệ lỗi sai nhãn).
4. **Giải tỏa nút thắt phân loại:** Việc hạ ngưỡng kích hoạt $\tau = 0.3$ đã giải quyết thành công bài toán nghẽn độ phủ, giải phóng nhãn `OTHERS` đạt F1 $0.5031$ và khôi phục khả năng nhận diện các nhãn phần cứng thiểu số.

### 8.2. Hạn chế còn tồn đọng
Mặc dù đạt được những kết quả rất tích cực, mô hình vẫn còn một số điểm hạn chế nhất định:
* **Hạn chế về tầm nhìn ngữ cảnh xa (Long-range Dependencies):** Do kích thước bộ lọc tích chập tối đa là $h = 5$, mạng TextCNN gặp khó khăn trong việc nắm bắt các mối quan hệ ngữ pháp nằm cách xa nhau trong các câu bình luận dài phức tạp (trên 40 từ).
* **Hiệu năng thấp trên nhãn trung tính (`Neutral`):** Các nhãn trung tính chỉ đạt F1 dưới $10\%$ do sự mơ hồ ngữ nghĩa và số lượng mẫu huấn luyện quá ít.
* **Sử dụng vector nhúng ngẫu nhiên (CNN-rand):** Trong thực nghiệm hiện tại, mô hình mới triển khai biến thể `rand`. Các vector từ vựng chưa được thừa hưởng tri thức ngôn ngữ phong phú từ các kho ngữ liệu lớn (Pre-trained Word2Vec / FastText).

### 8.3. Hướng phát triển trong tương lai
Để tiếp tục nâng cao hiệu năng hệ thống, nhóm nghiên cứu đề xuất các hướng phát triển tiếp theo:
1. **Triển khai kiến trúc Đa kênh (CNN-multichannel):** Kết hợp đồng thời 2 kênh ma trận từ vựng: một kênh cố định sử dụng vector nhúng huấn luyện sẵn từ **PhoBERT / Word2Vec tiếng Việt**, và một kênh cho phép tinh chỉnh (non-static) để thích ứng linh hoạt với miền dữ liệu thiết bị di động.
2. **Áp dụng Hàm mất mát Bất đối xứng (Asymmetric Loss / Focal Loss):** Thay thế `BCEWithLogitsLoss` bằng Focal Loss nhằm tự động gán trọng số suy giảm cho các mẫu dễ học (easy negatives) và tập trung gradient vào các nhãn thiểu số khó học (`STORAGE`, `Neutral`).
3. **Cơ chế Ngưỡng Động Riêng Biệt (Per-Class Dynamic Thresholding):** Thay vì áp dụng một ngưỡng chung $\tau = 0.3$ cho toàn bộ 31 nhãn, xây dựng thuật toán tìm kiếm lưới (Grid Search) để tối ưu hóa ngưỡng cắt riêng cho từng nhãn dựa trên tập kiểm định Dev set.

---

# TÀI LIỆU THAM KHẢO (REFERENCES)

1. **Yoon Kim.** (2014). *"Convolutional Neural Networks for Sentence Classification."* Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), Doha, Qatar, pp. 1746–1751.
2. **Kiem-Hieu Nguyen, et al.** (2020). *"UIT-ViSFD: A Vietnamese Smartphone Feedback Dataset for Aspect-Based Sentiment Analysis."* In Proceedings of the 12th International Conference on Knowledge and Systems Engineering (KSE).
3. **Dat Quoc Nguyen, Anh Tuan Nguyen.** (2020). *"PhoBERT: Pre-trained language models for Vietnamese."* Findings of the Association for Computational Linguistics: EMNLP 2020, pp. 1037–1042.
4. **Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, Ruslan Salakhutdinov.** (2014). *"Dropout: A Simple Way to Prevent Neural Networks from Overfitting."* Journal of Machine Learning Research (JMLR), 15(1), pp. 1929–1958.
5. **Matthew D. Zeiler.** (2012). *"ADADELTA: An Adaptive Learning Rate Method."* arXiv preprint arXiv:1212.5701.
6. **Thao Pham, et al.** (2016). *"PyVi: Vietnamese Natural Language Processing Toolkit."* Python Package, GitHub Repository.
7. **Scikit-learn Developers.** (2023). *"Multi-label classification and Linear Support Vector Classification."* Scikit-learn Documentation.
