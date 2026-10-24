# Đặc Tả Kỹ Thuật Kiến Trúc Mạng TextCNN (`cnn.py`)

> **Thư mục:** `Yoon Kim test/`  
> **Mã nguồn hiện thực:** [`cnn.py`](file:///F:/HK%20261/NLP/Train/Yoon%20Kim%20test/cnn.py) & [`deadpool.py`](file:///F:/HK%20261/NLP/Train/Yoon%20Kim%20test/deadpool.py)  
> **Nhiệm vụ:** Phân loại cảm xúc dựa trên khía cạnh đa nhãn (Aspect-Based Sentiment Analysis - ABSA) trên tập dữ liệu UIT-ViSFD (31 nhãn nhị phân độc lập).  
> **Tham chiếu học thuật:** Yoon Kim (EMNLP 2014) — *"Convolutional Neural Networks for Sentence Classification"*.

---

## 1. Sơ Đồ Khối Tổng Thể (Architecture Flowchart)

```
                       [ Câu văn bản đầu vào: n từ ]
                                     │
                                     ▼
                [ Tầng Embedding: Ma trận X ∈ ℝ^(n × k) ]
                           (k = 300, n = 80)
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         ▼                           ▼                           ▼
[ Conv2D: h = 3 ]           [ Conv2D: h = 4 ]           [ Conv2D: h = 5 ]
 (100 feature maps)          (100 feature maps)          (100 feature maps)
         │                           │                           │
         ▼                           ▼                           ▼
 [ Kích hoạt: ReLU ]         [ Kích hoạt: ReLU ]         [ Kích hoạt: ReLU ]
         │                           │                           │
         ▼                           ▼                           ▼
[ 1-Max Pooling ]           [ 1-Max Pooling ]           [ 1-Max Pooling ]
  (Vector: 100d)              (Vector: 100d)              (Vector: 100d)
         │                           │                           │
         └───────────────────────────┼───────────────────────────┘
                                     ▼
                  [ Vector đặc trưng z ∈ ℝ^300 ]
                                     │
                                     ▼
                  [ Penultimate Dropout: p = 0.5 ]
                                     │
                                     ▼
           [ Linear Fully Connected: 300 -> 31 logits (z_out) ]
                  (Kèm ràng buộc ||w||_2 ≤ 3.0)
                                     │
                                     ▼
                [ Hàm kích hoạt Sigmoid: p_j = σ(z_j) ]
                                     │
                                     ▼
          [ Quyết định nhãn theo Threshold: p_j ≥ 0.3 -> 1 ]
```

---

## 2. Chi Tiết Các Tầng & Phương Trình Toán Học (Mathematical Formulations)

### 2.1. Tầng Biểu Diễn Đầu Vào & Ma Trận Câu (Input Representation & Sentence Embedding Matrix)

Tầng Embedding là cầu nối chuyển đổi chuỗi ký tự ngôn ngữ tự nhiên thành một đối tượng hình học nhiều chiều $X \in \mathbb{R}^{n \times k}$ để các phép toán vi tích phân và tích chập có thể thực thi.

```
 [ Văn bản thô ] ──> [ Tiền xử lý Clean.md ] ──> [ Chuỗi token ] ──> [ Vector Chỉ số nguyên i ∈ ℤ^80 ]
                                                                                   │
                                                                   Tra cứu Lookup Table E ∈ ℝ^(|V| × 300)
                                                                                   ▼
                                                             [ Ma trận Câu X ∈ ℝ^(80 × 300) ]
```

---

#### A. Bản chất cấu trúc của Ma trận Câu $X \in \mathbb{R}^{n \times k}$
Giả sử câu văn bản sau bước tiền xử lý gồm $n = 80$ từ (được padding hoặc truncate):
$$\mathbf{s} = (w_1, w_2, \dots, w_n)$$

Mỗi từ $w_i$ được biểu diễn bởi một vector hàng đặc trưng phân tán (dense vector) trong không gian Euclid $k$ chiều ($k = 300$):
$$\mathbf{x}_i = [x_{i, 1}, x_{i, 2}, \dots, x_{i, 300}] \in \mathbb{R}^{300}$$

Theo Yoon Kim (2014), toàn bộ câu văn bản được hình thành bằng phép ghép nối (concatenation operator $\oplus$) các vector từ theo chiều dọc:
$$X = \mathbf{x}_{1:n} = \mathbf{x}_1 \oplus \mathbf{x}_2 \oplus \dots \oplus \mathbf{x}_n = \begin{bmatrix}
x_{1, 1} & x_{1, 2} & \dots & x_{1, 300} \\
x_{2, 1} & x_{2, 2} & \dots & x_{2, 300} \\
\vdots & \vdots & \ddots & \vdots \\
x_{80, 1} & x_{80, 2} & \dots & x_{80, 300}
\end{bmatrix} \in \mathbb{R}^{80 \times 300}$$

Ma trận $X$ có thể xem như một **"bức ảnh đơn sắc"** có chiều cao bằng độ dài câu ($H = 80$), chiều rộng bằng số chiều vector từ ($W = 300$) và số kênh màu bằng 1 ($C = 1$).

---

#### B. Quy trình Biến đổi Toán học từ Văn bản thành Ma trận $X$

Quá trình này trải qua 3 bước tuần tự trong [`deadpool.py`](file:///F:/HK%20261/NLP/Train/Yoon%20Kim%20test/deadpool.py):

1. **Bước 1: Tokenization & Word Segmentation:**
   * Văn bản thô được làm sạch qua 15 bước của `Clean.md` và phân đoạn từ ghép tiếng Việt bằng `pyvi`:
     $$\text{"Pin trâu chụp ảnh đẹp"} \xrightarrow{\text{Clean.md}} [\text{"pin"}, \text{"trâu"}, \text{"chụp\_ảnh"}, \text{"đẹp"}]$$
   * Mỗi cụm từ ghép có gạch dưới (như `chụp_ảnh`, `thế_giới`, `ổn_định`) được xem là **một token duy nhất**.

2. **Bước 2: Ánh xạ Chỉ số Từ điển (Vocabulary Indexing) & Padding:**
   * Từ điển $V$ gồm $|V| = 7,814$ từ vựng.
   * Ánh xạ từng từ $w_i \to \text{index } \text{id}_i \in \{0, 1, \dots, |V|-1\}$.
   * **Quy chuẩn Padding / Truncation về độ dài cố định $n = 80$:**
     - Nếu câu có $m < 80$ từ: Bổ sung $(80 - m)$ token `<pad>` (index $0$) vào cuối câu.
     - Nếu câu có $m > 80$ từ: Cắt lấy đúng 80 từ đầu tiên.
     $$\mathbf{i} = [\text{id}_1, \text{id}_2, \dots, \text{id}_m, \underbrace{0, 0, \dots, 0}_{80 - m \text{ số 0}}] \in \mathbb{Z}^{80}$$

3. **Bước 3: Phép toán Tra cứu Bảng (Lookup Table) qua Phép nhân Ma trận:**
   * Về mặt toán học hình thức: Mỗi chỉ số $\text{id}_t$ tương đương với một vector One-Hot $\mathbf{e}_{\text{id}_t} \in \{0, 1\}^{|V|}$ (chỉ có duy nhất vị trí $\text{id}_t$ bằng $1$, toàn bộ vị trí khác bằng $0$).
   * Cho ma trận trọng số nhúng toàn cục $E \in \mathbb{R}^{|V| \times 300}$, vector từ $\mathbf{x}_t$ là kết quả của phép chiếu:
     $$\mathbf{x}_t = \mathbf{e}_{\text{id}_t}^{\top} E \in \mathbb{R}^{300}$$
   * Trong PyTorch, thao tác này được tối ưu hóa thành phép lập chỉ mục mảng $O(1)$:
     ```python
     self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=300, padding_idx=0)
     ```

---

#### C. Quy tắc Đặc biệt dành cho Token `<pad>` (Index 0)
* Token `<pad>` (đệm câu) luôn được cố định bằng **vector toàn số 0**:
  $$E[0, :] = \mathbf{0} = [0, 0, \dots, 0] \in \mathbb{R}^{300}$$
* **Ý nghĩa toán học đối với mạng tích chập:**
  Khi một bộ lọc tích chập $\mathbf{w} \in \mathbb{R}^{h \times 300}$ trượt qua vùng chứa các token `<pad>`:
  $$\mathbf{w} \cdot \mathbf{0} + b = 0 + b = b$$
  Do bias $b$ thường mang giá trị âm hoặc rất nhỏ trong quá trình huấn luyện, sau khi đi qua hàm phi tuyến $\text{ReLU}(b) = \max(0, b) = 0$.
  $\implies$ Các vị trí đệm vô nghĩa sẽ cho ra kích hoạt bằng 0, **không thể cạnh tranh với các từ thực tế** trong phép toán 1-Max Pooling ($\hat{c} = \max\{c\}$).

---

#### D. Công thức Khởi tạo Từ Lạ (OOV) Bảo toàn Phương sai (Yoon Kim Section 4.3)

Đối với các từ xuất hiện trong tập dữ liệu nhưng **không có trong bộ vector pre-trained** (Out-Of-Vocabulary - OOV), Yoon Kim chỉ ra rằng việc khởi tạo ngẫu nhiên tùy tiện (như Gauss chuẩn $N(0, 1)$) sẽ làm sai lệch tương quan độ lớn (magnitude) giữa từ quen và từ lạ.

Tác giả đề xuất lấy mẫu từ phân phối đều $U[-a, a]$ sao cho:
$$\text{Var}(U[-a, a]) = \sigma_{\text{pretrained}}^2$$

**Chứng minh công thức xác định cận $a$:**
1. Với biến ngẫu nhiên $X \sim U[-a, a]$, kỳ vọng $\mathbb{E}[X] = 0$.
2. Phương sai của phân phối đều trên đoạn $[\alpha, \beta]$ là $\text{Var} = \frac{(\beta - \alpha)^2}{12}$.
3. Thay $\alpha = -a$ và $\beta = a$:
   $$\text{Var}(X) = \frac{(a - (-a))^2}{12} = \frac{(2a)^2}{12} = \frac{4a^2}{12} = \frac{a^2}{3}$$
4. Đặt phương sai này bằng phương sai thực tế $\sigma_{\text{pretrained}}^2$ của tập vector pre-trained:
   $$\frac{a^2}{3} = \sigma_{\text{pretrained}}^2 \iff a^2 = 3\sigma_{\text{pretrained}}^2 \implies a = \sqrt{3} \cdot \sigma_{\text{pretrained}}$$

Hàm `load_pretrained_vectors()` trong `deadpool.py` tính toán phương sai mẫu của các vector pre-trained nạp vào và khởi tạo các từ OOV chính xác theo đoạn $[-\sqrt{3}\sigma, \sqrt{3}\sigma]$. Nếu không có pre-trained vector (mô hình `rand`), ngưỡng mặc định là $a = 0.25$ theo Collobert et al. (2011).

---

#### E. Cơ chế 4 Biến thể Embedding của Yoon Kim (Section 3.3)

| Biến thể | Trạng thái Ma trận $E$ | Cơ chế hoạt động & Ý nghĩa học thuật |
| :--- | :--- | :--- |
| **`CNN-rand`** | Khởi tạo ngẫu nhiên $U[-0.25, 0.25]$ | Toàn bộ ma trận $E$ được cập nhật gradient. Đóng vai trò làm Baseline đối chuẩn. |
| **`CNN-static`** | Nạp Pre-trained, `freeze=True` | Ma trận $E$ bị khóa cố định. Mạng chỉ học các bộ lọc tích chập và tầng phân loại, giữ nguyên vẹn tri thức từ điển gốc. |
| **`CNN-non-static`** | Nạp Pre-trained, `freeze=False` | Các vector từ bắt đầu từ tri thức gốc, nhưng được phép di chuyển nhẹ (fine-tune) theo hướng tối ưu cho phân loại cảm xúc điện thoại. |
| **`CNN-multichannel`** | Sử dụng 2 ma trận $E_{\text{static}}$ và $E_{\text{non-static}}$ | Câu sinh ra 2 ma trận $X^{(1)}$ và $X^{(2)}$. Cả hai cùng đi qua một bộ lọc và cộng gộp kết quả: $c_i = f(\mathbf{w} \cdot \mathbf{x}^{(1)}_{i:i+h-1} + \mathbf{w} \cdot \mathbf{x}^{(2)}_{i:i+h-1} + b)$. Kênh static giữ vai trò mỏ neo chống trôi ngữ nghĩa, kênh non-static tự do học đặc trưng mới. |

---

#### F. Mở rộng Chiều Tensor cho Tầng Tích Chập 2D (Batch Dimension Handling)

Trong PyTorch, toán tử `nn.Conv2d` yêu cầu đầu vào dạng tensor 4 chiều:
$$(\text{Batch Size } B,\, \text{Channels } C,\, \text{Height } H,\, \text{Width } W)$$

1. Khi đưa batch gồm $B = 50$ câu vào tầng Embedding:
   $$\mathbf{i}_{\text{batch}} \in \mathbb{Z}^{50 \times 80} \xrightarrow{E} X_{\text{batch}} \in \mathbb{R}^{50 \times 80 \times 300}$$
2. Để tích chập 2D có thể trượt theo chiều cao (chiều câu), tensor được thêm một trục channel $C = 1$ thông qua phương thức `.unsqueeze(1)`:
   $$X_{\text{conv\_in}} = X_{\text{batch}}.\text{unsqueeze}(1) \in \mathbb{R}^{50 \times 1 \times 80 \times 300}$$
   *(Với `multichannel`, tensor được ghép qua `torch.stack` thành shape $(50, 2, 80, 300)$ với $C = 2$)*.

---

### 2.2. Tầng Tích Chập Đa Cửa Sổ (Multi-Filter Convolutional Layer)
Một bộ lọc tích chập (filter) $\mathbf{w} \in \mathbb{R}^{h \times k}$ quét qua một cửa sổ trượt gồm $h$ từ liên tiếp $\mathbf{x}_{i:i+h-1} \in \mathbb{R}^{h \times k}$ để trích xuất đặc trưng cụm từ (n-gram).

Đặc trưng $c_i$ được tạo ra tại vị trí $i$ theo công thức:
$$c_i = f\left( \mathbf{w} \cdot \mathbf{x}_{i:i+h-1} + b \right)$$

Trong đó:
* $\mathbf{w} \cdot \mathbf{x}_{i:i+h-1} = \sum_{u=1}^{h} \sum_{v=1}^{k} w_{u, v} \cdot x_{i+u-1, v}$ (tích vô hướng ma trận Frobenius).
* $b \in \mathbb{R}$ là hệ số điều chỉnh (bias).
* $f(\cdot)$ là hàm kích hoạt phi tuyến **ReLU (Rectified Linear Unit)** theo Yoon Kim Section 3.1:
  $$f(x) = \max(0, x)$$

*(Trong chế độ `multichannel`, bộ lọc tác động lên cả 2 kênh và cộng dồn: $c_i = f(\mathbf{w} \cdot \mathbf{x}^{(1)}_{i:i+h-1} + \mathbf{w} \cdot \mathbf{x}^{(2)}_{i:i+h-1} + b)$).*

#### Chiều dài của Feature Map:
Bộ lọc trượt qua toàn bộ các vị trí hợp lệ $i \in \{1, 2, \dots, n - h + 1\}$ để sinh ra một bản đồ đặc trưng (feature map) $\mathbf{c} \in \mathbb{R}^{n - h + 1}$:
$$\mathbf{c} = [c_1, c_2, \dots, c_{n-h+1}]$$

Với độ dài câu $n = 80$, số lượng phần tử của feature map ứng với từng cửa sổ $h$ là:
* Với $h = 3$: $L_3 = 80 - 3 + 1 = 78$
* Với $h = 4$: $L_4 = 80 - 4 + 1 = 77$
* Với $h = 5$: $L_5 = 80 - 5 + 1 = 76$

---

### 2.3. Tầng 1-Max Pooling (Max-Over-Time Pooling)
Để vector hóa câu về kích thước cố định bất kể độ dài văn bản và rút trích tín hiệu n-gram quan trọng nhất, phép toán lấy cực đại theo thời gian (Collobert et al., 2011) được áp dụng trên mỗi feature map:
$$\hat{c} = \max_{1 \le i \le n-h+1} \{c_i\}$$

Mô hình sử dụng $M = 100$ bộ lọc độc lập cho mỗi kích thước cửa sổ $h \in \{3, 4, 5\}$.  
Do đó, mỗi kích thước cửa sổ sẽ tạo ra một vector đặc trưng kích thước $100$:
$$\mathbf{z}^{(h)} = [\hat{c}_1^{(h)}, \hat{c}_2^{(h)}, \dots, \hat{c}_{100}^{(h)}] \in \mathbb{R}^{100}$$

---

### 2.4. Ghép Nối Đặc Trưng (Feature Concatenation)
Các vector cực đại từ cả 3 nhóm kích thước cửa sổ được nối lại với nhau tạo thành vector biểu diễn ngữ nghĩa toàn câu $\mathbf{z} \in \mathbb{R}^{300}$:
$$\mathbf{z} = \mathbf{z}^{(3)} \oplus \mathbf{z}^{(4)} \oplus \mathbf{z}^{(5)} = [\hat{c}_1^{(3)}, \dots, \hat{c}_{100}^{(3)}, \hat{c}_1^{(4)}, \dots, \hat{c}_{100}^{(4)}, \hat{c}_1^{(5)}, \dots, \hat{c}_{100}^{(5)}]$$

---

### 2.5. Tầng Điều Chuẩn Penultimate Dropout & Ràng Buộc $L_2$-Norm

#### A. Dropout ($p = 0.5$):
Để chống hiện tượng đồng thích nghi giữa các đặc trưng (co-adaptation), vector $\mathbf{z}$ đi qua tầng Dropout ở bước lan truyền xuôi:
$$\mathbf{z}_{\text{drop}} = \mathbf{z} \circ \mathbf{r}$$
Trong đó:
* $\circ$ là phép nhân từng phần tử (Hadamard product).
* $\mathbf{r} \in \{0, 1\}^{300}$ là vector mặt nạ ngẫu nhiên tuân theo phân phối Bernoulli: $P(r_j = 1) = 1 - p = 0.5$.

#### B. Ràng buộc chuẩn $L_2$-Norm ($s = 3.0$):
Theo Yoon Kim (Section 2.1), sau mỗi bước cập nhật trọng số bằng Gradient Descent, vector trọng số $\mathbf{w}_j$ của từng node phân loại trong tầng Fully Connected được kiểm tra và co tỷ lệ nếu vượt ngưỡng $s = 3.0$:
$$\mathbf{w}_j \leftarrow \begin{cases} 
\mathbf{w}_j & \text{nếu } \|\mathbf{w}_j\|_2 \le s \\
\mathbf{w}_j \cdot \frac{s}{\|\mathbf{w}_j\|_2} & \text{nếu } \|\mathbf{w}_j\|_2 > s 
\end{cases}$$
Hiện thực bằng hàm `model.clip_l2_norm(s=3.0)`.

---

### 2.6. Tầng Fully Connected & Đầu Ra Đa Nhãn (Multi-Label Classification)
Vector $\mathbf{z}_{\text{drop}}$ được chiếu tuyến tính qua tầng kết nối đầy đủ để tính ra 31 giá trị logit tương ứng với 31 khía cạnh - cảm xúc:
$$\mathbf{o} = W_{\text{fc}} \mathbf{z}_{\text{drop}} + \mathbf{b}_{\text{fc}} \in \mathbb{R}^{31}$$
Trong đó $W_{\text{fc}} \in \mathbb{R}^{31 \times 300}$ và $\mathbf{b}_{\text{fc}} \in \mathbb{R}^{31}$.

Khác với bài báo gốc dùng Softmax cho đơn nhãn, bài toán UIT-ViSFD là đa nhãn (Multi-label), do đó mỗi node đầu ra đi qua hàm kích hoạt **Sigmoid** độc lập:
$$\hat{p}_j = \sigma(o_j) = \frac{1}{1 + e^{-o_j}} \in [0, 1] \quad (\forall j = 1, \dots, 31)$$
Trong đó $\hat{p}_j$ là xác suất bình luận chứa nhãn thứ $j$.

---

### 2.7. Hàm Mất Mát & Thuật Toán Tối Ưu

#### Hàm mất mát Binary Cross-Entropy with Logits:
$$\mathcal{L} = -\frac{1}{31} \sum_{j=1}^{31} \left[ y_j \log \sigma(o_j) + (1 - y_j) \log (1 - \sigma(o_j)) \right]$$

#### Thuật toán tối ưu Adadelta (Zeiler, 2012):
Mô hình sử dụng Adadelta theo đúng chuẩn Yoon Kim Section 3.1 với $\rho = 0.95, \epsilon = 10^{-6}, \text{lr} = 1.0$:
$$\Delta \theta_t = -\frac{\text{RMS}[\Delta \theta]_{t-1}}{\text{RMS}[g]_t} \odot g_t$$
Không phụ thuộc vào việc điều chỉnh thủ công learning rate và thích nghi tốt trên dữ liệu thưa.

---

### 2.8. Quy Luật Ra Quyết Định Theo Ngưỡng Cắt ($\tau = 0.3$)
Nhãn dự đoán nhị phân $\hat{Y}_j \in \{0, 1\}$ được xác định dựa trên ngưỡng cắt $\tau = 0.3$:
$$\hat{Y}_j = \begin{cases} 
1 & \text{nếu } \hat{p}_j \ge 0.3 \\
0 & \text{nếu } \hat{p}_j < 0.3 
\end{cases}$$

---

## 3. Bảng Tổng Hợp Chiều Dữ Liệu (Tensor Dimensions)

| Bước thực thi | Tên toán tử / Tầng | Kích thước Tensor đầu vào | Kích thước Tensor đầu ra | Ghi chú |
| :---: | :--- | :---: | :---: | :--- |
| **1** | Token Indexing | Chuỗi text | $(B, 80)$ | $B = 50$ (Batch size), $n = 80$ |
| **2** | Embedding Layer | $(B, 80)$ | $(B, 1, 80, 300)$ | Bổ sung chiều channel $= 1$ |
| **3a** | Conv2D ($h=3$) + ReLU | $(B, 1, 80, 300)$ | $(B, 100, 78)$ | 100 filters kích thước $3 \times 300$ |
| **3b** | Conv2D ($h=4$) + ReLU | $(B, 1, 80, 300)$ | $(B, 100, 77)$ | 100 filters kích thước $4 \times 300$ |
| **3c** | Conv2D ($h=5$) + ReLU | $(B, 1, 80, 300)$ | $(B, 100, 76)$ | 100 filters kích thước $5 \times 300$ |
| **4a** | 1-Max Pooling ($h=3$) | $(B, 100, 78)$ | $(B, 100)$ | Lấy $\max$ dọc theo chiều $78$ |
| **4b** | 1-Max Pooling ($h=4$) | $(B, 100, 77)$ | $(B, 100)$ | Lấy $\max$ dọc theo chiều $77$ |
| **4c** | 1-Max Pooling ($h=5$) | $(B, 100, 76)$ | $(B, 100)$ | Lấy $\max$ dọc theo chiều $76$ |
| **5** | Concatenation | $3 \times (B, 100)$ | $(B, 300)$ | Ghép nối 3 vector đặc trưng |
| **6** | Dropout ($p=0.5$) | $(B, 300)$ | $(B, 300)$ | Tắt ngẫu nhiên 50% node |
| **7** | Linear FC Layer | $(B, 300)$ | $(B, 31)$ | Nhân ma trận $W_{\text{fc}} \in \mathbb{R}^{31 \times 300}$ |
| **8** | Sigmoid Activation | $(B, 31)$ | $(B, 31)$ | Xác suất $\hat{p} \in [0, 1]$ cho 31 nhãn |

---

## 4. Ví Dụ Minh Họa Tính Toán Từng Bước (End-to-End Walkthrough)

Giả sử ta đưa vào mô hình bình luận thực tế sau:
> *"Máy chụp ảnh đẹp, pin trâu nhưng chơi game hơi nóng"*

### Bước 1: Tiền xử lý theo `Clean.md`
Chuỗi được chuẩn hóa và tách từ ghép tiếng Việt:
$$\text{Cleaned text} = \text{"máy chụp\_ảnh đẹp , pin trâu nhưng chơi game hơi nóng"}$$
Tổng cộng có **10 tokens**.

---

### Bước 2: Padding về độ dài $n = 80$ & Tra cứu Index
Chuỗi 10 token được tra bảng `word2idx` và đệm thêm 70 token `<pad>` (index 0):
$$\mathbf{x} = [42, 105, 18, 5, 12, 88, 31, 204, 75, 150, \underbrace{0, 0, \dots, 0}_{70 \text{ tokens <pad>}}] \in \mathbb{Z}^{80}$$

---

### Bước 3: Tra cứu ma trận Embedding ($1 \times 1 \times 80 \times 300$)
Mỗi số nguyên được thay thế bằng vector 300 chiều:
$$X = \begin{bmatrix}
\mathbf{x}_{\text{máy}} \\
\mathbf{x}_{\text{chụp\_ảnh}} \\
\mathbf{x}_{\text{đẹp}} \\
\vdots \\
\mathbf{x}_{\text{<pad>}}
\end{bmatrix}_{80 \times 300}$$

---

### Bước 4: Tích chập trích xuất n-gram

* **Bộ lọc $h = 3$ (Bộ lọc số 15 chuyên bắt cụm camera tích cực):**
  Khi quét qua cụm `"máy chụp_ảnh đẹp"` (3 từ đầu tiên):
  $$c_1^{(h=3)} = \text{ReLU}\left( \mathbf{w}_{15} \cdot \mathbf{x}_{1:3} + b_{15} \right) = \text{ReLU}(3.42) = 3.42$$
  Khi quét qua các cụm từ đệm `<pad>` ở cuối:
  $$c_{78}^{(h=3)} = \text{ReLU}\left( \mathbf{w}_{15} \cdot \mathbf{0} + b_{15} \right) = \text{ReLU}(-0.85) = 0.0$$

* **Bộ lọc $h = 4$ (Bộ lọc số 42 chuyên bắt cụm hiệu năng chơi game nóng):**
  Khi quét qua cụm `"nhưng chơi game hơi nóng"`:
  $$c_7^{(h=4)} = \text{ReLU}(4.15) = 4.15$$

---

### Bước 5: 1-Max Pooling (Rút trích giá trị cực đại)
Trên toàn bộ feature map 78 phần tử của bộ lọc số 15 ($h=3$):
$$\hat{c}_{15}^{(h=3)} = \max\{3.42, 0.12, 0.0, \dots, 0.0\} = \mathbf{3.42}$$

Trên feature map của bộ lọc số 42 ($h=4$):
$$\hat{c}_{42}^{(h=4)} = \max\{-0.2, 0.5, \dots, 4.15, \dots, 0.0\} = \mathbf{4.15}$$

Gom toàn bộ $3 \times 100$ cực đại thành vector $\mathbf{z} \in \mathbb{R}^{300}$:
$$\mathbf{z} = [0.82, \dots, \underbrace{3.42}_{\text{vị trí 15}}, \dots, \underbrace{4.15}_{\text{vị trí 142}}, \dots, 1.05] \in \mathbb{R}^{300}$$

---

### Bước 6: Tuyến tính & Xác suất Sigmoid tại Tầng Đầu Ra
Vector $\mathbf{z}$ nhân với ma trận trọng số $W_{\text{fc}} \in \mathbb{R}^{31 \times 300}$ để sinh ra 31 logits $\mathbf{o} \in \mathbb{R}^{31}$:

| Chỉ mục $j$ | Tên Nhãn | Logit $o_j$ | Xác suất Sigmoid $\hat{p}_j = \sigma(o_j)$ | So sánh với Ngưỡng $\tau = 0.3$ | Quyết định nhãn $\hat{Y}_j$ |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 5 | `CAMERA#Positive` | $+2.15$ | $\frac{1}{1 + e^{-2.15}} = \mathbf{0.895}$ | $0.895 \ge 0.3$ | **1 (DƯƠNG TÍNH)** |
| 2 | `BATTERY#Positive` | $+1.45$ | $\frac{1}{1 + e^{-1.45}} = \mathbf{0.810}$ | $0.810 \ge 0.3$ | **1 (DƯƠNG TÍNH)** |
| 16 | `PERFORMANCE#Negative` | $+0.92$ | $\frac{1}{1 + e^{-0.92}} = \mathbf{0.715}$ | $0.715 \ge 0.3$ | **1 (DƯƠNG TÍNH)** |
| 15 | `OTHERS` | $-1.15$ | $\frac{1}{1 + e^{1.15}} = \mathbf{0.240}$ | $0.240 < 0.3$ | **0 (ÂM TÍNH)** |
| 0 | `BATTERY#Negative` | $-3.80$ | $\frac{1}{1 + e^{3.80}} = \mathbf{0.022}$ | $0.022 < 0.3$ | **0 (ÂM TÍNH)** |
| 12 | `GENERAL#Negative` | $-4.10$ | $\frac{1}{1 + e^{4.10}} = \mathbf{0.016}$ | $0.016 < 0.3$ | **0 (ÂM TÍNH)** |

---

### Bước 7: Kết Quả Dự Đoán Cuối Cùng
Tập nhãn dự đoán xuất ra cho mẫu văn bản:
$$\hat{Y} = \{\text{BATTERY\#Positive},\, \text{CAMERA\#Positive},\, \text{PERFORMANCE\#Negative}\}$$

Khớp hoàn toàn với cả 3 khía cạnh xuất hiện trong câu:
1. *"chụp ảnh đẹp"* $\rightarrow$ `CAMERA#Positive`
2. *"pin trâu"* $\rightarrow$ `BATTERY#Positive`
3. *"chơi game hơi nóng"* $\rightarrow$ `PERFORMANCE#Negative`
