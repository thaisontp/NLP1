# -*- coding: utf-8 -*-
"""
deadpool.py — Pipeline Tiền Xử Lý Dữ Liệu & Nạp Word Vectors Theo Yoon Kim (2014)
==================================================================================
Thư mục: Yoon Kim test/
Mục đích:
    Xây dựng toàn bộ pipeline chuẩn bị dữ liệu cho mô hình TextCNN:
    1. KẾT NỐI TRỰC TIẾP và GỌI THỰC THI bộ tiền xử lý TextPreprocessor từ module
       preprocessor.py theo đặc tả 15 bước trong Clean.md.
    2. Xây dựng Từ điển Từ vựng Vocabulary (<pad>=0, <unk>=1, min_freq).
    3. Nạp Pre-trained Word Embeddings (300d), xử lý từ OOV theo đúng công thức phân phối
       đều bảo toàn phương sai của Yoon Kim (2014) Section 4.3 (a = sqrt(3) * sigma).
    4. Sinh PyTorch Dataset & DataLoader cho cả tập TRAIN, DEV và TEST (Test.csv / Test_clean.csv).

Tham chiếu học thuật & quy chuẩn dự án:
    - Clean.md: Kiến trúc 2 cơ chế (Dictionary-based + Rule-based Regex 15 bước).
    - Yoon Kim (2014): "Convolutional Neural Networks for Sentence Classification", EMNLP 2014.
"""

import os
import sys
import math
import random
import logging
from typing import List, Dict, Tuple, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# Thiết lập encoding UTF-8 cho Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Thêm thư mục gốc vào sys.path để đảm bảo khả năng tương thích tuyệt đối
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# GỌI TRỰC TIẾP TextPreprocessor và parse_labels TỪ preprocessor.py
from preprocessor import TextPreprocessor, parse_labels

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("DeadpoolPreprocessor")


# ============================================================================
# 1. DANH SÁCH 31 NHÃN CHUẨN CỦA BÀI TOÁN UIT-ViSFD (ASPECT-BASED SENTIMENT)
# ============================================================================
LABEL_COLS_31: List[str] = [
    "BATTERY#Negative", "BATTERY#Neutral", "BATTERY#Positive",
    "CAMERA#Negative", "CAMERA#Neutral", "CAMERA#Positive",
    "DESIGN#Negative", "DESIGN#Neutral", "DESIGN#Positive",
    "FEATURES#Negative", "FEATURES#Neutral", "FEATURES#Positive",
    "GENERAL#Negative", "GENERAL#Neutral", "GENERAL#Positive",
    "OTHERS",
    "PERFORMANCE#Negative", "PERFORMANCE#Neutral", "PERFORMANCE#Positive",
    "PRICE#Negative", "PRICE#Neutral", "PRICE#Positive",
    "SCREEN#Negative", "SCREEN#Neutral", "SCREEN#Positive",
    "SER&ACC#Negative", "SER&ACC#Neutral", "SER&ACC#Positive",
    "STORAGE#Negative", "STORAGE#Neutral", "STORAGE#Positive",
]


# ============================================================================
# 2. BỘ LÀM SẠCH VĂN BẢN GỌI TRỰC TIẾP CLEAN.MD (CLEAN DATA POOL)
# ============================================================================
class CleanDataPool:
    """
    Lớp đóng gói (Wrapper) thực thi các cơ chế làm sạch của Clean.md:
    Gọi trực tiếp instance TextPreprocessor() từ preprocessor.py.
    """

    def __init__(self):
        logger.info("Khởi tạo bộ tiền xử lý TextPreprocessor theo đúng đặc tả Clean.md...")
        self.preprocessor = TextPreprocessor()

    def clean_single(self, text: str) -> str:
        """
        Thực thi toàn bộ 15 bước của Clean.md trên 1 câu văn bản thô.
        """
        if not isinstance(text, str):
            return ""
        return self.preprocessor.transform(text)

    def clean_batch(self, texts: List[str], log_interval: int = 2000) -> List[str]:
        """
        Làm sạch hàng loạt danh sách câu văn bản thô.
        """
        cleaned = []
        total = len(texts)
        logger.info(f"Bắt đầu làm sạch {total} văn bản thô bằng Clean.md...")
        for i, t in enumerate(texts):
            cleaned.append(self.clean_single(t))
            if (i + 1) % log_interval == 0 or (i + 1) == total:
                logger.info(f"Đã làm sạch {i + 1}/{total} câu...")
        return cleaned


# ============================================================================
# 3. XÂY DỰNG TỪ ĐIỂN TỪ VỰNG (VOCABULARY BUILDER)
# ============================================================================
class Vocabulary:
    """
    Quản lý từ vựng 2 chiều: từ (word) <-> chỉ số (index).
    Mặc định:
        - Index 0: <pad> (Padding token cho các câu ngắn hơn n)
        - Index 1: <unk> (Unknown token cho các từ ngoài từ điển OOV)
    Tích hợp trực tiếp CleanDataPool để tự động làm sạch câu thô khi băm token.
    """

    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    PAD_IDX = 0
    UNK_IDX = 1

    def __init__(
        self,
        min_freq: int = 1,
        max_vocab_size: Optional[int] = None,
        cleaner: Optional[CleanDataPool] = None,
    ):
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size
        self.cleaner = cleaner
        self.word2idx: Dict[str, int] = {
            self.PAD_TOKEN: self.PAD_IDX,
            self.UNK_TOKEN: self.UNK_IDX,
        }
        self.idx2word: Dict[int, str] = {
            self.PAD_IDX: self.PAD_TOKEN,
            self.UNK_IDX: self.UNK_TOKEN,
        }
        self.word_counts: Dict[str, int] = {}
        self.is_frozen = False

    def __len__(self) -> int:
        return len(self.word2idx)

    def tokenize(self, text: str) -> List[str]:
        """
        Tách từ tiếng Việt thành danh sách token.
        Nếu có bộ làm sạch cleaner, câu sẽ được chạy qua Clean.md trước khi tách.
        """
        if not isinstance(text, str):
            return []
        if self.cleaner is not None:
            text = self.cleaner.clean_single(text)

        text = text.strip()
        if not text:
            return []
        return text.split()

    def build_from_texts(self, texts: List[str]) -> "Vocabulary":
        """
        Đếm tần suất từ và xây dựng bảng ánh xạ index cho kho ngữ liệu.
        """
        logger.info(f"Đang đếm tần suất từ vựng từ {len(texts)} văn bản...")
        for text in texts:
            tokens = self.tokenize(text)
            for tok in tokens:
                self.word_counts[tok] = self.word_counts.get(tok, 0) + 1

        # Lọc theo min_freq và sắp xếp giảm dần theo tần số xuất hiện
        sorted_words = [
            (w, c) for w, c in self.word_counts.items() if c >= self.min_freq
        ]
        sorted_words.sort(key=lambda x: x[1], reverse=True)

        if self.max_vocab_size is not None:
            sorted_words = sorted_words[: self.max_vocab_size - len(self.word2idx)]

        curr_idx = len(self.word2idx)
        for word, _ in sorted_words:
            if word not in self.word2idx:
                self.word2idx[word] = curr_idx
                self.idx2word[curr_idx] = word
                curr_idx += 1

        self.is_frozen = True
        logger.info(
            f"Xây dựng từ điển hoàn tất! Tổng số từ vựng: {len(self.word2idx)} "
            f"(đã lọc min_freq={self.min_freq})"
        )
        return self

    def lookup_index(self, word: str) -> int:
        return self.word2idx.get(word, self.UNK_IDX)

    def lookup_word(self, idx: int) -> str:
        return self.idx2word.get(idx, self.UNK_TOKEN)

    def text_to_indices(self, text: str, max_len: int) -> List[int]:
        """
        Chuyển một câu văn bản thành chuỗi các index có độ dài cố định n (max_len).
        Nếu ngắn hơn n: Độn <pad> (0) vào cuối câu.
        Nếu dài hơn n: Cắt tỉa (truncate) về n từ đầu tiên.
        """
        tokens = self.tokenize(text)
        indices = [self.lookup_index(t) for t in tokens]

        if len(indices) < max_len:
            indices.extend([self.PAD_IDX] * (max_len - len(indices)))
        else:
            indices = indices[:max_len]
        return indices


# ============================================================================
# 4. NẠP PRE-TRAINED WORD VECTORS & XỬ LÝ OOV THEO CHUẨN YOON KIM (2014)
# ============================================================================
def load_pretrained_vectors(
    vocab: Vocabulary,
    vector_file_path: Optional[str] = None,
    embedding_dim: int = 300,
) -> torch.FloatTensor:
    """
    Nạp ma trận Pre-trained Word Vectors cho toàn bộ từ vựng Vocabulary.

    Triển khai chính xác quy tắc trong Yoon Kim (2014) Section 4.3:
    1. Với các từ CÓ trong Pre-trained Vectors:
       - Copy trực tiếp vector tương ứng.
    2. Với các từ KHÔNG có (Unknown Words / OOV):
       - Yoon Kim (2014): "When randomly initializing words not in word2vec,
         we obtained slight improvements by sampling each dimension from U[-a, a]
         where a was chosen such that the randomly initialized vectors have
         the same variance as the pre-trained ones."
       - Với phân phối đều U[-a, a], phương sai là: Var = a^2 / 3.
       - Do đó: a = sqrt(3 * Var(pre-trained)) = sqrt(3) * std(pre-trained).
    3. Với token <pad> (index 0):
       - Vector bằng 0 tuyệt đối để không gây nhiễu khi tính tích chập.
    """
    vocab_size = len(vocab)
    logger.info(f"Bắt đầu khởi tạo ma trận embedding: {vocab_size} x {embedding_dim}")

    embeddings_dict: Dict[str, np.ndarray] = {}

    if vector_file_path and os.path.exists(vector_file_path):
        logger.info(f"Đang nạp pre-trained vector từ file: {vector_file_path}")
        loaded_count = 0
        with open(vector_file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if line_idx == 0 and len(parts) <= 3:
                    continue
                word = parts[0]
                try:
                    vec = np.asarray(parts[1:], dtype=np.float32)
                    if len(vec) == embedding_dim:
                        embeddings_dict[word] = vec
                        loaded_count += 1
                except ValueError:
                    continue
        logger.info(f"Đã đọc {loaded_count} từ vựng từ tệp pre-trained.")
    else:
        if vector_file_path:
            logger.warning(
                f"Không tìm thấy file pre-trained tại '{vector_file_path}'. "
                f"Sẽ sử dụng phương thức khởi tạo ngẫu nhiên toàn bộ (CNN-rand)."
            )

    # Tính toán phương sai của các pre-trained vectors để làm chuẩn cho U[-a, a]
    if len(embeddings_dict) > 0:
        all_vecs = np.array(list(embeddings_dict.values()), dtype=np.float32)
        var_pretrained = float(np.var(all_vecs))
        std_pretrained = float(np.std(all_vecs))
        a_bound = math.sqrt(3.0 * var_pretrained)
        logger.info(
            f"Thống kê Pre-trained Vectors: Std={std_pretrained:.5f}, Var={var_pretrained:.5f} => "
            f"Khoảng phân phối đều cho từ OOV: U[-{a_bound:.5f}, {a_bound:.5f}] (Yoon Kim 2014)"
        )
    else:
        a_bound = 0.25
        logger.info(f"Khởi tạo ngẫu nhiên U[-{a_bound}, {a_bound}] cho toàn bộ từ điển.")

    weight_matrix = np.empty((vocab_size, embedding_dim), dtype=np.float32)

    found_count = 0
    oov_count = 0

    for word, idx in vocab.word2idx.items():
        if idx == vocab.PAD_IDX:
            weight_matrix[idx] = np.zeros(embedding_dim, dtype=np.float32)
        elif word in embeddings_dict:
            weight_matrix[idx] = embeddings_dict[word]
            found_count += 1
        else:
            weight_matrix[idx] = np.random.uniform(-a_bound, a_bound, size=embedding_dim).astype(np.float32)
            oov_count += 1

    logger.info(
        f"Hoàn thành ma trận Embedding: {found_count} từ tìm thấy trong pre-trained, "
        f"{oov_count} từ OOV được khởi tạo theo U[-{a_bound:.4f}, {a_bound:.4f}]."
    )
    return torch.from_numpy(weight_matrix).float()


# ============================================================================
# 5. PYTORCH DATASET CHO BÀI TOÁN SENTIMENT & ABSA
# ============================================================================
class KimSentimentDataset(Dataset):
    """
    Dataset tương thích với kiến trúc mạng tích chập Yoon Kim (2014).
    Hỗ trợ cả bài toán Multi-label (UIT-ViSFD 31 nhãn) và Single-label thông thường.
    """

    def __init__(
        self,
        texts: List[str],
        labels: np.ndarray,
        vocab: Vocabulary,
        max_len: int = 100,
        is_multilabel: bool = True,
    ):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
        self.is_multilabel = is_multilabel

        # Chuyển đổi toàn bộ câu sang chuỗi chỉ số index trước để tăng tốc nạp batch
        self.encoded_inputs: List[List[int]] = [
            vocab.text_to_indices(t, max_len=max_len) for t in texts
        ]

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.encoded_inputs[idx], dtype=torch.long)
        if self.is_multilabel:
            y = torch.tensor(self.labels[idx], dtype=torch.float32)
        else:
            y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


# ============================================================================
# 6. PIPELINE NẠP TẬP TRAIN & DEV
# ============================================================================
def prepare_data_pipeline(
    csv_path: str,
    text_col: str = "comment_clean",
    label_cols: Optional[List[str]] = None,
    pretrained_vec_path: Optional[str] = None,
    embedding_dim: int = 300,
    max_len: int = 100,
    batch_size: int = 50,  # 50 theo thiết lập Yoon Kim Section 3.1
    dev_ratio: float = 0.1,  # 10% dev set theo Yoon Kim Section 3.1
    min_word_freq: int = 1,
    random_seed: int = 42,
    force_apply_clean_md: bool = False,
) -> Tuple[DataLoader, DataLoader, Vocabulary, torch.Tensor, List[str]]:
    """
    Toàn bộ luồng dữ liệu huấn luyện:
        1. Đọc tệp CSV (mặc định Train_clean.csv).
        2. Tự động kiểm tra và làm sạch văn bản theo Clean.md nếu là câu thô.
        3. Tách tập Train (90%) và Dev (10%) ngẫu nhiên theo chuẩn Yoon Kim.
        4. Xây dựng Vocabulary từ tập Train.
        5. Nạp và khởi tạo ma trận Pre-trained Word Vectors (300d).
        6. Sinh 2 PyTorch DataLoaders (Train, Dev) với batch_size=50 và shuffle mini-batches.
    """
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại: {csv_path}")

    logger.info(f"Đang đọc dữ liệu Train từ: {csv_path}")
    df = pd.read_csv(csv_path)

    cleaner = None
    if text_col not in df.columns:
        if "comment" in df.columns:
            logger.warning(f"Không có cột '{text_col}', chuyển sang 'comment' thô và kích hoạt CleanDataPool (Clean.md).")
            text_col = "comment"
            cleaner = CleanDataPool()
        else:
            raise KeyError(f"Không tìm thấy cột văn bản '{text_col}' trong CSV!")
    elif text_col == "comment" or force_apply_clean_md:
        logger.info("Yêu cầu xử lý văn bản thô: Kích hoạt CleanDataPool (TextPreprocessor)...")
        cleaner = CleanDataPool()

    if label_cols is None:
        available_labels = [c for c in LABEL_COLS_31 if c in df.columns]
        if len(available_labels) > 0:
            label_cols = available_labels
        else:
            raise ValueError("Không tìm thấy các cột nhãn UIT-ViSFD trong file CSV!")

    logger.info(f"Số lượng nhãn phân loại: {len(label_cols)}")

    df[text_col] = df[text_col].fillna("").astype(str)
    valid_mask = df[text_col].str.strip().str.len() > 0
    df = df[valid_mask].reset_index(drop=True)

    texts = df[text_col].tolist()
    labels = df[label_cols].values.astype(np.float32)

    total_samples = len(texts)
    logger.info(f"Tổng số mẫu dữ liệu Train hợp lệ: {total_samples}")

    indices = np.arange(total_samples)
    np.random.shuffle(indices)

    dev_size = int(total_samples * dev_ratio)
    dev_idx = indices[:dev_size]
    train_idx = indices[dev_size:]

    train_texts = [texts[i] for i in train_idx]
    train_labels = labels[train_idx]

    dev_texts = [texts[i] for i in dev_idx]
    dev_labels = labels[dev_idx]

    logger.info(f"Tập Train: {len(train_texts)} mẫu | Tập Dev (10%): {len(dev_texts)} mẫu")

    vocab = Vocabulary(min_freq=min_word_freq, cleaner=cleaner)
    vocab.build_from_texts(train_texts)

    embedding_matrix = load_pretrained_vectors(
        vocab=vocab,
        vector_file_path=pretrained_vec_path,
        embedding_dim=embedding_dim,
    )

    train_dataset = KimSentimentDataset(
        texts=train_texts,
        labels=train_labels,
        vocab=vocab,
        max_len=max_len,
        is_multilabel=True,
    )
    dev_dataset = KimSentimentDataset(
        texts=dev_texts,
        labels=dev_labels,
        vocab=vocab,
        max_len=max_len,
        is_multilabel=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, dev_loader, vocab, embedding_matrix, label_cols


# ============================================================================
# 7. NẠP TẬP KIỂM THỬ ĐỘC LẬP (TEST SET: Test.csv / Test_clean.csv)
# ============================================================================
def prepare_test_dataloader(
    test_csv_path: str,
    vocab: Vocabulary,
    label_cols: List[str],
    text_col: str = "comment_clean",
    batch_size: int = 50,
    max_len: int = 100,
) -> Tuple[DataLoader, pd.DataFrame, np.ndarray]:
    """
    Nạp tập dữ liệu kiểm thử (Test set) và xây dựng PyTorch DataLoader:
    - Sử dụng cùng Vocabulary đã xây dựng từ tập Train.
    - Cố định danh sách và thứ tự 31 nhãn label_cols khớp 100% với tập Train.
    - Tự động tiền xử lý Clean.md nếu file Test là bản thô (Test.csv).

    Trả về:
        (test_loader, df_test, y_test_true)
    """
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"Không tìm thấy file Test tại: {test_csv_path}")

    logger.info(f"Đang nạp tập Test từ: {test_csv_path}")
    df_test = pd.read_csv(test_csv_path)

    # 1. Xử lý cột văn bản
    if text_col not in df_test.columns:
        if "comment" in df_test.columns:
            logger.info("Tập Test chưa có cột 'comment_clean', đang chạy tiền xử lý Clean.md...")
            cleaner = CleanDataPool()
            df_test["comment_clean"] = cleaner.clean_batch(df_test["comment"].fillna("").astype(str).tolist())
            text_col = "comment_clean"
        else:
            raise KeyError("File Test không chứa cột 'comment_clean' hoặc 'comment'!")

    # 2. Xử lý nhãn: nếu chưa có 31 cột nhị phân (vd file Test.csv thô), sinh từ cột 'label'
    has_binary_labels = all(col in df_test.columns for col in label_cols)
    if not has_binary_labels:
        if "label" in df_test.columns:
            logger.info("Đang chuyển đổi chuỗi nhãn sang 31 cột one-hot nhị phân...")
            df_test = parse_labels(df_test, label_col="label")
        else:
            raise KeyError("File Test không có các cột nhãn nhị phân và cũng không có cột 'label'!")

    df_test[text_col] = df_test[text_col].fillna("").astype(str)
    texts_test = df_test[text_col].tolist()
    y_test_true = df_test[label_cols].values.astype(np.float32)

    logger.info(f"Tập Test đã nạp thành công: {len(texts_test)} mẫu kiểm thử.")

    test_dataset = KimSentimentDataset(
        texts=texts_test,
        labels=y_test_true,
        vocab=vocab,
        max_len=max_len,
        is_multilabel=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    return test_loader, df_test, y_test_true


# ============================================================================
# 8. DEMO CHẠY THỬ NGHIỆM ĐỘC LẬP
# ============================================================================
if __name__ == "__main__":
    print("=" * 75)
    print("KIỂM TRA GỌI TRỰC TIẾP TEXTPREPROCESSOR & NẠP TẬP TEST TRONG DEADPOOL.PY")
    print("=" * 75)

    data_path = os.path.join(PROJECT_ROOT, "Train_clean.csv")
    test_path = os.path.join(PROJECT_ROOT, "Test", "Test_clean.csv")

    if os.path.exists(data_path) and os.path.exists(test_path):
        train_l, dev_l, vocab, emb_mat, labels = prepare_data_pipeline(
            csv_path=data_path,
            text_col="comment_clean",
            embedding_dim=300,
            max_len=80,
            batch_size=50,
            dev_ratio=0.1,
        )
        print(f"[OK] Train batches: {len(train_l)} | Dev batches: {len(dev_l)}")

        test_l, df_t, y_t = prepare_test_dataloader(
            test_csv_path=test_path,
            vocab=vocab,
            label_cols=labels,
            text_col="comment_clean",
            batch_size=50,
            max_len=80,
        )
        print(f"[OK] Test batches : {len(test_l)} | Số mẫu test: {len(df_t)}")
