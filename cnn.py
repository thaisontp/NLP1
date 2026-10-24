# -*- coding: utf-8 -*-
"""
cnn.py — Hiện Thực Mạng Tích Chập TextCNN Chuẩn Pipeline Yoon Kim (2014)
========================================================================
Thư mục: Yoon Kim test/
Mục đích:
    Triển khai mô hình Convolutional Neural Networks for Sentence Classification
    theo đúng kiến trúc, siêu tham số và cơ chế điều chuẩn của Yoon Kim (EMNLP 2014).
    Tích hợp đánh giá tự động trên tập TEST (Test.csv / Test_clean.csv) sau mỗi epoch:
    - Lưu file dự đoán chi tiết với cột 'True Label' và 'Predict Label' vào folder 'result epoch/'
    - Thống kê TOP 20 nhãn dự đoán sai nhiều nhất (False Positives + False Negatives) mỗi epoch.
"""

import os
import sys
import time
import argparse
import logging
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    hamming_loss,
)


THRESHOLD_DEFAULT = 0.3  # Ngưỡng mặc định để chuyển xác suất sang nhãn nhị phân (binary labels)

# Nạp pipeline từ deadpool trong cùng thư mục
try:
    from deadpool import (
        prepare_data_pipeline,
        prepare_test_dataloader,
        Vocabulary,
        LABEL_COLS_31,
    )
except ImportError:
    from .deadpool import (
        prepare_data_pipeline,
        prepare_test_dataloader,
        Vocabulary,
        LABEL_COLS_31,
    )

# Thiết lập encoding UTF-8
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("KimCNN")


# ============================================================================
# 1. KIẾN TRÚC MÔ HÌNH TEXTCNN CỦA YOON KIM (2014)
# ============================================================================
class YoonKimCNN(nn.Module):
    """
    Hiện thực chuẩn xác kiến trúc CNN của Yoon Kim (2014) cho phân loại câu.
    Hỗ trợ 4 biến thể: 'rand', 'static', 'non-static', 'multichannel'.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        num_classes: int = 31,
        filter_sizes: List[int] = [3, 4, 5],
        num_filters: int = 100,
        dropout_p: float = 0.5,
        l2_constraint: float = 3.0,
        mode: str = "non-static",
        pretrained_embeddings: Optional[torch.Tensor] = None,
        is_multilabel: bool = True,
    ):
        super(YoonKimCNN, self).__init__()

        self.mode = mode.lower()
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.dropout_p = dropout_p
        self.l2_constraint = l2_constraint
        self.is_multilabel = is_multilabel

        # Khởi tạo Embedding theo từng biến thể (Section 3.3)
        if self.mode == "rand":
            self.embedding = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=embedding_dim,
                padding_idx=0,
            )
            nn.init.uniform_(self.embedding.weight, -0.25, 0.25)
            self.embedding.weight.data[0].fill_(0)
            in_channels = 1

        elif self.mode == "static":
            if pretrained_embeddings is None:
                raise ValueError("Mode 'static' yêu cầu cung cấp pretrained_embeddings!")
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings,
                freeze=True,
                padding_idx=0,
            )
            in_channels = 1

        elif self.mode == "non-static":
            if pretrained_embeddings is None:
                raise ValueError("Mode 'non-static' yêu cầu cung cấp pretrained_embeddings!")
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings,
                freeze=False,
                padding_idx=0,
            )
            in_channels = 1

        elif self.mode == "multichannel":
            if pretrained_embeddings is None:
                raise ValueError("Mode 'multichannel' yêu cầu cung cấp pretrained_embeddings!")
            self.embedding_static = nn.Embedding.from_pretrained(
                pretrained_embeddings,
                freeze=True,
                padding_idx=0,
            )
            self.embedding_non_static = nn.Embedding.from_pretrained(
                pretrained_embeddings.clone(),
                freeze=False,
                padding_idx=0,
            )
            in_channels = 2
        else:
            raise ValueError(f"Mode '{mode}' không hợp lệ. Chọn: rand, static, non-static, multichannel.")

        # Tầng tích chập đa kích thước cửa sổ (Section 2 & 3.1)
        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_filters,
                kernel_size=(h, embedding_dim),
            )
            for h in filter_sizes
        ])

        # Tầng Điều chuẩn Dropout & Fully-Connected (Section 2.1)
        total_features = len(filter_sizes) * num_filters  # 3 * 100 = 300
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(total_features, num_classes)

        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Lan truyền xuôi (Forward pass):
            x: Tensor đầu vào có kích thước (batch_size, sentence_length)
        """
        if self.mode == "multichannel":
            x_static = self.embedding_static(x)
            x_non_static = self.embedding_non_static(x)
            emb = torch.stack([x_static, x_non_static], dim=1)
        else:
            emb = self.embedding(x).unsqueeze(1)

        pooled_outputs = []
        for conv in self.convs:
            c = F.relu(conv(emb)).squeeze(3)
            # 1-Max Pooling (Max-over-time) dọc theo câu
            c_hat = torch.max(c, dim=2)[0]
            pooled_outputs.append(c_hat)

        z = torch.cat(pooled_outputs, dim=1)
        z_drop = self.dropout(z)
        logits = self.fc(z_drop)
        return logits

    def clip_l2_norm(self, s: Optional[float] = None) -> None:
        """
        Ràng buộc chuẩn L2-norm của Yoon Kim (Section 2.1): ||w||_2 <= s (s = 3.0)
        """
        threshold = self.l2_constraint if s is None else s
        with torch.no_grad():
            norms = self.fc.weight.norm(p=2, dim=1, keepdim=True)
            scale = torch.clamp(norms / threshold, min=1.0)
            self.fc.weight.div_(scale)


# ============================================================================
# 2. ĐÁNH GIÁ & TÍNH TOÁN METRICS
# ============================================================================
def compute_metrics(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    threshold: float = THRESHOLD_DEFAULT,
) -> Dict[str, float]:
    y_pred = (y_pred_probs >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    h_loss = hamming_loss(y_true, y_pred)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    micro_p = precision_score(y_true, y_pred, average="micro", zero_division=0)
    micro_r = recall_score(y_true, y_pred, average="micro", zero_division=0)

    return {
        "accuracy": acc,
        "hamming_loss": h_loss,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = THRESHOLD_DEFAULT,
) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            total_loss += loss.item() * batch_x.size(0)

            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(batch_y.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    y_true = np.vstack(all_targets)
    y_pred_probs = np.vstack(all_probs)

    metrics = compute_metrics(y_true, y_pred_probs, threshold=threshold)
    return avg_loss, metrics, y_true, y_pred_probs


# ============================================================================
# 3. XỬ LÝ & XUẤT KẾT QUẢ TẬP TEST TẠI MỖI EPOCH (RESULT EPOCH)
# ============================================================================
def evaluate_and_export_test_epoch(
    model: nn.Module,
    test_loader: DataLoader,
    df_test: pd.DataFrame,
    label_cols: List[str],
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    results_dir: str,
    threshold: float = THRESHOLD_DEFAULT,
) -> Tuple[float, Dict[str, float], pd.DataFrame]:
    """
    Đánh giá mô hình trên tập Test sau mỗi epoch và lưu vào thư mục 'result epoch/':
    1. Xuất file 'epoch_{epoch}_predictions.csv' chứa:
       - index, comment, True Label, Predict Label, Is_Exact_Match
    2. Xuất file 'epoch_{epoch}_top20_errors.csv' chứa bảng thống kê TOP 20 nhãn sai nhiều nhất.
    3. In trực tiếp bảng TOP 20 lên màn hình terminal.
    """
    os.makedirs(results_dir, exist_ok=True)

    test_loss, test_metrics, y_true, y_probs = evaluate(
        model, test_loader, criterion, device, threshold=threshold
    )

    y_pred_bin = (y_probs >= threshold).astype(int)
    num_samples = len(y_true)

    # 1. Tạo chuỗi nhãn True Label và Predict Label cho từng mẫu
    true_labels_str = []
    pred_labels_str = []
    is_exact_match = []

    for i in range(num_samples):
        # Lấy các nhãn thực tế
        t_active = [label_cols[k] for k in range(len(label_cols)) if y_true[i, k] == 1]
        # Lấy các nhãn mô hình dự đoán
        p_active = [label_cols[k] for k in range(len(label_cols)) if y_pred_bin[i, k] == 1]

        true_str = "; ".join(t_active) if t_active else "NONE"
        pred_str = "; ".join(p_active) if p_active else "NONE"

        true_labels_str.append(true_str)
        pred_labels_str.append(pred_str)
        is_exact_match.append(np.array_equal(y_true[i], y_pred_bin[i]))

    # Tạo DataFrame dự đoán
    comment_col = "comment" if "comment" in df_test.columns else "comment_clean"
    pred_df = pd.DataFrame({
        "index": df_test["index"] if "index" in df_test.columns else list(range(num_samples)),
        "comment": df_test[comment_col],
        "True Label": true_labels_str,
        "Predict Label": pred_labels_str,
        "Is_Exact_Match": is_exact_match,
    })

    pred_file = os.path.join(results_dir, f"epoch_{epoch:02d}_predictions.csv")
    pred_df.to_csv(pred_file, index=False, encoding="utf-8-sig")

    # 2. Thống kê lỗi từng nhãn và chọn TOP 20 nhãn sai nhiều nhất
    label_error_stats = []
    for k, label_name in enumerate(label_cols):
        y_k = y_true[:, k]
        p_k = y_pred_bin[:, k]

        fp = int(np.sum((y_k == 0) & (p_k == 1)))  # Dự đoán thừa
        fn = int(np.sum((y_k == 1) & (p_k == 0)))  # Bỏ sót
        total_errors = fp + fn
        support = int(np.sum(y_k == 1))
        pred_count = int(np.sum(p_k == 1))
        f1_k = f1_score(y_k, p_k, zero_division=0)
        error_rate = (total_errors / num_samples) * 100.0

        label_error_stats.append({
            "Label": label_name,
            "Total Errors": total_errors,
            "False Positives (FP)": fp,
            "False Negatives (FN)": fn,
            "Support (True Count)": support,
            "Predicted Count": pred_count,
            "Error Rate (%)": round(error_rate, 2),
            "F1-Score": round(float(f1_k), 4),
        })

    # Sắp xếp giảm dần theo số lượng mẫu đoán sai (Total Errors)
    label_error_stats.sort(key=lambda x: x["Total Errors"], reverse=True)
    top20_list = label_error_stats[:20]

    for rank, item in enumerate(top20_list, 1):
        item["Rank"] = rank

    top20_df = pd.DataFrame(top20_list)
    # Đặt cột Rank lên đầu
    cols_order = ["Rank", "Label", "Total Errors", "False Positives (FP)", "False Negatives (FN)",
                  "Support (True Count)", "Predicted Count", "Error Rate (%)", "F1-Score"]
    top20_df = top20_df[cols_order]

    top20_file = os.path.join(results_dir, f"epoch_{epoch:02d}_top20_errors.csv")
    top20_df.to_csv(top20_file, index=False, encoding="utf-8-sig")

    # 3. In bảng thống kê Top 20 ra terminal
    print("\n" + "=" * 90)
    print(f" THỐNG KÊ TOP 20 NHÃN DỰ ĐOÁN SAI NHIỀU NHẤT TRÊN TẬP TEST (EPOCH {epoch:02d})")
    print(f"   (Tổng số mẫu kiểm thử: {num_samples} | Tập tin lưu: {os.path.basename(top20_file)})")
    print("=" * 90)
    print(f"{'Rank':<5} | {'Tên Nhãn (Label)':<24} | {'Tổng Lỗi':<9} | {'FP (Thừa)':<10} | {'FN (Thiếu)':<10} | {'True Count':<11} | {'F1-Score':<8}")
    print("-" * 90)
    for row in top20_list:
        print(f"{row['Rank']:<5} | {row['Label']:<24} | {row['Total Errors']:<9} | {row['False Positives (FP)']:<10} | {row['False Negatives (FN)']:<10} | {row['Support (True Count)']:<11} | {row['F1-Score']:<8.4f}")
    print("=" * 90 + "\n")

    return test_loss, test_metrics, top20_df


# ============================================================================
# 4. VÒNG LẶP HUẤN LUYỆN TOÀN DIỆN KÈM TEST SAU MỖI EPOCH
# ============================================================================
def train_kim_cnn(
    model: YoonKimCNN,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    test_loader: Optional[DataLoader] = None,
    df_test: Optional[pd.DataFrame] = None,
    label_cols: Optional[List[str]] = None,
    num_epochs: int = 15,
    learning_rate: float = 1.0,
    device_str: Optional[str] = None,
    patience: int = 4,
    save_path: Optional[str] = None,
    results_dir: str = "result epoch",
) -> Dict[str, any]:
    """
    Huấn luyện TextCNN và tự động đánh giá trên tập Test sau mỗi epoch.
    """
    device = torch.device(
        device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info(f"Khởi chạy huấn luyện trên thiết bị: {device}")
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adadelta(
        model.parameters(),
        lr=learning_rate,
        rho=0.95,
        eps=1e-6,
    )

    best_macro_f1 = -1.0
    best_epoch = -1
    no_improve_epochs = 0
    history = []

    logger.info("=" * 75)
    logger.info(f"BẮT ĐẦU HUẤN LUYỆN TEXTCNN — BIẾN THỂ [{model.mode.upper()}]")
    logger.info(f"Kết quả kiểm thử mỗi epoch sẽ được lưu vào: '{results_dir}'")
    logger.info("=" * 75)

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        model.train()
        train_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()

            optimizer.step()
            model.clip_l2_norm()

            train_loss += loss.item() * batch_x.size(0)

        train_loss /= len(train_loader.dataset)
        elapsed = time.time() - start_time

        # 1. Đánh giá trên tập Dev
        dev_loss, dev_metrics, _, _ = evaluate(model, dev_loader, criterion, device)

        # 2. Đánh giá trên tập Test và lưu vào folder 'result epoch'
        test_info = ""
        test_macro_f1 = 0.0
        if test_loader is not None and df_test is not None and label_cols is not None:
            t_loss, t_metrics, _ = evaluate_and_export_test_epoch(
                model=model,
                test_loader=test_loader,
                df_test=df_test,
                label_cols=label_cols,
                criterion=criterion,
                device=device,
                epoch=epoch,
                results_dir=results_dir,
            )
            test_macro_f1 = t_metrics["macro_f1"]
            test_info = f" | Test Loss: {t_loss:.4f} | Test Macro-F1: {test_macro_f1:.4f} | Test Acc: {t_metrics['accuracy']:.4f}"

        logger.info(
            f"Epoch [{epoch:02d}/{num_epochs:02d}] ({elapsed:.1f}s) | "
            f"Train Loss: {train_loss:.4f} | Dev Loss: {dev_loss:.4f} | "
            f"Dev Macro-F1: {dev_metrics['macro_f1']:.4f}{test_info}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "dev_loss": dev_loss,
            "dev_macro_f1": dev_metrics["macro_f1"],
            "test_macro_f1": test_macro_f1,
        })

        # Early Stopping theo Dev Macro-F1
        curr_macro_f1 = dev_metrics["macro_f1"]
        if curr_macro_f1 > best_macro_f1:
            best_macro_f1 = curr_macro_f1
            best_epoch = epoch
            no_improve_epochs = 0
            if save_path:
                os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_metrics": dev_metrics,
                    "model_config": {
                        "mode": model.mode,
                        "filter_sizes": model.filter_sizes,
                        "num_filters": model.num_filters,
                        "dropout_p": model.dropout_p,
                        "l2_constraint": model.l2_constraint,
                    },
                }, save_path)
                logger.info(f"[*] Đã lưu checkpoint tốt nhất vào '{save_path}' (Dev Macro-F1: {best_macro_f1:.4f})")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                logger.info(f"[!] Dừng sớm (Early Stopping) tại Epoch {epoch} do Dev Macro-F1 không cải thiện sau {patience} epochs.")
                break

    logger.info("=" * 75)
    logger.info(f"HUẤN LUYỆN HOÀN TẤT! Best Epoch: {best_epoch} | Best Dev Macro-F1: {best_macro_f1:.4f}")
    logger.info(f"Toàn bộ kết quả kiểm thử đã được lưu tại thư mục: '{results_dir}'")
    logger.info("=" * 75)

    return {
        "best_epoch": best_epoch,
        "best_macro_f1": best_macro_f1,
        "history": history,
    }


# ============================================================================
# 5. CHƯƠNG TRÌNH THỰC THI CHÍNH (CLI RUNNER)
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Huấn luyện TextCNN Yoon Kim (2014) và đánh giá trên tập Test mỗi epoch")
    parser.add_argument("--mode", type=str, default="non-static", choices=["rand", "static", "non-static", "multichannel"])
    parser.add_argument("--epochs", type=int, default=10, help="Số lượng epoch tối đa")
    parser.add_argument("--batch_size", type=int, default=50, help="Kích thước mini-batch")
    parser.add_argument("--embedding_dim", type=int, default=300, help="Số chiều vector từ")
    parser.add_argument("--max_len", type=int, default=80, help="Độ dài tối đa của câu")
    parser.add_argument("--patience", type=int, default=4, help="Patience cho Early Stopping")
    parser.add_argument("--pretrained_vec", type=str, default=None, help="Đường dẫn file word vectors")
    parser.add_argument("--test_csv", type=str, default=None, help="Đường dẫn file Test.csv / Test_clean.csv")
    parser.add_argument("--results_dir", type=str, default=None, help="Thư mục lưu kết quả kiểm thử mỗi epoch")
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    train_csv_path = os.path.join(project_root, "Train_clean.csv")

    # Xác định đường dẫn file Test
    if args.test_csv:
        test_csv_path = args.test_csv
    else:
        clean_test = os.path.join(project_root, "Test", "Test_clean.csv")
        raw_test = os.path.join(project_root, "Test", "Test.csv")
        test_csv_path = clean_test if os.path.exists(clean_test) else raw_test

    # Thư mục lưu kết quả mỗi epoch: 'Yoon Kim test/result epoch'
    results_dir = args.results_dir if args.results_dir else os.path.join(current_dir, "result epoch")

    # 1. Chuẩn bị tập Train & Dev
    train_loader, dev_loader, vocab, pretrained_matrix, label_cols = prepare_data_pipeline(
        csv_path=train_csv_path,
        text_col="comment_clean",
        pretrained_vec_path=args.pretrained_vec,
        embedding_dim=args.embedding_dim,
        max_len=args.max_len,
        batch_size=args.batch_size,
        dev_ratio=0.1,
    )

    # 2. Chuẩn bị tập Test độc lập
    test_loader, df_test, _ = prepare_test_dataloader(
        test_csv_path=test_csv_path,
        vocab=vocab,
        label_cols=label_cols,
        text_col="comment_clean",
        batch_size=args.batch_size,
        max_len=args.max_len,
    )

    # 3. Khởi tạo mô hình YoonKimCNN
    model = YoonKimCNN(
        vocab_size=len(vocab),
        embedding_dim=args.embedding_dim,
        num_classes=len(label_cols),
        filter_sizes=[3, 4, 5],
        num_filters=100,
        dropout_p=0.5,
        l2_constraint=3.0,
        mode=args.mode,
        pretrained_embeddings=pretrained_matrix,
        is_multilabel=True,
    )

    save_path = os.path.join(current_dir, f"kim_cnn_{args.mode}.pt")

    # 4. Huấn luyện và tự động đánh giá tập Test sau mỗi epoch
    train_kim_cnn(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        test_loader=test_loader,
        df_test=df_test,
        label_cols=label_cols,
        num_epochs=args.epochs,
        learning_rate=1.0,
        patience=args.patience,
        save_path=save_path,
        results_dir=results_dir,
    )


if __name__ == "__main__":
    main()
