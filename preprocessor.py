# -*- coding: utf-8 -*-
"""
preprocessor.py — Module cầu nối (Bridge/Proxy) tiền xử lý cho thư mục Yoon Kim test
=====================================================================================
Mục đích:
    Cho phép deadpool.py và cnn.py gọi trực tiếp module preprocessor của dự án gốc
    mà không bị lỗi "Import preprocessor could not be resolved" (lỗi gạch đỏ IDE Pylance)
    khi làm việc trong thư mục con.

    Toàn bộ logic tiền xử lý 15 bước theo đặc tả Clean.md được kế thừa trực tiếp
    từ preprocessor.py ở thư mục gốc.
"""

import os
import sys

# Thêm thư mục gốc vào đường dẫn tìm kiếm của Python
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Nạp trực tiếp từ preprocessor.py ở thư mục gốc
try:
    import preprocessor as _root_preprocessor

    # Tái xuất khẩu (re-export) đầy đủ các class và hàm theo Clean.md
    TextPreprocessor = _root_preprocessor.TextPreprocessor
    prepare_tfidf_dataset = _root_preprocessor.prepare_tfidf_dataset
    get_tfidf_vectorizer = _root_preprocessor.get_tfidf_vectorizer
    parse_labels = _root_preprocessor.parse_labels
    EMO = _root_preprocessor.EMO
except Exception as e:
    raise ImportError(
        f"Không thể nạp preprocessor.py từ thư mục gốc '{PROJECT_ROOT}': {e}"
    )

__all__ = [
    "TextPreprocessor",
    "prepare_tfidf_dataset",
    "get_tfidf_vectorizer",
    "parse_labels",
    "EMO",
]
