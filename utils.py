# utils.py
import os
import numpy as np

def text_to_line(text: str):
    """
    将文本按行分割为字符串列表。
    """
    return text.strip().splitlines()

def load_label_file(label_path):
    ids = []
    labels = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            ids.append(parts[0])
            labels.append(int(parts[1]))
    y = np.array(labels)
    return ids, y

def load_vec_file(file_path: str):
    """
    读取 .vec 文件，返回 id 列表和 numpy 向量矩阵。
    """
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    header = lines[0].strip().split()
    n_lines, dim = int(header[0]), int(header[1])

    ids = []
    vectors = []

    for line in lines[1:]:
        parts = line.strip().split()
        ids.append(parts[0])
        vectors.append([float(x) for x in parts[1:]])

    vectors = np.array(vectors)
    assert vectors.shape == (n_lines, dim), f"维度不匹配: got {vectors.shape}, expected ({n_lines}, {dim})"

    return ids, vectors

def save_vec_file(ids, vectors, file_path):
    """
    保存 id + 向量 到 .vec 文件格式。
    """
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"{len(ids)} {vectors.shape[1]}\n")
        for i, vec in enumerate(vectors):
            vec_str = " ".join(f"{x:.6f}" for x in vec)
            f.write(f"{ids[i]} {vec_str}\n")

def ensure_dir_exists(path):
    """
    确保目录存在，不存在则创建。
    """
    os.makedirs(path, exist_ok=True)
