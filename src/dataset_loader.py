"""
Dataset Loader - Download MNIST + EMNIST dari Kaggle pakai kagglehub.

MNIST  : digits 0-9 (10 classes)
EMNIST : letters A-Z (26 classes), uppercase only
"""

import os
import struct
import subprocess
import sys
import numpy as np


def _ensure_kagglehub():
    try:
        import kagglehub
        return kagglehub
    except ImportError:
        print('Installing kagglehub...')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'kagglehub'])
        import kagglehub
        return kagglehub


def _load_idx_images(filename):
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)


def _load_idx_labels(filename):
    with open(filename, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)


def _find_file(folder, must_contain, must_not_contain=None):
    """Cari file dengan substring. must_not_contain untuk exclude."""
    must_not_contain = must_not_contain or []
    for root, dirs, files in os.walk(folder):
        for f in files:
            fl = f.lower()
            if all(p in fl for p in must_contain) and not any(n in fl for n in must_not_contain):
                return os.path.join(root, f)
    raise FileNotFoundError(f'File {must_contain} tidak ketemu di {folder}')


# ============================================================
# MNIST — Digits 0-9
# ============================================================

def load_mnist():
    """Download dan load MNIST digits."""
    kagglehub = _ensure_kagglehub()
    print('Downloading MNIST...')
    path = kagglehub.dataset_download('hojjatk/mnist-dataset')
    
    x_train = _load_idx_images(_find_file(path, ['train', 'image']))
    y_train = _load_idx_labels(_find_file(path, ['train', 'label']))
    x_test  = _load_idx_images(_find_file(path, ['t10k', 'image']))
    y_test  = _load_idx_labels(_find_file(path, ['t10k', 'label']))
    
    print(f'  Train: {x_train.shape}, Test: {x_test.shape}')
    return (x_train, y_train), (x_test, y_test)


# ============================================================
# EMNIST — Letters A-Z
# ============================================================

def load_emnist_letters():
    """
    Download dan load EMNIST letters (A-Z, 26 classes).
    
    Note: EMNIST images perlu di-transpose dan flip karena format Kaggle 
    aslinya rotated. Labels awalnya 1-26, kita ubah jadi 0-25.
    """
    kagglehub = _ensure_kagglehub()
    print('Downloading EMNIST...')
    path = kagglehub.dataset_download('crawford/emnist')
    
    # File EMNIST naming: emnist-letters-train-images-idx3-ubyte
    train_img_path = _find_file(path, ['letters', 'train', 'image'], ['mapping'])
    train_lbl_path = _find_file(path, ['letters', 'train', 'label'], ['mapping'])
    test_img_path  = _find_file(path, ['letters', 'test', 'image'], ['mapping'])
    test_lbl_path  = _find_file(path, ['letters', 'test', 'label'], ['mapping'])
    
    x_train = _load_idx_images(train_img_path)
    y_train = _load_idx_labels(train_lbl_path)
    x_test  = _load_idx_images(test_img_path)
    y_test  = _load_idx_labels(test_lbl_path)
    
    # EMNIST images perlu di-transpose + flip horizontal (format aslinya rotated 90° + mirrored)
    x_train = np.transpose(x_train, (0, 2, 1))
    x_test = np.transpose(x_test, (0, 2, 1))
    
    # Labels EMNIST letters dimulai dari 1 (A=1, B=2, ..., Z=26). Ubah ke 0-25.
    y_train = y_train - 1
    y_test = y_test - 1
    
    print(f'  Train: {x_train.shape}, Test: {x_test.shape}')
    print(f'  Classes: {len(np.unique(y_train))} letters (A-Z)')
    return (x_train, y_train), (x_test, y_test)


# ============================================================
# Combined — Digits + Letters (36 classes: 0-9 + A-Z)
# ============================================================

def load_combined():
    """
    Gabungan MNIST + EMNIST letters jadi 36 classes:
      0-9   : digits 0-9
      10-35 : letters A-Z
    """
    print('Loading MNIST...')
    (x_d_train, y_d_train), (x_d_test, y_d_test) = load_mnist()
    
    print('\nLoading EMNIST letters...')
    (x_l_train, y_l_train), (x_l_test, y_l_test) = load_emnist_letters()
    
    # Shift letter labels: 0-25 → 10-35
    y_l_train = y_l_train + 10
    y_l_test = y_l_test + 10
    
    # Concatenate
    x_train = np.concatenate([x_d_train, x_l_train], axis=0)
    y_train = np.concatenate([y_d_train, y_l_train], axis=0)
    x_test = np.concatenate([x_d_test, x_l_test], axis=0)
    y_test = np.concatenate([y_d_test, y_l_test], axis=0)
    
    # Shuffle training set
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(x_train))
    x_train, y_train = x_train[idx], y_train[idx]
    
    print(f'\nCombined dataset:')
    print(f'  Train: {x_train.shape}, Test: {x_test.shape}')
    print(f'  Total classes: 36 (0-9 digits, 10-35 = A-Z)')
    return (x_train, y_train), (x_test, y_test)


# ============================================================
# Label mapping helper
# ============================================================

def index_to_char(idx):
    """Convert class index ke karakter."""
    if 0 <= idx <= 9:
        return str(idx)
    elif 10 <= idx <= 35:
        return chr(ord('A') + idx - 10)
    return '?'


def char_to_index(ch):
    """Convert karakter ke class index."""
    if ch.isdigit():
        return int(ch)
    elif ch.isalpha():
        return 10 + (ord(ch.upper()) - ord('A'))
    return -1


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='combined',
                        choices=['mnist', 'emnist', 'combined'])
    args = parser.parse_args()
    
    if args.dataset == 'mnist':
        load_mnist()
    elif args.dataset == 'emnist':
        load_emnist_letters()
    else:
        load_combined()
