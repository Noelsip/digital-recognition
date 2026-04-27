"""
Quick Start - Train (jika perlu) lalu buka drawing app.

Usage:
    python quickstart.py
    python quickstart.py --skip-train
"""

import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-train', action='store_true')
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()
    
    root = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(root, 'models', 'char_model.keras')
    
    if not args.skip_train and not os.path.exists(model_path):
        print('Training model...')
        result = subprocess.run(
            [sys.executable, os.path.join(root, 'src', 'train.py'),
             '--epochs', str(args.epochs),
             '--output-dir', os.path.join(root, 'models')],
            cwd=root
        )
        if result.returncode != 0:
            print('Training gagal.')
            return 1
    elif os.path.exists(model_path):
        print(f'✓ Model sudah ada: {model_path}')
    else:
        print(f'Model tidak ada di {model_path} dan --skip-train aktif.')
        return 1
    
    subprocess.run([sys.executable, os.path.join(root, 'interface', 'draw_app.py'),
                    '--model', model_path])
    return 0


if __name__ == '__main__':
    sys.exit(main())
