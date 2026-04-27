"""
Hand Character Recognition - Prediction Module

Predict 0-9 dan A-Z dari gambar.
Mode: 'digits' (0-9 only), 'letters' (A-Z only), 'mixed' (both).
"""

import argparse
import os
import sys
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_loader import index_to_char


class CharPredictor:
    """Predictor untuk 36 classes (0-9 + A-Z)."""
    
    def __init__(self, model_path='models/char_model.keras'):
        print(f'Loading model dari {model_path}...')
        self.model = tf.keras.models.load_model(model_path)
        print('Model loaded.')
    
    @staticmethod
    def preprocess_image(img):
        """Preprocess PIL image jadi array MNIST-style 28x28."""
        img = img.convert('L')
        arr = np.array(img)
        if arr.mean() > 127:
            img = ImageOps.invert(img)
        bbox = img.getbbox()
        if bbox:
            img = img.crop(bbox)
        img.thumbnail((20, 20), Image.Resampling.LANCZOS)
        new_img = Image.new('L', (28, 28), 0)
        offset = ((28 - img.width) // 2, (28 - img.height) // 2)
        new_img.paste(img, offset)
        arr = np.array(new_img).astype('float32') / 255.0
        return arr.reshape(1, 28, 28, 1)
    
    def predict_from_array(self, arr, mode='mixed'):
        """
        Predict dari array 28x28.
        
        Args:
            arr: numpy array 28x28
            mode: 'digits' (cuma 0-9), 'letters' (cuma A-Z), 'mixed' (semua)
        
        Returns:
            (char, confidence, all_36_probs)
        """
        if arr.shape != (28, 28):
            raise ValueError(f'Array harus 28x28, dapat {arr.shape}')
        
        x = arr.astype('float32')
        if x.max() > 1.0:
            x = x / 255.0
        x = x.reshape(1, 28, 28, 1)
        
        probs = self.model.predict(x, verbose=0)[0]
        
        # Filter berdasarkan mode
        if mode == 'digits':
            # Mask out letter classes
            masked = probs.copy()
            masked[10:] = 0
            idx = int(np.argmax(masked))
        elif mode == 'letters':
            masked = probs.copy()
            masked[:10] = 0
            idx = int(np.argmax(masked))
        else:  # mixed
            idx = int(np.argmax(probs))
        
        char = index_to_char(idx)
        confidence = float(probs[idx])
        return char, confidence, probs.tolist()
    
    def predict_from_file(self, image_path, mode='mixed'):
        """Predict dari file gambar."""
        img = Image.open(image_path)
        x = self.preprocess_image(img)
        probs = self.model.predict(x, verbose=0)[0]
        
        if mode == 'digits':
            masked = probs.copy()
            masked[10:] = 0
            idx = int(np.argmax(masked))
        elif mode == 'letters':
            masked = probs.copy()
            masked[:10] = 0
            idx = int(np.argmax(masked))
        else:
            idx = int(np.argmax(probs))
        
        char = index_to_char(idx)
        confidence = float(probs[idx])
        return char, confidence, probs.tolist()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict character from image')
    parser.add_argument('image', type=str)
    parser.add_argument('--model', type=str, default='models/char_model.keras')
    parser.add_argument('--mode', type=str, default='mixed',
                        choices=['digits', 'letters', 'mixed'])
    args = parser.parse_args()
    
    predictor = CharPredictor(args.model)
    char, conf, probs = predictor.predict_from_file(args.image, args.mode)
    
    print(f'\nPredicted: {char}')
    print(f'Confidence: {conf*100:.2f}%')
    
    # Top 5
    top5 = sorted(enumerate(probs), key=lambda x: -x[1])[:5]
    print('\nTop 5:')
    for idx, p in top5:
        bar = '█' * int(p * 30)
        print(f'  {index_to_char(idx)}: {p*100:5.2f}% {bar}')
