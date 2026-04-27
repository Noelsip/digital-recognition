"""
Hand Character Recognition - Training Script

Train CNN untuk recognize 36 karakter (0-9 + A-Z) dari MNIST + EMNIST.

Usage:
    python train.py
    python train.py --epochs 15
    python train.py --no-augment
"""

import argparse
import os
import sys
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_loader import load_combined

NUM_CLASSES = 36  # 0-9 (10) + A-Z (26)


def build_model():
    """CNN untuk 36 classes. Sedikit lebih besar dari versi MNIST-only."""
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.Dropout(0.25),
        
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train(epochs=10, batch_size=128, augment=True, output_dir='models'):
    # Load combined dataset (digits + letters)
    (x_train, y_train), (x_test, y_test) = load_combined()
    
    # Preprocess
    x_train = x_train.astype('float32').reshape(-1, 28, 28, 1) / 255.0
    x_test = x_test.astype('float32').reshape(-1, 28, 28, 1) / 255.0
    y_train = to_categorical(y_train, NUM_CLASSES)
    y_test = to_categorical(y_test, NUM_CLASSES)
    
    # Build & train
    model = build_model()
    model.summary()
    
    if augment:
        datagen = ImageDataGenerator(
            rotation_range=10, zoom_range=0.1,
            width_shift_range=0.1, height_shift_range=0.1
        )
        datagen.fit(x_train)
        history = model.fit(
            datagen.flow(x_train, y_train, batch_size=batch_size),
            epochs=epochs,
            validation_data=(x_test, y_test),
            verbose=1
        )
    else:
        history = model.fit(
            x_train, y_train,
            batch_size=batch_size, epochs=epochs,
            validation_data=(x_test, y_test),
            verbose=1
        )
    
    # Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f'\n🎯 Final Test Accuracy: {test_acc*100:.2f}%')
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    keras_path = os.path.join(output_dir, 'char_model.keras')
    h5_path = os.path.join(output_dir, 'char_model.h5')
    model.save(keras_path)
    model.save(h5_path)
    print(f'\n💾 Model saved:')
    print(f'  - {keras_path}')
    print(f'  - {h5_path}')
    
    return model, history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train character recognition model')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--no-augment', action='store_true')
    parser.add_argument('--output-dir', type=str, default='models')
    args = parser.parse_args()
    
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        augment=not args.no_augment,
        output_dir=args.output_dir
    )
