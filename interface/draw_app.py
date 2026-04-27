"""
Hand Character Recognition - Desktop Drawing Interface

Features:
- Smooth drawing dengan line interpolation
- Multi-character segmentation (tulis "11" → predict "11", tulis "AB" → predict "AB")
- Mode selector: Digits / Letters / Mixed
- Real-time prediction
"""

import argparse
import os
import sys
import tkinter as tk
from tkinter import messagebox

import numpy as np
from PIL import Image, ImageDraw

try:
    from scipy import ndimage
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from predict import CharPredictor
from dataset_loader import index_to_char


CANVAS_SIZE = 480  # Lebih lebar buat multi-character
CANVAS_HEIGHT = 280
BRUSH_RADIUS = 12


class CharDrawApp:
    def __init__(self, root, model_path='models/char_model.keras'):
        self.root = root
        self.root.title('Hand Character Recognition')
        self.root.resizable(False, False)
        
        if not HAS_SCIPY:
            messagebox.showerror(
                'scipy belum terinstall',
                'Install dulu: pip install scipy\n\nLalu jalankan ulang.'
            )
            sys.exit(1)
        
        if not os.path.exists(model_path):
            messagebox.showerror(
                'Model tidak ditemukan',
                f'Model tidak ada di:\n{model_path}\n\n'
                'Train dulu dengan: python src/train.py'
            )
            sys.exit(1)
        
        self.predictor = CharPredictor(model_path)
        self.image = Image.new('L', (CANVAS_SIZE, CANVAS_HEIGHT), 0)
        self.draw_obj = ImageDraw.Draw(self.image)
        self.last_x = None
        self.last_y = None
        self.predict_after_id = None
        self.mode = tk.StringVar(value='mixed')
        
        self._build_ui()
        self._bind_events()
    
    def _build_ui(self):
        main = tk.Frame(self.root, bg='#1e1e2e', padx=20, pady=20)
        main.pack()
        
        tk.Label(
            main, text='Hand Character Recognition',
            font=('Helvetica', 18, 'bold'),
            bg='#1e1e2e', fg='#cdd6f4'
        ).pack(pady=(0, 5))
        
        tk.Label(
            main, text='Tulis 0-9 atau A-Z di kanvas — pisahkan tiap karakter dengan jarak',
            font=('Helvetica', 10),
            bg='#1e1e2e', fg='#a6adc8'
        ).pack(pady=(0, 12))
        
        # Mode selector
        mode_frame = tk.Frame(main, bg='#1e1e2e')
        mode_frame.pack(pady=(0, 12))
        
        tk.Label(
            mode_frame, text='Mode:',
            font=('Helvetica', 10, 'bold'),
            bg='#1e1e2e', fg='#cdd6f4'
        ).pack(side='left', padx=(0, 8))
        
        for label, val in [('🔢 Digits (0-9)', 'digits'),
                            ('🔤 Letters (A-Z)', 'letters'),
                            ('🔀 Mixed (0-9 + A-Z)', 'mixed')]:
            rb = tk.Radiobutton(
                mode_frame, text=label, variable=self.mode, value=val,
                command=self.predict_now,
                font=('Helvetica', 10),
                bg='#1e1e2e', fg='#cdd6f4',
                selectcolor='#313244',
                activebackground='#1e1e2e',
                activeforeground='#89b4fa'
            )
            rb.pack(side='left', padx=4)
        
        # Content area
        content = tk.Frame(main, bg='#1e1e2e')
        content.pack()
        
        # Canvas
        canvas_frame = tk.Frame(content, bg='#313244', padx=3, pady=3)
        canvas_frame.grid(row=0, column=0, padx=(0, 20))
        
        self.canvas = tk.Canvas(
            canvas_frame,
            width=CANVAS_SIZE, height=CANVAS_HEIGHT,
            bg='black', highlightthickness=0,
            cursor='pencil'
        )
        self.canvas.pack()
        
        # Right panel
        right = tk.Frame(content, bg='#1e1e2e')
        right.grid(row=0, column=1, sticky='n')
        
        tk.Label(
            right, text='Prediksi:',
            font=('Helvetica', 11),
            bg='#1e1e2e', fg='#a6adc8'
        ).pack(anchor='w')
        
        self.result_label = tk.Label(
            right, text='—',
            font=('Helvetica', 48, 'bold'),
            bg='#1e1e2e', fg='#89b4fa'
        )
        self.result_label.pack(pady=(5, 5))
        
        self.confidence_label = tk.Label(
            right, text='Confidence: —',
            font=('Helvetica', 10),
            bg='#1e1e2e', fg='#a6adc8'
        )
        self.confidence_label.pack(pady=(0, 12))
        
        # Detail per-char
        tk.Label(
            right, text='Detail per karakter:',
            font=('Helvetica', 10, 'bold'),
            bg='#1e1e2e', fg='#cdd6f4'
        ).pack(anchor='w')
        
        self.detail_text = tk.Text(
            right, width=24, height=10,
            font=('Consolas', 10),
            bg='#313244', fg='#cdd6f4',
            relief='flat', padx=8, pady=8,
            wrap='word'
        )
        self.detail_text.pack(pady=(4, 0))
        self.detail_text.insert('1.0', 'Belum ada karakter terdeteksi')
        self.detail_text.config(state='disabled')
        
        # Buttons
        btn_frame = tk.Frame(main, bg='#1e1e2e')
        btn_frame.pack(pady=(15, 0))
        
        tk.Button(
            btn_frame, text='🗑  Clear',
            command=self.clear_canvas,
            font=('Helvetica', 11, 'bold'),
            bg='#f38ba8', fg='white',
            activebackground='#eb6f92', activeforeground='white',
            relief='flat', padx=20, pady=8, cursor='hand2'
        ).pack(side='left', padx=5)
        
        tk.Button(
            btn_frame, text='🔍  Predict',
            command=self.predict_now,
            font=('Helvetica', 11, 'bold'),
            bg='#a6e3a1', fg='#1e1e2e',
            activebackground='#94e2d5', activeforeground='#1e1e2e',
            relief='flat', padx=20, pady=8, cursor='hand2'
        ).pack(side='left', padx=5)
    
    def _bind_events(self):
        self.canvas.bind('<Button-1>', self.start_draw)
        self.canvas.bind('<B1-Motion>', self.draw_smooth)
        self.canvas.bind('<ButtonRelease-1>', self.end_draw)
    
    def start_draw(self, event):
        self.last_x = event.x
        self.last_y = event.y
        self._draw_dot(event.x, event.y)
    
    def draw_smooth(self, event):
        x, y = event.x, event.y
        if self.last_x is not None:
            self._draw_line(self.last_x, self.last_y, x, y)
        self.last_x = x
        self.last_y = y
        self._schedule_predict()
    
    def end_draw(self, event):
        self.last_x = None
        self.last_y = None
        self._schedule_predict()
    
    def _draw_dot(self, x, y):
        r = BRUSH_RADIUS
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill='white', outline='white')
        self.draw_obj.ellipse([x-r, y-r, x+r, y+r], fill=255)
    
    def _draw_line(self, x1, y1, x2, y2):
        r = BRUSH_RADIUS
        self.canvas.create_line(
            x1, y1, x2, y2,
            fill='white', width=r*2,
            capstyle=tk.ROUND, smooth=True, joinstyle=tk.ROUND
        )
        self.canvas.create_oval(x2-r, y2-r, x2+r, y2+r, fill='white', outline='white')
        self.draw_obj.line([x1, y1, x2, y2], fill=255, width=r*2)
        self.draw_obj.ellipse([x2-r, y2-r, x2+r, y2+r], fill=255)
    
    def _schedule_predict(self):
        if self.predict_after_id is not None:
            self.root.after_cancel(self.predict_after_id)
        self.predict_after_id = self.root.after(250, self.predict_now)
    
    def clear_canvas(self):
        self.canvas.delete('all')
        self.image = Image.new('L', (CANVAS_SIZE, CANVAS_HEIGHT), 0)
        self.draw_obj = ImageDraw.Draw(self.image)
        self.result_label.config(text='—', fg='#89b4fa', font=('Helvetica', 48, 'bold'))
        self.confidence_label.config(text='Confidence: —')
        self.detail_text.config(state='normal')
        self.detail_text.delete('1.0', tk.END)
        self.detail_text.insert('1.0', 'Belum ada karakter terdeteksi')
        self.detail_text.config(state='disabled')
    
    def _segment_characters(self):
        """
        Pisahkan multi-character pakai connected components.
        Returns: list of (28x28 array, bounding_box) untuk tiap karakter, sorted left-to-right.
        """
        arr = np.array(self.image)
        if arr.max() == 0:
            return []
        
        # Dilate sedikit dulu agar stroke yang dekat tapi tidak nyambung 
        # (misal stroke "4" yang terputus) bisa kebaca sebagai 1 karakter
        binary = (arr > 30).astype(np.uint8)
        binary = ndimage.binary_dilation(binary, iterations=2)
        
        labeled, num_features = ndimage.label(binary)
        if num_features == 0:
            return []
        
        # Get bounding box tiap component, filter noise
        components = []
        for i in range(1, num_features + 1):
            ys, xs = np.where(labeled == i)
            if len(xs) < 50:  # skip noise kecil
                continue
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            components.append((x_min, y_min, x_max, y_max))
        
        # Sort kiri ke kanan
        components.sort(key=lambda c: c[0])
        
        # Crop tiap component, preprocess ke 28x28
        results = []
        for x_min, y_min, x_max, y_max in components:
            pad = 8
            x1 = max(0, x_min - pad)
            y1 = max(0, y_min - pad)
            x2 = min(arr.shape[1], x_max + pad)
            y2 = min(arr.shape[0], y_max + pad)
            
            crop = self.image.crop((x1, y1, x2, y2))
            
            # Resize sambil jaga aspect ratio, pad ke 28x28 (MNIST/EMNIST style)
            crop.thumbnail((20, 20), Image.Resampling.LANCZOS)
            new_img = Image.new('L', (28, 28), 0)
            offset = ((28 - crop.width) // 2, (28 - crop.height) // 2)
            new_img.paste(crop, offset)
            
            results.append((np.array(new_img), (x1, y1, x2, y2)))
        
        return results
    
    def predict_now(self):
        """Predict semua karakter, gabungkan jadi string."""
        chars = self._segment_characters()
        if not chars:
            return
        
        mode = self.mode.get()
        predictions = []
        confidences = []
        
        for char_arr, bbox in chars:
            ch, conf, _ = self.predictor.predict_from_array(char_arr, mode=mode)
            predictions.append(ch)
            confidences.append(conf)
        
        # Combine jadi string
        result_str = ''.join(predictions)
        avg_conf = float(np.mean(confidences))
        
        # Adjust font size berdasarkan jumlah karakter
        if len(result_str) <= 2:
            font_size = 56
        elif len(result_str) <= 4:
            font_size = 40
        elif len(result_str) <= 7:
            font_size = 28
        else:
            font_size = 20
        
        self.result_label.config(text=result_str, font=('Helvetica', font_size, 'bold'))
        
        # Color by confidence
        if avg_conf > 0.85:
            self.result_label.config(fg='#a6e3a1')
        elif avg_conf > 0.6:
            self.result_label.config(fg='#f9e2af')
        else:
            self.result_label.config(fg='#f38ba8')
        
        if len(predictions) == 1:
            self.confidence_label.config(text=f'Confidence: {avg_conf*100:.1f}%')
        else:
            self.confidence_label.config(
                text=f'{len(predictions)} karakter • avg conf: {avg_conf*100:.1f}%'
            )
        
        # Detail panel
        self.detail_text.config(state='normal')
        self.detail_text.delete('1.0', tk.END)
        for i, (ch, conf) in enumerate(zip(predictions, confidences)):
            line = f'{i+1}. "{ch}"  →  {conf*100:.1f}%\n'
            self.detail_text.insert(tk.END, line)
        self.detail_text.config(state='disabled')


def main():
    parser = argparse.ArgumentParser(description='Character recognition drawing app')
    parser.add_argument('--model', type=str, default='models/char_model.keras')
    args = parser.parse_args()
    
    # Resolve path relatif
    if not os.path.isabs(args.model) and not os.path.exists(args.model):
        alt = os.path.join(os.path.dirname(__file__), '..', args.model)
        if os.path.exists(alt):
            args.model = alt
    
    root = tk.Tk()
    app = CharDrawApp(root, model_path=args.model)
    root.mainloop()


if __name__ == '__main__':
    main()
