
---

## 🛠️ Asset-Corrosion Image Classifier

This repository shows how to

1. Train a Convolutional Neural Network (CNN) that classifies photos of an asset into **normal** or **corroded**  
2. Evaluate the model on a held-out test set (Accuracy, Precision, Recall, F1, ROC-AUC)  
3. Serve the trained model through a lightweight **Flask** REST API (`/classify/`)  

> The code is intentionally minimal.  
> Swap in your favourite backbone, tune hyper-parameters, dockerise the API—everything still works.

---

## 📂 Project Layout

```bash
.
├── app.py                  # Flask REST API
├── model_wrapper.py        # Loads Keras model once; single-image inference
├── train_and_evaluate.py   # Training & evaluation script
├── requirements.txt        # Exact, pinned package versions
├── metrics.json            # Test metrics (created after training)
├── model_asset_classifier.h5 # Saved best model (created after training)
└── dataset/
    ├── train/
    │   ├── normal/      # images…
    │   └── corroded/
    └── test/
        ├── normal/
        └── corroded/
```

---

## 🖥️ System Requirements

| Category        | Minimum (works, slower)                                   | Recommended (faster / smoother)                                   |
|-----------------|-----------------------------------------------------------|-------------------------------------------------------------------|
| Operating System| Linux, macOS or Windows 10/11                             | Linux (Ubuntu 20.04+ LTS)                                         |
| Python Version  | 3.9 – 3.11                                                | 3.10 (64-bit)                                                     |
| CPU             | 2 cores, 64-bit                                           | 4+ cores, AVX-capable                                             |
| RAM             | 4 GB (training on small datasets)                         | 8–16 GB                                                           |
| GPU (optional)  | — (CPU-only training/inference)                           | NVIDIA GPU with CUDA 11.8 (≥ 4 GB VRAM) + latest driver + cuDNN 8 |
| Disk Space      | ~2 GB (code + virtual-env + model + few thousand images)  | 5 GB+ (room for checkpoints, logs, larger dataset)                |
| Network         | Not required for training; outbound access needed only to download pre-trained ImageNet weights on first run |

**Notes**

1. *CPU-only* training is fine for a few hundred images; expect minutes per epoch.  
   For tens of thousands of images, a CUDA-capable GPU is strongly advised.
2. If you use a GPU, install the matching `tensorflow-gpu` wheel or a recent `tensorflow` build with CUDA 11.8 support.  
3. Windows Subsystem for Linux 2 (WSL2) + CUDA can also be used if native Linux isn’t available.
4. Disk usage grows with dataset size and TensorBoard logs; adjust accordingly.

---

## 🚀 Quick-Start

1. **Create and activate a virtual environment (optional but recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate            # Linux / macOS
   # .\venv\Scripts\activate           # Windows
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Place your dataset** under `dataset/train/…` and `dataset/test/…` as shown above.

4. **Train the model**

   ```bash
   python train_and_evaluate.py \
         --data_dir   dataset \
         --epochs     10 \
         --batch_size 32
   ```

   After training you will have:

   * `model_asset_classifier.h5` (best checkpoint by `val_accuracy`)  
   * `metrics.json` with the evaluation metrics on the test set

5. **Serve the model**

   ```bash
   python app.py
   # ➜ http://localhost:5000/         # health-check
   # ➜ http://localhost:5000/classify/ (POST endpoint)
   ```

6. **Test the endpoint**

   ```bash
   curl -X POST http://localhost:5000/classify/ \
        -F "file=@/path/to/your_image.jpg"
   ```

   Response:

   ```json
   {
     "label": "corroded",
     "probabilities": {
       "corroded": 0.8764,
       "normal":   0.1236
     }
   }
   ```

---

## 📝 Script Arguments

```bash
python train_and_evaluate.py -h
```

| Flag             | Default | Description                                  |
|------------------|---------|----------------------------------------------|
| `--data_dir`     | dataset | Root folder containing `train/` & `test/`    |
| `--img_size`     | 224     | Image size (square)                          |
| `--batch_size`   | 32      | Mini-batch size                              |
| `--epochs`       | 10      | Number of training epochs                    |
| `--lr`           | 1e-4    | Learning-rate (Adam)                         |
| `--model_out`    | model_asset_classifier.h5 | Model filename            |

---

## ⚙️ Model Details

* **Backbone:** EfficientNet-B0 (ImageNet weights)  
* **Trainable layers:** Only the classification head (backbone frozen)  
* **Loss:** Binary cross-entropy  
* **Optimizer:** Adam (LR 1 × 10⁻⁴)  
* **Augmentations:** Rotation, shift, shear, zoom, horizontal flip  

Feel free to un-freeze layers for fine-tuning if you have enough data.

---

## 📈 Metrics Example (`metrics.json`)

```json
{
  "accuracy": 0.9467,
  "precision": 0.9231,
  "recall": 0.9231,
  "f1": 0.9231,
  "roc_auc": 0.9873
}
```

---

## 🌐 API Reference

| Method | Path          | Description                     |
|--------|---------------|---------------------------------|
| GET    | `/`           | Health-check (“API is live”)    |
| POST   | `/classify/`  | Classify one image (multipart)  |

### POST `/classify/`

* **Request (multipart/form-data)**  
  *Key:* `file` *Value:* binary image (JPG / PNG)

* **Response**

  ```json
  {
    "label": "normal" | "corroded",
    "probabilities": {
      "corroded": 0.0385,
      "normal":   0.9615
    }
  }
  ```

---

## 🛡️ Raising the Upload File-Size Limit (413 Error)

A 413 *Request Entity Too Large* can be triggered by Flask or your reverse-proxy:

1. **Flask / Werkzeug**

   ```python
   # app.py  – before defining routes
   app = Flask(__name__)
   app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024  # 4 MB
   ```

2. **Nginx**

   ```nginx
   http {
       client_max_body_size 5m;
   }
   # sudo nginx -s reload
   ```

3. **Apache**

   ```apache
   <Directory "/var/www/myapp">
       LimitRequestBody 5242880   # 5 MB
   </Directory>
   ```

Set the limit on every layer that processes the request, then reload/restart those services.

---

## 🐳 Docker (optional)

```Dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000
CMD ["python", "app.py"]
```

Build & run:

```bash
docker build -t corrosion-api .
docker run -p 5000:5000 corrosion-api
```

---

## 💡 Tips for Production

* **Fine-tuning:** Un-freeze the EfficientNet backbone after the head stabilises.  
* **Class imbalance:** Use `class_weight`, focal loss, or oversampling.  
* **Model formats:** Convert to TensorFlow SavedModel, ONNX, or TFLite for edge devices.  
* **Security:** Validate MIME types, restrict file size, enable HTTPS, add authentication / rate limiting.  
* **Scaling:** Use Gunicorn + Nginx or a managed service (GCP Cloud Run, AWS Fargate, etc.).

---

## 🤝 Contributing

PRs and issues are welcome! Please open a discussion if you want to add a new feature or improve the docs.

---

## 📜 License

MIT – do whatever you want, **no warranty**.

Happy hacking! 🚀