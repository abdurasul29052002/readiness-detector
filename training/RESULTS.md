# Training natijalari — Model solishtirish

## Dataset (Classification)
- **Balanced dataset**: 608,021 train / 45,080 val
- **Klasslar**: 7 (hand-raising, read, write, discuss, bow-head, turn-head, standing)
- **Balanslashtirish**: Katta klasslar 100K gacha undersampling

---

## 0. YOLOv8s Detection (Dastlabki model — to'liq sinf rasmlari)

**Dataset**: 70,750 rasm (to'liq sinf xonasi rasmlari, detection format)

| Metrika | Natija |
|---------|--------|
| **mAP50** | **50.74%** |
| mAP50-95 | 41.9% |
| Precision | 54.23% |
| Recall | 46.52% |
| Model size | 22 MB (YOLOv8s) |
| Image size | 640x640 |
| Model fayli | `models/best.pt` |

### Per-class natijalar

| Klass | Precision | Recall | mAP50 |
|-------|-----------|--------|-------|
| hand-raising | 85.7% | 85.4% | 92.2% |
| read | 59.0% | 52.2% | 58.3% |
| write | 82.8% | 73.0% | 82.1% |
| discuss | 33.5% | 22.7% | 21.9% |
| bow-head | 0.3% | 0.7% | 1.1% |
| turn-head | 42.7% | 12.3% | 17.0% |
| standing | 75.5% | 79.3% | 82.5% |

### Muammo
- **bow-head** deyarli aniqlanmagan (mAP50: 1.1%)
- **turn-head** va **discuss** past natija
- Shuning uchun individual classification ga o'tildi

---

## 1. YOLOv8s-cls (Baseline 1)

| Metrika | Natija |
|---------|--------|
| **Top-1 Accuracy** | **69.58%** |
| Top-5 Accuracy | 98.9% |
| Epochs | 35 (50 dan) |
| Best epoch | 31 |
| Model size | 30 MB (5.09M params) |
| Image size | 224x224 |
| Batch size | 64 |
| Optimizer | AdamW (lr=0.001) |
| Training vaqti | ~8 soat (RTX 3070 Laptop) |
| Model fayli | `models/best_cls.pt` |

### Kuzatishlar
- Epoch 20 dan keyin plateau — accuracy 69.2-69.6% orasida
- Loss 0.37 gacha tushdi
- Class weights qo'llanilmadi (Ultralytics cls parametri faqat scalar qabul qiladi)

---

## 2. CNN - ResNet50 (Baseline 2)

| Metrika | Natija |
|---------|--------|
| **Top-1 Accuracy** | *kutilmoqda* |
| Epochs | 35 |
| Model fayli | `models/best_resnet50.pt` |

---

## 3. CNN + Pose Fusion (Yangi algoritm)

| Metrika | Natija |
|---------|--------|
| **Top-1 Accuracy** | *kutilmoqda* |
| Epochs | 35 |
| Model fayli | `models/best_fusion.pt` |
