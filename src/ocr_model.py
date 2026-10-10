"""CRNN (CNN + BiLSTM + CTC) — наш распознаватель номера. Обучается на нашем
синтетическом датасете UZ-номеров. Вход: серый кроп 32x128. Выход: строка.
"""
from __future__ import annotations
import torch
import torch.nn as nn

# charset ДОЛЖЕН совпадать с генератором (data/gen_synth.py)
LETTERS = "ABCDEFGHJKLMNPRSTUVXYZ"
DIGITS = "0123456789"
CHARS = DIGITS + LETTERS          # индекс 0 зарезервирован под CTC-blank -> сдвиг +1
IDX2CH = {0: ""} | {i + 1: c for i, c in enumerate(CHARS)}
CH2IDX = {c: i + 1 for i, c in enumerate(CHARS)}
NUM_CLASSES = len(CHARS) + 1      # + blank


def encode(text: str) -> list[int]:
    return [CH2IDX[c] for c in text if c in CH2IDX]


def ctc_greedy_decode(logits: torch.Tensor) -> list[str]:
    """logits: (T, B, C) -> список строк (схлопывание повторов + удаление blank)."""
    idx = logits.argmax(2).permute(1, 0)  # (B, T)
    out = []
    for seq in idx.tolist():
        prev, s = 0, []
        for t in seq:
            if t != prev and t != 0:
                s.append(IDX2CH.get(t, ""))
            prev = t
        out.append("".join(s))
    return out


class CRNN(nn.Module):
    def __init__(self, n_classes: int = NUM_CLASSES, h: int = 32):
        super().__init__()
        def blk(i, o, k=3, s=1, p=1, pool=None):
            layers = [nn.Conv2d(i, o, k, s, p), nn.BatchNorm2d(o), nn.ReLU(True)]
            if pool:
                layers.append(nn.MaxPool2d(*pool))
            return layers
        self.cnn = nn.Sequential(
            *blk(1, 64, pool=(2, 2)),        # 32x128 -> 16x64
            *blk(64, 128, pool=(2, 2)),      # -> 8x32
            *blk(128, 256),
            *blk(256, 256, pool=((2, 1), (2, 1))),   # -> 4x32
            *blk(256, 512),
            *blk(512, 512, pool=((2, 1), (2, 1))),   # -> 2x32
            *blk(512, 512, k=2, s=1, p=0),           # -> 1x31
        )
        self.rnn = nn.LSTM(512, 256, num_layers=2, bidirectional=True, batch_first=False)
        self.fc = nn.Linear(512, n_classes)

    def forward(self, x):
        f = self.cnn(x)                 # (B, 512, 1, W)
        b, c, hh, w = f.size()
        assert hh == 1, f"height must collapse to 1, got {hh}"
        f = f.squeeze(2).permute(2, 0, 1)   # (W, B, 512)
        r, _ = self.rnn(f)
        return self.fc(r)               # (W, B, n_classes) -> logits для CTC
