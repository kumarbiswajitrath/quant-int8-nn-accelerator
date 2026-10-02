import numpy as np
import torch
import torchvision as tv

SHIFT = 24  # fixed-point shift used for requantization

# ---------- load float model ----------
sd = torch.load('float_model.pt')
W1 = sd['1.weight'].numpy()   # (32, 784)
b1 = sd['1.bias'].numpy()     # (32,)
W2 = sd['3.weight'].numpy()   # (10, 32)
b2 = sd['3.bias'].numpy()     # (10,)

# ---------- load MNIST as numpy (pixels in 0..1) ----------
train_ds = tv.datasets.MNIST('../data', train=True, download=True)
test_ds = tv.datasets.MNIST('../data', train=False, download=True)
Xtr = train_ds.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
Xte = test_ds.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
yte = test_ds.targets.numpy()

# ---------- float accuracy (sanity check) ----------
h_float = np.maximum(Xte @ W1.T + b1, 0)
float_acc = ((h_float @ W2.T + b2).argmax(1) == yte).mean() * 100
print(f"Float accuracy : {float_acc:.2f}%")

# ---------- scales ----------
s_x = 1.0 / 127.0                               # input: pixel*127 -> 0..127
s_w1 = np.abs(W1).max() / 127.0
s_w2 = np.abs(W2).max() / 127.0
h_train = np.maximum(Xtr @ W1.T + b1, 0)
s_h = h_train.max() / 127.0                     # hidden activation scale

# ---------- quantize parameters ----------
W1_q = np.round(W1 / s_w1).astype(np.int8)
W2_q = np.round(W2 / s_w2).astype(np.int8)
b1_q = np.round(b1 / (s_w1 * s_x)).astype(np.int32)
b2_q = np.round(b2 / (s_w2 * s_h)).astype(np.int32)
M1 = s_w1 * s_x / s_h
M1_int = int(round(M1 * (1 << SHIFT)))

print(f"s_w1={s_w1:.6f}  s_w2={s_w2:.6f}  s_h={s_h:.6f}")
print(f"M1={M1:.8f}  M1_int={M1_int}  (SHIFT={SHIFT})")

# ---------- integer-only inference (golden model) ----------
def infer_int(x_q):
    """x_q: (N,784) ints 0..127. Returns layer-2 int32 scores (N,10)."""
    acc1 = x_q.astype(np.int64) @ W1_q.T.astype(np.int64) + b1_q.astype(np.int64)
    h_q = (acc1 * M1_int + (1 << (SHIFT - 1))) >> SHIFT
    h_q = np.clip(h_q, 0, 127)                  # clip at 0 also acts as ReLU
    acc2 = h_q @ W2_q.T.astype(np.int64) + b2_q.astype(np.int64)
    return acc1, h_q, acc2

x_q = np.round(Xte * 127).astype(np.int64)
_, _, acc2 = infer_int(x_q)
int_acc = (acc2.argmax(1) == yte).mean() * 100
print(f"INT8 accuracy  : {int_acc:.2f}%")
print(f"Accuracy drop  : {float_acc - int_acc:.2f}%")

# ---------- checks useful for the hardware later ----------
acc1, _, _ = infer_int(x_q)
print(f"max |acc1| = {np.abs(acc1).max()}  (must fit in int32)")
print(f"max |acc1*M1_int| = {np.abs(acc1).max() * M1_int}  (must fit in 48 bits = {2**47})")
print(f"max |acc2| = {np.abs(acc2).max()}")

# ---------- save for the next step ----------
np.savez('quant_params.npz', W1_q=W1_q, W2_q=W2_q, b1_q=b1_q, b2_q=b2_q,
         M1_int=M1_int, SHIFT=SHIFT)
print("Saved quant_params.npz")
