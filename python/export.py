import numpy as np
import torchvision as tv

N_TEST = 200   # number of test images exported for the Verilog testbench

p = np.load('quant_params.npz')
W1_q, W2_q = p['W1_q'], p['W2_q']
b1_q, b2_q = p['b1_q'], p['b2_q']
M1_int, SHIFT = int(p['M1_int']), int(p['SHIFT'])

test_ds = tv.datasets.MNIST('../data', train=False, download=True)
Xte = test_ds.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
yte = test_ds.targets.numpy()

x_q = np.round(Xte[:N_TEST] * 127).astype(np.int64)
y = yte[:N_TEST]

# ---------- integer-only inference (same as quantize.py) ----------
acc1 = x_q @ W1_q.T.astype(np.int64) + b1_q.astype(np.int64)
h_q = np.clip((acc1 * M1_int + (1 << (SHIFT - 1))) >> SHIFT, 0, 127)
acc2 = h_q @ W2_q.T.astype(np.int64) + b2_q.astype(np.int64)
pred = acc2.argmax(1)
print(f"Integer model on first {N_TEST} test images: {(pred == y).sum()}/{N_TEST} correct")

# ---------- helpers ----------
def hex8(v):   # int8 -> 2 hex digits (two's complement)
    return f"{int(v) & 0xFF:02x}"

def hex32(v):  # int32 -> 8 hex digits (two's complement)
    return f"{int(v) & 0xFFFFFFFF:08x}"

def write_lines(path, lines):
    with open(path, 'w') as f:
        f.write("\n".join(lines) + "\n")

D = '../data/'

# weights: neuron-major order, index = neuron*num_inputs + input
write_lines(D + 'w1.mem', [hex8(v) for v in W1_q.flatten()])   # 32*784 lines
write_lines(D + 'b1.mem', [hex32(v) for v in b1_q])            # 32 lines
write_lines(D + 'w2.mem', [hex8(v) for v in W2_q.flatten()])   # 10*32 lines
write_lines(D + 'b2.mem', [hex32(v) for v in b2_q])            # 10 lines

# test images: one pixel (0..127) per line, image-major (N_TEST*784 lines)
write_lines(D + 'test_images.mem', [hex8(v) for v in x_q.flatten()])
write_lines(D + 'test_labels.txt', [str(int(v)) for v in y])

# golden references for the testbench comparison
write_lines(D + 'golden_scores.txt', [" ".join(str(int(v)) for v in row) for row in acc2])
write_lines(D + 'golden_h.txt', [" ".join(str(int(v)) for v in row) for row in h_q])

# constants for Verilog
with open(D + 'params.vh', 'w') as f:
    f.write(f"localparam M1_INT = {M1_int};\n")
    f.write(f"localparam SHIFT  = {SHIFT};\n")

print("Wrote: w1.mem b1.mem w2.mem b2.mem test_images.mem test_labels.txt "
      "golden_scores.txt golden_h.txt params.vh")
print(f"w1.mem lines = {W1_q.size} (expect 25088), test_images.mem lines = {x_q.size} (expect {N_TEST*784})")
