import numpy as np

fps, T = 30, 600                                   # 20 s clip
t = np.arange(T) / fps
w = 2 * np.pi * 0.15                               # slow, ~7 s per loop

# positions relative to each hand's own shoulder: x forward, y left, z up (metres)
left  = np.stack([0.25 + 0.08 * np.cos(w * t),
                  0.10 + 0.08 * np.sin(w * t),
                  -0.05 + 0.10 * np.sin(2 * w * t)], axis=1)
right = np.stack([0.25 + 0.08 * np.cos(w * t + 1.0),
                  -0.10 - 0.08 * np.sin(w * t + 1.0),
                  -0.05 + 0.10 * np.sin(2 * w * t + 1.0)], axis=1)

np.save("hands.npy", np.stack([left, right], axis=1).astype(np.float32))   # shape (T, 2, 3)
print("saved", np.load("hands.npy").shape)