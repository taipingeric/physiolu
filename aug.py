# PhysioNet Aug

import numpy as np

def norm(x):
    return (x-x.mean()) / x.std()
def flip(x):
    return -x
def scale(x):
    noise = np.random.normal(1, 0.1)
    return x*noise
def cutout(x, size=0.2):
    output = x.copy()
    sig_len = x.shape[-1]
    drop_window = int(sig_len*size)
    drop_idx = np.random.randint(0, sig_len-drop_window)
    output[:, drop_idx:drop_idx+drop_window] = np.random.normal(1, 0.1)
    return output
def reverse(x):
    return np.flip(x, axis=-1)
#     return x[:, ::-1]

# def sin_noise(x, max_wave_count=0, flip_noise=True):
#     sig_len = x.shape[-1]
# #     wave_count = np.random.randint(0, max_wave_count+1)
#     wave_count = 1
#     sin_noise = np.sin(np.linspace(-np.pi, np.pi, sig_len)*wave_count)
#     if np.random.choice([True, False]): sin_noise *= -1
#     return x + sin_noise
def sin_noise(x, max_wave_count=0, flip_noise=True):
    sig_len = x.shape[-1]
    wave_count = 1
    start_angle = np.random.uniform(-np.pi, np.pi)
    sin_noise = np.sin(np.linspace(start_angle, start_angle+np.pi/2, sig_len)*wave_count)
    scale = np.random.normal(2, 1)
    sin_noise *= scale
    if np.random.choice([True, False]): sin_noise *= -1
    return x + sin_noise[np.newaxis, ::]

def random_aug(x):
    all_aug = [flip, scale, sin_noise, reverse, cutout]
    for a in all_aug:
        if np.random.choice([True, False]):
            x = a(x) 
    return x