"""
Symulator lokalizacji źródła sygnału metodą TDOA w jednorodnym kanale hydroakustycznym.
Dodatek: obliczenie GDOP (Geometric Dilution of Precision).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.signal import correlate

# ============================================================
# 1. PARAMETRY
# ============================================================
c = 1500.0
fs = 10000
duration = 0.1
f0 = 500
SNR_dB = 25

hydrophones = np.array([
    [-50, -50],
    [ 50, -50],
    [ 50,  50],
    [-50,  50]
])
source_true = np.array([20.0, 25.0])

# ============================================================
# 2. GENEROWANIE SYGNAŁU I ODBIÓR (bez zmian)
# ============================================================
t = np.arange(0, duration, 1/fs)
source_signal = np.sin(2 * np.pi * f0 * t)

def add_noise(signal, snr_db):
    signal_power = np.mean(signal**2)
    snr_linear = 10**(snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power) * np.random.randn(len(signal))
    return signal + noise

received_signals = []
for pos in hydrophones:
    dist = np.linalg.norm(source_true - pos)
    delay = dist / c
    delay_samples = int(round(delay * fs))
    rx = np.zeros_like(source_signal)
    if delay_samples < len(source_signal):
        rx[delay_samples:] = source_signal[:-delay_samples]
    rx = add_noise(rx, SNR_dB)
    received_signals.append(rx)

# ============================================================
# 3. ESTYMACJA TDOA (korelacja)
# ============================================================
def estimate_tdoa(sig_ref, sig_other, fs):
    corr = correlate(sig_ref, sig_other, mode='same')
    lags = np.arange(-len(corr)//2, len(corr)//2)
    lag_samples = lags[np.argmax(corr)]
    return lag_samples / fs

ref_idx = 0
tdoa_estimated = []
for i in range(1, len(hydrophones)):
    tau = estimate_tdoa(received_signals[ref_idx], received_signals[i], fs)
    tdoa_estimated.append(tau)

# ============================================================
# 4. LOKALIZACJA (least_squares)
# ============================================================
def tdoa_residuals(x, sensors, ref_idx, measured_tdoa, c):
    residuals = []
    for i in range(len(sensors)):
        if i == ref_idx:
            continue
        dist_ref = np.linalg.norm(x - sensors[ref_idx])
        dist_i   = np.linalg.norm(x - sensors[i])
        model_tdoa = (dist_i - dist_ref) / c
        residuals.append(model_tdoa - measured_tdoa[i if i<ref_idx else i-1])
    return residuals

x0 = np.mean(hydrophones, axis=0)
result = least_squares(tdoa_residuals, x0, args=(hydrophones, ref_idx, tdoa_estimated, c), method='lm')
source_estimated = result.x
error_distance = np.linalg.norm(source_estimated - source_true)

print("=== Rzeczywiste TDOA (wzgl. H0) ===")
for i, tau in enumerate([(np.linalg.norm(source_true - hydrophones[i]) - np.linalg.norm(source_true - hydrophones[0]))/c for i in range(1,4)]):
    print(f"H{i+1}: {tau*1000:.3f} ms")
print("\n=== Estymowane TDOA ===")
for i, tau in enumerate(tdoa_estimated):
    print(f"H{i+1}: {tau*1000:.3f} ms")
print(f"\nŹródło rzeczywiste: {source_true}")
print(f"Źródło estymowane:  {source_estimated}")
print(f"Błąd odległości:    {error_distance:.2f} m")

# ============================================================
# 5. OBLICZENIE GDOP
# ============================================================
def calculate_gdop(point, sensors, ref_idx=0, c=1500.0):
    """
    Oblicza GDOP dla metody TDOA w 2D.
    point: [x, y] – punkt, dla którego liczymy GDOP.
    sensors: lista współrzędnych hydrofonów.
    ref_idx: indeks hydrofonu referencyjnego.
    Zwraca wartość GDOP (skalar).
    """
    n_sensors = len(sensors)
    # Liczba niezależnych TDOA = n_sensors - 1
    m = n_sensors - 1
    H = np.zeros((m, 2))   # Jacobian względem x, y
    
    # Wektor od punktu do każdego sensora
    vectors = sensors - point
    distances = np.linalg.norm(vectors, axis=1)
    # Wersory kierunkowe
    u = vectors / distances[:, np.newaxis]
    
    # Dla każdej pary (i, ref)
    idx = 0
    for i in range(n_sensors):
        if i == ref_idx:
            continue
        # Różnica wersorów: u_i - u_ref
        H[idx, :] = u[i] - u[ref_idx]
        idx += 1
    
    # Macierz kowariancji (Q = (H^T H)^{-1}) – zakładamy jednostkową wariancję pomiarów
    try:
        Q = np.linalg.inv(H.T @ H)
        gdop = np.sqrt(np.trace(Q))
    except np.linalg.LinAlgError:
        gdop = np.inf
    return gdop

# GDOP w punkcie rzeczywistego źródła
gdop_true = calculate_gdop(source_true, hydrophones, ref_idx, c)
print(f"\nGDOP dla rzeczywistego źródła: {gdop_true:.3f}")

# ============================================================
# 6. WIZUALIZACJA: mapa GDOP w obszarze
# ============================================================
x_range = np.linspace(-100, 100, 50)
y_range = np.linspace(-100, 100, 50)
X, Y = np.meshgrid(x_range, y_range)
gdop_map = np.zeros_like(X)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        gdop_map[i, j] = calculate_gdop([X[i,j], Y[i,j]], hydrophones, ref_idx, c)

plt.figure(figsize=(12, 5))

# GDOP mapa konturowa
plt.subplot(1, 2, 1)
contour = plt.contourf(X, Y, gdop_map, levels=20, cmap='viridis')
plt.colorbar(label='GDOP')
plt.scatter(hydrophones[:,0], hydrophones[:,1], c='red', marker='s', s=80, label='Hydrofony')
plt.scatter(source_true[0], source_true[1], c='blue', marker='o', s=100, edgecolors='white', label='Źródło')
plt.xlabel('X [m]')
plt.ylabel('Y [m]')
plt.title('Mapa GDOP dla układu 4 hydrofonów\n(im mniej, tym lepsza geometria)')
plt.legend()
plt.grid(alpha=0.3)
plt.axis('equal')

# Wyniki lokalizacji (jak poprzednio)
plt.subplot(1, 2, 2)
plt.scatter(hydrophones[:,0], hydrophones[:,1], c='blue', marker='s', s=100, label='Hydrofony')
plt.scatter(source_true[0], source_true[1], c='green', marker='o', s=120, label='Źródło rzeczywiste', edgecolors='black')
plt.scatter(source_estimated[0], source_estimated[1], c='red', marker='x', s=120, label='Źródło estymowane', linewidths=2)
plt.xlabel('X [m]')
plt.ylabel('Y [m]')
plt.title(f'Lokalizacja TDOA (SNR={SNR_dB} dB)\nBłąd = {error_distance:.2f} m, GDOP = {gdop_true:.2f}')
plt.legend()
plt.grid(True)
plt.axis('equal')

plt.tight_layout()
plt.savefig('tdoa_with_gdop.png', dpi=150)
plt.show()