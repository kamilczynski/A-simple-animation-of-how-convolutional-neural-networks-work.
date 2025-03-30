import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

plt.rcParams['font.family'] = 'Orbitron'
# Używamy ciemnego stylu
plt.style.use('dark_background')

# ----------------------------------------------------
# 1. Przygotowanie danych
# ----------------------------------------------------
# Losowy obraz binarny (0 lub 1), np. 10×10
image = np.random.randint(0, 2, (10, 10))

# Przykładowy filtr 3×3 (również binarny)
kernel = np.array([
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0]
], dtype=int)

# Rozmiary
n_rows, n_cols = image.shape
k_rows, k_cols = kernel.shape

# Macierz wynikowa (Feature Map) o wymiarach (n_rows-k_rows+1) × (n_cols-k_cols+1)
conv_result = np.zeros((n_rows - k_rows + 1, n_cols - k_cols + 1), dtype=int)

# ----------------------------------------------------
# 2. Przygotowanie figurek i subplotów
#    Układ 2×2:
#      - Lewy górny: Obraz wejściowy
#      - Prawy górny: Filtr (kernel)
#      - Lewy dolny: Aktualny fragment obrazu (z obliczeniami nad nim)
#      - Prawy dolny: Feature Map
# ----------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
ax_img    = axes[0, 0]
ax_kernel = axes[0, 1]  # Filtr (kernel) w prawym górnym oknie
ax_patch  = axes[1, 0]  # Aktualny fragment obrazu w lewym dolnym oknie
ax_conv   = axes[1, 1]

# Zmiana czcionki i stylu tekstu na futurystyczny (np. większa czcionka, neonowe kolory)
text_kwargs = dict(ha='center', va='center', fontsize=10, color='cyan')

# ----------------------------------------------------
# 2A. Obraz wejściowy (lewy górny subplot)
# ----------------------------------------------------
im1 = ax_img.imshow(
    image,
    cmap='gray',
    origin='upper',
    extent=[0, n_cols, n_rows, 0],
    interpolation='nearest',
    vmin=0,
    vmax=1
)
ax_img.set_title("Input Image", color='orangered', fontsize=12)

# Dodajemy wartości pikseli jako tekst z neonowym efektem
for r in range(n_rows):
    for c in range(n_cols):
        ax_img.text(
            c + 0.5, r + 0.5, str(image[r, c]),
            color='cyan', fontsize=10, ha='center', va='center'
        )

# Neonowy magenta prostokąt symbolizujący położenie filtra
rect = patches.Rectangle(
    (0, 0), k_cols, k_rows,
    edgecolor='magenta', facecolor='none', lw=2
)
ax_img.add_patch(rect)
ax_img.set_xlim(0, n_cols)
ax_img.set_ylim(n_rows, 0)

# ----------------------------------------------------
# 2B. Filtr (kernel) (prawy górny subplot)
# ----------------------------------------------------
im_kernel = ax_kernel.imshow(
    kernel,
    cmap='gray',
    origin='upper',
    extent=[0, k_cols, k_rows, 0],
    interpolation='nearest',
    vmin=0,
    vmax=1
)
ax_kernel.set_title("Filter (kernel)", color='lime', fontsize=12)

# Wyświetlamy wartości filtra z neonowym efektem
for rr in range(k_rows):
    for cc in range(k_cols):
        ax_kernel.text(
            cc + 0.5, rr + 0.5, str(kernel[rr, cc]),
            **text_kwargs
        )
ax_kernel.set_xlim(0, k_cols)
ax_kernel.set_ylim(k_rows, 0)

# ----------------------------------------------------
# 2C. Aktualny fragment obrazu (lewy dolny subplot)
#      - Dodajemy także pole tekstowe z obliczeniami nad tym oknem.
# ----------------------------------------------------
im_patch = ax_patch.imshow(
    np.zeros((k_rows, k_cols)),
    cmap='gray',
    origin='upper',
    extent=[0, k_cols, k_rows, 0],
    interpolation='nearest',
    vmin=0,
    vmax=1
)
ax_patch.set_title("Current Patch", color='lime', fontsize=12)

texts_patch = []
for rr in range(k_rows):
    row_texts = []
    for cc in range(k_cols):
        txt = ax_patch.text(
            cc + 0.5, rr + 0.5, '0', **text_kwargs
        )
        row_texts.append(txt)
    texts_patch.append(row_texts)
ax_patch.set_xlim(0, k_cols)
ax_patch.set_ylim(k_rows, 0)

# Przenosimy pole tekstowe dla obliczeń nad aktualnym fragmentem obrazu
calc_text = ax_patch.text(
    0.5, 1.1, '', transform=ax_patch.transAxes,
    ha='center', va='bottom', fontsize=10, color='lime'
)

# ----------------------------------------------------
# 2D. Feature Map (prawy dolny subplot)
# ----------------------------------------------------
im2 = ax_conv.imshow(
    conv_result,
    cmap='inferno',
    origin='upper',
    extent=[0, conv_result.shape[1], conv_result.shape[0], 0],
    interpolation='nearest',
    vmin=0,
    vmax=k_rows * k_cols
)
ax_conv.set_title("Feature Map", color='magenta', fontsize=12)

texts_conv = []
for r in range(conv_result.shape[0]):
    row_texts = []
    for c in range(conv_result.shape[1]):
        txt = ax_conv.text(
            c + 0.5, r + 0.5, '0',
            **text_kwargs
        )
        row_texts.append(txt)
    texts_conv.append(row_texts)
ax_conv.set_xlim(0, conv_result.shape[1])
ax_conv.set_ylim(conv_result.shape[0], 0)

# Najpierw automatyczne dopasowanie
plt.tight_layout()
# Następnie zwiększamy odstęp w pionie między wierszami
plt.subplots_adjust(hspace=0.4)  # Zwiększ/zmniejsz tę wartość w razie potrzeby

# ----------------------------------------------------
# 3. Przygotowanie animacji
# ----------------------------------------------------
output_rows = n_rows - k_rows + 1
output_cols = n_cols - k_cols + 1
total_frames = output_rows * output_cols

def init():
    """
    Inicjalizacja animacji – ustawiamy prostokąt i początkowe dane.
    """
    rect.set_xy((0, 0))
    return [rect, im_patch, im2]

def update(frame):
    """
    Dla każdej klatki:
      1. Ustalamy pozycję filtra.
      2. Wycinamy patch z obrazu.
      3. Obliczamy splot.
      4. Aktualizujemy Feature Map.
      5. Wyświetlamy szczegółowe obliczenia.
    """
    i = frame // output_cols
    j = frame % output_cols
    rect.set_xy((j, i))

    # Aktualny fragment obrazu (patch)
    patch = image[i: i + k_rows, j: j + k_cols]
    im_patch.set_data(patch)
    for rr in range(k_rows):
        for cc in range(k_cols):
            texts_patch[rr][cc].set_text(str(patch[rr, cc]))

    # Obliczenia konwolucji
    conv_value = np.sum(patch * kernel)
    conv_result[i, j] = conv_value
    im2.set_data(conv_result)
    texts_conv[i][j].set_text(str(conv_value))

    # Budujemy łańcuch z obliczeniami (np. "0*1 + 1*0 + 0*1 + ... = suma")
    calc_str = ""
    for rr in range(k_rows):
        for cc in range(k_cols):
            calc_str += f"{patch[rr, cc]}*{kernel[rr, cc]}"
            if not (rr == k_rows - 1 and cc == k_cols - 1):
                calc_str += " + "
    calc_str += f" = {conv_value}"
    calc_text.set_text(calc_str)

    ax_conv.set_title(f"Feature Map (step {frame + 1}/{total_frames})", color='magenta', fontsize=12)

    return [rect, im_patch, im2, calc_text] \
        + [t for row in texts_patch for t in row] \
        + [t for row in texts_conv for t in row]

ani = animation.FuncAnimation(
    fig,
    update,
    frames=total_frames,
    init_func=init,
    interval=500,
    blit=False,
    repeat=False
)

# ----------------------------------------------------
# Zapis do pliku (MP4)
# Aby to działało, musisz mieć zainstalowany FFmpeg i widoczny w PATH!
# ----------------------------------------------------
ani.save(
    r'C:\Users\topgu\PycharmProjects\obrazowanie\media\videos\cnn3.mp4',
    writer='ffmpeg',
    fps=2,
    dpi=150
)

plt.show()
