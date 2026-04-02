import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Reconstruct the dataframe
data = [
["ll2-7",   0.15, 0.35, 0.74, 0.76, 0.74, 0.73],
["gpt4o",   0.18, 0.19, 0.53, 0.54, 0.50, 0.55],
["gpt4",    0.14, 0.23, 0.52, 0.48, 0.49, 0.59],
["cl3h",    0.10, 0.22, 0.53, 0.53, 0.55, 0.51],
["cl3.7s",  0.08, 0.13, 0.56, 0.58, 0.52, 0.58],
["gptoss",  0.24, 0.11, 0.46, 0.51, 0.40, 0.48],
["cl3.5h",  0.13, 0.10, 0.50, 0.54, 0.40, 0.56],
["qw2-14",  0.09, 0.02, 0.53, 0.55, 0.52, 0.51],
["cl3.5s",  0.10, 0.19, 0.47, 0.50, 0.42, 0.49],
["gpt41m",  0.13, 0.06, 0.48, 0.54, 0.40, 0.51],
["qw1-7",   0.14, 0.02, 0.48, 0.47, 0.53, 0.46],
["gpt4om",  0.11, 0.02, 0.49, 0.50, 0.48, 0.50],
["gpt35",   0.14, 0.20, 0.38, 0.41, 0.46, 0.28],
["cl3o",    0.09, 0.22, 0.37, 0.37, 0.45, 0.30],
["ll3-8",   0.10, 0.03, 0.32, 0.30, 0.34, 0.31],
["ll2-13",  0.10, 0.02, 0.30, 0.40, 0.30, 0.21],
["ph3m",    0.18, 0.03, 0.25, 0.23, 0.27, 0.23],
["qw2-7",   0.12, 0.04, 0.24, 0.21, 0.17, 0.35],
["gem2-9",  0.10, 0.14, 0.10, 0.17, 0.10, 0.05],
]

cols = ["model","no_trust","zero_shot","two_mem","fire","farm","school"]
df = pd.DataFrame(data, columns=cols)

def family(m):
    if m.startswith("gpt"):
        return "GPT"
    if m.startswith("ll"):
        return "Llama"
    if m.startswith("cl"):
        return "Claude"
    if m.startswith("qw"):
        return "Qwen"
    if m.startswith("ph"):
        return "Phi"
    if m.startswith("gem"):
        return "Gemma"
    return "Other"

df["family"] = df["model"].apply(family)
df = df.sort_values(["family","model"]).reset_index(drop=True)

items = ["no_trust","zero_shot","two_mem","fire","farm","school"]
data_array = df[items].to_numpy()
nrows, ncols = data_array.shape

fig, ax = plt.subplots(figsize=(6.5,7))

im = ax.imshow(
    data_array,
    vmin=0, vmax=0.8,
    cmap="RdYlGn",
    interpolation="nearest"
)

ax.set_aspect('equal')

ax.set_xticks(np.arange(ncols))
ax.set_xticklabels(["no_trust","zero_shot","2-mem","fire","farm","school"],
                   rotation=45, ha="right", fontsize=9)
ax.set_yticks(np.arange(nrows))
ax.set_yticklabels(df["model"], fontsize=8)

ax.set_xticks(np.arange(-0.5, ncols, 1), minor=True)
ax.set_yticks(np.arange(-0.5, nrows, 1), minor=True)
ax.grid(which='minor', color='black', linestyle='-', linewidth=0.25)
ax.tick_params(which='minor', bottom=False, left=False)

# vertical separator between augmentation (0-2) and scenario (3-5)
ax.axvline(2.5, color="black", linewidth=1.0)

# Family separators
family_breaks = []
last = df["family"].iloc[0]
for i, f in enumerate(df["family"]):
    if f != last:
        family_breaks.append(i-0.5)
        last = f

for y in family_breaks:
    ax.hlines(y, -0.5, ncols-0.5, color="black", linewidth=1.0)

# Family labels
y0 = 0
labels = []
for b in family_breaks + [nrows-0.5]:
    y1 = int(b+0.5)
    mid = (y0 + y1) / 2
    fam = df["family"].iloc[y0]
    labels.append((mid, fam))
    y0 = y1

for y, fam in labels:
    ax.text(-1.1, y, fam, va="center", ha="right", fontsize=9, rotation=90)

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Fraction of decisions placing trust", fontsize=10)

plt.title("Decision Heatmap: Augmentations and Scenarios", fontsize=12)
plt.tight_layout()

fig.savefig("/mnt/data/decision_heatmap_red_green_squares.pdf")
plt.show()

