import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Recreate dataframe
data = [
["qw2-14",6.00,7.00,7.00,7.00,5.00,7.00,7.00,7.00,7.00,5.10,7.00,7.00],
["qw2-7",7.00,6.50,6.80,6.00,5.60,7.00,7.00,6.80,7.00,5.50,6.10,7.00],
["ll2-13",7.00,6.90,7.00,7.00,7.00,7.00,7.00,6.10,7.00,7.00,7.00,0.70],
["gpt4om",6.00,6.00,6.00,6.00,5.40,6.00,6.00,6.00,6.00,5.00,6.00,6.00],
["gpt35",5.90,6.00,6.90,7.00,5.00,5.80,6.60,5.10,5.30,3.30,5.90,6.70],
["gpt41m",6.00,6.00,6.00,6.00,5.00,6.00,6.00,6.00,5.00,4.70,6.00,6.00],
["gptoss",5.20,5.90,5.90,6.00,4.80,5.60,5.90,5.90,6.40,4.70,5.30,6.50],
["ll3-8",5.80,6.10,5.90,6.00,5.70,6.10,6.00,5.50,6.00,2.40,6.10,5.30],
["cl3.7s",5.00,5.30,5.00,6.00,4.90,5.00,5.00,5.00,4.30,4.00,5.00,5.00],
["cl3.5s",5.00,5.00,5.00,5.00,5.00,5.00,5.00,5.00,4.60,4.00,5.00,5.10],
["gem2-9",5.10,5.10,5.00,5.00,4.00,5.00,5.40,4.00,5.00,4.00,5.10,6.00],
["cl3h",5.00,5.00,5.00,5.10,5.00,5.00,5.00,5.00,4.40,4.00,5.00,5.00],
["ph3m",5.60,6.00,6.10,4.50,4.90,5.90,6.20,3.90,2.00,1.20,6.60,5.20],
["cl3o",4.80,5.00,5.00,5.00,4.00,5.00,5.70,4.90,5.00,3.60,5.00,5.00],
["gpt4",4.70,5.00,5.00,5.50,4.00,5.00,5.00,5.30,4.00,3.70,5.00,5.00],
["qw1-7",4.60,6.30,7.00,5.00,4.60,4.00,6.30,5.10,1.00,1.00,4.60,7.00],
["gpt4o",4.00,5.00,5.00,5.00,3.00,5.00,5.00,5.00,4.00,3.00,4.00,4.10],
["cl3.5h",4.00,4.60,4.30,5.00,4.00,4.00,4.10,5.00,4.00,4.00,4.00,4.80],
["ll2-7",3.40,6.80,7.00,0.50,4.20,0.70,7.00,0.20,1.80,4.00,3.30,0.30],
]
cols=["model"]+[f"item{i}" for i in range(1,13)]
df=pd.DataFrame(data,columns=cols)

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

df["family"]=df["model"].apply(family)
df=df.sort_values(["family","model"]).reset_index(drop=True)

items=[f"item{i}" for i in range(1,13)]
data_array=df[items].to_numpy()
nrows, ncols = data_array.shape

fig, ax = plt.subplots(figsize=(8,8))  # make it roughly square

im=ax.imshow(
    data_array,
    vmin=0, vmax=7,
    cmap="RdYlGn",
    interpolation="nearest"
)

# Make each cell square
ax.set_aspect('equal')

# Main ticks for labels
ax.set_xticks(np.arange(ncols))
ax.set_xticklabels(range(1, ncols+1), fontsize=9)
ax.set_yticks(np.arange(nrows))
ax.set_yticklabels(df["model"], fontsize=8)

# Gridlines only at cell boundaries (not inside cells)
ax.set_xticks(np.arange(-0.5, ncols, 1), minor=True)
ax.set_yticks(np.arange(-0.5, nrows, 1), minor=True)
ax.grid(which='minor', color='black', linestyle='-', linewidth=0.25)
ax.tick_params(which='minor', bottom=False, left=False)

# Highlight high-loading items (4,7,8,12) at column boundaries
high=[4,7,8,12]
high_idx=[h-1 for h in high]
for j in high_idx:
    # slightly thicker boundary lines on those columns
    ax.axvline(j-0.5, color="white", linewidth=1.5)
    ax.axvline(j+0.5, color="white", linewidth=1.5)
    ax.get_xticklabels()[j].set_fontweight("bold")

# Family separators (between rows)
family_breaks=[]
last=df["family"][0]
for i,f in enumerate(df["family"]):
    if f!=last:
        family_breaks.append(i-0.5)
        last=f
for y in family_breaks:
    ax.hlines(y,-0.5,ncols-0.5,color="black",linewidth=1.0)

# Family labels
y0=0
labels=[]
for b in family_breaks+[nrows-0.5]:
    y1=int(b+0.5)
    mid=(y0+y1)/2
    fam=df["family"][y0]
    labels.append((mid,fam))
    y0=y1
for y,fam in labels:
    ax.text(-1.3,y,fam,va="center",ha="right",fontsize=9,rotation=90)

cbar=fig.colorbar(im,ax=ax)
cbar.set_label("Trust score (1–7)",fontsize=10)

plt.title("Trust Item Heatmap (Grouped by Model Family)",fontsize=12)
plt.tight_layout()

fig.savefig("heatmap.pdf")
plt.show()

