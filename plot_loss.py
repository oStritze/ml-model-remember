import pandas as pd
import seaborn as sns
import numpy
import matplotlib.pyplot as plt

fname = "results/train_output_cor.txt"
with open(fname) as f:
    s = f.readlines()

df = pd.DataFrame(columns=["acc", "train_loss", "test_loss"])
print(df)
for line in s:
    if "training loss:" in line:
         l = line.split("\t")
         trainloss = float(l[2][:8])
    if "validation loss:" in line:
         l = line.split("\t")
         testloss = float(l[2][:8])
    if "accuracy" in line:
        l = line.split("\t")
        acc = float(l[2][:5])/100
    if line.startswith("Epoch"):
        df.loc[-1] = [acc, trainloss, testloss]
        df.index = df.index+1
        df = df.sort_index()

print(df.head())
idx = df.index
df = df.sort_index(ascending=False)
df.index = idx
plt.plot(df.index, df["train_loss"], label="train loss")
plt.plot(df.index, df["test_loss"], label="test loss")
plt.plot(df.index, df["acc"], label="test accuracy")
plt.plot(df.index, numpy.zeros(df.shape[0]), color="black")
plt.legend(loc='lower right', bbox_to_anchor=(1.0, 0.2))
plt.title("COR attack training process on 50x50 images")
plt.xlabel("Epochs")
#plt.show()
plt.savefig("results/COR_loss.png", dpi=150)

#plt.clear()
plt.close()

### same for cap
fname = "results/train_output_cap.txt"
with open(fname) as f:
    s = f.readlines()

df = pd.DataFrame(columns=["acc", "train_loss", "test_loss"])
print(df)
for line in s:
    if "training loss:" in line:
         l = line.split("\t")
         trainloss = float(l[2][:8])
    if "validation loss:" in line:
         l = line.split("\t")
         testloss = float(l[2][:8])
    if "accuracy" in line:
        l = line.split("\t")
        acc = float(l[2][:5])/100
    if line.startswith("Epoch"):
        df.loc[-1] = [acc, trainloss, testloss]
        df.index = df.index+1
        df = df.sort_index()

print(df.head())
idx = df.index
df = df.sort_index(ascending=False)
df.index = idx
plt.plot(df.index, df["train_loss"], label="train loss")
plt.plot(df.index, df["test_loss"], label="test loss")
plt.plot(df.index, df["acc"], label="test accuracy")
plt.plot(df.index, numpy.zeros(df.shape[0]), color="black")
plt.legend(loc='lower right', bbox_to_anchor=(1.0, 0.2))
plt.title("COP attack training process on 250x250 images")
plt.xlabel("Epochs")
#plt.show()
plt.savefig("results/CAP_loss.png", dpi=150)
