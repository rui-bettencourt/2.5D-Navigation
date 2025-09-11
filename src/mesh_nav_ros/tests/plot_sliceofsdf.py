import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/home/rui/ds/testsiros2025/slice_z_0_5.csv", index_col=0)
x = df.index.astype(float)
y = df.columns.astype(float)

plt.figure(figsize=(8,6))
plt.imshow(df.values.T, origin='lower', extent=[x.min(), x.max(), y.min(), y.max()], cmap='coolwarm')
plt.colorbar(label="SDF distance (m)")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("SDF slice at z=0.5m")
plt.show()