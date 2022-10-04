import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate

excel_data_df = pd.read_excel('test1.xlsx')

X= excel_data_df['x'].tolist()
Y= excel_data_df['y'].tolist()
Z= excel_data_df['z'].tolist()

X = np.array(X)
Y = np.array(Y)
Z = np.array(Z)
# Flatten trial data to meet your requirement:
x = X.ravel()
y = Y.ravel()
z = Z.ravel()

# Resampling on as square grid with given resolution:
resolution = 8
xlin = np.linspace(min(x), max(x), resolution)
ylin = np.linspace(min(y), max(y), resolution)
Xlin, Ylin = np.meshgrid(xlin, ylin)

# Linear multi-dimensional interpolation:
interpolant = interpolate.NearestNDInterpolator([r for r in zip(x, y)], z)
Zhat = interpolant(Xlin.ravel(), Ylin.ravel()).reshape(Xlin.shape)
cmap = 'jet'

# Render and interpolate again if necessary:
fig, axe = plt.subplots()
axe.imshow(Zhat, origin="lower", cmap=cmap, interpolation='bicubic',extent=[min(x),max(x),min(y),max(y)])

#plt.xlabel('X Values', fontsize = 15)
#plt.ylabel('Y Values', fontsize = 15)
plt.title('Interpolated Heatmap', fontsize = 20)

plt.show()
