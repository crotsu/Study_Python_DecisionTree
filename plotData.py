# 2次元プロットデータ（2クラス）
# 表示

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# データ読み込み
df = pd.read_csv('training.csv')

# 散布図をプロットする
for i in range(len(df)):
    if df.gender[i]==1:
        plt.scatter(df.height[i],df.weight[i], color='r',marker='o', s=30)
    else:
        plt.scatter(df.height[i],df.weight[i], color='b',marker='x', s=30)

# グリッド表示
plt.grid(True)

# EPSファイルとして出力する
plt.savefig('data.eps')

# 表示
plt.show()
