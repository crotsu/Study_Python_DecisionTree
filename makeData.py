# データを作成する
# 2次元プロットデータ（2クラス）
# 確率密度関数で生成
# 分散共分散行列で楕円を指定
# pandasの練習

import numpy as np
import pandas as pd

# 乱数の種を初期化
np.random.seed(0)

# データ数
num1 = 100
num2 = 100
num = num1 + num2

# 平均
mu1 = [168,63]
mu2 = [157,52]

# 共分散
cov = [[25,5],[5,25]]

# 入力信号を生成
x1, y1 = np.random.multivariate_normal(mu1, cov, num1).T
x2, y2 = np.random.multivariate_normal(mu2, cov, num2).T


# 教師信号を生成
cls1 = [1 for i in range(num1)]
cls2 = [0 for i in range(num1)]

# DataFrameに変換
name = ['height','weight','gender']
df1 = pd.DataFrame(np.array([x1,y1,cls1]).T, columns=name)
df2 = pd.DataFrame(np.array([x2,y2,cls2]).T, columns=name)
df = pd.concat([df1,df2],axis=0)

# ファイル出力
df.to_csv('training.csv', index=None)
