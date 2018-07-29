# 決定木を学ぶ

# 身長，体重から男性と女性に分ける

import pandas as pd
from sklearn import tree

# CSVファイルを取得
data = pd.read_csv("training.csv", sep=",")

# 説明変数(x1, x2に設定)
variables = ['height', 'weight']

# 決定木の分類器を生成
clf = tree.DecisionTreeClassifier(max_depth=3)

# 分類器にサンプルデータを入れて学習(目的変数はx）
clf = clf.fit(data[['height','weight']], data['gender'])

import pydotplus
from sklearn.externals.six import StringIO
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data, feature_names=['height','weight'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
 
# PDFファイルに出力
graph.write_pdf("gender.pdf")
