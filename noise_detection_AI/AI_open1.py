import os
import pickle
import sys

base_path = os.path.dirname(__file__)
open_path = os.path.join(base_path, 'pickle/')
# 予測（テスト）データ読み込み
with open(open_path + 'test_feature.pickle', mode='rb') as f:
    test_feature = pickle.load(f)

# 学習結果読み込み
with open(open_path + 'model2.pickle', mode='rb') as f:
    model = pickle.load(f)

pred = model.predict(test_feature)
print(f"predict_data = {pred}")
