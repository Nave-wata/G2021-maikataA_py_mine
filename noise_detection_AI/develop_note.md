# Noise detection AI

騒音検知 AI 作ります．

開発状況は気が向き次第ここに記入します．

[公開用](https://github.com/maikataA/G2021-hirakataA/blob/v1.2.0/save_AI/develop_note.md)

## インストール

- FFmpeg のインストール [参考](https://nagaragawa-r.com/download-install-pass-ffmpeg/)

## webサイト

- [やさしい AI の始め方](https://poncotuki.com/ai-ml/ai-voicedlml/)
- [ディープラーニングで音声分類](https://qiita.com/cvusk/items/61cdbce80785eaf28349#augmentation)
- [音声分類をいろいろなモデルや特徴量でやってみた](https://qiita.com/kshina76/items/5686923dee2889beba7c)
- (予定)[Pythonの音声処理ライブラリ【LibROSA】で音声読み込み⇒スペクトログラム変換・表示⇒位相推定して音声復元](https://qiita.com/lilacs/items/a331a8933ec135f63ab1)
- [Pythonの並列処理を理解したい［マルチスレッド編］](https://zenn.dev/ryo_kawamata/articles/python-concurrent-thread)
- [機械学習を用いてリアルタイムにギター のコードを分類してみた](https://qiita.com/ha161553/items/80f3056544a3d4ae3352)
- https://qiita.com/yutalfa/items/dbd172138db60d461a56
- https://qiita.com/FujiedaTaro/items/3e469688ad6f0dca62c0

### 使用データ

- [ESC-50](https://github.com/karolpiczak/ESC-50)

#### Animals

- dog -> other
- rooster -> other
- pig -> other
- cow -> other
- frog -> other
- cat -> other
- hen -> other
- insects(flying) -> other
- sheep -> other
- crow -> other

#### Natural soundscapes & water sounds

- rain -> other
- sea_waves -> other
- cracking fire -> other
- crickets -> other
- chirping birds -> other
- wind -> other
- pouring water -> other
- toilet flush -> other
- thunderstorm -> other

#### Human, non-speech sounds

- crying_baby -> other
- sneezing -> other
- clapping -> other
- breathing -> other
- coughing -> other
- footsteps -> other
- laughing -> other
- brushing teeth -> other
- snoring -> snoring
- drinking, sipping -> other

#### Interior/domestic sounds

- door_knock -> other
- mouse_click -> other
- keyboard_typing -> other
- door, wood_creaks -> other
- can_opening -> other
- washing_machine -> washing_machine
- vacuum_cleaner -> vacuum_cleaner
- clock_tick -> other
- glass_breaking -> other

#### Exterior/urban noises

- helicopter -> helicopter
- chainsaw -> chainsaw
- siren -> other
- car_horn -> other
- engine -> engine
- train -> other
- church bells -> other
- airplane -> airplane
- fireworks -> other
- hand_saw -> other

other と ラベル付いてるやつで分けて判定

## その他

1. `AI_save1.py` で学習して，学習結果・前処理したデータを保存する．

2. `AI_open1.py` で学習結果から予測を行う．

学習結果とかは `pickle` ディレクトリ内に保存した．学習結果自体はすでに保存済み．予測自体は `AI_open1.py` の実行でできる．感覚的には学習に 20 秒くらいかかった．
