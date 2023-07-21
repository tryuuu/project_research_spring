# しりとりのfinetuning
## 概要
Jumandicからしりとりのデータセットを作成し、それを用いてfine tuning及びLoRA tuningを行った。

そして、学習データの量がどう精度を影響を与えるか、finetuningとLoRAtuningの精度の比較、使用する日本語LLM(cyberagent/open-calm-1b,3b,7b)及びChatGPTとの精度の差を検証した。

## コードについて
### 学習データの作成
noun.txt・・・Jumandicから名詞のみを取り出して.txtファイルに保存したもの
makedict.py・・・noun.txtに格納されている名詞から、辞書のように索引(先頭のひらがな)->単語という形で名詞データを再保存するためのコード
noun_dict_sorted.txt・・・makedict.pyにより順番・構造を辞書式に並び替えられた名刺のデータが保存されたもの
tojson.py・・・noun_dict_sorted.txtからN個分のしりとりのデータセットを取り出して、しりとりのinstruction tuningができるようにしたjsonファイル
