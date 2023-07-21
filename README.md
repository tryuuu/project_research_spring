# しりとりのfinetuning
## 概要
Jumandicからしりとりのデータセットを作成し、それを用いてfine tuning及びLoRA tuningを行った。

そして、学習データの量がどう精度を影響を与えるか、finetuningとLoRAtuningの精度の比較、使用する日本語LLM(cyberagent/open-calm-1b,3b,7b)及びChatGPTとの精度の差を検証した。

## コードについて
### 学習データの作成
