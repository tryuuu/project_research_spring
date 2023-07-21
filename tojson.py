from pyknp import Juman
juman = Juman()
import random
import json
file_path = "/Users/ryutsuruyoshi/Desktop/プロ研final/noun_dict_sorted.txt"

# ファイルを開く
with open(file_path, 'r') as file:
    # ファイルの内容を読み込む
    content = file.read()

# 辞書形式に変換
dictionary = {}
for line in content.split('\n'):
    if line:
        key, value = line.split(': ')
        dictionary[key] = value.split(', ')
#print(dictionary)

def change_small(s):
    if s=="ぁ" or s=="ァ":
        return "あ"
    elif s=="ぃ" or s=="ィ":
        return "い"
    elif s=="ぅ" or s=="ゥ":
        return "う"
    elif s=="ぇ" or s=="ェ":
        return "え"
    elif s=="ぉ" or s=="ォ":
        return "お"
    elif s=="ゃ" or s=="ャ":
        return "や"
    elif s=="ゅ" or s=="ュ":
        return "ゆ"
    elif s=="ょ" or s=="ョ":
        return "よ"
    return s


def get_lastletter(current_word):
    result = juman.analysis(current_word)
    mrph_list = result.mrph_list()
    mrph = mrph_list[-1]
    if mrph.yomi[-1] == "ー":
        return change_small(mrph.yomi[-2:-1])
    return change_small(mrph.yomi[-1])

def get_nextword(word):
    s = get_lastletter(word)
    next_words = dictionary.get(s, [])  # 索引に対応する単語リストを取得し、存在しない場合は空のリストを返す
    if next_words:
        return random.choice(next_words)  # ランダムに1つの単語を選択して返す
    else:
        return None
    
n=0
word_list = []
s="あいつ"
N=5000
while(n<N):
    word_list.append(s)
    s = get_nextword(s)
    n+=1

pairs = []

for i in range(len(word_list) - 1):
    pair = {"instruction": "しりとりをしましょう。つまり、次の言葉をまずひらがなに直してください。そして、そのひらがなに直した単語について一番最後の文字から始まる言葉を一つだけ言ってください。例えば、「しりとり」ならば「りんご」、「酸素」なら「創作」などとなります。", "input": word_list[i], "output": word_list[i + 1]}
    pairs.append(pair)

# JSON形式で保存
with open("shiritori_data.json", "w") as json_file:
    json.dump(pairs, json_file, ensure_ascii=False, indent=4)
