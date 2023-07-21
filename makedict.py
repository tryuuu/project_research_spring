from pyknp import Juman
juman = Juman()

file_path = "/Users/ryutsuruyoshi/Desktop/プロ研final/noun.txt"
output_path = "/Users/ryutsuruyoshi/Desktop/プロ研final/noun_dict_sorted.txt.txt"

def get_initial(word):
    if word == "":
        return True
    result1 = juman.analysis(word)
    mrph_list1 = result1.mrph_list()
    mrph1 = mrph_list1[0]
    return mrph1.yomi[0]

def create_dictionary(noun_list):
    dictionary = {}

    for noun in noun_list:
        initial = get_initial(noun)
        if initial not in dictionary:
            dictionary[initial] = []
        dictionary[initial].append(noun)

    return dictionary

noun_list = []

with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        noun = line.strip()
        noun_list.append(noun)

dictionary = create_dictionary(noun_list)

sorted_data = {k: sorted(v, key=str.lower) for k, v in sorted(dictionary.items(), key=lambda x: ord(x[0]))}
with open(output_path, 'w') as f:
    for key, values in sorted_data.items():
        f.write(f"{key}: {', '.join(values)}\n")

