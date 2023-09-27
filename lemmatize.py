from pymorphy2 import MorphAnalyzer
from nltk.stem import WordNetLemmatizer


with open('stopwordsRU.txt', 'r') as ru, open('stopwordsEN.txt', 'r') as en:
    STOPWORDSEN = [w.strip() for w in en.readlines()]
    STOPWORDSRU = [w.strip() for w in ru.readlines()]


morphRU = MorphAnalyzer(lang='ru')
ENlemmatizer = WordNetLemmatizer()


ea = 'abcdefghijklmnopqrstuvwxyz'
ra = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'


def lemmatize(tokens):
    res = []
    for token in tokens:
        token = token.strip()

        if token[0] in ea and token not in STOPWORDSEN:
            token = ENlemmatizer.lemmatize(token)
        elif token[0] in ra and token not in STOPWORDSRU:
            token = morphRU.normal_forms(token)[0]

        res.append(token)
    return res


if __name__ == '__main__':
    path = 'mem2captureRuEn.txt'
    res = []
    with open(path, 'r') as fin, open('mem2captureLem.txt', 'w') as fout:
        lines = fin.readlines()
        for line in lines:
            out_line = lemmatize(line.strip().split())
            res.append(' '.join(out_line) + '\n')
        fout.writelines(res)
