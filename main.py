import numpy as np
from gensim.models import Word2Vec


INPUT_FILE_NAME = 'ja.bin'
OUTPUT_FILE_NAME = 'ja.tsv'


def main():
    model = Word2Vec.load(INPUT_FILE_NAME)
    with open(OUTPUT_FILE_NAME, mode='w', encoding='utf-8') as f:
        for i, word in enumerate(model.wv.index2word):
            f.write('{}\t{}\t{}\n'.format(
                str(i),
                word.encode('utf-8').decode('utf-8'),
                np.array2string(
                    model[word],
                    separator=',', # for json format
                    floatmode='fixed', # for cases such as `1.`
                    max_line_width=np.inf # prevent new line in a vector
                )
            ))


if __name__ == '__main__':
    main()