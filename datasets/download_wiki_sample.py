from datasets import load_dataset, ReadInstruction
from sklearn.model_selection import train_test_split


def download_and_split_wiki_dataset():
    dataset = load_dataset('wikipedia', '20220301.en', split=ReadInstruction('train', to=5, unit='%'))

    # The text of each article is stored in the 'text' field
    texts = [example['text'] for example in dataset]

    # Split the data into training and evaluation sets
    train_texts, eval_texts = train_test_split(texts, test_size=0.1)

    return train_texts, eval_texts


if __name__ == "__main__":
    train_texts, eval_texts = download_and_split_wiki_dataset()

    print(len(train_texts))
    print(len(eval_texts))
    print(train_texts[0])
