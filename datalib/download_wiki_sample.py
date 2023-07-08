from datasets import load_dataset, ReadInstruction
from sklearn.model_selection import train_test_split


def download_and_split_wiki_dataset(train_len=100000, test_len=1000):
    split = "train[0:" + str(train_len) + "]+train[-" + str(test_len) + ":]"
    dataset = load_dataset('wikipedia', '20220301.en', split=split)

    assert(len(dataset) == train_len + test_len)

    texts = [example['text'] for example in dataset]

    return texts[0:train_len], texts[train_len:]


if __name__ == "__main__":
    train_texts, eval_texts = download_and_split_wiki_dataset()