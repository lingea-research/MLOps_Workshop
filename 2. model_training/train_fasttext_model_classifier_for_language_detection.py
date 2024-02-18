import fasttext


def train_fasttext_model_classifier(model_num=1, epoch=None, lr=None):
    if epoch is None and lr is None:
        model = fasttext.train_supervised(input="artifacts/all_train.txt")
    elif epoch is not None and lr is None:
        model = fasttext.train_supervised(input="artifacts/all_train.txt", epoch=epoch)
    elif epoch is None and lr is not None:
        model = fasttext.train_supervised(input="artifacts/all_train.txt", lr=lr)
    else:
        model = fasttext.train_supervised(input="artifacts/all_train.txt", epoch=epoch, lr=lr)

    model.save(f"models/model{model_num}.bin")
    print(model.predict("Jmenuji se Petr", k=-1, threshold=0.5))
    print(model.predict("Volam sa Peter", k=-1, threshold=0.5))
    print(model.predict("My name is Peter", k=-1, threshold=0.5))
    num_of_samples, precision, recall = model.test("artifacts/all_valid.txt")
    print(num_of_samples, precision, recall)
