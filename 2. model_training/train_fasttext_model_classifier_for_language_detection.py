import fasttext
import mlflow
import time
from unidecode import unidecode


def normalize_datasets():
    with (open("artifacts/all_train.txt") as f_in,
          open("artifacts/all_train_normalized.txt", "w") as f_out):
        text = f_in.read()
        text = text.lower()
        text = unidecode(text)
        f_out.write(text)
    with (open("artifacts/all_valid.txt") as f_in,
          open("artifacts/all_valid_normalized.txt", "w") as f_out):
        text = f_in.read()
        text = text.lower()
        text = unidecode(text)
        f_out.write(text)


def train_fasttext_model_classifier(model_num=1, epoch=None, lr=None, normalized_dataset=False):
    with mlflow.start_run():
        input = "artifacts/all_train.txt"
        if normalized_dataset:
            input = "artifacts/all_train_normalized.txt"

        mlflow.log_artifact(input)

        mlflow.log_param("lr", 0.1 if lr is None else lr)
        mlflow.log_param("epoch", 5 if epoch is None else epoch)
        mlflow.log_param("normalized", True if normalized_dataset else False)

        start_time = time.time()
        if epoch is None and lr is None:
            model = fasttext.train_supervised(input=input)
        elif epoch is not None and lr is None:
            model = fasttext.train_supervised(input=input, epoch=epoch)
        elif epoch is None and lr is not None:
            model = fasttext.train_supervised(input=input, lr=lr)
        else:
            model = fasttext.train_supervised(input=input, epoch=epoch, lr=lr)
        end_time = time.time()

        output = f"models/model{model_num}.bin"
        if normalized_dataset:
            output = f"models/normalized_model{model_num}.bin"
        model.save_model(output)

        # mlflow.register_model(f"models/model{model_num}.bin", f"model{model_num}")

        print(model.predict("Jmenuji se Petr", k=-1, threshold=0.5))
        print(model.predict("Volam sa Peter", k=-1, threshold=0.5))
        print(model.predict("My name is Peter", k=-1, threshold=0.5))
        if normalized_dataset is False:
            test = "artifacts/all_valid.txt"
        else:
            test = "artifacts/all_valid_normalized.txt"
        num_of_samples, precision, recall = model.test(test)
        print(num_of_samples, precision, recall)

        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("accuracy", recall)
        mlflow.log_metric("TTC", end_time - start_time)


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://localhost:8080")

    train_fasttext_model_classifier(1)
    train_fasttext_model_classifier(2, 25)
    train_fasttext_model_classifier(3, None, 0.1)
    train_fasttext_model_classifier(4, 25, 0.1)

    normalize_datasets()
    train_fasttext_model_classifier(1, normalized_dataset=True)
    train_fasttext_model_classifier(2, 25, normalized_dataset=True)
    train_fasttext_model_classifier(3, None, 0.1, normalized_dataset=True)
    train_fasttext_model_classifier(4, 25, 0.1, normalized_dataset=True)

    print(mlflow.get_artifact_uri())
