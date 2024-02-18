import fasttext
import mlflow


def train_fasttext_model_classifier(model_num=1, epoch=None, lr=None):
    with mlflow.start_run():
        mlflow.log_artifact(f"artifacts/all_train.txt")

        mlflow.log_param("lr", 0.1 if lr is None else lr)
        mlflow.log_param("epoch", 5 if epoch is None else epoch)

        if epoch is None and lr is None:
            model = fasttext.train_supervised(input="artifacts/all_train.txt")
        elif epoch is not None and lr is None:
            model = fasttext.train_supervised(input="artifacts/all_train.txt", epoch=epoch)
        elif epoch is None and lr is not None:
            model = fasttext.train_supervised(input="artifacts/all_train.txt", lr=lr)
        else:
            model = fasttext.train_supervised(input="artifacts/all_train.txt", epoch=epoch, lr=lr)

        model.save_model(f"models/model{model_num}.bin")

        # mlflow.register_model(f"models/model{model_num}.bin", f"model{model_num}")

        print(model.predict("Jmenuji se Petr", k=-1, threshold=0.5))
        print(model.predict("Volam sa Peter", k=-1, threshold=0.5))
        print(model.predict("My name is Peter", k=-1, threshold=0.5))
        num_of_samples, precision, recall = model.test("artifacts/all_valid.txt")
        print(num_of_samples, precision, recall)

        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://localhost:8080")
    train_fasttext_model_classifier(1)
    train_fasttext_model_classifier(2, 25)
    train_fasttext_model_classifier(3, None, 0.1)
    train_fasttext_model_classifier(4, 25, 0.1)
    print(mlflow.get_artifact_uri())
