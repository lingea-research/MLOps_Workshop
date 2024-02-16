from mlflow import log_param, log_metric, log_artifact, set_tracking_uri


if __name__ == '__main__':
    set_tracking_uri("http://localhost:8080")

    for i in range(5):
        log_param("threshold", 3)

        log_metric("accuracy", 0.8)
        log_metric("TTC", 33)

        log_artifact(f"./artifacts/artifact{i+1}.csv")
