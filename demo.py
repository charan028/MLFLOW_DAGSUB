import os, warnings, sys, logging
import pandas as pd, numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import mlflow, mlflow.sklearn
import dagshub

# Initialize DagsHub & MLflow correctly
dagshub.init(repo_owner='mohamedabrar786786', repo_name='MLFLOW_DAGSUB', mlflow=True)
# If you prefer to be explicit, uncomment the next line (but it's redundant with mlflow=True above):
# mlflow.set_tracking_uri("https://dagshub.com/mohamedabrar786786/MLFLOW_DAGSUB.mlflow")

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    csv_url = "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception("Unable to download CSV: %s", e)
        raise

    train, test = train_test_split(data, test_size=0.25, random_state=42)
    X_train, X_test = train.drop(columns=["quality"]), test.drop(columns=["quality"])
    y_train, y_test = train["quality"].values.ravel(), test["quality"].values.ravel()

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    # (Optional) pick a consistent experiment
    mlflow.set_experiment("default")

    # Start the run AFTER the tracking URI is set
    with mlflow.start_run():
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        rmse, mae, r2 = eval_metrics(y_test, preds)
        print(f"Elasticnet model (alpha={alpha:.6f}, l1_ratio={l1_ratio:.6f}):")
        print(f"  RMSE: {rmse}")
        print(f"  MAE:  {mae}")
        print(f"  R2:   {r2}")

        mlflow.log_params({"alpha": alpha, "l1_ratio": l1_ratio})
        mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})

        # Log & (optionally) register the model
        mlflow.sklearn.log_model(model, artifact_path="model",
                                 registered_model_name="ElasticnetWineModel")
