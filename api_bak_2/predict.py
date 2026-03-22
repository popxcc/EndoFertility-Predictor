import json
import numpy as np
import onnxruntime as rt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
LR_PATH = BASE_DIR / "models" / "lr_pregnancy.onnx"
RF_PATH = BASE_DIR / "models" / "rf_livebirth.onnx"

lr_sess = rt.InferenceSession(str(LR_PATH))
rf_sess = rt.InferenceSession(str(RF_PATH))

def _predict(features: list) -> dict:
    x = np.array([features], dtype=np.float32)
    lr_prob = lr_sess.run(None, {"float_input": x})[1][0][1]
    rf_prob = rf_sess.run(None, {"float_input": x})[1][0][1]
    return {
        "pregnancy_prob": float(lr_prob),
        "livebirth_prob": float(rf_prob),
    }

def handler(request):
    if request.method != "POST":
        return {
            "statusCode": 405,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": "Method not allowed"})
        }

    try:
        body = json.loads(request.body)

        features = [
            body["age"],
            body["inf_dur"],
            body["inf_type"],
            body["dysm"],
            body["amh"],
            body["afc"],
            body["endo_size"],
            body["bilateral"],
            body["treatment"],
        ]

        result = _predict(features)
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps(result)
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": str(e)})
        }
