import logging
import requests
import pickle


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Request to the Render server
url = "https://fastapi-ml-model.onrender.com/infer"
[encoder, lb, model] = pickle.load(open("model/lr_model.pkl", "rb"))
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

response = requests.post(
    url=url,
    json={
        "age": 52,
        "workclass": "Self-emp-not-inc",
        "fnlgt": 200000,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 45,
        "native-country": "United-States"
    }
)
logger.info(f"Status code: {response.status_code}")
logger.info(response.json())