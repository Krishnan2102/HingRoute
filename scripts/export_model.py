import pandas as pd
import json
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

train_df=pd.read_csv("data/processed/train_clean.csv")
val_df=pd.read_csv("data/processed/val_clean.csv")
full_train=pd.concat([train_df,val_df],ignore_index=True)

pipeline=make_pipeline(
    TfidfVectorizer(analyzer="char_wb", ngram_range=(2,5),lowercase=True,sublinear_tf=True),
    LogisticRegression(max_iter=2000,C=10.0,random_state=42)
)
pipeline.fit(full_train["text"],full_train["label"])

joblib.dump(pipeline,"models/tfidf_pipeline.pkl")

config={
    "tau":0.65,
    "risk_intents":["card_lost","compromised_card","unrecognized_payment"],
    "classes":list(pipeline.classes_)
}

with open("models/config.json","w") as f:
    json.dump(config,f,indent=2)

print("Model and config successlfully exported to models/ directory.")
