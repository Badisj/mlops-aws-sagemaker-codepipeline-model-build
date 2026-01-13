"""Feature engineers the abalone dataset."""
import argparse
import logging
import os
from pathlib import Path

import boto3
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


# Since we get a headerless CSV file we specify the column names here.
feature_columns_names = [
    # Customer Demographics
    "Age",
    "Gender",
    "Country",
    "City",
    "Membership_Years",

    # Platform Engagement
    "Login_Frequency",
    "Session_Duration_Avg",
    "Pages_Per_Session",
    "Cart_Abandonment_Rate",
    "Wishlist_Items",
    "Email_Open_Rate",
    "Mobile_App_Usage",
    "Social_Media_Engagement_Score",

    # Purchase Behavior
    "Total_Purchases",
    "Average_Order_Value",
    "Days_Since_Last_Purchase",
    "Discount_Usage_Rate",
    "Returns_Rate",
    "Payment_Method_Diversity",

    # Customer Service & Value
    "Customer_Service_Calls",
    "Product_Reviews_Written",
    "Lifetime_Value",

    # Financial / Temporal
    "Credit_Balance",
    "Signup_Quarter",
]

label_column = "Churned"

feature_columns_dtype = {
    # Customer Demographics
    "Age": np.float64,
    "Gender": str,
    "Country": str,
    "City": str,
    "Membership_Years": np.float64,

    # Platform Engagement
    "Login_Frequency": np.float64,
    "Session_Duration_Avg": np.float64,
    "Pages_Per_Session": np.float64,
    "Cart_Abandonment_Rate": np.float64,
    "Wishlist_Items": np.float64,
    "Email_Open_Rate": np.float64,
    "Mobile_App_Usage": np.float64,
    "Social_Media_Engagement_Score": np.float64,

    # Purchase Behavior
    "Total_Purchases": np.float64,
    "Average_Order_Value": np.float64,
    "Days_Since_Last_Purchase": np.float64,
    "Discount_Usage_Rate": np.float64,
    "Returns_Rate": np.float64,
    "Payment_Method_Diversity": np.float64,

    # Customer Service & Value
    "Customer_Service_Calls": np.float64,
    "Product_Reviews_Written": np.float64,
    "Lifetime_Value": np.float64,

    # Financial / Temporal
    "Credit_Balance": np.float64,
    "Signup_Quarter": str,
}

label_column_dtype = {"Churned": np.float64}


def merge_two_dicts(x, y):
    """Merges two dicts, returning a new copy."""
    z = x.copy()
    z.update(y)
    return z


if __name__ == "__main__":
    logger.debug("Starting preprocessing.")
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--input-data", type=str, required=True)
    # args = parser.parse_args()
    base_dir = Path("/opt/ml/processing")
    data_dir = base_dir / "input" / "data"
    files = list(data_dir.rglob("*.csv"))
    if len(files) == 0:
        raise RuntimeError("No input CSV files provided")
    # pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)
    # input_data = args.input_data
    # bucket = input_data.split("/")[2]
    # key = "/".join(input_data.split("/")[3:])

    logger.info("Reading data from dir", data_dir)
    # fn = f"{base_dir}/data/ecommerce_customer_churn_dataset.csv"
    # s3 = boto3.resource("s3")
    # s3.Bucket(bucket).download_file(key, fn)

    logger.info("Reading downloaded data.")
    df = pd.concat(pd.read_csv(f) for f in files)
    # df = pd.read_csv(
    #     input_dir,
    #     header=0,
    #     names=feature_columns_names + [label_column],
    #     dtype=merge_two_dicts(feature_columns_dtype, label_column_dtype),
    # )
    # os.unlink(fn)
    logger.info("Completed data read.")

    logger.debug("Defining transformers.")
    categorical_features = ["Gender", "Country", "City", "Signup_Quarter"]
    numeric_features = [elt for elt in feature_columns_names if elt not in categorical_features]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    logger.info("Applying transforms.")
    y = df.pop("Churned")
    X_pre = preprocess.fit_transform(df)
    y_pre = y.to_numpy().reshape(len(y), 1)

    X = np.concatenate((y_pre, X_pre), axis=1)

    logger.info("Splitting %d rows of data into train, validation, test datasets.", len(X))
    np.random.shuffle(X)
    train, validation, test = np.split(X, [int(0.7 * len(X)), int(0.85 * len(X))])

    logger.info("Writing out datasets to %s.", base_dir)
    pd.DataFrame(train).to_csv(f"{base_dir}/train/train.csv", header=False, index=False)
    pd.DataFrame(validation).to_csv(
        f"{base_dir}/validation/validation.csv", header=False, index=False
    )
    pd.DataFrame(test).to_csv(f"{base_dir}/test/test.csv", header=False, index=False)
    logger.info("Processing complete.")
