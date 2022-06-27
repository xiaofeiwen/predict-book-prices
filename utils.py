import re
import pandas as pd


def first_num(in_val):
    num_string = in_val.split(" ")[0]
    digits = re.sub(r"[^0-9\.]", "", num_string)
    return float(digits)


def year_get(in_val):
    m = re.compile(r"\d{4}").findall(in_val)
    # print(in_val, m)
    if len(m) > 0:
        return int(m[0])
    else:
        return None


def month_get(in_val):
    m = re.compile(r"Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec").findall(in_val)
    # print(in_val, m)
    mapping = {
        "Jan": 1,
        "Feb": 2,
        "Mar": 3,
        "Apr": 4,
        "May": 5,
        "Jun": 6,
        "Jul": 7,
        "Aug": 8,
        "Sep": 9,
        "Oct": 10,
        "Nov": 11,
        "Dec": 12,
    }
    if len(m) > 0:
        return mapping[m[0]]
    else:
        return None


def rename_genre(genre_str):
    genre_list = genre_str.split()
    if genre_list[-1] in ["(Books)", "Books"]:
        genre_list.pop()
    return " ".join(genre_list)


def feature_engineering_v1(df):
    df = df.copy()
    # CLEAN FEATURES
    df["Reviews-n"] = df["Ratings"].apply(lambda x: int(first_num(x)))
    df["Ratings-n"] = df["Reviews"].apply(first_num)
    df["hard-paper"] = df["Edition"].apply(lambda x: x.split(",")[0])
    df["year"] = df["Edition"].apply(year_get)
    df["month"] = df["Edition"].apply(month_get)
    # DROPPING ORIGINAL FEATURES
    df.drop(["Edition", "Ratings", "Reviews"], axis=1, inplace=True)
    return df


def predict(df, feature_engineer, model, output_name):
    df_fe = feature_engineer(df)
    prediction = model.predict(df_fe)
    df_submission = pd.DataFrame(columns=["ID", "Price"])
    # Creating ID column from ID list
    df_submission["ID"] = df["ID"].tolist()
    # Creating label column from price prediction list
    df_submission["Price"] = prediction
    # saving your csv file for Leaderboard submission
    df_submission.to_csv(f"predictions/{output_name}", index=False)
