import numpy as np
import pandas as pd
import math
from src import get_expected_score

from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, SplineTransformer, OneHotEncoder
import pickle
### Assumes you have created the cleaned_ttc_scores object from clean_data.py


def compute_massey_ratings(df: pd.DataFrame, keep_steps: bool = True) -> pd.DataFrame:
    """
    Compute Massey ratings from dataframe with:
    Winner, Loser, WinnerGames, LoserGames
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values(by="Date", ascending=True, inplace=True)

    players = sorted(set(df["Winner"]).union(set(df["Loser"])))
    n = len(players)

    if n < 2:
        return pd.DataFrame()

    # Getting dictionary of player name and their playerid
    # Eg: {"Peter Tea": 0, "Kelvin Lu":1, ...}
    idx = {player_name: index_int for index_int, player_name in enumerate(players)}

    M = np.zeros((n, n), float)
    b = np.zeros(n, float)

    ratings_by_time_arr = []

    for row_index, row in df.iterrows():
        # for iterrows, we get the following in each iteration:
        # (index, Series)
        # row = df.iloc[0,:]
        player1_name, player2_name = row["Winner"], row["Loser"]
        games_player1, games_player2 = row["WinnerGames"], row["LoserGames"]
        margin = games_player1 - games_player2

        player1_id, player2_id = idx[player1_name], idx[player2_name]
        M[player1_id, player1_id] += 1
        M[player2_id, player2_id] += 1
        M[player1_id, player2_id] -= 1
        M[player2_id, player1_id] -= 1
        b[player1_id] += margin
        b[player2_id] -= margin

        if keep_steps:
            # Impose constraint: sum(ratings)=0
            M[-1, :] = 1.0
            b[-1] = 0.0
            # Compute least-squares solution to linear system || b-ax|| is minimized
            # Finds "x" such that M * x = b
            try:
                player1_prev_rating = ratings[player1_id]
                player2_prev_rating = ratings[player2_id]
            except NameError:
                player1_prev_rating = 0.0
                player2_prev_rating = 0.0

            ratings = np.linalg.lstsq(M, b, rcond=None)[0]
            # | player | date | rating | score | result | opponent |
            expected_result_player1 = player1_prev_rating - player2_prev_rating

            ratings_by_time_arr.append(
                {
                    "Date": row["Date"],
                    "player": player1_name,
                    "prev_rating": round(player1_prev_rating, 3),
                    "rating": round(ratings[player1_id], 3),
                    "score": row["Score"],
                    "expected_result": get_expected_score(
                        player1_prev_rating, player2_prev_rating
                    ),
                    "result": "W" if row["IsDraw"] == False else "D",
                    "opponent": player2_name,
                    "opponent_prev_rating": round(player2_prev_rating, 3),
                }
            )
            ratings_by_time_arr.append(
                {
                    "Date": row["Date"],
                    "player": player2_name,
                    "rating": round(ratings[player2_id], 3),
                    "prev_rating": round(player2_prev_rating, 3),
                    "score": row["Score"],
                    "expected_result": get_expected_score(
                        player2_prev_rating, player1_prev_rating
                    ),
                    "result": "L" if row["IsDraw"] == False else "D",
                    "opponent": player1_name,
                    "opponent_prev_rating": round(player1_prev_rating, 3),
                }
            )

    # Impose constraint: sum(ratings)=0
    M[-1, :] = 1.0
    b[-1] = 0.0

    # Compute least-squares solution to linear system || b-ax|| is minimized
    # Finds "x" such that M * x = b
    ratings = np.linalg.lstsq(M, b, rcond=None)[0]

    player_ratings_dict = {player: rating for player, rating in zip(players, ratings)}
    player_ratings_df = pd.DataFrame.from_dict(
        player_ratings_dict, orient="index", columns=["massey_rating"]
    )
    player_ratings_df.reset_index(inplace=True)
    player_ratings_df.rename(columns={"index": "player"}, inplace=True)
    player_ratings_df["massey_rating"] = player_ratings_df["massey_rating"].round(2)
    player_ratings_df.sort_values(by="massey_rating", ascending=False, inplace=True)

    matches_played = pd.concat([df["Winner"], df["Loser"]]).value_counts()
    matches_played.index.name = "player"
    matches_played.name = "matches_played"

    full_df = player_ratings_df.merge(
        matches_played, left_on="player", right_on="player", how="left"
    )

    # Minimum matches played filter
    full_df = full_df[full_df["matches_played"] >= 5]
    full_df.reset_index(drop=True, inplace=True)

    ratings_by_time_arr_df = pd.DataFrame(ratings_by_time_arr)
    # ratings_by_time_arr_df[ratings_by_time_arr_df["player"]=="Peter Tea"]
    # ratings_by_time_arr_df[ratings_by_time_arr_df["player"]=="shane kafka"]

    return full_df, ratings_by_time_arr_df


df = pd.read_csv("/Users/p.tea/Documents/ttc_ratings/data/cleaned_ttc_scores.csv")

player_summary_df = pd.read_csv(
    "/Users/p.tea/Documents/ttc_ratings/data/player_summary.csv"
)

final_massey_rating_df, all_massey_ratings_df = compute_massey_ratings(df)

all_massey_ratings_df.to_csv(
    "/Users/p.tea/Documents/ttc_ratings/data/all_massey_ratings_over_time.csv",
    index=False,
)

massey_df = final_massey_rating_df.merge(
    player_summary_df, left_on="player", right_on="player", how="left"
)
massey_df = massey_df[
    ["player", "massey_rating", "record", "points_percentage", "last_match_date"]
]


massey_df.to_csv(
    "/Users/p.tea/Documents/ttc_ratings/data/final_massey_ratings.csv", index=False
)


# ---------- PREDICT MATCH OUTCOME ----------
spline_fit_df = all_massey_ratings_df[["prev_rating", "opponent_prev_rating", "result"]]

spline_fit_df["diff_rating"] = (
    spline_fit_df["prev_rating"] - spline_fit_df["opponent_prev_rating"]
)
spline_fit_df["result_binary"] = np.where(spline_fit_df["result"] == "W", 1, 0)
spline_fit_df = spline_fit_df[spline_fit_df["result"] != "D"]


# spline_fit_df1 = all_massey_ratings_df[
#     ["player", "opponent", "Date", "prev_rating", "opponent_prev_rating"]
# ]
# spline_fit_df2 = all_massey_ratings_df[
#     ["player", "opponent", "Date", "prev_rating", "opponent_prev_rating"]
# ].rename(
#     columns={
#         "player": "opponent",
#         "prev_rating": "opponent_prev_rating",
#         "opponent": "player",
#         "opponent_prev_rating": "prev_rating",
#     }
# )

# spline_fit_df = (
#     pd.concat([spline_fit_df1, spline_fit_df2])
#     .reset_index(drop=True)
#     .sort_values(by=["Date"], ascending=[True])
# )


X = spline_fit_df[["diff_rating"]]
y = spline_fit_df["result_binary"]


# 2. Define the Spline Transformer
# degree=3 for cubic splines, n_knots sets the number of segments
# knots='uniform' places knots evenly, but you can also define custom knot locations
spline_transformer = SplineTransformer(
    degree=2, n_knots=5, knots="uniform", include_bias=False
)

# 3. Define the Logistic Regression model
# We set C to a high value to reduce regularization, focusing on the spline fit
logistic_regression = LogisticRegression(solver="liblinear", C=1000)

# 4. Create a pipeline
# The pipeline first transforms the data, then fits the logistic model
model = Pipeline([("spline", spline_transformer), ("logistic", logistic_regression)])

# 5. Fit the model
model.fit(X, y)

# Evaluate log-loss on training data
y_pred_proba = model.predict_proba(X)

# Calculate the log loss
loss = log_loss(y, y_pred_proba)

# st.success(f"Training set Log Loss: {loss}")
print(f"Training set Log Loss: {loss}")


filename = (
    "/Users/p.tea/Documents/ttc_ratings/models/massey_ratings_to_pred_win_model.pkl"
)
# Save the model to disk
with open(filename, "wb") as file:
    pickle.dump(model, file)
