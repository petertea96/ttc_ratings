import streamlit as st
import pandas as pd
import numpy as np
import math

from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, SplineTransformer, OneHotEncoder
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# ---------- Helper: compute Massey ratings ----------
OUTPUT_FILE = "./data/cleaned_ttc_scores.csv"
df = read_data = pd.read_csv(OUTPUT_FILE)


def get_expected_score(player_a_rating: float, player_b_rating: float) -> str:
    """
    Given two player ratings, compute expected score for player A
    using the Massey method formula.
    """
    rating_diff = player_a_rating - player_b_rating

    rating_diff = max(-8.0, rating_diff)
    rating_diff = min(8.0, rating_diff)

    if rating_diff < 0:
        rating_diff = abs(math.floor(rating_diff))
        match_result = "L"

    else:
        rating_diff = math.ceil(rating_diff)
        match_result = "W"

    expected_score_loser = 8 - rating_diff

    expected_score = match_result + " " + str(8) + "-" + str(expected_score_loser)
    return expected_score


def match_results_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a dataframe with columns:
    ['player_1', 'player_2', 'player_1_games_won', 'player_2_games_won']

    Returns a dataframe with one row per player containing:
    ['matches_won', 'matches_lost', 'matches_drawn', ...]
    """

    # Results from player_1 perspective
    matches_won = df[~df.IsDraw].groupby("Winner").size()
    matches_won.index.name = "player"
    matches_won.name = "matches_won"

    matches_lost = df[~df.IsDraw].groupby("Loser").size()
    matches_lost.index.name = "player"
    matches_lost.name = "matches_lost"

    matches_drawn = pd.concat(
        [df[df.IsDraw]["Winner"], df[df.IsDraw]["Loser"]]
    ).value_counts()
    matches_drawn.index.name = "player"
    matches_drawn.name = "matches_drawn"

    match_dates = pd.concat(
        [
            df[["Date", "Winner"]].rename(columns={"Winner": "player"}),
            df[["Date", "Loser"]].rename(columns={"Loser": "player"}),
        ]
    )
    last_match_date = match_dates.groupby("player")["Date"].max()
    last_match_date.name = "last_match_date"
    # .dt.date

    summary = pd.concat(
        [matches_won, matches_lost, matches_drawn, last_match_date], axis=1
    )
    summary = summary.fillna(0).reset_index()
    summary = summary.astype(
        {
            "matches_won": "int64",
            "matches_lost": "int64",
            "matches_drawn": "int64",
            "last_match_date": "string",
        }
    )
    summary["num_matches"] = (
        summary["matches_won"] + summary["matches_lost"] + summary["matches_drawn"]
    )
    summary["win_percentage"] = summary["matches_won"] / summary["num_matches"]
    summary["win_percentage"] = 100 * summary["win_percentage"].round(3)
    summary["record"] = (
        summary["matches_won"].astype(str)
        + "-"
        + summary["matches_lost"].astype(str)
        + "-"
        + summary["matches_drawn"].astype(str)
    )
    summary["points_percentage"] = (
        3 * summary["matches_won"] + 1 * summary["matches_drawn"]
    ) / (summary["num_matches"] * 3)

    summary.sort_values(
        by=["points_percentage", "matches_won"], ascending=[False, False], inplace=True
    )

    return summary


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


# ---------- Streamlit App ----------

st.set_page_config(
    page_title="Toronto Tennis City Massey Ratings – Single Row Input",
    layout="wide",
)

st.title("🎾 Toronto Tennis City Player Ratings – 8 game pro set matches")

# ---------- Table of current player ratings ----------

st.write(
    """
### Current player ratings
These ratings are computed using the **Massey method**, based on all recorded first-to-8 match results from the Toronto Tennis City Singles ladder.
The ratings are updated based on the quality of opponents faced, as well as the margin of victory. The method predicts an expected score for future matches between any two players, and then updates ratings based on the observed results.
    """
)

final_massey_rating_df, all_massey_ratings_df = compute_massey_ratings(df)
match_results_summary_df = match_results_summary(df)

massey_df = final_massey_rating_df.merge(
    match_results_summary_df, left_on="player", right_on="player", how="left"
)
massey_df = massey_df[
    ["player", "massey_rating", "record", "points_percentage", "last_match_date"]
]

st.caption(
    "points_percentage is the % of points won from all matches played. Ladder rules state that a Win earns 3 points, a Draw earns 1 point and a Loss earns 0 points. "
)

st.dataframe(massey_df, use_container_width=True)


# ---------- Show evolution ----------
st.write(
    """
### Player Ratings Over Time

    """
)

select_player_ratings_plot = st.multiselect(
    "Select players to plot their rating over time:",
    set(final_massey_rating_df["player"]),
    max_selections=5,
    accept_new_options=True,
    placeholder="Select contact method...",
)
st.space("small")

# players_to_plot = ["Peter Tea", "Will Dove"]
player_ratings_over_time_df = all_massey_ratings_df[
    all_massey_ratings_df["player"].isin(select_player_ratings_plot)
]

player_ratings_over_time_df["match_result"] = (
    player_ratings_over_time_df["result"]
    + " "
    + player_ratings_over_time_df["score"]
    + " vs. "
    + player_ratings_over_time_df["opponent"]
)

# player_ratings_over_time_df[
#     [
#         "Date",
#         "player",
#         "opponent",
#         "prev_rating",
#         "opponent_prev_rating",
#         "match_result",
#         "expected_result",
#         "result",
#     ]
# ]

# basic line chart:
# st.line_chart(player_ratings_over_time_df, x="Date", y="rating", color="player")


# --- Create Plotly Figure with Graph Objects ---
fig = go.Figure()

# Add a scatter trace for actual data points

for each_player in select_player_ratings_plot:
    player_data = player_ratings_over_time_df[
        player_ratings_over_time_df["player"] == each_player
    ]
    fig.add_trace(
        go.Scatter(
            x=player_data["Date"],
            y=player_data["rating"],
            mode="markers+lines",  # Combines both markers and lines
            name=each_player,
            text=player_data["match_result"],
            marker=dict(size=8),
            line=dict(width=2),
        )
    )

# Update layout for better presentation
fig.update_layout(
    title="Player ratings over time",
    xaxis_title="Date",
    yaxis_title="Rating",
    legend_title="Player",
)

# --- 3. Display the chart in Streamlit ---
st.plotly_chart(fig, use_container_width=True)


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


# 6. (Optional) Visualize the results
# Plotting the predicted probability curve over the range of X


# X_plot = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
# y_prob_plot = model.predict_proba(X_plot)[:, 1]


# fig, ax = plt.subplots(figsize=(6, 4))

# ax.scatter(X, y, color="gray", alpha=0.5, label="Actual Data (0/1)")
# ax.plot(
#     X_plot,
#     y_prob_plot,
#     color="red",
#     linewidth=2,
#     label="Spline Logistic Fit (Probability)",
# )
# ax.set_title("Spline Model with Logistic Regression")
# ax.set_xlabel("Difference in Rating")
# ax.set_ylabel("Prob(Win) (Draws counted as loss)")
# ax.legend()
# ax.grid(True)
# fig.show()


# model.predict_proba(pd.DataFrame({"diff_rating": [-0.5, 0.0, 0.5, 0.55]}))
# model.predict_proba(pd.DataFrame({"diff_rating": [-0.5]}))

# st.pyplot(fig, use_container_width=False)

# pred_win = model.predict_proba(pd.DataFrame({"diff_rating": [0.5]}))
# pred_loss = model.predict_proba(pd.DataFrame({"diff_rating": [-0.5]}))

# st.success(f"Diff rating 0.5. Pred Win: {pred_win}")
# st.warning(f"Diff rating -0.5. Pred Win: {pred_loss}")


# st.dataframe(
#     all_massey_ratings_df.sort_values(by="Date", ascending=False),
#     # .astype(
#     #    {"Date": "string", "IsDraw": "string"}
#     # )
#     use_container_width=True,
# )

# ---------- Predict next match ----------
st.write(
    """
### Predicting Future Match Outcomes
Use latest ratings to predict match outcomes:
    """
)


def round_to_nearest_5(n):
    return 5 * round(n / 5)


col1, col2 = st.columns(2)

predict_player_choices = sorted(set(massey_df["player"]))

with col1:
    select_player1_match_prediction = st.selectbox(
        "Player 1:", predict_player_choices, placeholder="Peter Tea"
    )

with col2:
    select_player2_match_prediction = st.selectbox(
        "Player 2:", predict_player_choices, placeholder="Select Player 2..."
    )

    rating1 = massey_df[massey_df["player"] == select_player1_match_prediction][
        "massey_rating"
    ].values[0]
    rating2 = massey_df[massey_df["player"] == select_player2_match_prediction][
        "massey_rating"
    ].values[0]

if rating1 >= rating2:
    diff_rating = rating1 - rating2
    pred_score = get_expected_score(rating1, rating2)
    pred_match_win = model.predict_proba(pd.DataFrame({"diff_rating": [diff_rating]}))[
        0
    ][1]
    # st.success(f"Difference in rating: {round(diff_rating, 2)}")
    st.success(
        f"{select_player1_match_prediction} ({rating1}) is predicted to win against {select_player2_match_prediction} ({rating2}) with probability {100 * pred_match_win:.2f}%."
    )
    st.success(f"Expected Score: {select_player1_match_prediction} {pred_score}")
    st.markdown(
        f":green-badge[:material/favorite: {select_player1_match_prediction} to win: {round(1 / pred_match_win, 2)}] :red-badge[:material/favorite: {select_player2_match_prediction} to win: {round(1 / (1 - pred_match_win), 2)}]"
    )
    st.markdown(
        f":green-badge[:material/favorite: {select_player1_match_prediction} to win: {round_to_nearest_5(-(pred_match_win / (1 - pred_match_win)) * 100) - 20}] :red-badge[:material/favorite: {select_player2_match_prediction} to win: +{round_to_nearest_5(100 * pred_match_win / (1 - pred_match_win)) + 5}]"
    )

else:
    diff_rating = rating2 - rating1
    pred_score = get_expected_score(rating2, rating1)
    pred_match_win = model.predict_proba(pd.DataFrame({"diff_rating": [diff_rating]}))[
        0
    ][1]
    # st.success(f"Difference in rating: {round(diff_rating, 2)}")
    st.success(
        f"{select_player2_match_prediction} ({rating2}) is predicted to win against {select_player1_match_prediction} ({rating1}) with probability {100 * pred_match_win:.2f}%."
    )
    st.success(f"Expected Score: {select_player2_match_prediction} {pred_score}")
    st.markdown(
        f":green-badge[:material/favorite: {select_player2_match_prediction} to win: {round_to_nearest_5(-(pred_match_win / (1 - pred_match_win)) * 100) - 20}] :red-badge[:material/favorite: {select_player1_match_prediction} to win: +{round_to_nearest_5(100 * pred_match_win / (1 - pred_match_win)) + 5}]"
    )


# ---------- Manual data entry ----------

# st.write(
#     """
# Enter match results **one at a time**, and this app will:
# 1. Store the match in memory
# 2. Recompute **Massey ratings**
# 3. Let you **predict future first-to-8 scores**

# ### Required fields for each match:
# - Player A name
# - Player B name
# - Score for Player A (0–8)
# - Score for Player B (0–8)
# """
# )

# # Create session state to store match history
# if "match_history" not in st.session_state:
#     st.session_state.match_history = pd.DataFrame(
#         columns=["Winner", "Loser", "WinnerGames", "LoserGames"]
#     )


# # ---------- Input Form for One New Row ----------

# with st.form("add_match_form"):
#     st.subheader("Add a New Match Result")

#     col1, col2 = st.columns(2)
#     with col1:
#         Winner = st.text_input("Player A Name")
#     with col2:
#         Loser = st.text_input("Player B Name")

#     col3, col4 = st.columns(2)
#     with col3:
#         WinnerGames = st.number_input("Score A (0–8)", min_value=0, max_value=8, step=1)
#     with col4:
#         LoserGames = st.number_input("Score B (0–8)", min_value=0, max_value=8, step=1)

#     submitted = st.form_submit_button("Add Match")

#     if submitted:
#         if not Winner or not Loser:
#             st.warning("Please enter both player names.")
#         elif Winner == Loser:
#             st.warning("Players must be different.")
#         else:
#             new_row = {
#                 "Winner": Winner,
#                 "Loser": Loser,
#                 "WinnerGames": WinnerGames,
#                 "LoserGames": LoserGames,
#             }
