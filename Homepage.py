import streamlit as st
import pandas as pd
import numpy as np
import math
import pickle
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler, SplineTransformer, OneHotEncoder
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from src.src import get_expected_score

# ---------- Helper: compute Massey ratings ----------
OUTPUT_FILE = "./data/cleaned_ttc_scores.csv"
df = read_data = pd.read_csv(OUTPUT_FILE)
player_summary_df = pd.read_csv("./data/player_summary.csv")

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


massey_df = pd.read_csv("./data/final_massey_ratings.csv")

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

all_massey_ratings_df = pd.read_csv("./data/all_massey_ratings_over_time.csv")

select_player_ratings_plot = st.multiselect(
    "Select players to plot their rating over time:",
    set(massey_df["player"]),
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

filename = "./models/massey_ratings_to_pred_win_model.pkl"
# Load the model from disk
with open(filename, "rb") as file:
    model = pickle.load(file)


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
