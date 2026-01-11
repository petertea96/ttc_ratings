import pandas as pd

# P: pewJoz-jumnym-1narru
# process: Copy and paste scores to google sheets from ttc scores website
# then, download as ttc_scores.csv

# this script cleans ttc_scores.csv and generates clean_ttc_scores.csv

# remove matches with an incorrect score
# remove duplicates
# Fix scores that are draws.

# Date | Winner | Loser | Score | WinnerGames | LoserGames | IsDraw |
df = pd.read_csv("/Users/p.tea/Documents/ttc_ratings/data/ttc_scores.csv")
df.head()

df.shape
df.drop_duplicates().shape
df = df.drop_duplicates()

df[df["Score"] != "1-"].shape
df = df[df["Score"] != "1-"]


def fix_score_format(score: str):
    assert isinstance(score, str)

    score_split = score.split("-")
    if len(score_split) >= 2:
        score1 = score_split[0]
        score2 = score_split[1]

        return f"{score1}-{score2}"
    else:
        return score


df["Score"] = df["Score"].apply(fix_score_format)


def get_winner_loser_games(score: str):
    score_split = score.split("-")
    if len(score_split) >= 2:
        score1 = int(score_split[0])
        score2 = int(score_split[1])

        if score1 > score2:
            return score1, score2, False
        elif score1 == score2:
            return score1, score2, True
        else:
            return score2, score1, False
    else:
        return 0, 0, True


assert get_winner_loser_games("3-1") == (3, 1, False)
assert get_winner_loser_games("3-3") == (3, 3, True)

df_a = df.copy()
df_a["Score"].apply(get_winner_loser_games).apply(pd.Series)
df_a[["WinnerGames", "LoserGames", "IsDraw"]] = (
    df_a["Score"].apply(get_winner_loser_games).apply(pd.Series)
)

df_a.to_csv(
    "/Users/p.tea/Documents/ttc_ratings/data/cleaned_ttc_scores.csv", index=False
)


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


player_summary_df = match_results_summary(df_a)


player_summary_df.to_csv(
    "/Users/p.tea/Documents/ttc_ratings/data/player_summary.csv", index=False
)
