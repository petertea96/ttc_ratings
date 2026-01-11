import math


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
