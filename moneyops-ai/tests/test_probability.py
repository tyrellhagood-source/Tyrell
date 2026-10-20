from datetime import timedelta

import numpy as np
import pytest

from data.sports_registry import SPORTS
from models.probability import TeamRatingModel
from models.stats import hit_rate, streaks, wilson_interval
from tests.conftest import synthetic_league


def test_gaussian_model_recovers_strength_order_and_no_leakage():
    teams, games = synthetic_league(rounds=10)
    cut = games[len(games) // 2].date
    m = TeamRatingModel(SPORTS["NBA"], half_life_days=None).fit(games, cut)
    assert m.trained_through < cut
    assert m.n_games == sum(1 for g in games if g.date < cut)
    ratings = [m.team_rating(t) for t in teams]
    assert np.corrcoef(ratings, range(len(teams)))[0, 1] > 0.8
    with pytest.raises(ValueError):
        TeamRatingModel(SPORTS["NBA"]).fit(games, games[0].date)  # nothing before first game


def test_probabilities_are_coherent():
    teams, games = synthetic_league(rounds=10)
    m = TeamRatingModel(SPORTS["NBA"]).fit(games, games[-1].date + timedelta(days=1))
    strong, weak = teams[-1], teams[0]
    ph, _ = m.probability("h2h", "home", None, strong, weak)
    pa, _ = m.probability("h2h", "away", None, strong, weak)
    assert ph + pa == pytest.approx(1.0, abs=1e-9)
    assert ph > 0.6
    # spreads on a half-point line have no push; both sides sum to 1
    w1, p1 = m.probability("spreads", "home", -4.5, strong, weak)
    w2, p2 = m.probability("spreads", "away", 4.5, strong, weak)
    assert p1 == 0 and w1 + w2 == pytest.approx(1.0, abs=1e-9)
    # whole-number line has a push probability; win+push+loss = 1 consistent
    w3, p3 = m.probability("spreads", "home", -4.0, strong, weak)
    w4, p4 = m.probability("spreads", "away", 4.0, strong, weak)
    assert p3 > 0 and p3 == pytest.approx(p4) and w3 + w4 + p3 == pytest.approx(1.0, abs=1e-9)
    o, _ = m.probability("totals", "over", 200.5, strong, weak)
    u, _ = m.probability("totals", "under", 200.5, strong, weak)
    assert o + u == pytest.approx(1.0, abs=1e-9)
    assert m.probability("spreads", "home", -4.25, strong, weak) is None  # quarter line unsupported


def test_poisson_soccer_three_way_and_nhl_ties():
    teams, games = synthetic_league(rounds=10, poisson=True)
    after = games[-1].date + timedelta(days=1)
    soc = TeamRatingModel(SPORTS["EPL"]).fit(games, after)
    h, _ = soc.probability("h2h", "home", None, teams[3], teams[5])
    d, _ = soc.probability("h2h", "draw", None, teams[3], teams[5])
    a, _ = soc.probability("h2h", "away", None, teams[3], teams[5])
    assert h + d + a == pytest.approx(1.0, abs=1e-9) and d > 0.05
    nhl = TeamRatingModel(SPORTS["NHL"]).fit(games, after)
    grid = nhl.score_grid(teams[3], teams[5])
    assert np.trace(grid.prob) == pytest.approx(0.0, abs=1e-12)  # no ties after OT/shootout
    h2, _ = nhl.probability("h2h", "home", None, teams[3], teams[5])
    a2, _ = nhl.probability("h2h", "away", None, teams[3], teams[5])
    assert h2 + a2 == pytest.approx(1.0, abs=1e-9)


def test_joint_probability_same_game():
    teams, games = synthetic_league(rounds=10)
    m = TeamRatingModel(SPORTS["NBA"]).fit(games, games[-1].date + timedelta(days=1))
    legs = [("h2h", "home", None), ("totals", "over", 200.5)]
    joint = m.joint_probability(legs, teams[5], teams[2])
    p1 = m.probability("h2h", "home", None, teams[5], teams[2])[0]
    p2 = m.probability("totals", "over", 200.5, teams[5], teams[2])[0]
    assert 0 < joint <= min(p1, p2)
    # mutually exclusive legs -> zero joint probability
    assert m.joint_probability([("h2h", "home", None), ("h2h", "away", None)], teams[5], teams[2]) == pytest.approx(
        0.0, abs=0.01)


def test_bootstrap_gives_spread_of_estimates():
    teams, games = synthetic_league(rounds=6)
    m = TeamRatingModel(SPORTS["NBA"]).fit(games, games[-1].date + timedelta(days=1))
    boots = m.bootstrap(10, seed=3)
    ps = [b.probability("h2h", "home", None, teams[4], teams[6])[0] for b in boots
          if b.is_rated(teams[4]) and b.is_rated(teams[6])]
    assert len(ps) >= 5 and np.std(ps) > 0


def test_wilson_and_hit_rate():
    lo, hi = wilson_interval(7, 10)
    assert 0.39 < lo < 0.40 and 0.89 < hi < 0.90
    assert wilson_interval(0, 0) == (0.0, 1.0)
    hr = hit_rate(["W", "W", "L", "P", "W"])
    assert hr["wins"] == 3 and hr["losses"] == 1 and hr["pushes"] == 1 and hr["n"] == 4
    assert hr["rate"] == 0.75


def test_streaks():
    s = streaks(["W", "W", "P", "W", "L", "L", "W"])
    assert s["longest_win"] == 3 and s["longest_loss"] == 2 and s["current"] == "W1"
