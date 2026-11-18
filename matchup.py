"""
Matchup detail - side-by-side team comparison for a single game.

Every stat is computed only from games played *before* kickoff, so a past
week's page shows what was known going into that game.
"""

import math
import pandas as pd

from nfl_predictor import get_team_stats_from_history, PRIOR_SEASON_GAMES


def _num(value):
    """Convert pandas/numpy scalars to plain Python, NaN -> None."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, 'item'):
        return value.item()
    return value


def _team_result(game, team):
    """(points_for, points_against, 'W'|'L'|'T', opponent, was_home) for team in game."""
    home = game['home_team'] == team
    pf = game['home_score'] if home else game['away_score']
    pa = game['away_score'] if home else game['home_score']
    outcome = 'W' if pf > pa else 'L' if pf < pa else 'T'
    opponent = game['away_team'] if home else game['home_team']
    return int(pf), int(pa), outcome, opponent, home


def _record(results):
    w = sum(r[2] == 'W' for r in results)
    l = sum(r[2] == 'L' for r in results)
    t = sum(r[2] == 'T' for r in results)
    return w, l, t


def _record_str(results):
    w, l, t = _record(results)
    return f"{w}-{l}" + (f"-{t}" if t else "")


def _win_pct(results):
    if not results:
        return None
    w, _, t = _record(results)
    return (w + 0.5 * t) / len(results)


def _covered(game, team):
    """True/False/None (push or no line) for whether team beat the closing spread."""
    spread = game.get('spread_line')
    if spread is None or pd.isna(spread):
        return None
    margin = game['home_score'] - game['away_score']  # positive = home won by that much
    if margin == spread:
        return None
    home_covered = margin > spread
    return home_covered if game['home_team'] == team else not home_covered


def _implied_prob(moneyline):
    if moneyline is None or pd.isna(moneyline) or moneyline == 0:
        return None
    return (-moneyline) / (-moneyline + 100) if moneyline < 0 else 100 / (moneyline + 100)


def _team_games(games, team):
    return games[(games['home_team'] == team) | (games['away_team'] == team)]


def _team_profile(team, season_games, prior_season_games, all_before, date_col, game_date, season, game_row, side):
    """All per-team numbers shown on the matchup page."""
    games = _team_games(season_games, team)
    results = [_team_result(g, team) for _, g in games.iterrows()]
    home_results = [r for r in results if r[4]]
    away_results = [r for r in results if not r[4]]
    last5 = results[-5:]

    n = len(results)
    ppg = sum(r[0] for r in results) / n if n else None
    papg = sum(r[1] for r in results) / n if n else None
    last5_diff = sum(r[0] - r[1] for r in last5) / len(last5) if last5 else None

    covers = [c for c in (_covered(g, team) for _, g in games.iterrows()) if c is not None]

    # Streak, e.g. "W3"
    streak = ""
    if results:
        kind = results[-1][2]
        count = 0
        for r in reversed(results):
            if r[2] != kind:
                break
            count += 1
        streak = f"{kind}{count}"

    # Strength of schedule: average season win% of opponents faced so far
    opp_pcts = []
    for r in results:
        opp_results = [_team_result(g, r[3]) for _, g in _team_games(season_games, r[3]).iterrows()]
        pct = _win_pct(opp_results)
        if pct is not None:
            opp_pcts.append(pct)
    sos = sum(opp_pcts) / len(opp_pcts) if opp_pcts else None

    prior_results = [_team_result(g, team) for _, g in _team_games(prior_season_games, team).iterrows()]

    blended = get_team_stats_from_history(team, all_before, date_col, game_date, season=season)

    return {
        'team': team,
        'games': n,
        'record': _record_str(results) if results else '0-0',
        'win_pct': _win_pct(results),
        'home_record': _record_str(home_results) if home_results else '0-0',
        'away_record': _record_str(away_results) if away_results else '0-0',
        'ppg': ppg,
        'papg': papg,
        'diff': (ppg - papg) if n else None,
        'last5_diff': last5_diff,
        'ats': f"{sum(covers)}-{len(covers) - sum(covers)}" if covers else '—',
        'ats_pct': sum(covers) / len(covers) if covers else None,
        'streak': streak,
        'sos': sos,
        'prior_record': _record_str(prior_results) if prior_results else '—',
        'rest': _num(game_row.get(f'{side}_rest')),
        'qb': _num(game_row.get(f'{side}_qb_name')),
        'coach': _num(game_row.get(f'{side}_coach')),
        'form': [
            {'result': r[2], 'opp': r[3], 'home': r[4], 'score': f"{r[0]}-{r[1]}"}
            for r in last5
        ],
        'model_ppg': blended['avg_points_scored'],
        'model_papg': blended['avg_points_allowed'],
        'model_win_rate': blended['win_rate'],
        'current_weight': n / (n + PRIOR_SEASON_GAMES),
    }


# (key, label, higher_is_better, scale, formatter, phrase for key edges)
# scale = the difference that counts as a big edge; it sizes the bars and ranks key edges.
COMPARISONS = [
    ('win_pct', 'Win %', True, 0.30, lambda v: f"{v * 100:.0f}%", None),
    ('ppg', 'Points per game', True, 7.0, lambda v: f"{v:.1f}", "scores {d:.1f} more points per game"),
    ('papg', 'Points allowed per game', False, 7.0, lambda v: f"{v:.1f}", "allows {d:.1f} fewer points per game"),
    ('diff', 'Point differential / game', True, 10.0, lambda v: f"{v:+.1f}", "has a {d:.1f}-point better margin per game"),
    ('last5_diff', 'Last 5 margin / game', True, 10.0, lambda v: f"{v:+.1f}", "has been {d:.1f} points per game better over the last 5"),
    ('sos', 'Strength of schedule', True, 0.12, lambda v: f"{v:.3f}", "has faced tougher opponents (opponents' win % {d:.3f} higher)"),
    ('ats_pct', 'Against the spread', True, 0.30, lambda v: f"{v * 100:.0f}%", "has covered the spread {d:.0%} more often"),
    ('rest', 'Days of rest', True, 3.0, lambda v: f"{v:.0f}", "has {d:.0f} more days of rest"),
]

DISPLAY_TEXT = {'ats_pct': 'ats'}  # show the W-L string instead of the percentage


def _compare(away, home):
    rows, edges = [], []
    for key, label, higher_better, scale, fmt, phrase in COMPARISONS:
        a, h = away.get(key), home.get(key)
        row = {
            'label': label,
            'away': '—' if a is None else (away[DISPLAY_TEXT[key]] if key in DISPLAY_TEXT else fmt(a)),
            'home': '—' if h is None else (home[DISPLAY_TEXT[key]] if key in DISPLAY_TEXT else fmt(h)),
            'edge': None,
            'away_share': 50.0,
        }
        if a is not None and h is not None and a != h:
            # positive = away team is better on this stat
            signed = (a - h) if higher_better else (h - a)
            row['edge'] = 'away' if signed > 0 else 'home'
            row['away_share'] = round(50 + 45 * math.tanh(signed / scale), 1)
            if phrase:
                leader = away['team'] if signed > 0 else home['team']
                edges.append({
                    'team': leader,
                    'strength': abs(signed) / scale,
                    'text': phrase.format(d=abs(a - h)),
                })
        rows.append(row)
    edges.sort(key=lambda e: e['strength'], reverse=True)
    return rows, [e for e in edges if e['strength'] >= 0.25][:4]


def build_matchup(game_row, completed_games, date_col, home_win_prob):
    """Assemble everything the matchup page needs for one game."""
    game_date = pd.to_datetime(game_row[date_col])
    season = int(game_row['season'])
    home, away = game_row['home_team'], game_row['away_team']

    before = completed_games[completed_games[date_col] < game_date]
    season_games = before[before['season'] == season]
    prior_season_games = before[before['season'] == season - 1]

    away_p = _team_profile(away, season_games, prior_season_games, before, date_col, game_date, season, game_row, 'away')
    home_p = _team_profile(home, season_games, prior_season_games, before, date_col, game_date, season, game_row, 'home')
    rows, key_edges = _compare(away_p, home_p)

    # Head-to-head: most recent meetings in the loaded history
    h2h_games = before[
        ((before['home_team'] == home) & (before['away_team'] == away)) |
        ((before['home_team'] == away) & (before['away_team'] == home))
    ].tail(5)
    h2h = []
    for _, g in h2h_games.iloc[::-1].iterrows():
        winner = g['home_team'] if g['home_score'] > g['away_score'] else g['away_team'] if g['away_score'] > g['home_score'] else None
        h2h.append({
            'season': int(g['season']),
            'week': _num(g.get('week')),
            'matchup': f"{g['away_team']} @ {g['home_team']}",
            'score': f"{int(g['away_score'])}-{int(g['home_score'])}",
            'winner': winner,
        })

    # Vegas
    away_imp = _implied_prob(game_row.get('away_moneyline'))
    home_imp = _implied_prob(game_row.get('home_moneyline'))
    vegas_home_prob = home_imp / (home_imp + away_imp) if away_imp and home_imp else None
    spread = _num(game_row.get('spread_line'))
    if spread is None:
        spread_text = None
    elif spread == 0:
        spread_text = "Pick'em"
    else:
        spread_text = f"{home if spread > 0 else away} -{abs(spread):g}"

    home_score, away_score = _num(game_row.get('home_score')), _num(game_row.get('away_score'))
    completed = home_score is not None and away_score is not None
    model_pick = home if home_win_prob >= 0.5 else away
    final = None
    if completed:
        actual = home if home_score > away_score else away if away_score > home_score else None
        final = {
            'home_score': int(home_score),
            'away_score': int(away_score),
            'winner': actual,
            'correct': actual == model_pick if actual else None,
        }

    return {
        'game_id': game_row.get('game_id'),
        'season': season,
        'week': _num(game_row.get('week')),
        'game_type': _num(game_row.get('game_type')),
        'home': home_p,
        'away': away_p,
        'home_win_prob': float(home_win_prob),
        'model_pick': model_pick,
        'rows': rows,
        'key_edges': key_edges,
        'edge_count': {
            'away': sum(r['edge'] == 'away' for r in rows),
            'home': sum(r['edge'] == 'home' for r in rows),
        },
        'h2h': h2h,
        'vegas_home_prob': vegas_home_prob,
        'spread_text': spread_text,
        'total_line': _num(game_row.get('total_line')),
        'div_game': bool(_num(game_row.get('div_game'))),
        'stadium': _num(game_row.get('stadium')),
        'roof': _num(game_row.get('roof')),
        'surface': (_num(game_row.get('surface')) or '').replace('_', ' ').title() or None,
        'temp': _num(game_row.get('temp')),
        'wind': _num(game_row.get('wind')),
        'final': final,
    }
