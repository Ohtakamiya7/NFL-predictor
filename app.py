"""
NFL Game Predictor - Flask Web Application
"""

from flask import Flask, render_template, jsonify
from datetime import datetime
from collections import OrderedDict
import os
import time
import pandas as pd
from nfl_predictor import (
    load_historical_games,
    load_upcoming_games,
    get_next_week_games,
    train_model,
    predict_upcoming_games,
    load_model,
    save_model,
)

app = Flask(__name__)
MODEL_FILE = 'nfl_model.pkl'

# ---------------------------------------------------------------------------
# Simple in-process cache to avoid re-loading large dataframes on every request
# ---------------------------------------------------------------------------
_cache = {}

def _cached(key, loader, ttl=1800):
    entry = _cache.get(key)
    if entry and (time.time() - entry['ts']) < ttl:
        return entry['data']
    data = loader()
    _cache[key] = {'data': data, 'ts': time.time()}
    return data

def _invalidate_cache():
    _cache.clear()

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_now():
    now = datetime.now()
    return f"{now.strftime('%b')} {now.day} · {now.hour % 12 or 12}:{now.strftime('%M')} {now.strftime('%p')}"


def _format_kickoff(game_date_dt, gametime_str):
    """Return e.g. 'Thu Sep 10 · 8:20 PM' from a date object + gametime string like '20:20'."""
    if game_date_dt is None:
        return "TBD"
    dt = game_date_dt
    if gametime_str and ':' in str(gametime_str):
        try:
            h, m = map(int, str(gametime_str).split(':')[:2])
            dt = dt.replace(hour=h, minute=m)
        except (ValueError, AttributeError):
            pass
    if dt.hour == 0 and dt.minute == 0:
        return f"{dt.strftime('%a %b')} {dt.day} · TBD"
    hour_12 = dt.hour % 12 or 12
    ampm = 'PM' if dt.hour >= 12 else 'AM'
    return f"{dt.strftime('%a %b')} {dt.day} · {hour_12}:{dt.strftime('%M')} {ampm}"


def _slate_label(dt):
    if dt is None:
        return "Upcoming Games"
    day = dt.strftime('%A')
    h = dt.hour
    if day == 'Thursday':
        return "Thursday Night Football"
    if day == 'Friday':
        return "Friday Night Football"
    if day == 'Saturday':
        return "Saturday Games"
    if day == 'Sunday':
        if h == 0:
            return "Sunday Games"
        if h < 14:
            return "Sunday Early"
        if h < 17:
            return "Sunday Late"
        return "Sunday Night Football"
    if day == 'Monday':
        return "Monday Night Football"
    return f"{day} Games"


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_all_data():
    """Load (cached) model, historical games, and full season schedule."""
    if not os.path.exists(MODEL_FILE):
        raise RuntimeError("Model not trained yet. Visit /retrain first.")
    model, feature_cols, accuracy, games_count = load_model(MODEL_FILE)
    completed_games, date_col = _cached('completed_games', load_historical_games)
    upcoming_games, upcoming_date_col = _cached('upcoming_games', load_upcoming_games)
    return model, feature_cols, accuracy, games_count, completed_games, date_col, upcoming_games, upcoming_date_col


def _reg_season_games(upcoming_games):
    """Return only regular-season rows (weeks 1-18)."""
    if upcoming_games is None or upcoming_games.empty:
        return pd.DataFrame()
    if 'game_type' in upcoming_games.columns:
        return upcoming_games[upcoming_games['game_type'] == 'REG'].copy()
    # Fallback: filter to week <= 18
    if 'week' in upcoming_games.columns:
        return upcoming_games[pd.to_numeric(upcoming_games['week'], errors='coerce') <= 18].copy()
    return upcoming_games.copy()


def _predictions_for_games(model, feature_cols, games_df, completed_games, date_col, upcoming_date_col):
    """Run model on games_df, merge actual scores where available."""
    use_date_col = date_col or upcoming_date_col
    preds_df = predict_upcoming_games(model, feature_cols, games_df, completed_games, use_date_col)
    if preds_df.empty:
        return []

    formatted = []
    for _, pred in preds_df.iterrows():
        match = games_df[
            (games_df['home_team'] == pred['home_team']) &
            (games_df['away_team'] == pred['away_team'])
        ]
        completed, home_score, away_score = False, None, None
        if not match.empty:
            row = match.iloc[0]
            hs, as_ = row.get('home_score'), row.get('away_score')
            if pd.notna(hs) and pd.notna(as_):
                home_score, away_score, completed = int(hs), int(as_), True

        game_date_dt = None
        raw = pred.get('game_date')
        if pd.notna(raw):
            game_date_dt = raw.to_pydatetime() if hasattr(raw, 'to_pydatetime') else raw

        formatted.append({
            'away_team': str(pred['away_team']),
            'home_team': str(pred['home_team']),
            'home_win_prob': float(pred['home_win_probability']),
            'game_date_dt': game_date_dt,
            'gametime': str(pred.get('gametime') or ''),
            'week': pred.get('week'),
            'completed': completed,
            'home_score': home_score,
            'away_score': away_score,
        })
    return formatted


def _build_slates(predictions):
    """Group predictions into time-slot slate sections."""
    buckets = OrderedDict()
    for pred in predictions:
        dt = pred.get('game_date_dt')
        gametime = pred.get('gametime', '')
        label_dt = dt
        if dt and gametime and ':' in str(gametime):
            try:
                h, m = map(int, str(gametime).split(':')[:2])
                label_dt = dt.replace(hour=h, minute=m)
            except (ValueError, AttributeError):
                pass
        label = _slate_label(label_dt)
        kick = _format_kickoff(dt, gametime)
        buckets.setdefault(label, []).append({
            'away': pred['away_team'],
            'home': pred['home_team'],
            'homeProb': round(pred['home_win_prob'] * 100, 1),
            'kick': kick,
            'completed': pred.get('completed', False),
            'homeScore': pred.get('home_score'),
            'awayScore': pred.get('away_score'),
        })
    return [{'label': k, 'games': v} for k, v in buckets.items()]


def _week_label(predictions):
    for pred in predictions:
        w = pred.get('week')
        if w is not None:
            try:
                return f"Week {int(w)}"
            except (TypeError, ValueError):
                pass
    return "Current Week"


def _build_week_meta(reg_games):
    """Return list of {week, status} for weeks 1-18.

    Status values:
      past     — all games played (scores exist)
      current  — some games played
      upcoming — games scheduled, none played yet
      empty    — no games scheduled this week
    """
    if reg_games is None or reg_games.empty or 'week' not in reg_games.columns:
        return []

    meta = []
    for w in range(1, 19):
        wg = reg_games[pd.to_numeric(reg_games['week'], errors='coerce') == w]
        if wg.empty:
            meta.append({'week': w, 'status': 'empty'})
            continue
        n_completed = int(wg['home_score'].notna().sum()) if 'home_score' in wg.columns else 0
        total = len(wg)
        if n_completed == total:
            status = 'past'
        elif n_completed > 0:
            status = 'current'
        else:
            status = 'upcoming'
        meta.append({'week': w, 'status': status})
    return meta


def _current_week_num(week_meta):
    for m in week_meta:
        if m['status'] == 'current':
            return m['week']
    for m in week_meta:
        if m['status'] == 'upcoming':
            return m['week']
    for m in reversed(week_meta):
        if m['status'] == 'past':
            return m['week']
    return 1


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    try:
        model, feature_cols, accuracy, games_count, completed_games, date_col, upcoming_games, upcoming_date_col = _load_all_data()
        reg_games = _reg_season_games(upcoming_games)
        use_date_col = date_col or upcoming_date_col

        next_week = get_next_week_games(upcoming_games, use_date_col)
        if not next_week.empty:
            preds = _predictions_for_games(model, feature_cols, next_week, completed_games, date_col, upcoming_date_col)
        else:
            preds = []

        slates = _build_slates(preds)
        week_label = _week_label(preds)
        week_meta = _build_week_meta(reg_games)

        # Derive the active week number from predictions, fall back to meta
        active_week = None
        for p in preds:
            w = p.get('week')
            if w is not None:
                try:
                    active_week = int(w)
                    break
                except (TypeError, ValueError):
                    pass
        if active_week is None:
            active_week = _current_week_num(week_meta)

        return render_template('index.html',
            slates=slates,
            model_accuracy=round(accuracy * 100, 1) if accuracy else None,
            games_trained=games_count,
            last_updated=_format_now(),
            week_label=week_label,
            week_meta=week_meta,
            active_week=active_week,
            error=None if preds else "No games found for the current week.",
        )
    except Exception as e:
        return render_template('index.html',
            slates=[], model_accuracy=None, games_trained=None,
            last_updated=_format_now(), week_label="Current Week",
            week_meta=[], active_week=1,
            error=f"Error loading predictions: {str(e)}",
        )


@app.route('/api/week/<int:week_num>')
def week_api(week_num):
    try:
        model, feature_cols, _, _, completed_games, date_col, upcoming_games, upcoming_date_col = _load_all_data()
        reg_games = _reg_season_games(upcoming_games)

        if reg_games.empty or 'week' not in reg_games.columns:
            return jsonify({'slates': [], 'week_label': f'Week {week_num}'})

        week_games = reg_games[pd.to_numeric(reg_games['week'], errors='coerce') == week_num].copy()
        if week_games.empty:
            return jsonify({'slates': [], 'week_label': f'Week {week_num}'})

        preds = _predictions_for_games(model, feature_cols, week_games, completed_games, date_col, upcoming_date_col)
        slates = _build_slates(preds)
        return jsonify({'slates': slates, 'week_label': f'Week {week_num}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/retrain')
def retrain():
    try:
        completed_games, date_col = load_historical_games()
        model, feature_cols, accuracy, games_count = train_model(completed_games, date_col)
        save_model(model, feature_cols, MODEL_FILE, accuracy=accuracy, games_count=games_count)
        _invalidate_cache()
        return render_template('index.html',
            slates=[], model_accuracy=round(accuracy * 100, 1) if accuracy else None,
            games_trained=games_count, last_updated=_format_now(),
            week_label="Current Week", week_meta=[], active_week=1,
            error="Model retrained successfully! Refresh the page to see new predictions.",
        )
    except Exception as e:
        return render_template('index.html',
            slates=[], model_accuracy=None, games_trained=None,
            last_updated=_format_now(), week_label="Current Week",
            week_meta=[], active_week=1,
            error=f"Error retraining model: {str(e)}",
        )


if __name__ == '__main__':
    print("Starting NFL Predictor web app...")
    print("Visit http://localhost:5001 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5001)
