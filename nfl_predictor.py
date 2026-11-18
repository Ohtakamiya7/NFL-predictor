"""
NFL Game Predictor - Core prediction functions

This module contains functions for loading data, training models, and making predictions.
"""

import pandas as pd 
import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import warnings
import pickle
import os
warnings.filterwarnings('ignore')

# nflverse's schedule + scores file (regular season and playoffs, 1999-present,
# updated during the season). This is the same file nfl_data_py.import_schedules
# downloaded; reading it directly avoids that package, which pins pandas < 2.
SCHEDULE_URL = 'https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv'


def fetch_schedule():
    """Download the full NFL schedule (all seasons) as a DataFrame."""
    return pd.read_csv(SCHEDULE_URL, low_memory=False)

# How quickly the current season takes over from last season. The current
# season's weight is n / (n + PRIOR_SEASON_GAMES), where n is games played this
# season: 1 game -> 20%, 4 games -> 50%, 10 games -> ~71%, 17 games -> ~81%.
PRIOR_SEASON_GAMES = 4

LEAGUE_AVERAGE = {'avg_points_scored': 21.0, 'avg_points_allowed': 21.0, 'win_rate': 0.5}


def infer_season(game_date):
    """NFL season a date belongs to (Jan/Feb playoff games belong to the prior year)."""
    game_date = pd.to_datetime(game_date)
    return game_date.year - 1 if game_date.month <= 2 else game_date.year


def _summarize_games(team, team_games):
    """Average points scored/allowed and win rate over team_games, or None if empty."""
    points_scored = []
    points_allowed = []
    wins = 0
    for _, game in team_games.iterrows():
        if game['home_team'] == team:
            points_scored.append(game['home_score'])
            points_allowed.append(game['away_score'])
            wins += game['home_win']
        else:
            points_scored.append(game['away_score'])
            points_allowed.append(game['home_score'])
            wins += (1 - game['home_win'])
    if not points_scored:
        return None
    return {
        'avg_points_scored': float(np.mean(points_scored)),
        'avg_points_allowed': float(np.mean(points_allowed)),
        'win_rate': wins / len(points_scored),
    }


def get_team_stats_from_history(team, games_df, date_col, current_date, window_size=16, season=None):
    """Calculate team statistics from games played before current_date.

    Stats are season-aware: this season's games are blended with the team's
    previous season, and this season's weight grows with every game played
    (see PRIOR_SEASON_GAMES), so last year's results fade out as the season goes on.
    window_size is only used when the data has no 'season' column.
    """
    team_mask = (games_df['home_team'] == team) | (games_df['away_team'] == team)
    if date_col and date_col in games_df.columns:
        team_games = games_df[team_mask & (games_df[date_col] < current_date)]
    else:
        team_games = games_df[team_mask]

    if 'season' in team_games.columns:
        if season is None or pd.isna(season):
            season = infer_season(current_date)
        this_season = team_games[team_games['season'] == season]
        earlier = team_games[team_games['season'] < season]
        prior_season = earlier[earlier['season'] == earlier['season'].max()]
    else:
        this_season = team_games.tail(window_size)
        prior_season = team_games.iloc[:0]

    previous = _summarize_games(team, prior_season) or LEAGUE_AVERAGE
    current = _summarize_games(team, this_season) or previous

    n = len(this_season)
    weight = n / (n + PRIOR_SEASON_GAMES)
    stats = {key: weight * current[key] + (1 - weight) * previous[key] for key in LEAGUE_AVERAGE}
    stats['point_differential'] = stats['avg_points_scored'] - stats['avg_points_allowed']
    return stats

def calculate_features_for_game(game, completed_games_df, date_col, window_size=16):
    """Calculate features for a single game based on historical data."""
    home_team = game.get('home_team')
    away_team = game.get('away_team')
    
    # Get game date if available
    if date_col and date_col in game:
        game_date = pd.to_datetime(game.get(date_col, datetime.now()))
    else:
        game_date = datetime.now()
    
    # Get stats for both teams
    season = game.get('season')
    home_stats = get_team_stats_from_history(home_team, completed_games_df, date_col, game_date, window_size, season)
    away_stats = get_team_stats_from_history(away_team, completed_games_df, date_col, game_date, window_size, season)
    
    # Create feature dictionary
    features = {
        'home_avg_points_scored': home_stats['avg_points_scored'],
        'away_avg_points_scored': away_stats['avg_points_scored'],
        'home_avg_points_allowed': home_stats['avg_points_allowed'],
        'away_avg_points_allowed': away_stats['avg_points_allowed'],
        'home_win_rate': home_stats['win_rate'],
        'away_win_rate': away_stats['win_rate'],
        'home_point_differential': home_stats['point_differential'],
        'away_point_differential': away_stats['point_differential'],
    }
    
    # Add difference features
    features['points_scored_diff'] = features['home_avg_points_scored'] - features['away_avg_points_scored']
    features['points_allowed_diff'] = features['home_avg_points_allowed'] - features['away_avg_points_allowed']
    features['win_rate_diff'] = features['home_win_rate'] - features['away_win_rate']
    features['point_differential_diff'] = features['home_point_differential'] - features['away_point_differential']
    
    return features

def load_historical_games(seasons=None):
    """Load completed NFL games (regular season and playoffs) from nflverse."""
    if seasons is None:
        current_year = datetime.now().year
        seasons = list(range(2020, current_year + 1))

    schedule = fetch_schedule()
    all_games = []
    for season in seasons:
        season_games = schedule[schedule['season'] == season]
        if len(season_games) > 0:
            all_games.append(season_games)
        else:
            print(f"Warning: no schedule data for season {season}")

    if not all_games:
        raise ValueError("Could not load any game data")
    
    games = pd.concat(all_games, ignore_index=True)
    
    # Filter to completed games only (includes regular season and playoff games)
    games = games[games['home_score'].notna() & games['away_score'].notna()].copy()
    
    # Sort by date
    date_col = None
    if 'gameday' in games.columns:
        date_col = 'gameday'
        games[date_col] = pd.to_datetime(games[date_col], errors='coerce')
    elif 'game_date' in games.columns:
        date_col = 'game_date'
        games[date_col] = pd.to_datetime(games[date_col], errors='coerce')
    
    if date_col:
        games = games.sort_values(date_col).reset_index(drop=True)
    
    games['home_win'] = (games['home_score'] > games['away_score']).astype(int)
    
    return games, date_col

def load_upcoming_games(season=None):
    """Load NFL games for predictions - includes current week games even if they've started.
    Includes both regular season and postseason/playoff games.
    Note: Playoff games for a season occur in January/February of the next year."""
    if season is None:
        current_date = datetime.now()
        current_month = current_date.month
        current_year = current_date.year
        
        # If we're in January or February, we might be in postseason
        # Playoff games belong to the previous year's season (e.g., Jan 2025 = 2024 season playoffs)
        if current_month <= 2:
            # Check both current year and previous year for playoff games
            seasons_to_load = [current_year - 1, current_year]
        else:
            # Regular season, use current year
            seasons_to_load = [current_year]
    else:
        seasons_to_load = [season]
    
    try:
        schedule = fetch_schedule()
        all_games_list = []
        for season_to_load in seasons_to_load:
            season_games = schedule[schedule['season'] == season_to_load]
            if len(season_games) > 0:
                all_games_list.append(season_games)
        
        if not all_games_list:
            return pd.DataFrame(), None
        
        all_games = pd.concat(all_games_list, ignore_index=True)
        
        # Get date column
        date_col = None
        if 'gameday' in all_games.columns:
            date_col = 'gameday'
            all_games[date_col] = pd.to_datetime(all_games[date_col], errors='coerce')
        elif 'game_date' in all_games.columns:
            date_col = 'game_date'
            all_games[date_col] = pd.to_datetime(all_games[date_col], errors='coerce')
        
        if date_col:
            all_games = all_games.sort_values(date_col).reset_index(drop=True)
        
        # Return all games (we'll filter by date in get_next_week_games)
        # This includes both regular season and postseason games
        return all_games, date_col
    except Exception as e:
        print(f"Error loading games: {e}")
        return pd.DataFrame(), None

def get_next_week_games(upcoming_games_df, date_col):
    """Get games for the current week. 
    Shows games even after they start, until Tuesday night when we update for next week.
    On Tuesday night, filters out completed games from previous week."""
    if upcoming_games_df.empty:
        return pd.DataFrame()
    
    if date_col and date_col in upcoming_games_df.columns:
        now = datetime.now()
        today = now.date()
        current_weekday = now.weekday()  # 0=Monday, 1=Tuesday, ..., 6=Sunday
        
        # Convert date column to date
        upcoming_games_df = upcoming_games_df.copy()
        upcoming_games_df['game_date_only'] = pd.to_datetime(upcoming_games_df[date_col], errors='coerce').dt.date
        
        # NFL week typically runs Tuesday (after MNF) to Monday (MNF)
        # Determine the start of current week based on day
        # On Tuesday 8pm+, we transition to the new week
        # Otherwise, we're showing the current/ongoing week
        
        if current_weekday == 1 and now.hour >= 20:  # Tuesday 8pm or later
            # New week starts - show games from this Tuesday through next Monday
            week_start = today  # Today (Tuesday)
            week_end = today + timedelta(days=6)  # Next Monday
        elif current_weekday == 1:  # Tuesday before 8pm
            # Still showing previous week's games (last Tuesday through last Monday)
            week_start = today - timedelta(days=7)  # Last Tuesday
            week_end = today - timedelta(days=1)  # Last Monday
        elif current_weekday == 0:  # Monday
            # Show current week (last Tuesday to this Monday)
            week_start = today - timedelta(days=6)  # Last Tuesday
            week_end = today  # Today (Monday)
        else:  # Wednesday-Sunday
            # Show current week (last Tuesday to next Monday)
            days_since_tuesday = (current_weekday - 1) if current_weekday > 1 else 0
            week_start = today - timedelta(days=days_since_tuesday)
            week_end = today + timedelta(days=(8 - current_weekday))  # Next Monday
        
        # Filter games in the current week (including games that have started)
        current_week_games = upcoming_games_df[
            (upcoming_games_df['game_date_only'] >= week_start) & 
            (upcoming_games_df['game_date_only'] <= week_end)
        ].copy()
        
        # On Tuesday night (8pm+), we've already filtered to the new week
        # so no need to filter out old games - they're already excluded
        
        if not current_week_games.empty:
            return current_week_games
    
    # Fallback: if no date column or no games in current week, return upcoming games
    # Include games from recent past to near future
    if date_col and date_col in upcoming_games_df.columns:
        today = datetime.now().date()
        recent_games = upcoming_games_df[
            (upcoming_games_df['game_date_only'] >= today - timedelta(days=7)) &
            (upcoming_games_df['game_date_only'] <= today + timedelta(days=7))
        ].copy()
        if not recent_games.empty:
            return recent_games
    
    # Last resort: return first 16 games
    return upcoming_games_df.head(16).copy()

def train_model(completed_games_df, date_col, window_size=16):
    """Train a logistic regression model on historical games."""
    print("Calculating features for historical games...")
    
    game_features_list = []
    games_sorted = completed_games_df.sort_values(date_col if date_col else 'game_id').reset_index(drop=True)
    
    for idx, game in games_sorted.iterrows():
        game_date = pd.to_datetime(game.get(date_col, datetime.now()))
        features = calculate_features_for_game(game, games_sorted, date_col, window_size)
        
        game_features_list.append({
            'game_id': game.get('game_id', idx),
            'season': game.get('season'),
            'home_team': game['home_team'],
            'away_team': game['away_team'],
            'home_win': game['home_win'],
            **features
        })
        
        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{len(games_sorted)} games...")
    
    game_features = pd.DataFrame(game_features_list)
    
    # Define feature columns
    feature_columns = [
        'home_avg_points_scored', 'away_avg_points_scored',
        'home_avg_points_allowed', 'away_avg_points_allowed',
        'home_win_rate', 'away_win_rate',
        'home_point_differential', 'away_point_differential',
        'points_scored_diff', 'points_allowed_diff',
        'win_rate_diff', 'point_differential_diff'
    ]
    
    X = game_features[feature_columns].fillna(0)
    y = game_features['home_win']
    
    # Filter invalid rows
    valid_mask = ~(X.isna().any(axis=1) | np.isinf(X).any(axis=1))
    X = X[valid_mask]
    y = y[valid_mask]
    
    print(f"\nTraining model on {len(X)} games...")
    
    # Train model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)
    
    # Evaluate
    train_accuracy = model.score(X, y)
    print(f"Model accuracy: {train_accuracy:.3f}")

    return model, feature_columns, float(train_accuracy), len(X)

def predict_upcoming_games(model, feature_columns, upcoming_games_df, completed_games_df, date_col, window_size=16):
    """Make predictions for upcoming games."""
    predictions = []
    
    for _, game in upcoming_games_df.iterrows():
        try:
            features_dict = calculate_features_for_game(game, completed_games_df, date_col, window_size)
            
            # Convert to DataFrame for prediction
            features_df = pd.DataFrame([features_dict])
            features_df = features_df[feature_columns].fillna(0)
            
            # Make prediction
            home_win_prob = model.predict_proba(features_df)[0][1]
            predicted_winner = game['home_team'] if home_win_prob > 0.5 else game['away_team']
            
            predictions.append({
                'game_id': game.get('game_id', ''),
                'home_team': game['home_team'],
                'away_team': game['away_team'],
                'game_date': game.get(date_col, ''),
                'gametime': game.get('gametime', ''),
                'week': game.get('week', None),
                'predicted_winner': predicted_winner,
                'home_win_probability': home_win_prob,
                'away_win_probability': 1 - home_win_prob,
                'confidence': abs(home_win_prob - 0.5) * 2
            })
        except Exception as e:
            print(f"Error predicting game {game.get('game_id', 'unknown')}: {e}")
    
    return pd.DataFrame(predictions)

def save_model(model, feature_columns, filepath='nfl_model.pkl', accuracy=None, games_count=None):
    """Save trained model and feature columns."""
    model_data = {
        'model': model,
        'feature_columns': feature_columns,
        'accuracy': accuracy,
        'games_count': games_count,
    }
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"Model saved to {filepath}")

def load_model(filepath='nfl_model.pkl'):
    """Load trained model and feature columns."""
    with open(filepath, 'rb') as f:
        model_data = pickle.load(f)
    return (
        model_data['model'],
        model_data['feature_columns'],
        model_data.get('accuracy'),
        model_data.get('games_count'),
    )

