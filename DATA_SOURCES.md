# NFL Data Sources for Predictor

## Current Implementation

The predictor now uses **nfl_data_py** which:

- Sources data from nflfastR (a comprehensive NFL data repository)
- Updates **nightly** during the season with completed games
- Provides data going back to 1999
- Includes schedules, scores, play-by-play, and team statistics

## Current Season Data Availability

**nfl_data_py DOES have current season data**, but:

- Data updates nightly after games complete
- Early in the season (September), there may be limited data
- The script automatically detects what seasons are available

### What the Script Does Now

1. **Auto-detects available seasons**: Tries to load seasons from 2020 to current year
2. **Checks for completed games**: Only uses games with final scores
3. **Handles missing data gracefully**: Works with whatever data is available
4. **Uses rolling window statistics**: Uses last 16 games (prevents data leakage, better predictions)

## Alternative Data Sources

If nfl_data_py doesn't have the data you need, here are alternatives:

### Option 1: sportsdataverse.nfl

```bash
pip install sportsdataverse
```

```python
from sportsdataverse.nfl import nfl_load_pbp

# Load play-by-play data
pbp_data = nfl_load_pbp(seasons=[2021, 2022, 2023, 2024, 2025])
```

### Option 2: NFL Official API (requires API key)

```python
import requests

# You'll need to sign up for an API key
api_key = "YOUR_API_KEY"
url = f"https://api.sportsdata.io/v3/nfl/scores/json/Games/2025"
headers = {"Ocp-Apim-Subscription-Key": api_key}
response = requests.get(url, headers=headers)
games = response.json()
```

### Option 3: Web Scraping (for current games)

You could scrape from:

- NFL.com official website
- ESPN.com NFL scores
- Pro Football Reference

## Recommendations

**For most use cases**: Stick with **nfl_data_py** - it's well-maintained and updates regularly.

**If you need real-time data during games**: Consider the NFL official API or web scraping.

**If nfl_data_py is missing specific data**: Check the [nflfastR repository](https://github.com/nflverse/nflfastR-data) directly.

## Running the Script

The script will automatically:

1. Detect what seasons have data available
2. Load all available completed games
3. Calculate team statistics using rolling windows
4. Prepare data for logistic regression
5. Train and evaluate the model

You'll see output like:

```
Attempting to load data from seasons: [2020, 2021, 2022, 2023, 2024, 2025]
  ✓ Season 2021: 272 completed games
  ✓ Season 2022: 272 completed games
  ✓ Season 2023: 272 completed games
  ✓ Season 2024: 272 completed games
  ⚠ Season 2025: 45 completed games  (if season is in progress)
```

This tells you exactly what data is available!
