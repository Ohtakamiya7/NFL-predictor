# NFL Game Predictor 🏈

A simple web application that uses machine learning (Logistic Regression) to predict the outcomes of upcoming NFL games.

## Features

- **AI-Powered Predictions**: Uses logistic regression trained on historical game data
- **Rolling Window Statistics**: Uses recent team performance (last 16 games) for more accurate predictions
- **Simple Web Interface**: Clean, modern UI showing predictions for upcoming games
- **Automatic Updates**: Fetches current season data and upcoming games automatically

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Required packages:**
   - pandas
   - numpy
   - scikit-learn
   - nfl-data-py
   - flask

## Usage

### Starting the Web Application

Simply run:
```bash
python app.py
```

Then open your browser and visit:
```
http://localhost:5000
```

### How It Works

1. **First Run**: The app will automatically train a model on historical NFL games (2020-present)
2. **Model Training**: Uses rolling window statistics to calculate team features
3. **Predictions**: Loads upcoming games and makes predictions using the trained model
4. **Display**: Shows predictions with win probabilities and confidence levels

### Retraining the Model

To retrain the model with the latest data, click the "🔄 Retrain Model" button on the web page, or visit:
```
http://localhost:5000/retrain
```

**Note**: Retraining can take a few minutes as it processes thousands of historical games.

## File Structure

```
AI Football Predictor/
├── app.py                 # Flask web application
├── nfl_predictor.py      # Core prediction functions
├── predictor.py          # Original training script (legacy)
├── templates/
│   └── index.html        # Web interface
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── DATA_SOURCES.md      # Information about data sources
```

## Prediction Features

The model uses the following features to make predictions:

- **Team Offense**: Average points scored (last 16 games)
- **Team Defense**: Average points allowed (last 16 games)
- **Win Rate**: Recent win percentage
- **Point Differential**: Average scoring margin
- **Home/Away Differences**: Feature differences between home and away teams

## Notes

- Predictions are based on historical performance and may not account for:
  - Injuries
  - Weather conditions
  - Recent trades or roster changes
  - Motivation factors (playoff implications, etc.)

- The model accuracy is typically around 60-65% (slightly better than coin flip!)

- Data is sourced from `nfl-data-py` which updates nightly during the season

## Troubleshooting

**No upcoming games shown?**
- The season may have ended
- Games may not be scheduled yet
- Try retraining the model

**Error loading data?**
- Check your internet connection
- Ensure `nfl-data-py` is installed and updated
- Try: `pip install --upgrade nfl-data-py`

**Model not training?**
- Ensure you have enough historical data (at least a few seasons)
- Check that you have sufficient disk space
- Review error messages in the console

## License

This is a personal project for educational purposes.

