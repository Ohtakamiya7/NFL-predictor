# Quick Start Guide

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Run the Application

```bash
python app.py
```

## Step 3: Open in Browser

Visit: **http://localhost:5000**

That's it! The application will:
- Automatically train a model on first run (may take a few minutes)
- Load upcoming games
- Display predictions with win probabilities

## What You'll See

- **Game Cards**: Each upcoming game with teams and predictions
- **Win Probabilities**: Percentage chance each team has to win
- **Predicted Winner**: Highlighted in purple
- **Confidence Level**: How confident the model is in the prediction

## Tips

- **First Run**: Be patient - model training takes 2-5 minutes
- **No Games**: If no games appear, the season may have ended
- **Retrain**: Click "Retrain Model" to update with latest data
- **Refresh**: Refresh the page to get updated predictions

## Troubleshooting

**Port already in use?**
```bash
# Change port in app.py (last line) to something else like 5001
app.run(debug=True, host='0.0.0.0', port=5001)
```

**Module not found errors?**
```bash
pip install --upgrade -r requirements.txt
```

**No games showing?**
- Check if NFL season is in progress
- Try retraining the model
- Check console for error messages

