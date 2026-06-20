# AgriVerse

> An LLM crop advisor wired to LSTM price forecasting, so farmers get guidance and a price outlook in one place.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![LSTM](https://img.shields.io/badge/LSTM-time--series-00897B?style=flat-square)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![pandas](https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white)

## What it does

Smallholder farmers rarely get localized, data-driven guidance or any view of where prices are heading. AgriVerse pairs an LLM advisor with LSTM-based price forecasting across **200 stations**, putting crop guidance and a price outlook in the same tool.

It's a working example of wiring an LLM into a real data and forecasting pipeline rather than treating the model as a standalone chatbot.

## Demo

<!-- Add a screenshot or short GIF — the advice view plus a forecast chart works well here. -->
<!-- ![AgriVerse advice and forecast](docs/demo.gif) -->

_Screenshot / demo GIF coming soon._

## How it works

1. **Forecasting** — an LSTM trained on historical price series predicts the near-term outlook per station.
2. **Advisor** — an LLM takes the crop, location, and forecast and produces plain-language guidance.
3. **Together** — the user sees the recommendation and the price trend that informs it, side by side.

## Stack

Python · PyTorch (LSTM) · an LLM backend (Hugging Face / API) · pandas / NumPy (data) · a charting library for the forecast view.

## Results

- Price forecasting across **200 stations** nationwide.

<!-- Add a forecasting-quality number once you have it, e.g. MAE / RMSE / MAPE on a
     held-out window. Don't invent it. -->

_Add forecast accuracy here (e.g. MAE or MAPE on a held-out period) once measured._

## Running it

```bash
git clone https://github.com/oa07610/AgriVerse.git
cd AgriVerse
pip install -r requirements.txt
python train_forecaster.py --data ./data/prices.csv
python app.py
```

## Notes

Demonstration of an LLM + time-series pipeline. MIT licensed.
