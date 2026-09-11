# Dataset generation

**Original execution environment:** Python in PyCharm.

The script measures TLS connection/handshake elapsed time and associates each observation with meteorological information retrieved from OpenWeatherMap.

## Run

1. Install dependencies from the repository `requirements.txt` (or at minimum `requests`).
2. Set `OPENWEATHERMAP_API_KEY` and `OPENWEATHERMAP_CITY`.
3. Run:

```bash
python clanak_dataset_generation.py
```

4. Enter a target such as `example.com:443` when prompted.
5. The program writes observations to `mycsv.csv`.

The source program originally used placeholders for the API key and city. The publication-ready version keeps credentials outside the source code and handles an absent `rain.1h` field safely.
