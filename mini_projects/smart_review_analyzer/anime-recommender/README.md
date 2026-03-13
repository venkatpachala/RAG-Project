# Anime Recommender

A machine learning project to build an intelligent anime recommendation system using AniList data.

## Project Structure

```
anime-recommender/
├── data/
│   ├── raw/            -- Put your original "AniList" file here (Never edit this!)
│   └── processed/      -- We will save our cleaned data here later
├── notebooks/
│   ├── 01_problem_framing.ipynb  -- What we are doing today
│   ├── 02_eda_and_cleaning.ipynb -- Visualizing the "shitload of anime"
│   └── 03_model_training.ipynb   -- Building the actual brain
├── src/                -- (Optional) For helper scripts as you get advanced
└── README.md           -- Documentation for your future self
```

## Getting Started

1. **Data Collection**: Place your AniList dataset in `data/raw/`
2. **Problem Framing**: Start with `01_problem_framing.ipynb`
3. **EDA & Cleaning**: Run `02_eda_and_cleaning.ipynb` for data exploration
4. **Model Training**: Use `03_model_training.ipynb` to build the recommender

## Notes

- Never edit files in `data/raw/` - treat it as read-only
- Processed data will be saved to `data/processed/`
- Add helper functions to `src/` as needed
