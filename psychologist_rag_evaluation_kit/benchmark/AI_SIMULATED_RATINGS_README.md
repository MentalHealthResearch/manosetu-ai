# AI-Simulated Provisional Ratings

`create_ai_simulated_ratings.py` creates five simulated-rater scores for each generated answer using a transparent text rubric. The output is **not** psychologist feedback and must remain separate from `rater_packets/` and the Table 5 human-expert analysis.

After all eight generated-response CSVs are present, run:

```powershell
& 'C:\Users\goura\anaconda3\envs\narrative-rag\python.exe' create_ai_simulated_ratings.py
```

It writes 1,600 rows to `ai_simulated_provisional_ratings.csv`: 320 responses multiplied by five simulated raters.
