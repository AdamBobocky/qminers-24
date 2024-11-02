Implementation of Elo on team level.

```
me_market_corr 0.9738798138290375
elo_mse: 0.20311014538397673
odds_mse: 0.19998107525362585

{
    'pnl_0%': -2152.6590635785833,
    'bets_0%': 33275,
    'vig_0%': 1768.2773225556116,
    'odds_0%': 60237.57243434554,
    'pnl_10%': -581.9797800957278,
    'bets_10%': 4732,
    'vig_10%': 274.02726764321676,
    'odds_10%': 12428.61796129406,
    'pnl_20%': -246.88778113540062,
    'bets_20%': 1181,
    'vig_20%': 62.21427516585448,
    'odds_20%': 4554.754579536465
}
```

With ~97.4% correlation to odds it seems that odds are generated with team-level
Elo model and home field advantage. So finding edges should likely come from
player level and trends.
