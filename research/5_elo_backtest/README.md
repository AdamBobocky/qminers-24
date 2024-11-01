Implementation of Elo on team level.

```
elo_mse: 0.21778113727211543
odds_mse: 0.19998107525362585

{
    'pnl_0%': -4492.269477784996,
    'bets_0%': 55087,
    'vig_0%': 2986.702532235045,
    'pnl_10%': -4229.4687274209355,
    'bets_10%': 47875,
    'vig_10%': 2598.242502574007,
    'pnl_20%': -3716.8593184159195,
    'bets_20%': 35010,
    'vig_20%': 1846.1450898358962
}
```

Interestingly, PnL is more negative than just the vig, implying that Elo predictions have negative expected value even after devigging.
