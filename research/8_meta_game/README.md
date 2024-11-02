# Meta game

## Attempt at finding edges through how the games are posted, understanding what intentional holes are left in on purpose

### odds_changes.py

The observation is that odds are *ALWAYS* posted 1 day in advance, and the odds don't change.
So there is no opportunity to get better odds and it is beneficial to place bets at last moment
to optimize capital allocation.

### odds_accuracy.py

Purpose of this is to take odds skill estimates as the ground truth and see how they evolve,
whether we can bet from deviations from past skill estimates, etc...

```
mse_my 0.2165402057431469
mse_market 0.19757044947928307
r2_me_market 0.41709342198096666
avg_odds 3.9019497793605686
avg_pnl -0.09162748481099192 avg_overround 0.052344765100685156 bets 13816
```

The code works like this:
It iterates all games sequentially and at every one logs what the odds opinion of team strength is.
It then uses average opinion of team strength from past odds for the team for the season to make a prediction.
And the result is very low r2 correlation (0.41), negative pnl. Which means the odds reflect a lot of information
that is not just team strength, and team strength does not have much power here. Which is contrary to how
it usually works in sports. Pretty weird.
