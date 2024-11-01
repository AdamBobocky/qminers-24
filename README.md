# Qminers Quant Hackathon 2024

Vítej na Qminers Quant Hackathonu!

V tomto repozitáři najdeš vše potřebné k odladění tvého modelu, aby Tě odevzdávací systém ničím nepřekvapil.

Odevzdávací systém beží na webu [hyperion.felk.cvut.cz](http://hyperion.felk.cvut.cz/).

## Struktura repozitáře
* `problem_info.md` obsahuje detailní [zadání](problem_info.md) úlohy včetně popisu dat, které doporučujeme důkladně pročíst než se pustíš do tvorby vlastního modelu.
* `qqh-2024-env.yml` obsahuje stejné prostředí jaké používá odevzdávací systém.
* `data/games.csv` a `data/players.csv` obsahují trénovací data.
* `src/environment.py` obsahuje evaluační smyčku, která běží v odevzdávacím systému.
* `src/evaluate.py` skript pro spuštění lokáního vyhodnocení tvého modelu na trénovacích datech.
* `src/model.py` obsahuje ukázkový model s metodou `place_bets`.

## Runtime prostředí

Soubor `qqh-2024-env.yml` obsahuje balíčky, které budou dostupné při evaluaci. Vřele doporučujeme toto prostředí replikovat lokálně.

1. Nainstaluj si nástroj miniconda: https://docs.anaconda.com/miniconda/
2. Importuj environment: `conda env create -f qqh-2024-env.yml`
3. Aktivuj enviroment: `conda activate qqh-2024` 

Vývojová prostředí (IDE) často umí s conda environmenty pracovat (například PyCharm či VS Code).

Pro ověření funčnosti můžeš rovnou spustit evaluaci modelu, který sází náhodně: `python src/evaluate.py`

## Vlastní řešení

Třída, kterou budeš odevzdávat do odevzdávacího systému, se musí jmenovat `Model` a musí obsahovat implementaci metody `place_bets(self, summary: pd.DataFrame, opps: pd.DataFrame, inc: tuple[pd.DataFrame, pd.DataFrame])`. Ukázkový model náhodného sázkaře najdeš [zde](src/model.py).

Bližší info najdeš v [zadání](problem_info.md).

**Je zákázáno:**

- Používat k ladění modelů jakákoliv jiná než námi dodaná data.
- Spolupracovat s jinými týmy.
- Manipulovat s odevzdávacím systémem.
- Jakkoliv dolovat data z odevzdávacího systému.
- Spouštět vlastní procesy, komunikovat přes síť a vytvářet soubory v odevzdávacím systému mimo složku `/tmp`.

Porušení pravidel bude vést k vyloučení ze soutěže. Všechna odeslaná řešení se ukládají a mohou být zpětně přezkoumána.

## Evaluace

Evaluace v odevzdávacím systému probíhá na skrytých (validačních) datech. Trénovací data obsahují [zápasy](data/games.csv) a statistiky [jednotlivých hráčů](data/players.csv) ze sezón 1975/76-1998/99.
Validační data obsahují zápasy ze sezón 1999/00-2004/05.
V první iteraci [evaluační smyčky](src/environment.py#L68) obdržíš jako [inkrement](problem_info.md#dataframe-inkrement%C3%A1ln%C3%ADch-dat) všechna trénovací data.

V odevzdávacím systému budeš mít k dispozici pouze 1 cpu (1 vlákno) a 5 GB RAM. Současně může běžet pouze jedna evaluce tvého řešení.

Počet odevzdání je omezen, doporučujeme tedy kód pečlivě odladit na lokálním stroji.

V případě neúspěšného doběhnutí tvého programu se odevzdávací sytém pokouší napovědět příčinu pádu.

- `RTE` = Runtime Error. Nejčastěji kód spadnul na vyjímku, jejíž jméno ti odevzdávací systém (po rozkliknutí testcasu) ukáže. 
- `MLE` = Memory Limit Exceeded. Překročil jsi  pamětový limit. (Může být rozpoznáno jako RTE)
- `TLE` = Time Limit Exceeded. Překročil jsi časový limit.

