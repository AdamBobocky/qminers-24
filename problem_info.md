## Úvod do sportovního sázení

Sportovní sázení je hra dvou hráčů: bookmakera a sázkaře.
Úlohou bookmakera je nabízet sázkařům příležitosti k sázkám s cílem maximalizovat svůj vlastní získ.
Blíží-li se tedy například nějaké hokejové utkání dvou týmů, bookmaker vypíše tzv. kurzy na možné výsledky.
Sázkař si může zvolit, na které zápasy a výsledky by si chtěl vsadit.
V případě, že si sázkař vsadí na výsledek, který nastane, vyhrává sázkař vloženou sázku vynásobenou kurzem bookmakera.
Pokud daný výsledek nenastane, sázkař svou sázku prohrál.

**Příklad:**
Pro první zápas Děčín - Nymburk vypíše bookmaker kurzy 2.08 na výhru domácích (Děčín) a 1.72 na vítězství Nymburku.
Pro druhý zápas Pardubice - Slavia Praha vypíše bookmaker kurzy 1.19 na výhru Pardubic a 4.55 na vítězství Slavie.
Sázkař, který má k dispozici 1000 Kč, se rozhodne vsadit 100 Kč na vítězství domácího Děčína, tedy na kurz 2.08, a 50 Kč na vítězství hostující Slavie, tedy na kurz 4.55. Po vsazení vkladů má na svém kontě 850 Kč. Předpokládejmě, že obě utkání skončí výhrou domácích, tedy Děčína a Pardubic. Z prvního zápasu sázkař vyhrává 2.08 x 100 = 208 Kč. Sázka 50 Kč na druhý zápas propadá, jelikož vyhráli Pardubice a nikoliv Slavie. Ve finále má sázkař na kontě 1058 Kč. 

## Zadání

Vaším úkolem bude naprogramovat co nejúspešnějšího sázkaře. V naší soutěži však nebude rozhodovat počet shlédnutých zápasů, nýbrž vaše schopnost navrhnout, implementovat a otestovat model, který se naučí predikovat výsledky nadcházejících zápasů z historických dat. 

Váš sázkař je reprezentován třídou [`Model`](src/model.py) s metodou `place_bets`, skrze kterou každý den, kdy jsou na trhu nějaké sázkařské příležitosti, obdržíte shrnutí, sázkařské příležitosti a inkrement dat. Od vás budeme očekávat sázky, které si přejete uskutečnit. Jelikož jednotlivé ročníky soutěží (sezóny) trvají několik měsíců, bude se váš model muset v průběhu sezóny adaptovat na nové výsledky a formu hráčů.

## Datové typy

V metodě `place_bets` se v jednotlivých argumentech setkáte celkem se čtyřmi datovými typy - `summary` obsahující [Summary DataFrame](#summary-dataframe), `opps` obsahující [Opps DataFrame](#opps-dataframe) a `inc` složený z dvojice [Games DataFrame](#games-dataframe) a [Players DataFrame](#players-dataframe). Posledním pátým typem je odevzdávaný [Bets DataFrame](#bets-dataframe)

### Summary DataFrame

Summary dataframe obsahuje informace o současném stavu sázkařského prostředí. Dozvíte se v něm aktuální datum a jaké prostředky máte k dispozici.

|    |   Bankroll | Date                |   Min_bet |   Max_bet |
|---:|-----------:|:--------------------|----------:|----------:|
|  0 |    1000    | 1975-11-06 00:00    |    1      |    100    |

<table>
<tr>
<td>

- **Bankroll** - Aktuální stav vašeho konta
- **Date** - Aktuální datum 
</td>
<td>

- **Min_bet** - Minimální možná nenulová sázka
- **Max_bet** - Maximální možná sázka
</td>
</tr>
</table>

### Opps DataFrame

Sázkařské příležitosti obsahují zápasy hrané v nadcházejících dnech. Příležitosti jsou platné do data konání zápasu (včetně). Je tedy možné vsadit na danou příležitost **vícekrát**. Vámi dříve vsazené částky budete mít k dispozici ve sloupcích `BetH` a `BetA`.

| ID | Season|     Date   | HID  | AID  |  N  |POFF |  OddsH   |   OddsA  |  BetH | BetA  |
|---:|------:|-----------:|-----:|-----:|----:|----:|---------:|---------:|------:|------:|
| 15 |   1   | 1975-11-08 |  22  |  41  |  0  |  0  | 1.930235 | 1.916195 |  10.0 |  0.0  |
| 16 |   1   | 1975-11-08 |  12  |  24  |  0  |  0  | 1.418280 | 3.048582 |  0.0  |  7.0  |
| 17 |   1   | 1975-11-08 |   1  |  17  |  0  |  0  | 1.520244 | 2.650582 |  0.0  |  0.0  |
| 18 |   1   | 1975-11-08 |   2  |  42  |  0  |  0  | 1.491004 | 2.748333 |  5.0  |  0.0  |
| 19 |   1   | 1975-11-08 |  19  |  35  |  0  |  0  | 1.302708 | 3.808664 |  5.0  |  0.0  |
| 20 |   1   | 1975-11-09 |  13  |  19  |  0  |  0  | 1.471073 | 2.821692 |  0.0  |  0.0  |
| 21 |   1   | 1975-11-09 |  21  |  11  |  0  |  0  | 1.505497 | 2.698513 |  0.0  |  0.0  |

<table>
<tr>
<td>

- **ID** - Unikátní identifikátor zápasu (index tabulky)
- **Season** - Označení sezóny zápasu
- **Date** - Datum konání zápasu 
- **HID** - Unikátní identifikátor domácího týmu
- **AID** - Unikátní identifikátor hostujícího týmu
- **N** - Přepínač zda se jedná o zápas na neutrální půdě

</td>
<td>

- **POFF** - Přepínač zda se jedná o zápas playoffs
- **OddsH** - Kurz na výhru domácího týmu
- **OddsA** - Kurz na výhru hostujícího týmu
- **BetH** - Vaše dřívější sázky na výhru domácího týmu
- **BetA** - Vaše dřívější sázky na výhru hostujícího týmu

</td>
</tr>
</table>


### Inkrementální data
Inkrementální data obsahují výsledky a statistiky, které jste doposud neviděli. Tato data jsou složena ze dvou tabulek (DataFrames). [Games DataFrame](#games-dataframe) obsahující informace o parametrech odehraných zápasů a [Players DataFrame](#players-dataframe) obsahující přehled o výkonu jednotlivých hráčů v odehraných zápasech. Počítejte s tím, že první inkrement který uvidíte, bude obsahovat i zápasy staršího data (z předcházejících sezón). 

#### Games DataFrame

| ID | Season | Date | HID | AID | N | POFF | OddsH | OddsA | H | A | HSC | ASC | HFGM | AFGM | HFGA | AFGA | HFG3M | AFG3M | HFG3A | AFG3A | HFTM | AFTM | HFTA | AFTA | HORB | AORB | HDRB | ADRB | HRB | ARB | HAST | AAST | HSTL | ASTL | HBLK | ABLK | HTOV | ATOV | HPF | APF |
|--:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
| 15 | 1 | 1975-11-08 | 22 | 41 | 0 | 0 | 1.930235 | 1.916195 | 1 | 0 | 111 | 105 | 46.0 | 41.0 | 93.0 | 85.0 | 3.0 | 0.0 | 7.0 | 4.0 | 16.0 | 23.0 | 23.0 | 28.0 | 11.0 | 12.0 | 30.0 | 33.0 | 41.0 | 45.0 | 27.0 | 23.0 | 12.0 | 4.0 | 2.0 | 7.0 | 11.0 | 19.0 | 24.0 | 20.0 |
| 16 | 1 | 1975-11-08 | 12 | 24 | 0 | 0 | 1.418280 | 3.048582 | 1 | 0 | 119 | 110 | 50.0 | 40.0 | 111.0 | 93.0 | 1.0 | 3.0 | 7.0 | 7.0 | 18.0 | 27.0 | 29.0 | 41.0 | 28.0 | 18.0 | 38.0 | 31.0 | 66.0 | 49.0 | 29.0 | 26.0 | 13.0 | 8.0 | 16.0 | 9.0 | 12.0 | 19.0 | 31.0 | 23.0 |
| 17 | 1 | 1975-11-08 | 1 | 17 | 0 | 0 | 1.520244 | 2.650582 | 1 | 0 | 112 | 102 | 45.0 | 38.0 | 88.0 | 80.0 | 1.0 | 2.0 | 5.0 | 4.0 | 21.0 | 24.0 | 34.0 | 29.0 | 13.0 | 16.0 | 22.0 | 28.0 | 35.0 | 44.0 | 27.0 | 29.0 | 7.0 | 5.0 | 6.0 | 1.0 | 11.0 | 20.0 | 22.0 | 24.0 |
| 18 | 1 | 1975-11-08 | 2 | 42 | 0 | 0 | 1.491004 | 2.748333 | 1 | 0 | 114 | 103 | 49.0 | 38.0 | 102.0 | 98.0 | 2.0 | 1.0 | 4.0 | 5.0 | 14.0 | 26.0 | 19.0 | 33.0 | 19.0 | 14.0 | 41.0 | 27.0 | 60.0 | 41.0 | 31.0 | 19.0 | 7.0 | 9.0 | 5.0 | 3.0 | 19.0 | 10.0 | 28.0 | 20.0 |
| 19 | 1 | 1975-11-08 | 19 | 35 | 0 | 0 | 1.302708 | 3.808664 | 1 | 0 | 131 | 111 | 50.0 | 38.0 | 92.0 | 77.0 | 2.0 | 0.0 | 3.0 | 3.0 | 29.0 | 35.0 | 39.0 | 42.0 | 16.0 | 9.0 | 27.0 | 21.0 | 43.0 | 30.0 | 27.0 | 17.0 | 10.0 | 5.0 | 2.0 | 5.0 | 22.0 | 22.0 | 35.0 | 31.0 |

<table>
<tr>
<td>

- **ID** - Unikátní identifikátor zápasu (index tabulky)
- **Season** - Označení sezóny zápasu
- **Date** - Datum konání zápasu 
- **HID** - Unikátní identifikátor domácího týmu
- **AID** - Unikátní identifikátor hostujícího týmu
- **N** - Přepínač zda se jedná o zápas na neutrální půdě
- **POFF** - Přepínač zda se jedná o zápas playoffs
- **OddsH** - Kurz na výhru domácího týmu
- **OddsA** - Kurz na výhru hostujícího týmu
- **H** - Přepínač zda vyhrál domácí tým
- **A** - Přepínač zda vyhrál hostující tým
- **HSC** - Skóre domácího týmu
- **ASC** - Skóre hostujícího týmu
- **HFGM** - Počet úspěšných pokusů o koš (mimo trestných hodů) domácího týmu
- **AFGM** - Počet úspěšných pokusů o koš (mimo trestných hodů) hostujícího týmu
- **HFGA** - Celkový počet pokusů o koš (mimo trestných hodů) domácího týmu
- **AFGA** - Celkový počet pokusů o koš (mimo trestných hodů) hostujícího týmu
- **HFG3M** - Počet úspěšných pokusů o trojku domácího týmu
- **AFG3M** - Počet úspěšných pokusů o trojku hostujícího týmu
- **HFG3A** - Celkový počet pokusů o trojku domácího týmu
- **AFG3A** - Celkový počet pokusů o trojku hostujícího týmu

</td>
<td>

- **HFTM** - Počet úspěšných trestných hodů domácího týmu
- **AFTM** - Počet úspěšných trestných hodů hostujícího týmu
- **HFTA** - Celkový počet trestných hodů domácího týmu
- **AFTA** - Celkový počet trestných hodů hostujícího týmu 
- **HORB** - Počet útočných doskoků domácího týmu
- **AORB** - Počet útočných doskoků hostujícího týmu
- **HDRB** - Počet obranných doskoků domácího týmu
- **ADRB** - Počet obranných doskoků hostujícího týmu
- **HRB** - Celkový počet doskoků domácího týmu
- **ARB** - Celkový počet doskoků hostujícího týmu
- **HAST** - Počet asistovaných košů domácího týmu
- **AAST** - Počet asistovaných košů hostujícího týmu
- **HSTL** - Počet způsobených útočných ztrát domácího týmu
- **ASTL** - Počet způsobených útočných ztrát hostujícího týmu
- **HBLK** - Počet bloků domácího týmu
- **ABLK** - Počet bloků hostujícího týmu
- **HTOV** - Počet útočných ztrát míče domácího týmu
- **ATOV** - Počet útočných ztrát míče hostujícího týmu
- **HPF** - Počet osobních faulů domácího týmu
- **APF** - Počet osobních faulů hostujícího týmu

</td>
</tr>
</table>

#### Players DataFrame

| | Season | Date | Player | Team | Game | MIN | FGM | FGA | FG3M | FG3A | FTM | FTA | ORB | DRB | RB | AST | STL | BLK | TOV | PF | PTS
|--:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
| 198 | 1 | 1975-11-08 | 3048 | 12 | 16 | 16.0 | 2 | 3.0 | 0.0 | 0.0 | 1 | 2.0 | 0.0 | 3.0 | 3.0 | 1.0 | 0.0 | 0.0 | 0.0 | 2.0 | 5
| 199 | 1 | 1975-11-08 | 4000 | 24 | 16 | 33.0 | 11 | 21.0 | 0.0 | 1.0 | 3 | 3.0 | 3.0 | 2.0 | 5.0 | 1.0 | 0.0 | 1.0 | 1.0 | 1.0 | 25
| 200 | 1 | 1975-11-08 | 1994 | 24 | 16 | 21.0 | 0 | 9.0 | 0.0 | 0.0 | 0 | 0.0 | 3.0 | 9.0 | 12.0 | 1.0 | 1.0 | 0.0 | 2.0 | 3.0 | 0
| 201 | 1 | 1975-11-08 | 1082 | 12 | 16 | 16.0 | 0 | 2.0 | 0.0 | 1.0 | 3 | 4.0 | 3.0 | 1.0 | 4.0 | 4.0 | 2.0 | 0.0 | 0.0 | 3.0 | 3
| 202 | 1 | 1975-11-08 | 2621 | 24 | 16 | 24.0 | 0 | 5.0 | 0.0 | 0.0 | 1 | 2.0 | 0.0 | 5.0 | 5.0 | 2.0 | 1.0 | 1.0 | 1.0 | 2.0 | 1

<table>
<tr>
<td>

- **Season** - Označení sezóny zápasu
- **Date** - Datum konání zápasu 
- **Player** - Unikátní identifikátor hráče
- **Team** - Unikátní identifikátor týmu
- **Game** - Unikátní identifikátor zápasu
- **MIN** - Počet minut strávených hráčem na hřišti během zápasu
- **FGM** - Počet úspěšných pokusů o koš (mimo trestných hodů)
- **FGA** - Celkový počet pokusů o koš (mimo trestných hodů) 
- **FG3M** - Počet úspěšných pokusů o trojku
- **FG3A** - Celkový počet pokusů o trojku

</td>
<td>

- **FTM** - Počet úspěšných trestných hodů
- **FTA** - Celkový počet trestných hodů
- **ORB** - Počet útočných doskoků
- **DRB** - Počet obranných doskoků
- **RB** - Celkový počet doskoků
- **AST** - Počet asistovaných košů
- **STL** - Počet způsobených útočných ztrát
- **BLK** - Počet bloků
- **TOV** - Počet útočných ztrát míče
- **PF** - Počet osobních faulů
- **PTS** - Počet získaných bodů

</td>
</tr>
</table>


### Bets DataFrame

Tento DataFrame očekáváme jako vaší odpověď z metody `place_bets`. I pokud nechcete pokládat žádné sázky, musíte poslat (alespoň prázdný) `DataFrame`.

|   ID |     BetH |     BetA |
|----------:|---------:|---------:|
|     15 | 0.0  | 0.0 |
|     16 | 0.0  | 0.0 |
|     16 | 0.0  | 0.0 |
|     17 | 0.0  | 0.0 |
|     18 | 10.0 | 0.0 |
|     19 | 0.0  | 7.5 |

- **ID** - Index odpovídajícího zápasu z [Opps DataFrame](#opps-dataframe) (index tabulky)
- **BetH** - Vaše sázky na výhru domácího týmu
- **BetA** - Vaše sázky na výhru hostujícího týmu


## Technické náležitosti

- Veškerá data kolující mezi vaším modelem a bookmakerem jsou uložená v `pandas.DataFrame`.
- Komunikace mezi vámi a bookmakerem probíhá přes std in/out. Nemusíte se ničeho obávat, serializaci jsme vyřešili za vás, ale **raději se vyvarujte používání stdout v odevzdávaném řešení**.
- **Pokud bookmakerovi pošlete sázky na jiné příležitosti, než které vám přisly, budou ignorovány.**
- **Pokud nebudete mít dostatek prostředků pro vaše sázky, budou ignorovány.**
- **Sázky > max_bet a sázky < min_bet budou ignorovány.**

## Časté problémy

- Nejčastější chyba v upload systému je spojena s tím, že se váš kód v evaluační smyčce na hyperionu dostane do stavu, na který není připraven.
  - Například "RTE: KeyError" - zkontrolujte si zda je váš kód připravený na prázdný Dataframe příležitostí, nově se vyskytující ID týmu, apod.

Veškeré dotazy, problémy a náměty směřujte na qqh@fel.cvut.cz
