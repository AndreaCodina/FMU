# Guida all'interfaccia del prototipo AMD

## 1. Avvio

Dalla cartella `amd_fmu`:

```bash
python gui.py
```

(la prima volta: `pip install -r requirements.txt`). Si apre una finestra con:

* **a sinistra** il pannello dei parametri del modello;
* **a destra** tre schede: **Simulazione**, **Validazione test**, **FRF**;
* **in basso a destra** la barra di avanzamento del calcolo.

Ogni simulazione gira in sottofondo: la finestra resta utilizzabile. Tempi indicativi:
~0.5 s ogni 10 s di profilo.

## 2. Pannello parametri (sinistra)

I parametri più usati sono in cima al gruppo *Meccanica / bumper*:

| Parametro | Significato |
|---|---|
| Massa mobile [kg] | massa mobile totale dell'AMD, forcer e carrello compresi (modulabile) |
| Semicorsa / contatto bumper [m] | oltre ±questo valore la massa tocca il bumper |
| Stringhe in parallelo | 1 o 2 stringhe di batterie (gruppo *Batterie / bus DC*) |
| Tensione iniziale pacco [V] | tensione a vuoto delle batterie a inizio prova (*Simulazione*) |
| Kp / Ki velocità (Pr 03.010 / 03.011) | guadagni dell'anello di velocità in unità del drive |
| Anello di corrente | attiva la dinamica dell'anello di corrente (modello v2). Per il modello v1 completo: *Carica…* → `versioni/parametri_v1_senza_anello_corrente.json` |

* Il menu **Motore** in alto carica i dati Hiwin LMFA64L o Siemens 1FN3900.
* **Default** riporta tutti i parametri ai valori identificati; **Salva… / Carica…** salvano e
  ricaricano un set di parametri in JSON (utile per confrontare configurazioni).
* Un parametro modificato vale per tutte le schede alla simulazione successiva.

## 3. Scheda «Simulazione» — provare un profilo di SetPosition

In alto si sceglie il **Profilo SetPosition**; i campi accanto cambiano in base al profilo.

### a) Profili generati dall'interfaccia (in forza o in spostamento)

| Profilo | Campi |
|---|---|
| **Seno in forza** | ampiezza forza [kN], frequenza [Hz], durata [s], rampa di ingresso/uscita [s] |
| **Sweep in forza** | ampiezza forza [kN], frequenza iniziale f0 e finale f1, durata, rampa |
| Seno (spostamento) | ampiezza [m], frequenza [Hz], durata [s], rampa di ingresso/uscita [s] |
| Sweep (spostamento) | ampiezza, frequenza iniziale f0 e finale f1, durata |
| Gradino | ampiezza (posizione finale), durata |
| Rampa (urto bumper) | posizione finale, velocità [m/s], durata — per provare l'urto sul bumper |

I profili partono dopo 0.5 s di fermo e hanno una rampa a coseno in ampiezza (default 1 s) per
non chiedere un'accelerazione infinita all'istante iniziale.

**Profili in forza.** Si imposta la *forza scaricata* desiderata (stessa convenzione di
`rActualForce`: F = −M·ẍ, con M la massa mobile del pannello). L'interfaccia costruisce la
SetPosition x(t) = F/(M·ω²)·rampa·sin(ωt) (per lo sweep con ω variabile): a regime la forza
richiesta è esattamente quella impostata e la posizione parte e torna a zero. Nel grafico della
forza compaiono la **forza richiesta** (tratteggiata) e quella **ottenuta**; nella sintesi il
rapporto di ampiezza ottenuta/richiesta e il ritardo (≈ 17–18 ms: ≈ 13 ms sono il filtro con cui
il PLC calcola `rActualForce`, il resto è drive e comunicazione). Lo spostamento necessario cresce
come 1/f²: alle basse frequenze può superare la corsa (il controllo di fattibilità lo segnala).

### b) File sperimentale TwinCAT (come i test che avete registrato)

Profilo **Da file CSV** → **Carica CSV…** → scegliere un file di prova (es.
`02_File FMU per claude\Test sperimentali\...\Test8_Sisma1320SLC2_....csv`).
Il file viene riconosciuto in automatico: si usa `rSetPosition` come riferimento e, se c'è,
`rActualPosition` viene sovrapposta ai risultati per il confronto.
**Attenzione:** impostare a mano nel pannello la **massa** e il **n° di stringhe** del test
(la scheda *Validazione test* invece lo fa da sola).

### c) Un CSV con la propria SetPosition

Profilo **Da file CSV** → **Carica CSV…**. Formati accettati:

* due colonne **tempo [s]** e **posizione**, con o senza intestazione (se l'intestazione
  contiene nomi come `tempo`/`t`/`time` e `posizione`/`x`/`setpos` le colonne vengono
  riconosciute anche in ordine diverso);
* una sola colonna di **posizione**: il passo di campionamento si imposta in *Passo (se 1 colonna)*;
* separatore `;` `,` o tabulazione; decimali con punto o con virgola (va bene l'export di Excel);
* righe che iniziano con `#` sono ignorate;
* **Unità posizione**: m oppure mm.

Il segnale viene ricampionato a 1 ms (interpolazione lineare). Dopo il caricamento compare
un'anteprima del profilo e una riga con durata e range. Esempi pronti nella cartella `esempi/`:

| File | Formato |
|---|---|
| `setpos_due_armoniche_m.csv` | `tempo_s;posizione_m`, decimali con virgola (stile Excel) |
| `setpos_trapezio_mm_10ms.csv` | `t,x` in **mm**, campionato a 10 ms |
| `setpos_una_colonna_1ms.txt` | una colonna in m, passo 1 ms |

Consiglio: il profilo deve essere "liscio". Un salto di posizione o di velocità nel file
richiede un'accelerazione infinita e il modello risponderà (correttamente) con la forza massima.

### Eseguire e leggere i risultati

* **▶ Simula** avvia il calcolo. *Simula precarica bus* aggiunge 1 s iniziale con la sequenza
  contattore di precarica (47 Ω) → contattore principale → abilitazione drive.
* Riga di sintesi: posizione e velocità massime, errore massimo rispetto alla SetPosition, forza
  massima scaricata, Vbus min/max, % di tempo in saturazione di corrente / tensione, energia
  dissipata sulla resistenza di frenatura, eventuale **URTO BUMPER** con la forza di picco.
* **Controllo di fattibilità**: forza richiesta dal profilo (massa × accelerazione del riferimento),
  velocità e spostamento richiesti, confrontati con i limiti dell'attuatore (≈ 23 kN a 200 A, calo
  di corrente oltre ~2 m/s, corsa). Se il profilo è oltre i limiti la riga diventa **rossa**: il
  sistema non può seguirlo e l'errore sarà grande.
* Se il file conteneva la posizione misurata compare anche l'errore modello − misura (RMSE e max).

Grafici (4 × 2):

| | sinistra | destra |
|---|---|---|
| 1 | SetPosition, ActualPosition del modello (+ misura), linee del bumper | errore [mm] |
| 2 | corrente Iq e limite (corrente / tensione) | forza scaricata (come `rActualForce`) e bumper |
| 3 | velocità | Vbus e tensione a vuoto del pacco |
| 4 | potenza al bus DC e sulla resistenza di frenatura | corrente di batteria e SOC |

La barra sopra i grafici permette zoom (lente), spostamento (croce), ritorno alla vista
iniziale (casa) e salvataggio dell'immagine (dischetto). Gli assi dei tempi sono collegati.

**Esporta risultati CSV…** salva tutte le grandezze campionate a 1 ms (separatore `;`,
decimali con virgola, apribile in Excel): tempo, SetPosition, posizione, velocità,
accelerazione, Iq, limite, forze, Vbus, Voc, corrente batteria, SOC, potenze, flag di saturazione.

*Ricalcola automaticamente al cambio parametri*: se attivo, ogni modifica nel pannello di
sinistra rilancia la simulazione (comodo per vedere l'effetto di massa, corsa, guadagni…).

## 4. Scheda «Validazione test» — confronto completo con le misure

* A sinistra l'elenco delle prove (CVT, Sismi, FRF) raggruppate per massa;
  **doppio click** su una prova (o *Confronta modello / test*).
* Con *Imposta massa e n° stringhe dal test* attivo, massa e stringhe vengono presi dalla prova.
* Grafici: posizione, Iq, forza scaricata, Vbus, potenza del drive, **energia ∫vG·F** reale e
  del modello; in alto le metriche e lo **scarto di energia finale** in %.
* *Cartella dati…* per puntare a un'altra cartella con la stessa struttura.

## 5. Scheda «FRF»

Si sceglie una prova sweep e *Calcola FRF*: confronto modulo/fase sperimentale con il modello
(simulazione nel tempo, che include i limiti del drive, e modello lineare tratteggiato).

## 6. Mini simulatore

```bash
python mini_gui.py
```

Finestra con soli cursori (massa, corsa, ampiezza in forza [kN] o in spostamento [m], frequenza,
durata, stringhe) e un selettore *Seno in forza / Seno in spostamento*: ricalcola da sola mezzo
secondo dopo che si rilascia un cursore. Utile per esplorazioni veloci.

## 7. Uso da script / FMU

* `python run_validation.py` ripete il confronto su tutti i test → `results/validazione.csv`.
* `python build_fmu.py` crea `AMD_Actuator.fmu` (ingressi `x_ref`, `enable`, contattori;
  uscite posizione, forza `F_actual`, Vbus, …), usabile da qualunque simulatore FMI 2.0 che
  abbia Python con numpy/scipy disponibile.
