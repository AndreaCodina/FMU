# Modello FMU — attuatore AMD (prototipo)

Modello Python dell'attuatore dell'Active Mass Damper: **SetPosition → ActualPosition**
con motore lineare, drive Unidrive M701 (RFC-S + AMC), pacco batterie e bus DC, bumper di fine corsa.
Il modello è pensato per diventare una FMU (FMI 2.0 Co-Simulation) da integrare con la logica PLC.

## Avvio rapido

```bash
pip install -r requirements.txt
python gui.py                 # interfaccia completa (simulazione, validazione con energia, FRF)
python mini_gui.py            # mini simulatore a cursori
python run_validation.py      # confronto con tutti i test -> results/validazione.csv
python identification.py      # ri-identifica massa, attriti, kf, potenza, batteria
python tune_overload.py       # taratura curve limite di tensione sui test in sovraccarico (~25 min)
python build_fmu.py           # genera AMD_Actuator.fmu e la testa con FMPy
```

**Guida dettagliata all'interfaccia: [GUIDA_INTERFACCIA.md](GUIDA_INTERFACCIA.md).**
Esempi di file di SetPosition: cartella `esempi/`.

I dati sperimentali vengono letti da `..\..\02_File FMU per claude` (cartella fratella);
dalla GUI si può scegliere un'altra cartella.

## File

| File | Contenuto |
|---|---|
| `params.py` | Tutti i parametri (dataclass). **Massa e corsa sono parametri liberi.** |
| `amd_model.py` | Modello completo, interfaccia `reset()` / `do_step()` stile FMU + `simulate()` |
| `battery.py` | Pacco batterie + bus DC (porting del modello MATLAB), chopper 780 V su 8 Ω, precarica 47 Ω |
| `force_energy.py` | Forza scaricata come la calcola il PLC (`rActualForce`) e metrica energetica |
| `fmu_wrapper.py`, `build_fmu.py` | Wrapper FMI 2.0 (pythonfmu) e build/test della FMU |
| `gui.py` | Interfaccia: simulazione, validazione sui test (forza ed energia), FRF |
| `mini_gui.py` | Mini simulatore a cursori (massa, corsa, ampiezza, frequenza, durata, stringhe) |
| `identification.py` | Identificazione parametri dai test |
| `tune_overload.py` | Taratura nel tempo sui test in sovraccarico con obiettivo l'energia scaricata |
| `validation.py`, `run_validation.py` | Confronto modello/test e metriche |
| `references.py` | Generatori di SetPosition (seno, sweep, gradino, rampa, da file) |
| `data_io.py` | Lettura CSV TwinCAT Scope |
| `data/` | Curva di scarica Ovonic (estratta dal .fig) e parametri identificati |

## Struttura del modello

```
x_ref ─[T_ref 4 ms: EtherCAT + AMC]─► AMC: FF velocità ×1.0 + Kp 25·e (limitato ±0.077 m/s)
      ─► anello velocità PI (Ki/Kp = 65 1/s) ─► filtro rif. corrente 4 ms ─► limite 200 A
      ─► limite tensione I(v, Vbus) ─► F = kf_eff(|Iq|)·kf(x)·Iq
      ─► massa + attrito guide (μ·m·g) + viscoso + bumper ─► x, v
P_dc = k·F·v + 3·R·Iq² + P_v·|v|  ─► batterie (Voc(Ah), R(I, dI/dt), ramo RC) ─► Vbus ─┐
      ▲                       chopper 780 V su 8 Ω, rigenerazione nel pacco             │
      └─────────────────────────── Vbus (limite di tensione) ◄──────────────────────────┘
```

* **Drive (dai parametri reali, `ParametersExport_all.pdf`)**: controllo di posizione nell'AMC
  (menu 31–41, 250 µs) con feed-forward di velocità pieno (Pr 39.010 = 1), Kp posizione 25
  (Pr 39.007), uscita P limitata a 109.2 UU/ms = 0.077 m/s (Pr 39.009), velocità max 6000 rpm
  = 4.6 m/s (Pr 39.011; 1 giro = passo polare 46 mm = 65536 UU); anello di velocità PI con
  Ki/Kp = 4.80/0.0735 (Pr 03.010/011); filtro 4 ms sul riferimento di corrente (Pr 04.012);
  limite 860% × 23.1 A = 199 A (Pr 04.005-007); chopper 780 V (Pr 06.073/074) su 8 Ω
  (Pr 10.061); sottotensione 330 V (Pr 06.065).
  I guadagni di velocità si impostano in unità del drive (Pr 03.010 [s/rad], 03.011 [s²/rad]);
  dalle sweep (simulate nel tempo) si ricavano solo la scala di conversione
  (0.0735 s/rad → 884 A/(m/s)), il ritardo sul riferimento (4 ms) e un ritardo residuo d'anello (0.9 ms);
  il rapporto Ki/Kp, lasciato libero, è convergito a 64.6 (drive: 65.3).
* **Limite di tensione applicato prima del filtro di corrente** (dopo il filtro resta solo il limite
  di 200 A del drive): in campo di tensione la corrente segue la dinamica dell'anello di corrente,
  quindi i passaggi "limite di tensione ↔ comando del regolatore" sono raccordati come nel reale
  e non presentano punti angolosi.
* **Anti-windup "a inseguimento"** (back-calculation, Tt = 20 ms) sull'integrale di velocità,
  attivo sia sul limite di corrente sia su quello di tensione.
* **Saturazione magnetica**: kf ≈ 148 N/A a bassa corrente → ≈ 114 N/A a 200 A
  (forza a saturazione ≈ 22 kN, uguale per tutte le masse).
* **kf in funzione della posizione**: ≈ +1% per x < 0, −2…−5% per x > 0 (misurato dai test).
* **Limite di tensione / velocità**: curva liscia e monotona I = I_max / (1 + ((v − v1)/vc)^n)
  per v > v1 (v scalata con Vbus/660 V), tarata nel tempo sull'energia dei test in sovraccarico:
  in trazione 200 A fino a 1.83 m/s, 100 A a 2.5 m/s, ~30 A a 3.2 m/s; in frenata 200 A fino a
  3.5 m/s, 100 A a 4.7 m/s.
* **Bumper**: molla-smorzatore solo a compressione oltre ±corsa; rigidezza tarata su
  100 kN @ 4 m/s con 1320 kg, smorzamento dal coefficiente di restituzione (default 0.3).
* **Bus DC**: porting del modello MATLAB (Voc da curva di scarica ×32, R di stringa
  funzione di corrente e dI/dt, ramo RC), cavi 2 m 16 mm², n stringhe in parallelo,
  precarica su 47 Ω e capacità DC link stimata 0.9 mF.
* La massa mobile impostata comprende già forcer e carrello (nessuna massa aggiuntiva).
* Altri parametri identificati: μ guide 0.013, viscoso ≈ 245 N·s/m,
  potenza k_mot 1.04 / k_gen 0.92 / R 0.42 Ω, batteria R×0.89 e ramo RC 0.39 Ω / 1.2 s.

## Metrica energetica (forza scaricata)

* `rActualForce` del PLC = −M_nom · LP(ẍ), LP = Butterworth 2° ordine 20 Hz causale + 2 ms
  (ricostruita dalla posizione con errore 0.3%). L'FMU dà la stessa grandezza in `F_actual`.
* vG = ∫ rAccGround, passa-alto Butterworth 2° ordine 0.1 Hz (filtfilt).
* E = ∫ vG · F dt; si confronta il valore finale reale (da `rActualForce`) con quello del modello.

## Validazione (179 test CVT + sismi, `results/validazione.csv`)

| | valore |
|---|---|
| errore energia, test con energia > 1 kJ (69): mediana | **0.23 %** |
| … entro 1% / 2% | **86%** / 94% (max 9.8%) |
| … solo i 53 test non usati nella taratura: entro 1% | 85% (mediana 0.19%) |
| 11 test in sovraccarico | tutti entro ±1.0% |
| errore posizione (% della corsa del test): mediana / max | 0.4 % / 9.4 % |
| errore Iq / Vbus / potenza (mediana) | 8 A / 7 V / 2.6 kW |

I test ad alta frequenza (7.5–10 Hz) hanno energia finale quasi nulla (±0.1 kJ): lì lo
scarto percentuale non è significativo.

Limiti noti del prototipo:
* Due test a 3–4 Hz in saturazione piena e ampiezza massima (`Test17_Seno780_f4`,
  `Test17_Seno1050_f3`, energia ~1–1.7 kJ) restano fuori (fino a ≈ ±10%): la commutazione ±200 A
  avviene qualche ms dopo il reale; da affinare la dinamica d'anello a 3–5 Hz.
* Prove 510 kg: indicati Pr 03.010/011 = 0.06 / 3.2, ma la sweep "GainGiusti" e i 16 test CVT
  a 510 kg tornano molto meglio con 0.0735 / 4.80 (costo FRF 0.013 contro 0.083; errore di
  posizione nei test lineari 2–4 volte più basso). La sweep "GainAlti" risulta ≈ 0.09–0.11.
  Per ora la 510 kg usa 0.0735 / 4.80; per cambiarlo: `TEST_SPEED_GAINS` in `validation.py`.
* La curva limite di tensione è empirica (forma liscia a 3 parametri per verso, scalata con Vbus).
* Ripetibilità sperimentale in sovraccarico: prove ripetute differiscono dell'1.4–5.8%.
* Rumore e incertezze non sono ancora modellati (passo successivo).

## Ingressi / uscite FMU

* Ingressi: `x_ref`, `enable`, `precharge_contactor`, `main_contactor`
* Uscite: `x, v, a, Iq, I_lim, F_motor, F_bumper, F_actual, Vbus, I_batt, Voc, SOC, P_drive, P_brake, P_mech`,
  flag `sat_current, sat_voltage, bumper_contact, drive_enabled`
* Parametri: `mass, stroke, bumper_e, mu_coulomb, kf, I_max, Kp_pos, K_ff,
  Kp_speed_Pr03010, Ki_speed_Pr03011` (unità del drive),
  T_delay, V_brake, R_scale_batt, V0, n_strings, bus_precharged`
