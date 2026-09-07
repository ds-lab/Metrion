# METRION Optimierungs-Specs

Sechs Schritte, in 36 Inkremente zerlegt. Jedes Inkrement ist so geschnitten, dass es

* **eine** Sache ändert,
* einen automatischen Test hat, der vorher rot und nachher grün ist,
* und in wenigen Minuten von Hand verifizierbar ist.

## Reihenfolge

| Schritt | Datei | Inkremente | Warum an dieser Stelle |
|---|---|---|---|
| S1 | [S1-shutdown.md](S1-shutdown.md) | 7 | Ohne sauber beendbaren Prozess ist nichts reproduzierbar messbar |
| S2 | [S2-latenz-config-drift.md](S2-latenz-config-drift.md) | 6 | Macht 60 s Live-Verzug überhaupt erst möglich |
| S3 | [S3-parquet-senke.md](S3-parquet-senke.md) | 5 | Nimmt die Schreiblast weg, bevor die Attribution dazukommt |
| S4 | [S4-live-attribution.md](S4-live-attribution.md) | 7 | Braucht S2 (Latenz) und S3 (Senke) |
| S5 | [S5-bpf-event.md](S5-bpf-event.md) | 5 | Genauigkeit; messbar erst wenn S4 Ergebnisse liefert |
| S6 | [S6-python-cleanup.md](S6-python-cleanup.md) | 6 | Unabhängig, jederzeit zwischendurch |

S1 und S6 sind voneinander und von allem anderen unabhängig. S2 → S3 → S4 ist eine echte Kette.

## Konventionen für die Umsetzung

**Ein Inkrement = ein Commit.** Commit-Message: `S1.3 sched_switch: Sentinel im Normalbetrieb erkennen`.

**Nichts umbenennen, was nicht in der Spec steht.** Die Specs referenzieren Zeilennummern des
Stands `main @ 249f62a`. Wenn eine Zeilennummer nicht passt, nach dem genannten Funktions- oder
Variablennamen suchen — der ist die verlässliche Referenz, die Zeilennummer nur ein Hinweis.

**Tests laufen ohne root, ohne eBPF, ohne GPU.** Wenn ein Inkrement etwas ändert, das eBPF
braucht, testet der automatische Test die reine Logik (Queue-Verhalten, Parser, Schema) mit
Attrappen; die eBPF-Seite verifizierst du von Hand auf dem Linux-Rechner.

```bash
# Alle Tests
pytest tests/unit -q

# Tests eines Inkrements
pytest tests/unit/test_queue_util.py -q
```

**Bench-Harness als Regressionsgate.** Vor S1 einmal die Baseline aufnehmen:

```bash
cd bench
for s in bench_db_write bench_entity_upsert bench_python_overhead bench_ringbuf_callback; do
    python $s.py 2>&1 | tee "baseline_$s.txt"
done
```

## Struktur jedes Inkrements

```
### S<n>.<m> — Titel
Ziel            Ein Satz: was danach anders ist.
Dateien         Genaue Pfade.
Vorher lesen    Was das umsetzende Modell gelesen haben muss, bevor es etwas ändert.
Änderung        Präzise Anweisung, oft mit Code.
Automatischer Test   Neue oder erweiterte Testdatei + Kommando.
Deine Verifikation   Kommando, das du ausführst, und was du sehen musst.
Fertig wenn     Abnahmekriterium.
```

## Zwei Entscheidungen, die in allen Specs vorausgesetzt werden

1. **Der Ergebnisstrom ist breit.** Eine Zeile pro (Thread, Zeitfenster) mit neun Energiespalten
   (`cpu_total_j`, `cpu_idle_j`, `cpu_dynamic_j`, `dram_*`, `gpu_*`), plus denormalisierte
   Identität (`native_pid`, `executable_name`, `cgroup_name`, …). Über Sockel bzw. GPUs wird
   summiert; die Aufschlüsselung pro Gerät bleibt im Rohstrom.
2. **Der Rohstrom ist abschaltbar.** `storage.store_raw_measurements: false` schreibt nur noch den
   Ergebnisstrom. Das ist der größte verbleibende Hebel auf die Systemlast — und der Preis ist,
   dass sich der Lauf nachträglich nicht anders attribuieren lässt.
