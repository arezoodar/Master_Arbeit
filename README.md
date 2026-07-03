# KI-basierte Bildanalyse zur Zustandsbewertung von Gegenständen

**Masterarbeit — Hochschule Flensburg, 2026**

## Projektbeschreibung

Dieses Repository enthält den Quellcode und die Rohdaten 
zur Masterarbeit mit dem Titel:

> *Anwendung KI-basierter Bildanalyseverfahren zur 
> Zustandsbewertung und Komponentenklassifikation 
> von Gegenständen*

---


## Installation

```bash
# Repository klonen
https://github.com/arezoodar/Master_Arbeit.git
cd Master_Arbeit

# Virtuelle Umgebung erstellen
python -m venv venv
source venv/bin/activate

# Pakete installieren
pip install -r requirements.txt
```

## API Keys konfigurieren

Vor der Ausführung müssen die API-Schlüssel in `app.py` 
eingetragen werden:

```python
groq_client    = Groq(api_key="DEIN_GROQ_KEY")
openai_client  = OpenAI(api_key="DEIN_OPENAI_KEY")
claude_client  = anthropic.Anthropic(api_key="DEIN_ANTHROPIC_KEY")
```

---

## Web-Applikation starten

```bash
python app.py
```

Anschließend im Browser öffnen: **http://localhost:5000**

---

## Evaluation ausführen

```bash
python Evaluation_Script.py
```

Die Ergebnisse werden automatisch in `evaluation_results.csv` 
auf dem Desktop gespeichert.


## Autorin

**Arezoo Darvishi**  
Hochschule Flensburg  
Masterarbeit, 2026