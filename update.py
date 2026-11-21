import os
import json
import datetime
import urllib.request
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# 1. API-Keys aus den GitHub Secrets (Umgebungsvariablen) laden
FINNHUB_KEY = os.environ["FINNHUB_API_KEY"]
GEMINI_KEY = os.environ["GEMINI_API_KEY"]


# 2. Aktuelle News von Finnhub abrufen (General + Crypto für alle 5 Sparten)
def fetch_finnhub_news(category="general", limit=20):
    url = f"https://finnhub.io/api/v1/news?category={category}&token={FINNHUB_KEY}"
    req = urllib.request.urlopen(url, timeout=20)
    articles = json.loads(req.read().decode("utf-8"))
    return [
        {
            "headline": a.get("headline", ""),
            "summary": a.get("summary", ""),
            "source": a.get("source", ""),
            "url": a.get("url", ""),
        }
        for a in articles[:limit]
    ]


news_general = fetch_finnhub_news("general", 20)
news_crypto = fetch_finnhub_news("crypto", 8)
all_news = news_general + news_crypto


# 3. Strukturierte Ausgabe für das Dashboard festlegen
class TopNewsItem(BaseModel):
    sector: str = Field(
        description="Genau eine von: Tech, Rohstoffe, Währungen, Krypto, Industrie"
    )
    title: str = Field(description="Prägnante deutsche Überschrift")
    reason: str = Field(
        description="Kurze 1-2 Satz Erklärung, warum diese Meldung den Markt bewegt"
    )
    url: str = Field(description="Original-URL aus den bereitgestellten News")


class DailyMarketAnalysis(BaseModel):
    tech: float = Field(description="Stimmungswert von -5.0 bis +5.0")
    rohstoffe: float = Field(description="Stimmungswert von -5.0 bis +5.0")
    waehrungen: float = Field(description="Stimmungswert von -5.0 bis +5.0")
    krypto: float = Field(description="Stimmungswert von -5.0 bis +5.0")
    industrie: float = Field(description="Stimmungswert von -5.0 bis +5.0")
    top_news: list[TopNewsItem] = Field(
        description="Exakt die 3 wichtigsten Markt-Treiber des Tages"
    )


# 4. Gemini API zur Auswertung aufrufen
client = genai.Client(api_key=GEMINI_KEY)

prompt = f"""
Analysiere die folgenden aktuellen Finanz- und Wirtschaftsnachrichten.
Bewerte die heutige Marktstimmung für jede der 5 Sparten (Tech, Rohstoffe, Währungen, Krypto, Industrie)
auf einer Skala von -5.0 (stark bärisch/negativ) bis +5.0 (stark bullisch/positiv), wobei 0.0 neutral ist.
Wähle außerdem die 3 wichtigsten Markt-Treiber des Tages aus, fasse sie auf Deutsch zusammen und übernimm die Original-URL.

Nachrichten:
{json.dumps(all_news, ensure_ascii=False)}
"""

response = client.models.generate_content(
   model="gemini-3.8-flash",
    contents=prompt,
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=DailyMarketAnalysis,
        temperature=0.2,
    ),
)

today_result = json.loads(response.text)
today_str = datetime.date.today().isoformat()


def clamp_score(val):
    return round(max(-5.0, min(5.0, float(val))), 2)


# 5. Bestehende data.json laden und heutigen Tag anhängen/aktualisieren
data_file = "data.json"
if os.path.exists(data_file):
    with open(data_file, "r", encoding="utf-8") as f:
        history = json.load(f)
else:
    history = {"days": []}

history["days"] = [d for d in history["days"] if d["date"] != today_str]
history["days"].append(
    {
        "date": today_str,
        "scores": {
            "tech": clamp_score(today_result["tech"]),
            "rohstoffe": clamp_score(today_result["rohstoffe"]),
            "waehrungen": clamp_score(today_result["waehrungen"]),
            "krypto": clamp_score(today_result["krypto"]),
            "industrie": clamp_score(today_result["industrie"]),
        },
        "top_news": [item.model_dump() if hasattr(item, "model_dump") else item for item in today_result["top_news"]],
    }
)

# Maximal die letzten 365 Tage behalten
history["days"] = sorted(history["days"], key=lambda x: x["date"])[-365:]

with open(data_file, "w", encoding="utf-8") as f:
    json.dump(history, f, ensure_ascii=False, indent=2)

print(f"Update für {today_str} erfolgreich gespeichert.")
