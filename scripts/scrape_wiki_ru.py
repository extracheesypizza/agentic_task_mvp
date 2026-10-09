# scripts/scrape_wiki_ru.py
"""
Fetch Russian Wikipedia intro extracts for medical/lab terms.
Uses the MediaWiki API — no HTML parsing, no BeautifulSoup.
"""

import requests
import json
import time
from pathlib import Path
from tqdm import tqdm

OUTPUT = Path("data/raw/wikipedia")
OUTPUT.mkdir(parents=True, exist_ok=True)

WIKI_API = "https://ru.wikipedia.org/w/api.php"
HEADERS = {
    "User-Agent": "MedTranslateBot/1.0 (university student project)",
    "Accept": "application/json",
}

# ── Expanded seed list: 120+ common Russian medical/lab terms ────────
SEED_TERMS = [
    # Общий анализ крови
    "Гемоглобин", "Эритроциты", "Лейкоциты", "Тромбоциты",
    "Гематокрит", "СОЭ", "Ретикулоциты", "Цветовой показатель",
    "Нейтрофилы", "Лимфоциты", "Моноциты", "Эозинофилы", "Базофилы",
    "Средний объём эритроцита", "Среднее содержание гемоглобина в эритроците",

    # Биохимия крови
    "Глюкоза", "Креатинин", "Мочевина", "Мочевая кислота",
    "Билирубин", "Аланинаминотрансфераза", "Аспартатаминотрансфераза",
    "Щелочная фосфатаза", "Гамма-глутамилтрансфераза",
    "Общий белок", "Альбумин", "Холестерин", "Триглицериды",
    "Липопротеины низкой плотности", "Липопротеины высокой плотности",
    "Амилаза", "Липаза", "Лактатдегидрогеназа",
    "Калий", "Натрий", "Хлор", "Кальций", "Магний", "Фосфор", "Железо",
    "Ферритин", "Трансферрин",

    # Гормоны
    "Тиреотропный гормон", "Трийодтиронин", "Тироксин",
    "Инсулин", "Кортизол", "Тестостерон", "Эстрадиол",
    "Прогестерон", "Пролактин", "Фолликулостимулирующий гормон",
    "Лютеинизирующий гормон", "Паратгормон",

    # Коагулограмма
    "Коагулограмма", "Протромбиновое время", "Международное нормализованное отношение",
    "Активированное частичное тромбопластиновое время",
    "Фибриноген", "Тромбиновое время", "D-димер",

    # Общий анализ мочи
    "Общий анализ мочи", "Удельный вес мочи", "Белок в моче",
    "Лейкоциты в моче", "Эритроциты в моче", "Цилиндры в моче",

    # Маркеры и онкомаркеры
    "С-реактивный белок", "Ревматоидный фактор",
    "Простатический специфический антиген",
    "Раково-эмбриональный антиген", "Альфа-фетопротеин",
    "Хорионический гонадотропин человека",

    # Инфекции / иммунология
    "Иммуноглобулин", "Антитела", "Интерферон",
    "ВИЧ", "Гепатит B", "Гепатит C",

    # Витамины и микроэлементы
    "Витамин D", "Витамин B12", "Фолиевая кислота",

    # Инструментальные исследования
    "Электрокардиография", "Эхокардиография",
    "Ультразвуковое исследование", "Магнитно-резонансная томография",
    "Компьютерная томография", "Рентгенография",
    "Флюорография", "Маммография",

    # Общие медицинские термины
    "Биопсия", "Пункция", "Анамнез", "Диагноз",
    "Прогноз", "Ремиссия", "Рецидив", "Метастаз",
    "Воспаление", "Некроз", "Фиброз", "Склероз",
    "Анемия", "Лейкоз", "Тромбоз", "Эмболия",
    "Ишемия", "Инфаркт", "Инсульт", "Сепсис",
    "Аллергия", "Анафилаксия", "Аутоиммунное заболевание",
    "Гипертония", "Гипотония", "Аритмия", "Тахикардия", "Брадикардия",
    "Сахарный диабет", "Гипотиреоз", "Гипертиреоз",
    "Гастрит", "Панкреатит", "Холецистит", "Гепатит",
    "Пиелонефрит", "Цистит", "Уретрит",
    "Остеопороз", "Артрит", "Артроз",
    "Бронхит", "Пневмония", "Астма",
    "Дерматит", "Экзема", "Псориаз",
]


def fetch_article(title: str, max_retries: int = 3) -> dict | None:
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "exintro": True,
        "explaintext": True,
        "format": "json",
        "formatversion": 2,
    }

    for attempt in range(max_retries):
        try:
            resp = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=15)

            # ── Handle rate limit ─────────────────────────────────
            if resp.status_code == 429:
                wait = 2 ** (attempt + 2)  # 4s, 8s, 16s
                print(f"  ⏳ Rate limited on '{title}', waiting {wait}s...")
                time.sleep(wait)
                continue

            if resp.status_code != 200:
                print(f"  ⚠ HTTP {resp.status_code}: {title}")
                return None

            if not resp.text.strip():
                return None

            data = resp.json()

        except requests.exceptions.RequestException as e:
            print(f"  ✗ Network error ({title}): {e}")
            time.sleep(3)
            continue
        except json.JSONDecodeError:
            return None

        # ── Parse ─────────────────────────────────────────────────
        pages = data.get("query", {}).get("pages", [])
        if not pages:
            return None

        page = pages[0]
        if page.get("missing"):
            return None

        extract = page.get("extract", "").strip()
        if len(extract) < 30:
            return None

        return {
            "term": page.get("title", title),
            "plain": extract,
            "source": "wikipedia_ru",
        }

    print(f"  ✗ Gave up on '{title}' after {max_retries} retries")
    return None


if __name__ == "__main__":
    results = []
    skipped = []

    for term in tqdm(SEED_TERMS, desc="Wikipedia RU"):
        article = fetch_article(term)
        if article:
            results.append(article)
        else:
            skipped.append(term)
        time.sleep(1.5)  # ← was 0.3, now 1.5 seconds

    out = OUTPUT / "wiki_medical_terms.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Saved {len(results)} articles → {out}")
    if skipped:
        print(f"⊘ Skipped {len(skipped)}:")
        for s in skipped:
            print(f"   - {s}")