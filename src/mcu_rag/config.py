"""
Central configuration — single source of truth for all constants.
Edit here to change models, paths, or chunking parameters.
"""
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]  # project/

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_DIR        = ROOT / "data"
RAW_DIR         = DATA_DIR / "raw"
WIKI_DIR        = DATA_DIR / "wikipedia"
FANDOM_DIR      = DATA_DIR / "fandom"
UPLOADS_DIR     = DATA_DIR / "uploads"
CHROMA_DIR      = DATA_DIR / "chroma_db"
BM25_INDEX_PATH = DATA_DIR / "bm25_index.pkl"

# ── Ollama models ─────────────────────────────────────────────────────────────
LLM_MODEL       = "llama3.1:8b"      # swap to "qwen2.5:7b" if pulled
EMBED_MODEL     = "nomic-embed-text"
OLLAMA_BASE_URL = "http://localhost:11434"

# ── ChromaDB ──────────────────────────────────────────────────────────────────
CHROMA_COLLECTION = "mcu_knowledge"

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 512
CHUNK_OVERLAP = 64

# ── Retrieval ─────────────────────────────────────────────────────────────────
BM25_TOP_K      = 10
VECTOR_TOP_K    = 10
RERANK_TOP_K    = 5   # final chunks passed to LLM after CrossEncoder

# ── CrossEncoder model (downloads automatically on first run) ─────────────────
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── Query rewriting ───────────────────────────────────────────────────────────
QUERY_REWRITE_ENABLED = True   # set False to skip rewriting entirely (debug)
MULTI_QUERY_ENABLED   = True   # set False to use single-query retrieval only
MULTI_QUERY_VARIANTS  = 2      # number of extra query variants to generate

# ── Wikipedia scraping targets ────────────────────────────────────────────────
WIKI_CHARACTERS = [
    "Iron Man", "Captain America", "Thor", "Hulk", "Black Widow",
    "Hawkeye", "Spider-Man", "Doctor Strange", "Black Panther", "Captain Marvel",
    "Ant-Man", "Scarlet Witch", "Vision", "Falcon", "War Machine",
    "Guardians of the Galaxy", "Star-Lord", "Gamora", "Thanos", "Loki",
    "Nick Fury", "Shang-Chi", "Eternals", "Wolverine",
    "Deadpool", "She-Hulk", "Moon Knight", "America Chavez", "Nebula",
    "Okoye", "Pepper Potts", "Phil Coulson", "Maria Hill", "Happy Hogan",
    "Mantis", "Shuri", "Kamala Khan",
    # Post-cutoff characters (after Dec 2023 — LLM has no training data on these)
    "Cassandra Nova",           # main villain of Deadpool & Wolverine (2024)
    "Agatha Harkness",          # lead of Agatha All Along (2024)
    "Matthew Murdock",          # Daredevil: Born Again (2025)
    "Bob Reynolds",             # Sentry — Thunderbolts* (2025)
    "Thaddeus Ross",            # Red Hulk — Captain America: Brave New World (2025)
    # Phase 6 characters (MCU-specific versions the LLM has never seen)
    "Reed Richards",            # Mr. Fantastic — The Fantastic Four: First Steps (2025)
    "Sue Storm",                # Invisible Woman — The Fantastic Four: First Steps (2025)
    "Johnny Storm",             # Human Torch — The Fantastic Four: First Steps (2025)
    "Ben Grimm",                # The Thing — The Fantastic Four: First Steps (2025)
    "Victor von Doom",          # Doctor Doom / RDJ — Avengers: Doomsday (2026)
]

WIKI_FILMS = [
    "Iron Man (film)", "Iron Man 2", "Iron Man 3",
    "The Incredible Hulk (film)",
    "Thor (film)", "Thor: The Dark World", "Thor: Ragnarok", "Thor: Love and Thunder",
    "Captain America: The First Avenger", "Captain America: The Winter Soldier",
    "Captain America: Civil War",
    "The Avengers (2012 film)", "Avengers: Age of Ultron",
    "Avengers: Infinity War", "Avengers: Endgame",
    "Guardians of the Galaxy (film)", "Guardians of the Galaxy Vol. 2",
    "Guardians of the Galaxy Vol. 3",
    "Black Panther (film)", "Black Panther: Wakanda Forever",
    "Spider-Man: Homecoming", "Spider-Man: Far From Home", "Spider-Man: No Way Home",
    "Doctor Strange (film)", "Doctor Strange in the Multiverse of Madness",
    "Ant-Man (film)", "Ant-Man and the Wasp", "Ant-Man and the Wasp: Quantumania",
    "Captain Marvel (film)", "The Marvels (film)",
    "Black Widow (film)", "Shang-Chi and the Legend of the Ten Rings",
    "Eternals (film)", "Deadpool & Wolverine",
    # Phase 4 Disney+ shows (major MCU content, previously missing)
    "WandaVision",
    "The Falcon and the Winter Soldier",
    "Loki (TV series)",
    "Hawkeye (TV series)",
    "Ms. Marvel (TV series)",
    "Moon Knight (TV series)",
    "She-Hulk: Attorney at Law",
    # Post-cutoff releases (after Dec 2023 — perfect RAG-vs-No-RAG demo targets)
    "Secret Invasion (TV series)",           # Jun 2023 (Phase 5 show)
    "Echo (TV series)",                      # Jan 2024
    "Agatha All Along",                      # Sep 2024
    "Captain America: Brave New World",      # Feb 2025
    "Daredevil: Born Again",                 # Mar 2025
    "Thunderbolts* (film)",                  # May 2025
    # Phase 6
    "The Fantastic Four: First Steps",       # Jul 2025 — Pedro Pascal, Vanessa Kirby, Ralph Ineson as Doom
    "Avengers: Doomsday",                    # May 2026 — RDJ returns as Doctor Doom (announced)
]

WIKI_PAGES = [
    "Marvel Cinematic Universe", "Infinity Stones", "S.H.I.E.L.D.",
    "Hydra (Marvel Comics)", "Avengers (Marvel Cinematic Universe)",
    # Removed: "Battle of New York (MCU)" — redirects to same page as "Avengers (MCU)"
    # Removed: "Infinity War" — same article as "Avengers: Infinity War" in WIKI_FILMS
    "Blip (Marvel Cinematic Universe)",
    "Multiverse (Marvel Cinematic Universe)",
    "Ten Rings (organization)",
]

# ── MCU Fandom Wiki articles to scrape via MediaWiki API ─────────────────────
# Source: https://marvelcinematicuniverse.fandom.com
FANDOM_ARTICLES = [
    # Core characters — use MCU Fandom exact page titles (with redirects=1 in API call)
    "Iron Man",         # Tony Stark's Fandom article title
    "Steve Rogers", "Thor", "Hulk", "Black Widow",
    "Clint Barton", "Spider-Man", "Doctor Strange", "T'Challa", "Captain Marvel",
    "Scott Lang", "Wanda Maximoff", "Vision", "Sam Wilson", "James Rhodes",
    "Peter Quill", "Gamora", "Drax the Destroyer", "Rocket Raccoon", "Groot",
    "Thanos", "Loki", "Nick Fury", "Pepper Potts", "Happy Hogan",
    "Nebula", "Okoye", "Shuri", "Yelena Belova", "Kate Bishop",
    "Shang-Chi", "America Chavez", "Kamala Khan", "Riri Williams",
    "Mantis",
    # Key events
    "Battle of New York", "Infinity War", "The Snap", "Time Heist",
    "Battle of Earth", "Multiverse", "Decimation",
    # Key objects & places
    "Infinity Gauntlet", "Infinity Stones", "Mind Stone", "Soul Stone",
    "Time Stone", "Space Stone", "Reality Stone", "Power Stone",
    "Wakanda", "Asgard", "Quantum Realm", "Ten Rings",
    "S.H.I.E.L.D.", "HYDRA", "Avengers Compound", "Sanctum Sanctorum",
    # Phase 4 Disney+ shows (major MCU content, previously missing)
    "WandaVision", "The Falcon and the Winter Soldier", "Loki",
    "Hawkeye", "Ms. Marvel", "Moon Knight", "She-Hulk: Attorney at Law",
    # Films (plot summaries)
    "The Avengers", "Avengers: Age of Ultron", "Avengers: Infinity War",
    "Avengers: Endgame", "Spider-Man: No Way Home",
    "Doctor Strange in the Multiverse of Madness",
    # ── Post-cutoff content (after Dec 2023) — LLM blind spots, ideal for RAG demo ──
    # Films / shows — Phase 5
    "Secret Invasion",                       # Jun 2023
    "Deadpool & Wolverine",              # Aug 2024
    "Captain America: Brave New World",  # Feb 2025
    "Thunderbolts*",                     # May 2025
    # Disney+ shows — Phase 5
    "Echo",                              # Jan 2024
    "Agatha All Along",                  # Sep 2024
    "Daredevil: Born Again",             # Mar 2025
    # Post-cutoff characters — Phase 5
    "Cassandra Nova",                    # main villain, Deadpool & Wolverine
    "Agatha Harkness",                   # lead, Agatha All Along
    "Matthew Murdock",                   # Daredevil: Born Again
    "Bob Reynolds",                      # Sentry, Thunderbolts*
    "Thaddeus Ross",                     # Red Hulk, Captain America: Brave New World
    "Valentina Allegra de Fontaine",     # director, Thunderbolts*
    # ── Phase 6 (further beyond LLM cutoff) ──
    "The Fantastic Four: First Steps",   # Jul 2025 — Pedro Pascal as Mr. Fantastic
    "Avengers: Doomsday",               # May 2026 — RDJ returns as Doctor Doom
    # Phase 6 characters
    "Reed Richards",                     # Mr. Fantastic (Pedro Pascal)
    "Susan Storm",                       # Invisible Woman (Vanessa Kirby)
    "Johnny Storm",                      # Human Torch (Joseph Quinn)
    "Ben Grimm",                         # The Thing (Ebon Moss-Bachrach)
    "Victor von Doom",                   # Doctor Doom — RDJ's shocking return
]

# ── HuggingFace datasets ──────────────────────────────────────────────────────
# Note: jrtec/Superheroes removed — too generic, causes retrieval noise.
HF_DATASETS = [
    {
        "name":    "Manvith/Marvel_dataset",
        "split":   "train",
        "handler": "marvel",
    },
    {
        "name":    "ismaildlml/Jarvis-MCU-Dialogues",
        "split":   "train",
        "handler": "dialogues",
    },
    # rohitsaxena/MovieSum removed — returned 0 MCU docs (title column mismatch, 1800 non-MCU skipped)
]

# ── MCU film title list (used to filter MovieSum dataset) ────────────────────
MCU_FILM_TITLES = {
    t.lower() for films in {
        1: ["Iron Man", "The Incredible Hulk", "Iron Man 2", "Thor",
            "Captain America: The First Avenger", "The Avengers"],
        2: ["Iron Man 3", "Thor: The Dark World", "Captain America: The Winter Soldier",
            "Guardians of the Galaxy", "Avengers: Age of Ultron", "Ant-Man"],
        3: ["Captain America: Civil War", "Doctor Strange", "Guardians of the Galaxy Vol. 2",
            "Spider-Man: Homecoming", "Thor: Ragnarok", "Black Panther",
            "Avengers: Infinity War", "Ant-Man and the Wasp", "Captain Marvel",
            "Avengers: Endgame", "Spider-Man: Far From Home"],
        4: ["Black Widow", "Shang-Chi and the Legend of the Ten Rings", "Eternals",
            "Spider-Man: No Way Home", "Doctor Strange in the Multiverse of Madness",
            "Thor: Love and Thunder", "Black Panther: Wakanda Forever"],
        5: ["Ant-Man and the Wasp: Quantumania", "Guardians of the Galaxy Vol. 3",
            "The Marvels", "Deadpool & Wolverine",
            "Captain America: Brave New World", "Thunderbolts*"],
        6: ["The Fantastic Four: First Steps", "Avengers: Doomsday"],
    }.values()
    for t in films
}

# ── MCU Phase metadata (for filtering) ───────────────────────────────────────
MCU_PHASES = {
    1: ["Iron Man", "The Incredible Hulk", "Iron Man 2", "Thor",
        "Captain America: The First Avenger", "The Avengers"],
    2: ["Iron Man 3", "Thor: The Dark World", "Captain America: The Winter Soldier",
        "Guardians of the Galaxy", "Avengers: Age of Ultron", "Ant-Man"],
    3: ["Captain America: Civil War", "Doctor Strange", "Guardians of the Galaxy Vol. 2",
        "Spider-Man: Homecoming", "Thor: Ragnarok", "Black Panther",
        "Avengers: Infinity War", "Ant-Man and the Wasp", "Captain Marvel",
        "Avengers: Endgame", "Spider-Man: Far From Home"],
    4: ["Black Widow", "Shang-Chi and the Legend of the Ten Rings", "Eternals",
        "Spider-Man: No Way Home", "Doctor Strange in the Multiverse of Madness",
        "Thor: Love and Thunder", "Black Panther: Wakanda Forever"],
    5: ["Ant-Man and the Wasp: Quantumania", "Guardians of the Galaxy Vol. 3",
        "The Marvels", "Deadpool & Wolverine",
        "Captain America: Brave New World", "Thunderbolts*"],
    6: ["The Fantastic Four: First Steps", "Avengers: Doomsday"],
}
