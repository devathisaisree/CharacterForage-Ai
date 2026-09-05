
import os
import json
import sqlite3
import time
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# ============================================================
# CHARACTERFORGE AI - FINAL VERSION
# Gemini = dialogue generation
# ElevenLabs = single voice + multi-speaker voice
# SQLite = local character memory
# ============================================================

load_dotenv()

APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "characters.db"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash").strip()

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "").strip()
ARJUN_VOICE_ID = os.getenv("ARJUN_VOICE_ID", "").strip()
MAYA_VOICE_ID = os.getenv("MAYA_VOICE_ID", "").strip()
ELEVENLABS_MODEL = "eleven_v3"

ELEVENLABS_BASE = "https://api.elevenlabs.io/v1"
ELEVENLABS_TIMEOUT = 120
MAX_DIALOGUE_CHARS = 2000

st.set_page_config(
    page_title="CharacterForge AI",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# DATABASE
# ============================================================

def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS characters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                age TEXT DEFAULT '',
                role TEXT DEFAULT '',
                personality TEXT DEFAULT '',
                backstory TEXT DEFAULT '',
                speech_style TEXT DEFAULT '',
                quirks TEXT DEFAULT '',
                core_values TEXT DEFAULT '',
                example_lines TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()


def list_characters():
    with db() as conn:
        return conn.execute(
            "SELECT * FROM characters ORDER BY name COLLATE NOCASE"
        ).fetchall()


def get_character(name):
    with db() as conn:
        return conn.execute(
            "SELECT * FROM characters WHERE name = ?",
            (name,),
        ).fetchone()


def save_character(data):
    with db() as conn:
        conn.execute(
            """
            INSERT INTO characters
            (
                name, age, role, personality, backstory,
                speech_style, quirks, core_values, example_lines,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(name) DO UPDATE SET
                age = excluded.age,
                role = excluded.role,
                personality = excluded.personality,
                backstory = excluded.backstory,
                speech_style = excluded.speech_style,
                quirks = excluded.quirks,
                core_values = excluded.core_values,
                example_lines = excluded.example_lines,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                data["name"].strip(),
                data.get("age", "").strip(),
                data.get("role", "").strip(),
                data.get("personality", "").strip(),
                data.get("backstory", "").strip(),
                data.get("speech_style", "").strip(),
                data.get("quirks", "").strip(),
                data.get("core_values", "").strip(),
                data.get("example_lines", "").strip(),
            ),
        )
        conn.commit()


def delete_character(name):
    with db() as conn:
        conn.execute("DELETE FROM characters WHERE name = ?", (name,))
        conn.commit()


def row_to_dict(row):
    if row is None:
        return {}
    return {key: row[key] or "" for key in row.keys()}


init_db()


# ============================================================
# DEMO CHARACTERS
# ============================================================

DEMO_CHARACTERS = [
    {
        "name": "Arjun Varma",
        "age": "24",
        "role": "Analytical investigator",
        "personality": (
            "Analytical, calm, serious, cautious, determined, "
            "emotionally controlled."
        ),
        "backstory": (
            "His closest friend betrayed him by selling important "
            "information to a rival."
        ),
        "speech_style": (
            "Clear, measured and serious. Uses short purposeful "
            "sentences. Prefers facts over emotional exaggeration."
        ),
        "quirks": (
            "Asks for evidence, pauses before important decisions, "
            "focuses on details and consequences."
        ),
        "core_values": "Truth, loyalty, evidence, trust, responsibility.",
        "example_lines": (
            '"I don\'t need excuses. I need the truth."\n'
            '"Trust isn\'t something I give away easily."\n'
            '"I\'m angry, but anger doesn\'t change the facts."'
        ),
    },
    {
        "name": "Maya Varma",
        "age": "19",
        "role": "Fearless strategist and loyal confidante",
        "personality": (
            "Confident, empathetic, observant, quick-thinking, "
            "courageous, emotionally intelligent, direct and determined."
        ),
        "backstory": (
            "Grew up alongside Arjun and is one of the few people "
            "he deeply trusts. She relies on observation and intuition."
        ),
        "speech_style": (
            "Clear, direct, confident and conversational. Short "
            "purposeful sentences. Uses pointed questions and becomes "
            "intense when protecting someone."
        ),
        "quirks": (
            'Often says "Think about it", "Be honest with me", and '
            '"We don\'t have time for this". Pauses before important '
            "points and uses occasional dry humor."
        ),
        "core_values": "Loyalty, honesty, justice, courage, trust, protection.",
        "example_lines": (
            '"Don\'t give me a perfect answer. Give me the honest one."\n'
            '"Think about it, Arjun. Something about this doesn\'t add up."\n'
            '"You don\'t have to face this alone."\n'
            '"Be honest with me. What are you really afraid of?"'
        ),
    },
]


def seed_demo_data():
    for character in DEMO_CHARACTERS:
        if get_character(character["name"]) is None:
            save_character(character)


# ============================================================
# GEMINI
# ============================================================

def gemini_client():
    if not GEMINI_API_KEY:
        return None

    return OpenAI(
        api_key=GEMINI_API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )


def call_ai(system_prompt, user_prompt, retries=3):
    client = gemini_client()

    if client is None:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Add it to your .env file."
        )

    last_error = None

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=GEMINI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )

            if not response.choices:
                raise RuntimeError("Gemini returned no response.")

            text = response.choices[0].message.content or ""

            if not text.strip():
                raise RuntimeError("Gemini returned an empty response.")

            return text.strip()

        except Exception as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))

    raise RuntimeError(f"Gemini generation failed: {last_error}")


def call_ai_json(system_prompt, user_prompt):
    text = call_ai(
        system_prompt
        + """

IMPORTANT:
Return ONLY valid JSON.
Do not use markdown code fences.
Do not add explanations before or after the JSON.
""",
        user_prompt,
    )

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    raise RuntimeError(
        "Gemini returned invalid JSON. Please click Generate again."
    )


def profile_prompt(c):
    return f"""
CHARACTER PROFILE

Name:
{c.get("name", "")}

Age:
{c.get("age", "")}

Role:
{c.get("role", "")}

Personality:
{c.get("personality", "")}

Backstory:
{c.get("backstory", "")}

Speech style:
{c.get("speech_style", "")}

Speech quirks:
{c.get("quirks", "")}

Core values / beliefs:
{c.get("core_values", "")}

Example lines:
{c.get("example_lines", "")}
""".strip()


# ============================================================
# ELEVENLABS HELPERS
# ============================================================

def eleven_headers():
    return {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
    }


def eleven_error(response):
    try:
        data = response.json()
        detail = data.get("detail", data)

        if isinstance(detail, dict):
            status = detail.get("status", "")
            message = detail.get("message", "")
            if status and message:
                return f"{status}: {message}"
            return json.dumps(detail)

        return str(detail)
    except Exception:
        return response.text[:1000] or "Unknown ElevenLabs error."


def validate_elevenlabs_setup():
    if not ELEVENLABS_API_KEY:
        return False, "ELEVENLABS_API_KEY is missing."

    try:
        response = requests.get(
            f"{ELEVENLABS_BASE}/models",
            headers=eleven_headers(),
            timeout=20,
        )

        if response.status_code == 200:
            return True, "ElevenLabs API key is valid."

        return False, (
            f"ElevenLabs API check failed ({response.status_code}): "
            f"{eleven_error(response)}"
        )

    except requests.RequestException as exc:
        return False, f"Cannot connect to ElevenLabs: {exc}"


def generate_voice(text, voice_id):
    """Single-speaker ElevenLabs voice."""
    if not ELEVENLABS_API_KEY:
        raise RuntimeError("ELEVENLABS_API_KEY is missing in .env.")

    if not voice_id:
        raise RuntimeError("ElevenLabs voice ID is missing in .env.")

    text = str(text).strip()

    if not text:
        raise RuntimeError("There is no text to convert to voice.")

    url = (
        f"{ELEVENLABS_BASE}/text-to-speech/"
        f"{voice_id}?output_format=mp3_44100_128"
    )

    payload = {
        "text": text,
        "model_id": ELEVENLABS_MODEL,
    }

    try:
        response = requests.post(
            url,
            headers=eleven_headers(),
            json=payload,
            timeout=ELEVENLABS_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"ElevenLabs connection error: {exc}")

    if response.status_code != 200:
        raise RuntimeError(
            f"ElevenLabs voice error ({response.status_code}): "
            f"{eleven_error(response)}"
        )

    if not response.content:
        raise RuntimeError("ElevenLabs returned empty audio.")

    return response.content


def parse_conversation_items(items):
    """
    Convert generated dialogue into ElevenLabs Text-to-Dialogue inputs.
    Only Arjun and Maya are mapped because the project voice demo uses
    these two voices.
    """
    voice_map = {
        "arjun": ARJUN_VOICE_ID,
        "arjun varma": ARJUN_VOICE_ID,
        "maya": MAYA_VOICE_ID,
        "maya varma": MAYA_VOICE_ID,
    }

    inputs = []

    for item in items:
        speaker = str(item.get("speaker", "")).strip()
        line = str(item.get("line", "")).strip()

        if not speaker or not line:
            continue

        voice_id = voice_map.get(speaker.lower())

        if not voice_id:
            continue

        inputs.append(
            {
                "text": line,
                "voice_id": voice_id,
            }
        )

    return inputs


def generate_conversation_voice(items):
    """
    Generate the whole Arjun + Maya conversation using ElevenLabs
    Text-to-Dialogue.

    ElevenLabs recommends keeping the combined text at or below
    2,000 characters per request, so this function rejects oversized
    conversations instead of sending a request that may fail.
    """
    if not ELEVENLABS_API_KEY:
        raise RuntimeError("ELEVENLABS_API_KEY is missing in .env.")

    if not ARJUN_VOICE_ID:
        raise RuntimeError("ARJUN_VOICE_ID is missing in .env.")

    if not MAYA_VOICE_ID:
        raise RuntimeError("MAYA_VOICE_ID is missing in .env.")

    inputs = parse_conversation_items(items)

    if not inputs:
        raise RuntimeError(
            "No Arjun/Maya dialogue lines were found."
        )

    total_chars = sum(len(x["text"]) for x in inputs)

    if total_chars > MAX_DIALOGUE_CHARS:
        raise RuntimeError(
            f"Conversation is {total_chars} characters. "
            f"Keep it at or below {MAX_DIALOGUE_CHARS} characters. "
            f"Use 4 turns for the safest demo."
        )

    payload = {
        "inputs": inputs,
        "model_id": ELEVENLABS_MODEL,
        "seed": 12345,
    }

    url = (
        f"{ELEVENLABS_BASE}/text-to-dialogue"
        "?output_format=mp3_44100_128"
    )

    temporary_errors = {408, 429, 500, 502, 503, 504}
    last_error = None

    for attempt in range(4):
        try:
            response = requests.post(
                url,
                headers=eleven_headers(),
                json=payload,
                timeout=ELEVENLABS_TIMEOUT,
            )

            if response.status_code == 200:
                if not response.content:
                    raise RuntimeError(
                        "ElevenLabs returned empty conversation audio."
                    )
                return response.content

            message = (
                f"ElevenLabs conversation error "
                f"({response.status_code}): {eleven_error(response)}"
            )
            last_error = message

            if response.status_code not in temporary_errors:
                raise RuntimeError(message)

        except requests.RequestException as exc:
            last_error = f"ElevenLabs connection error: {exc}"

        if attempt < 3:
            time.sleep(2 ** attempt)

    raise RuntimeError(last_error or "ElevenLabs request failed.")


# ============================================================
# UI HELPERS
# ============================================================

def render_dialogue(items):
    if not items:
        st.warning("No dialogue was returned.")
        return

    for item in items:
        speaker = item.get("speaker", "Character")
        line = item.get("line", "")
        note = item.get("voice_note", "")

        st.markdown(f"### **{speaker}**")
        st.markdown(f"> {line}")

        if note:
            st.caption(f"Voice fit: {note}")

        st.divider()


def save_last_conversation(items, title=""):
    st.session_state["last_multi_dialogue"] = items
    st.session_state["last_multi_title"] = title


# ============================================================
# SESSION STATE
# ============================================================

if "last_multi_dialogue" not in st.session_state:
    st.session_state["last_multi_dialogue"] = []

if "last_multi_title" not in st.session_state:
    st.session_state["last_multi_title"] = ""


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("⚙️ Setup")

    if GEMINI_API_KEY:
        st.success("Gemini API key detected")
    else:
        st.error("GEMINI_API_KEY missing")

    st.caption(f"Gemini model: `{GEMINI_MODEL}`")

    st.divider()

    if ELEVENLABS_API_KEY:
        st.success("ElevenLabs key detected")
    else:
        st.error("ElevenLabs key missing")

    if ARJUN_VOICE_ID and MAYA_VOICE_ID:
        st.success("Arjun + Maya voice IDs detected")
    else:
        st.warning("Add both voice IDs")

    if st.button(
        "🔎 Test ElevenLabs Connection",
        use_container_width=True,
    ):
        with st.spinner("Checking ElevenLabs..."):
            ok, message = validate_elevenlabs_setup()

        if ok:
            st.success(message)
        else:
            st.error(message)

    st.divider()

    if st.button(
        "🎬 Load Arjun + Maya Demo",
        use_container_width=True,
    ):
        seed_demo_data()
        st.success("Demo characters loaded.")
        st.rerun()

    st.divider()

    rows = list_characters()

    st.subheader("📚 Saved Cast")

    if rows:
        for row in rows:
            st.write(f"• {row['name']}")
    else:
        st.caption("No characters saved yet.")

    st.caption("Characters are stored locally in characters.db.")


# ============================================================
# MAIN
# ============================================================

st.title("🎭 CharacterForge AI")
st.caption(
    "AI-powered fictional character dialogue, voice consistency, "
    "multi-character scenes and AI character voices."
)

tabs = st.tabs(
    [
        "🏠 Dashboard",
        "👤 Character Profiles",
        "✍️ Generate Dialogue",
        "🔍 Consistency Checker",
        "🎬 Multi-Character Scene",
        "📖 Saved Cast",
    ]
)


# ============================================================
# DASHBOARD
# ============================================================

with tabs[0]:
    st.subheader("A writing partner that remembers how characters speak")

    c1, c2, c3 = st.columns(3)
    c1.metric("Characters", len(list_characters()))
    c2.metric("Generation modes", "3")
    c3.metric("Voice engine", "ElevenLabs")

    st.markdown(
        """
### Core workflow

1. Create or load a character profile.
2. Enter a scene.
3. Generate character dialogue with Gemini.
4. Check dialogue consistency.
5. Generate a multi-character scene.
6. Give each character a different ElevenLabs voice.
7. Play the complete conversation.

### Voice architecture

**Gemini → dialogue JSON → speaker detection → ElevenLabs Text-to-Dialogue → MP3 → Streamlit audio player**
"""
    )


# ============================================================
# CHARACTER PROFILES
# ============================================================

with tabs[1]:
    st.subheader("👤 Character Profile")

    existing = ["— New Character —"] + [
        row["name"] for row in list_characters()
    ]

    selected = st.selectbox("Load existing profile", existing)

    loaded = {}
    if selected != "— New Character —":
        loaded = row_to_dict(get_character(selected))

    with st.form("character_form"):
        col1, col2 = st.columns(2)

        name = col1.text_input(
            "Character name *",
            value=loaded.get("name", ""),
        )

        age = col2.text_input(
            "Age",
            value=loaded.get("age", ""),
        )

        role = st.text_input(
            "Role / archetype",
            value=loaded.get("role", ""),
        )

        personality = st.text_area(
            "Personality traits *",
            value=loaded.get("personality", ""),
            height=90,
        )

        backstory = st.text_area(
            "Backstory",
            value=loaded.get("backstory", ""),
            height=100,
        )

        speech_style = st.text_area(
            "Speech style *",
            value=loaded.get("speech_style", ""),
            height=100,
        )

        quirks = st.text_area(
            "Speech quirks",
            value=loaded.get("quirks", ""),
            height=90,
        )

        values = st.text_area(
            "Core values / beliefs",
            value=loaded.get("core_values", ""),
            height=80,
        )

        examples = st.text_area(
            "Example lines",
            value=loaded.get("example_lines", ""),
            height=100,
        )

        submitted = st.form_submit_button(
            "💾 Save Character",
            use_container_width=True,
        )

    if submitted:
        if not name.strip():
            st.error("Character name is required.")
        elif not personality.strip():
            st.error("Personality traits are required.")
        elif not speech_style.strip():
            st.error("Speech style is required.")
        else:
            save_character(
                {
                    "name": name,
                    "age": age,
                    "role": role,
                    "personality": personality,
                    "backstory": backstory,
                    "speech_style": speech_style,
                    "quirks": quirks,
                    "core_values": values,
                    "example_lines": examples,
                }
            )
            st.success(f"Saved profile: {name}")
            st.rerun()


# ============================================================
# SINGLE CHARACTER DIALOGUE
# ============================================================

with tabs[2]:
    st.subheader("✍️ Generate Character Dialogue")

    rows = list_characters()

    if rows:
        names = [row["name"] for row in rows]
        char_name = st.selectbox("Choose character", names)
        char = row_to_dict(get_character(char_name))

        scene = st.text_area(
            "Scene context *",
            placeholder=(
                "Describe what is happening, location, stakes, "
                "and what the character wants."
            ),
            height=140,
        )

        col1, col2 = st.columns(2)

        emotional = col1.slider(
            "Emotional intensity",
            0,
            100,
            50,
        )

        formality = col2.slider(
            "Formality",
            0,
            100,
            50,
        )

        num_lines = st.slider(
            "Number of dialogue lines",
            1,
            8,
            4,
        )

        if st.button(
            "✨ Generate Dialogue",
            type="primary",
            use_container_width=True,
        ):
            if not scene.strip():
                st.error("Please describe the scene.")
            elif not GEMINI_API_KEY:
                st.error("Add GEMINI_API_KEY to .env.")
            else:
                system = """
You are a professional fiction dialogue editor.

Generate dialogue for ONE fictional character.
The character profile is a hard constraint.

Prioritize:
1. Personality and beliefs.
2. Relevant backstory.
3. Speech style, vocabulary, rhythm and quirks.
4. Scene objective and emotional stakes.

Do not sound like a generic AI assistant.
Do not dump backstory into dialogue.
Keep the character distinct and human.

Return:
{
  "dialogue": [
    {
      "speaker": "Character Name",
      "line": "...",
      "voice_note": "..."
    }
  ]
}
""".strip()

                user = f"""
{profile_prompt(char)}

SCENE:
{scene}

Emotional intensity: {emotional}/100
Formality: {formality}/100

Generate exactly {num_lines} natural dialogue lines.
""".strip()

                try:
                    with st.spinner("Writing in character..."):
                        result = call_ai_json(system, user)

                    st.success("Dialogue generated.")
                    render_dialogue(result.get("dialogue", []))

                except Exception as exc:
                    st.error(str(exc))
    else:
        st.info("Load the demo cast or create a character first.")


# ============================================================
# CONSISTENCY CHECKER
# ============================================================

with tabs[3]:
    st.subheader("🔍 Character Voice Consistency Checker")

    rows = list_characters()

    if rows:
        names = [row["name"] for row in rows]

        char_name = st.selectbox(
            "Check dialogue against",
            names,
            key="checker_character",
        )

        char = row_to_dict(get_character(char_name))

        draft = st.text_area(
            "Paste dialogue draft *",
            height=220,
        )

        if st.button(
            "🔎 Check Voice Consistency",
            type="primary",
            use_container_width=True,
        ):
            if not draft.strip():
                st.error("Paste dialogue first.")
            elif not GEMINI_API_KEY:
                st.error("Add GEMINI_API_KEY to .env.")
            else:
                system = """
You are a strict but fair fiction dialogue continuity editor.

Compare the dialogue with the established character profile.

Check:
- personality
- emotional behavior
- vocabulary
- formality
- sentence rhythm
- beliefs and values
- speech quirks
- relevant backstory

Normal emotional variation is not automatically inconsistent.

Return:
{
  "overall_score": 0,
  "summary": "...",
  "checks": [
    {
      "line_number": 1,
      "line": "...",
      "status": "fits",
      "reason": "...",
      "suggestion": "..."
    }
  ]
}

Valid status values:
fits
warning
out_of_character

Score must be 0 to 100.
""".strip()

                user = f"""
{profile_prompt(char)}

DIALOGUE TO CHECK:
{draft}
""".strip()

                try:
                    with st.spinner("Checking character voice..."):
                        result = call_ai_json(system, user)

                    score = int(result.get("overall_score", 0))
                    score = max(0, min(100, score))

                    st.metric(
                        "Voice consistency score",
                        f"{score}/100",
                    )

                    st.write(result.get("summary", ""))

                    for item in result.get("checks", []):
                        status = item.get("status", "warning")

                        icon = {
                            "fits": "✅",
                            "warning": "⚠️",
                            "out_of_character": "❌",
                        }.get(status, "⚠️")

                        st.markdown(
                            f"### {icon} Line "
                            f"{item.get('line_number', '?')}: "
                            f"{item.get('line', '')}"
                        )

                        st.write(
                            f"**Reason:** {item.get('reason', '')}"
                        )

                        suggestion = item.get("suggestion", "")
                        if suggestion:
                            st.info(
                                f"**Suggestion:** {suggestion}"
                            )

                except Exception as exc:
                    st.error(str(exc))
    else:
        st.info("Create or load a character profile first.")


# ============================================================
# MULTI-CHARACTER SCENE + VOICE
# ============================================================

with tabs[4]:
    st.subheader("🎬 Multi-Character Conversation Generator")

    rows = list_characters()

    if len(rows) >= 2:
        names = [row["name"] for row in rows]

        selected_chars = st.multiselect(
            "Select characters",
            names,
            default=names[:2],
            max_selections=5,
        )

        scene = st.text_area(
            "Shared scene context *",
            placeholder=(
                "Example: Arjun and Maya discover that their "
                "closest friend betrayed them at an abandoned "
                "railway station. The rival is arriving soon."
            ),
            height=140,
        )

        relationship = st.text_area(
            "Optional relationship context",
            height=90,
        )

        # 4 turns is the safe default for the voice demo.
        turns = st.slider(
            "Conversation turns",
            2,
            8,
            4,
            help=(
                "Four turns is recommended for the ElevenLabs "
                "voice demo because Text-to-Dialogue works best "
                "with <= 2,000 total characters."
            ),
        )

        if st.button(
            "🎭 Generate Scene",
            type="primary",
            use_container_width=True,
        ):
            if len(selected_chars) < 2:
                st.error("Select at least two characters.")
            elif not scene.strip():
                st.error("Describe the shared scene.")
            elif not GEMINI_API_KEY:
                st.error("Add GEMINI_API_KEY to .env.")
            else:
                profiles = [
                    row_to_dict(get_character(name))
                    for name in selected_chars
                ]

                profile_text = "\n\n".join(
                    profile_prompt(c) for c in profiles
                )

                system = """
You are a screenplay and fiction dialogue specialist.

Generate a multi-character scene where every speaker has
a distinct voice.

Treat every character profile as a hard constraint.
Re-check the correct profile before writing each turn.

Do not make all characters use the same vocabulary,
sentence rhythm, emotional style or level of directness.

Characters may disagree, hesitate, interrupt or react
emotionally when natural.

Use ONLY the exact selected character names.

Return:
{
  "scene_title": "...",
  "dialogue": [
    {
      "speaker": "Exact Character Name",
      "line": "...",
      "voice_note": "..."
    }
  ]
}
""".strip()

                user = f"""
SELECTED CHARACTER PROFILES:

{profile_text}

SHARED SCENE:
{scene}

RELATIONSHIP CONTEXT:
{relationship or "No additional relationship context provided."}

Generate exactly {turns} alternating conversation turns.
Each turn should advance the scene.
Keep the voices sharply distinct.
""".strip()

                try:
                    with st.spinner("Drafting the scene..."):
                        result = call_ai_json(system, user)

                    scene_title = result.get(
                        "scene_title",
                        "Generated Scene",
                    )

                    items = result.get("dialogue", [])

                    # Keep only valid generated dialogue objects.
                    clean_items = []

                    allowed = {
                        name.lower(): name
                        for name in selected_chars
                    }

                    for item in items:
                        speaker = str(
                            item.get("speaker", "")
                        ).strip()

                        line = str(
                            item.get("line", "")
                        ).strip()

                        if (
                            speaker.lower() in allowed
                            and line
                        ):
                            clean_items.append(
                                {
                                    "speaker": allowed[
                                        speaker.lower()
                                    ],
                                    "line": line,
                                    "voice_note": str(
                                        item.get(
                                            "voice_note",
                                            "",
                                        )
                                    ),
                                }
                            )

                    if not clean_items:
                        raise RuntimeError(
                            "Gemini did not return usable dialogue."
                        )

                    save_last_conversation(
                        clean_items,
                        scene_title,
                    )

                    st.markdown(f"## {scene_title}")
                    render_dialogue(clean_items)

                except Exception as exc:
                    st.error(str(exc))

        # --------------------------------------------------------
        # PERSISTENT VOICE AREA
        # --------------------------------------------------------

        saved_items = st.session_state["last_multi_dialogue"]

        if saved_items:
            st.divider()
            st.subheader("🎙️ AI Character Voices")

            voice_col1, voice_col2 = st.columns(2)

            with voice_col1:
                test_arjun = st.button(
                    "🔊 Test Arjun Voice",
                    use_container_width=True,
                    key="test_arjun_voice_final",
                )

            with voice_col2:
                test_maya = st.button(
                    "🔊 Test Maya Voice",
                    use_container_width=True,
                    key="test_maya_voice_final",
                )

            if test_arjun:
                try:
                    with st.spinner(
                        "Generating Arjun voice..."
                    ):
                        audio = generate_voice(
                            "I need the truth. Start from the beginning.",
                            ARJUN_VOICE_ID,
                        )

                    st.audio(audio, format="audio/mpeg")
                    st.success(
                        "Arjun voice works. Press play above."
                    )

                except Exception as exc:
                    st.error(f"Arjun voice failed: {exc}")

            if test_maya:
                try:
                    with st.spinner(
                        "Generating Maya voice..."
                    ):
                        audio = generate_voice(
                            "Be honest with me. We do not have time to waste.",
                            MAYA_VOICE_ID,
                        )

                    st.audio(audio, format="audio/mpeg")
                    st.success(
                        "Maya voice works. Press play above."
                    )

                except Exception as exc:
                    st.error(f"Maya voice failed: {exc}")

            voice_inputs = parse_conversation_items(
                saved_items
            )

            total_chars = sum(
                len(x["text"]) for x in voice_inputs
            )

            st.caption(
                f"ElevenLabs input: {len(voice_inputs)} turns • "
                f"{total_chars} characters • "
                f"limit {MAX_DIALOGUE_CHARS}"
            )

            if total_chars > MAX_DIALOGUE_CHARS:
                st.warning(
                    "This conversation is too long for one "
                    "reliable Text-to-Dialogue request. "
                    "Regenerate with fewer/shorter turns."
                )

            if not ARJUN_VOICE_ID or not MAYA_VOICE_ID:
                st.info(
                    "Add ARJUN_VOICE_ID and MAYA_VOICE_ID "
                    "to .env before using the full conversation voice."
                )
            else:
                if st.button(
                    "▶️ Generate & Play Full Conversation",
                    type="primary",
                    use_container_width=True,
                    key="full_conversation_voice_final",
                ):
                    try:
                        if total_chars > MAX_DIALOGUE_CHARS:
                            raise RuntimeError(
                                "Conversation exceeds 2,000 characters. "
                                "Regenerate using 4 turns or shorter dialogue."
                            )

                        if len(voice_inputs) < 2:
                            raise RuntimeError(
                                "At least two Arjun/Maya turns are required."
                            )

                        with st.spinner(
                            "Generating full conversation with ElevenLabs..."
                        ):
                            audio = generate_conversation_voice(
                                saved_items
                            )

                        st.audio(
                            audio,
                            format="audio/mpeg",
                        )

                        st.success(
                            "Voice conversation generated successfully. "
                            "Press ▶ on the audio player."
                        )

                    except Exception as exc:
                        st.error(
                            "Conversation voice failed: "
                            f"{exc}"
                        )

    else:
        st.info(
            "Create or load at least two characters."
        )


# ============================================================
# SAVED CAST
# ============================================================

with tabs[5]:
    st.subheader("📖 Saved Cast")

    rows = list_characters()

    if rows:
        table = pd.DataFrame(
            [
                {
                    "Name": row["name"],
                    "Role": row["role"],
                    "Personality": row["personality"],
                    "Speech style": row["speech_style"],
                }
                for row in rows
            ]
        )

        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
        )

        st.divider()

        delete_name = st.selectbox(
            "Select character to delete",
            [row["name"] for row in rows],
            key="delete_character",
        )

        if st.button(
            "🗑️ Delete Selected Character",
            key="delete_character_button",
        ):
            delete_character(delete_name)
            st.success(f"Deleted {delete_name}")
            st.rerun()

    else:
        st.info("No saved characters yet.")


# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption(
    "CharacterForge AI • Gemini dialogue generation • "
    "SQLite character memory • ElevenLabs AI voices"
)
