import os
import io
import json
import sqlite3
import time
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

APP_TITLE = "CharacterForge AI"
DB_PATH = Path("characters.db")
MAX_DIALOGUE_CHARS = 2000
MAX_VOICES = 10

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash").strip()
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "").strip()
ELEVENLABS_MODEL = os.getenv("ELEVENLABS_MODEL", "eleven_v3").strip()

st.set_page_config(page_title=APP_TITLE, page_icon="🎭", layout="wide")


# -----------------------------
# Database
# -----------------------------
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS characters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                age TEXT,
                role TEXT,
                personality TEXT,
                backstory TEXT,
                speech_style TEXT,
                quirks TEXT,
                core_values TEXT,
                example_lines TEXT,
                voice_id TEXT DEFAULT '',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(characters)")}
        if "voice_id" not in columns:
            conn.execute("ALTER TABLE characters ADD COLUMN voice_id TEXT DEFAULT ''")
        conn.commit()


def get_characters():
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM characters ORDER BY name COLLATE NOCASE"
        ).fetchall()
    return [dict(row) for row in rows]


def get_character(name):
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM characters WHERE name = ?", (name,)
        ).fetchone()
    return dict(row) if row else None


def save_character(data):
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO characters
            (name, age, role, personality, backstory, speech_style,
             quirks, core_values, example_lines, voice_id, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(name) DO UPDATE SET
                age=excluded.age,
                role=excluded.role,
                personality=excluded.personality,
                backstory=excluded.backstory,
                speech_style=excluded.speech_style,
                quirks=excluded.quirks,
                core_values=excluded.core_values,
                example_lines=excluded.example_lines,
                voice_id=excluded.voice_id,
                updated_at=CURRENT_TIMESTAMP
        """, (
            data["name"], data["age"], data["role"], data["personality"],
            data["backstory"], data["speech_style"], data["quirks"],
            data["core_values"], data["example_lines"], data.get("voice_id", "")
        ))
        conn.commit()


def delete_character(name):
    with get_conn() as conn:
        conn.execute("DELETE FROM characters WHERE name = ?", (name,))
        conn.commit()


init_db()


# -----------------------------
# API helpers
# -----------------------------
def gemini_client():
    if not GEMINI_API_KEY:
        return None
    return OpenAI(
        api_key=GEMINI_API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )


@st.cache_data(ttl=300, show_spinner=False)
def fetch_elevenlabs_voices():
    """Return the voices available to the configured ElevenLabs account."""
    if not ELEVENLABS_API_KEY:
        return []

    try:
        response = requests.get(
            "https://api.elevenlabs.io/v1/voices",
            headers={"xi-api-key": ELEVENLABS_API_KEY},
            params={"show_legacy": "true"},
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()
        voices = data.get("voices", [])

        cleaned = []
        for voice in voices:
            voice_id = voice.get("voice_id")
            name = voice.get("name")
            if voice_id and name:
                labels = voice.get("labels") or {}
                category = voice.get("category") or labels.get("accent") or ""
                cleaned.append({
                    "id": voice_id,
                    "name": name,
                    "category": category,
                    "preview_url": voice.get("preview_url", ""),
                })

        cleaned.sort(key=lambda x: x["name"].lower())
        return cleaned
    except Exception:
        return []


def voice_options():
    voices = fetch_elevenlabs_voices()
    return voices


def voice_label(v):
    return f'{v["name"]} — {v["category"]}' if v["category"] else v["name"]


def selected_voice_id(voices, selected_label):
    for voice in voices:
        if voice_label(voice) == selected_label:
            return voice["id"]
    return ""


def current_voice_label(voices, voice_id):
    for voice in voices:
        if voice["id"] == voice_id:
            return voice_label(voice)
    return None


def call_gemini(system_prompt, user_prompt, temperature=0.8, retries=3):
    client = gemini_client()
    if client is None:
        raise RuntimeError("GEMINI_API_KEY is missing. Add it to your .env file.")

    last_error = None
    for attempt in range(retries):
        try:
            result = client.chat.completions.create(
                model=GEMINI_MODEL,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return result.choices[0].message.content
        except Exception as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    raise last_error


def extract_json(text):
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.replace("```json", "", 1).replace("```", "", 1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


# -----------------------------
# Dialogue generation
# -----------------------------
def build_character_context(characters):
    blocks = []
    for c in characters:
        blocks.append(
            f"""
CHARACTER: {c['name']}
Age: {c.get('age', '')}
Role: {c.get('role', '')}
Personality: {c.get('personality', '')}
Backstory: {c.get('backstory', '')}
Speech style: {c.get('speech_style', '')}
Quirks: {c.get('quirks', '')}
Core values: {c.get('core_values', '')}
Example lines: {c.get('example_lines', '')}
""".strip()
        )
    return "\n\n".join(blocks)


def generate_single_dialogue(character, scene, emotion, turns, language):
    system = f"""
You are CharacterForge AI, a professional dialogue writer.
Write dialogue strictly in the personality and speech style of the supplied character.
Do not change the character's identity or invent conflicting traits.
The requested output language is {language}.
Return valid JSON only.
Schema:
{{
  "scene_title": "string",
  "dialogue": [
    {{"speaker": "{character['name']}", "line": "string", "voice_note": "short emotion/delivery note"}}
  ]
}}
""".strip()

    user = f"""
CHARACTER PROFILE:
{build_character_context([character])}

SCENE:
{scene}

EMOTION / TONE:
{emotion}

NUMBER OF TURNS:
{turns}

Generate natural, readable dialogue.
""".strip()

    return extract_json(call_gemini(system, user))


def generate_multi_dialogue(characters, scene, relationship, emotion, turns, language):
    names = [c["name"] for c in characters]
    system = f"""
You are CharacterForge AI, a professional multi-character dialogue writer.

Selected characters, in this exact set:
{", ".join(names)}

Rules:
1. Use ONLY these characters as speakers.
2. Preserve every character's personality, backstory, values and speech style.
3. Make the voices clearly distinct.
4. Do not rename speakers.
5. Include natural reactions, interruptions or disagreement where appropriate.
6. Keep the scene coherent.
7. Write in {language}.
8. Return valid JSON only.

Schema:
{{
  "scene_title": "string",
  "dialogue": [
    {{"speaker": "EXACT CHARACTER NAME", "line": "string", "voice_note": "short delivery note"}}
  ]
}}
""".strip()

    user = f"""
CHARACTER PROFILES:
{build_character_context(characters)}

SCENE:
{scene}

RELATIONSHIP / DYNAMIC:
{relationship}

EMOTION / TONE:
{emotion}

NUMBER OF TURNS:
{turns}
""".strip()

    data = extract_json(call_gemini(system, user))

    allowed = set(names)
    cleaned = []
    for item in data.get("dialogue", []):
        speaker = str(item.get("speaker", "")).strip()
        line = str(item.get("line", "")).strip()
        if speaker in allowed and line:
            cleaned.append({
                "speaker": speaker,
                "line": line,
                "voice_note": str(item.get("voice_note", "")).strip()
            })

    data["dialogue"] = cleaned
    return data


def normalize_dialogue(data):
    if not isinstance(data, dict):
        return {"scene_title": "Generated Scene", "dialogue": []}
    dialogue = []
    for item in data.get("dialogue", []):
        if isinstance(item, dict):
            dialogue.append({
                "speaker": str(item.get("speaker", "")).strip(),
                "line": str(item.get("line", "")).strip(),
                "voice_note": str(item.get("voice_note", "")).strip(),
            })
    data["dialogue"] = dialogue
    return data


# -----------------------------
# Consistency checker
# -----------------------------
def check_consistency(character, dialogue_text):
    system = """
You are a fictional-character consistency checker.
Compare the generated dialogue with the supplied character profile.
Return valid JSON only:
{
  "score": 0,
  "issues": ["..."],
  "suggestions": ["..."]
}
Score from 0 to 100 based only on consistency with the supplied profile.
Do not judge grammar or literary quality unless it affects character consistency.
""".strip()

    user = f"""
CHARACTER PROFILE:
{build_character_context([character])}

GENERATED DIALOGUE:
{dialogue_text}
""".strip()

    return extract_json(call_gemini(system, user, temperature=0.2))


# -----------------------------
# ElevenLabs audio
# -----------------------------
def parse_conversation_items(data):
    items = []
    for item in data.get("dialogue", []):
        speaker = item.get("speaker", "").strip()
        line = item.get("line", "").strip()
        if speaker and line:
            items.append((speaker, line))
    return items


def generate_conversation_voice(data, voice_map):
    if not ELEVENLABS_API_KEY:
        raise RuntimeError("ELEVENLABS_API_KEY is missing. Add it to .env.")

    items = parse_conversation_items(data)
    if not items:
        raise ValueError("No dialogue lines were generated.")

    missing = sorted({speaker for speaker, _ in items if not voice_map.get(speaker)})
    if missing:
        raise ValueError("Select a voice for: " + ", ".join(missing))

    unique_ids = {voice_map[speaker] for speaker, _ in items}
    if len(unique_ids) > MAX_VOICES:
        raise ValueError(f"ElevenLabs supports at most {MAX_VOICES} unique voices per request.")

    total_chars = sum(len(text) for _, text in items)
    if total_chars > MAX_DIALOGUE_CHARS:
        raise ValueError(
            f"Dialogue is {total_chars} characters. Please keep it at or below "
            f"{MAX_DIALOGUE_CHARS} characters for one audio request."
        )

    payload = {
        "model_id": ELEVENLABS_MODEL,
        "inputs": [
            {"text": text, "voice_id": voice_map[speaker]}
            for speaker, text in items
        ]
    }

    response = requests.post(
        "https://api.elevenlabs.io/v1/text-to-dialogue",
        headers={
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    return response.content


# -----------------------------
# Export helpers
# -----------------------------
def dialogue_as_text(data):
    title = data.get("scene_title", "Generated Scene")
    lines = [title, "=" * len(title), ""]
    for item in data.get("dialogue", []):
        note = f" [{item['voice_note']}]" if item.get("voice_note") else ""
        lines.append(f"{item['speaker']}: {item['line']}{note}")
    return "\n".join(lines)


def make_pdf(data):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.units import inch
    except ImportError as exc:
        raise RuntimeError("Install reportlab with: pip install reportlab") from exc

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = [
        Paragraph(data.get("scene_title", "Generated Scene"), styles["Title"]),
        Spacer(1, 0.2 * inch),
    ]

    for item in data.get("dialogue", []):
        speaker = item.get("speaker", "")
        line = item.get("line", "")
        note = item.get("voice_note", "")
        story.append(Paragraph(f"<b>{speaker}:</b> {line}", styles["BodyText"]))
        if note:
            story.append(Paragraph(f"<i>Delivery: {note}</i>", styles["BodyText"]))
        story.append(Spacer(1, 0.1 * inch))

    doc.build(story)
    return buffer.getvalue()


# -----------------------------
# UI
# -----------------------------
st.title("🎭 CharacterForge AI")
st.caption("Create fictional characters, generate consistent dialogue, and turn multi-character scenes into voice.")

with st.sidebar:
    st.header("⚙️ Configuration")
    st.write("Gemini:", "✅ Ready" if GEMINI_API_KEY else "❌ Missing API key")
    st.write("ElevenLabs:", "✅ Ready" if ELEVENLABS_API_KEY else "❌ Missing API key")
    st.caption("API keys are read from your local .env file.")

    # -----------------------------
    # Saved Cast / Characters
    # -----------------------------
    st.divider()
    st.subheader("📚 Saved Cast")

    sidebar_characters = get_characters()

    if sidebar_characters:
        for character in sidebar_characters:
            st.markdown(f"• **{character['name']}**")
    else:
        st.caption("No characters saved yet.")

    st.caption("Characters are stored locally in characters.db.")

tabs = st.tabs([
    "👤 Character Profiles",
    "💬 Single Character",
    "🎬 Multi-Character",
    "🔎 Consistency Checker",
    "📚 Saved Characters",
])

# Character Profiles
with tabs[0]:
    st.subheader("Create / Update a Character")
    voices = voice_options()

    if not ELEVENLABS_API_KEY:
        st.info("Add ELEVENLABS_API_KEY to .env to enable voice selection.")
    elif not voices:
        st.warning("No ElevenLabs voices could be loaded. Check your API key and internet connection.")

    existing = get_characters()
    existing_names = [c["name"] for c in existing]

    selected_existing = st.selectbox(
        "Edit an existing character (optional)",
        ["New character"] + existing_names
    )

    old = get_character(selected_existing) if selected_existing != "New character" else {}

    with st.form("character_form"):
        name = st.text_input("Character name *", value=old.get("name", ""))
        age = st.text_input("Age", value=old.get("age", ""))
        role = st.text_input("Role", value=old.get("role", ""))
        personality = st.text_area("Personality *", value=old.get("personality", ""))
        backstory = st.text_area("Backstory", value=old.get("backstory", ""))
        speech_style = st.text_area(
            "Speech style",
            value=old.get("speech_style", ""),
            placeholder="Example: short sentences, formal, sarcastic, uses simple words..."
        )
        quirks = st.text_area("Quirks / habits", value=old.get("quirks", ""))
        core_values = st.text_area("Core values", value=old.get("core_values", ""))
        example_lines = st.text_area("Example lines", value=old.get("example_lines", ""))

        voice_id = old.get("voice_id", "")
        if voices:
            labels = [voice_label(v) for v in voices]
            previous = current_voice_label(voices, voice_id)
            index = labels.index(previous) if previous in labels else 0
            voice_choice = st.selectbox("Voice", labels, index=index)
            voice_id = selected_voice_id(voices, voice_choice)
        else:
            st.text_input(
                "Voice",
                value="Voice selection unavailable",
                disabled=True
            )

        save = st.form_submit_button("💾 Save Character", use_container_width=True)

    if save:
        if not name.strip() or not personality.strip():
            st.error("Character name and personality are required.")
        else:
            save_character({
                "name": name.strip(),
                "age": age.strip(),
                "role": role.strip(),
                "personality": personality.strip(),
                "backstory": backstory.strip(),
                "speech_style": speech_style.strip(),
                "quirks": quirks.strip(),
                "core_values": core_values.strip(),
                "example_lines": example_lines.strip(),
                "voice_id": voice_id,
            })
            st.success(f"Character '{name.strip()}' saved successfully.")
            st.rerun()

# Single Character
with tabs[1]:
    st.subheader("Generate Single-Character Dialogue")
    characters = get_characters()

    if not characters:
        st.info("Create a character first.")
    else:
        names = [c["name"] for c in characters]
        name = st.selectbox("Character", names, key="single_character")
        character = get_character(name)

        scene = st.text_area(
            "Scene / situation",
            placeholder="Example: The character discovers that their closest friend lied to them."
        )
        emotion = st.text_input("Emotion / tone", value="natural and emotionally believable")
        turns = st.slider("Number of turns", 2, 20, 8)
        language = st.selectbox("Output language", ["English", "Telugu", "Hindi"])

        if st.button("✨ Generate Dialogue", key="single_generate", use_container_width=True):
            if not scene.strip():
                st.error("Enter a scene first.")
            else:
                with st.spinner("Generating dialogue..."):
                    try:
                        st.session_state["last_dialogue"] = normalize_dialogue(
                            generate_single_dialogue(character, scene, emotion, turns, language)
                        )
                        st.success("Dialogue generated.")
                    except Exception as exc:
                        st.error(f"Generation failed: {exc}")

# Multi Character
with tabs[2]:
    st.subheader("🎬 Multi-Character Scene")
    characters = get_characters()

    if len(characters) < 2:
        st.info("Create at least 2 characters to generate a multi-character scene.")
    else:
        names = [c["name"] for c in characters]
        selected = st.multiselect(
            "Select 2–5 characters",
            names,
            max_selections=5,
            key="multi_characters"
        )

        scene = st.text_area(
            "Scene / situation",
            placeholder="Example: Two friends confront a detective about a missing document."
        )
        relationship = st.text_input(
            "Relationship / dynamic",
            placeholder="Example: close friends with hidden tension"
        )
        emotion = st.text_input("Overall emotion / tone", value="natural, dramatic")
        turns = st.slider("Number of turns", 4, 30, 12, key="multi_turns")
        language = st.selectbox(
            "Output language",
            ["English", "Telugu", "Hindi"],
            key="multi_language"
        )

        if st.button("🎬 Generate Multi-Character Dialogue", use_container_width=True):
            if len(selected) < 2:
                st.error("Select at least 2 characters.")
            elif not scene.strip():
                st.error("Enter a scene first.")
            else:
                selected_chars = [get_character(n) for n in selected]
                with st.spinner("Generating multi-character dialogue..."):
                    try:
                        st.session_state["last_dialogue"] = normalize_dialogue(
                            generate_multi_dialogue(
                                selected_chars, scene, relationship, emotion, turns, language
                            )
                        )
                        st.success("Multi-character dialogue generated.")
                    except Exception as exc:
                        st.error(f"Generation failed: {exc}")

# Display latest dialogue
if st.session_state.get("last_dialogue"):
    data = normalize_dialogue(st.session_state["last_dialogue"])
    st.divider()
    st.subheader("📝 Generated Output")

    for item in data["dialogue"]:
        st.markdown(f"**{item['speaker']}:** {item['line']}")
        if item.get("voice_note"):
            st.caption(f"🎙️ {item['voice_note']}")

    st.download_button(
        "⬇️ Download TXT",
        dialogue_as_text(data).encode("utf-8"),
        file_name="characterforge_dialogue.txt",
        mime="text/plain"
    )

    try:
        pdf_bytes = make_pdf(data)
        st.download_button(
            "📄 Download PDF",
            pdf_bytes,
            file_name="characterforge_dialogue.pdf",
            mime="application/pdf"
        )
    except Exception:
        st.caption("PDF export requires reportlab.")

    # Dynamic voice selection for the generated speakers
    available_voices = voice_options()
    if available_voices:
        st.subheader("🎙️ Choose voices for this scene")
        voice_map = {}

        for speaker in sorted({x["speaker"] for x in data["dialogue"]}):
            character = get_character(speaker)
            saved_id = character.get("voice_id", "") if character else ""
            previous = current_voice_label(available_voices, saved_id)
            labels = [voice_label(v) for v in available_voices]
            idx = labels.index(previous) if previous in labels else 0

            choice = st.selectbox(
                f"Voice for {speaker}",
                labels,
                index=idx,
                key=f"generated_voice_{speaker}"
            )
            voice_map[speaker] = selected_voice_id(available_voices, choice)

        if st.button("🔊 Generate Voice Conversation", use_container_width=True):
            with st.spinner("Creating voice conversation..."):
                try:
                    audio_bytes = generate_conversation_voice(data, voice_map)
                    st.audio(audio_bytes, format="audio/mpeg")
                    st.download_button(
                        "⬇️ Download MP3",
                        audio_bytes,
                        file_name="characterforge_conversation.mp3",
                        mime="audio/mpeg"
                    )
                except Exception as exc:
                    st.error(f"Voice generation failed: {exc}")
    elif ELEVENLABS_API_KEY:
        st.warning("No ElevenLabs voices are available. Check the API key.")

# Consistency Checker
with tabs[3]:
    st.subheader("🔎 Character Consistency Checker")
    characters = get_characters()

    if not characters:
        st.info("Create a character first.")
    else:
        name = st.selectbox(
            "Character to check",
            [c["name"] for c in characters],
            key="consistency_character"
        )
        character = get_character(name)
        text = st.text_area(
            "Paste generated dialogue",
            height=220,
            placeholder="Paste the character's dialogue here..."
        )

        if st.button("🔎 Check Consistency", use_container_width=True):
            if not text.strip():
                st.error("Paste dialogue first.")
            else:
                with st.spinner("Checking character consistency..."):
                    try:
                        result = check_consistency(character, text)
                        score = int(result.get("score", 0))
                        st.metric("Consistency Score", f"{score}/100")

                        if result.get("issues"):
                            st.markdown("**Issues**")
                            for issue in result["issues"]:
                                st.write("•", issue)

                        if result.get("suggestions"):
                            st.markdown("**Suggestions**")
                            for suggestion in result["suggestions"]:
                                st.write("•", suggestion)
                    except Exception as exc:
                        st.error(f"Consistency check failed: {exc}")

# Saved Characters
with tabs[4]:
    st.subheader("📚 Saved Characters")
    characters = get_characters()

    if not characters:
        st.info("No characters saved yet.")
    else:
        rows = []
        for c in characters:
            rows.append({
                "Name": c["name"],
                "Age": c["age"],
                "Role": c["role"],
                "Personality": c["personality"],
                "Voice": "Assigned" if c.get("voice_id") else "Not assigned",
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        delete_name = st.selectbox(
            "Delete character",
            [c["name"] for c in characters],
            key="delete_character"
        )
        if st.button("🗑️ Delete Character"):
            delete_character(delete_name)
            st.success(f"Deleted {delete_name}.")
            st.rerun()

st.divider()
st.caption("CharacterForge AI • Dynamic characters • Multi-character dialogue • Optional ElevenLabs voice generation")
