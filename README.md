# 🎭 CharacterForge AI

## AI-Powered Fictional Character Dialogue & Voice Generator for Writers

CharacterForge AI is an AI-powered application designed to help writers create **personality-consistent fictional character dialogues and multi-character voice conversations**.

The application allows users to create detailed character profiles, provide scene and relationship context, generate AI-powered dialogue, check character consistency, and convert conversations into realistic AI-generated voices.

---

## ✨ Features

### 👤 1. Character Profile Creation

Users can create and manage multiple fictional characters with detailed profiles, including:

- Character Name
- Age
- Role / Archetype
- Personality
- Backstory
- Speech Style
- Quirks
- Core Values
- Example Dialogue Lines
- ElevenLabs Voice ID

Each character can have a unique personality, speaking style, and voice.

---

### 📝 2. AI Dialogue Generation

The system generates dialogue based on the character profile and scene context.

Users can provide:

- Character Profile
- Scene Context
- Relationship Context
- Desired Number of Dialogue Lines

The AI considers:

- Personality
- Backstory
- Speech patterns
- Core values
- Character quirks
- Relationships
- Scene situation

This helps generate dialogue that feels consistent with the selected character.

---

### 🔍 3. Character Consistency Checker

The application analyzes generated dialogue and checks whether the lines are consistent with the character's personality and established profile.

It can identify:

- Out-of-character dialogue
- Personality mismatches
- Inconsistent speech patterns
- Conflicts with character values

It also provides a brief explanation for why a line may not fit the character.

---

### 🎬 4. Multi-Character Conversation Generator

CharacterForge AI supports conversations between multiple fictional characters.

Users can select multiple saved character profiles and provide a scene.

The AI generates a conversation while considering each character's individual:

- Personality
- Speech style
- Background
- Values
- Quirks
- Relationship with other characters

Example:

```text
Arjun Varma: I don't need excuses. I need the truth.

Maya Varma: Think about it, Arjun. Something about this doesn't add up.

Arjun Varma: Anger won't help us. We need to understand what happened.

Maya Varma: Then let's find the truth together.

🎙️ 5. AI Character Voice Generation

Each character can be assigned a unique ElevenLabs Voice ID.

The application dynamically maps each speaker to their assigned voice.

Character
    ↓
Voice ID
    ↓
ElevenLabs
    ↓
Generated Speech

For example:

Arjun Varma → Voice A
Maya Varma  → Voice B
Rahul       → Voice C
Ananya      → Voice D

This allows different characters to have different voices in the same conversation.

🔊 6. Multi-Voice Conversation

The complete generated conversation can be converted into an audio dialogue.

The system identifies each speaker and automatically selects their corresponding ElevenLabs voice.

Example:

Arjun Varma: "We need to stay calm."

        ↓

Arjun Varma → Arjun Voice ID

        ↓

ElevenLabs

        ↓

Generated Audio

The same process is performed for every character in the conversation.

💾 7. Character Management

Character profiles are stored locally using SQLite.

Users can:

Create characters
Save characters
Edit characters
Reuse characters
Manage multiple character profiles
Assign different voices
View saved character information

This allows writers to maintain a reusable character library.

🏗️ System Architecture
                    ┌───────────────────────┐
                    │     Streamlit UI      │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Character Profiles  │
                    │   Scene Context       │
                    │   Relationships       │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │       Gemini AI       │
                    │   Dialogue Generation │
                    └───────────┬───────────┘
                                │
                    ┌───────────┴────────────┐
                    │                        │
                    ▼                        ▼
          ┌──────────────────┐     ┌───────────────────┐
          │  Consistency     │     │ Multi-Character   │
          │     Checker      │     │   Conversation    │
          └──────────────────┘     └─────────┬─────────┘
                                             │
                                             ▼
                                  ┌─────────────────────┐
                                  │     ElevenLabs      │
                                  │   Voice Generation  │
                                  └──────────┬──────────┘
                                             │
                                             ▼
                                  🔊 Multi-Voice Audio
🔄 Application Workflow
Create Character
       ↓
Save Character Profile
       ↓
Add Scene Context
       ↓
Add Relationship Context
       ↓
Generate Dialogue
       ↓
Check Character Consistency
       ↓
Generate Multi-Character Conversation
       ↓
Map Characters to Voice IDs
       ↓
ElevenLabs Voice Generation
       ↓
Play Audio Conversation
🧠 AI Dialogue Pipeline

The dialogue generation process follows this pipeline:

Character Profile
       +
Scene Context
       +
Relationship Context
       +
Writing Instructions
       ↓
     Gemini AI
       ↓
Generated Dialogue
       ↓
Consistency Analysis
       ↓
Final Character Dialogue

The character profile acts as the foundation for maintaining the character's identity and speaking style.

🎙️ Voice Generation Pipeline
Generated Conversation
          ↓
    Speaker Detection
          ↓
 Character Name Matching
          ↓
    Voice ID Lookup
          ↓
     ElevenLabs API
          ↓
    Audio Generation
          ↓
      Audio Playback
🛠️ Technologies Used
Technology	Purpose
Python	Core application logic
Streamlit	Web application interface
Google Gemini	AI dialogue generation
ElevenLabs	AI voice generation
SQLite	Character profile storage
Requests	API communication
python-dotenv	Environment variable management
📁 Project Structure
CharacterForge-AI/
│
├── app.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
│
└── characters.db

characters.db is created locally when the application runs.
