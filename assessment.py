"""
assessment.py
─────────────────────────────────────────────────────────────────────────────
Assessment engine: Ikigai-structured multi-select questions + hidden
behavioral signal tracking. Users see a natural conversation; the system
silently builds a behavioral fingerprint in the background.

Behavioral dimensions tracked:
  structure_preference   — how much the user favors order vs ambiguity
  risk_tolerance         — comfort with uncertainty and novelty
  motivation_type        — intrinsic (passion) vs extrinsic (reward)
  social_orientation     — solo deep work vs collaborative energy
  consistency_score      — do stated preferences align across questions?
  ambiguity_tolerance    — comfort selecting fewer vs more options
  analytical_bias        — logic-first vs feeling-first patterns
  execution_bias         — ideas person vs finisher
"""

import numpy as np
import textwrap

FEATURE_NAMES = [
    "creativity", "analytical", "social", "technical", "leadership",
    "detail_oriented", "business", "science", "arts", "physical",
    "communication", "market_demand"
]

IKIGAI_QUADRANTS = ["love", "good_at", "world_needs", "paid_for"]

BEHAVIORAL_DIMS = [
    "structure_preference",
    "risk_tolerance",
    "motivation_type",
    "social_orientation",
    "ambiguity_tolerance",
    "analytical_bias",
    "execution_bias",
    "curiosity_breadth",
]

# ─────────────────────────────────────────────────────────────────────────────
# QUESTION BANK
# Each option carries:
#   weights   → 12D feature vector delta
#   ikigai    → [love, good_at, world_needs, paid_for] delta
#   behavioral→ 8D behavioral signal delta
# ─────────────────────────────────────────────────────────────────────────────

QUESTIONS = [
    # ── Q1: WHAT YOU LOVE (Ikigai) ───────────────────────────────────────────
    {
        "id": "love",
        "quadrant": "love",
        "category": "❤️  What You Love",
        "question": "Which of these genuinely light you up? (Select all that apply)",
        "options": [
            {"label": "Building & engineering things",
             "weights":   [0.2,0.5,0.1,0.9,0.1,0.6,0.1,0.5,0.1,0.5,0.1,0.0],
             "ikigai":    [0.9,0.3,0.2,0.2],
             "behavioral":[0.7,0.3,0.3,0.3,0.4,0.6,0.8,0.4]},
            {"label": "Creating art, music, or visual work",
             "weights":   [0.9,0.2,0.4,0.2,0.1,0.5,0.1,0.1,0.9,0.1,0.5,0.0],
             "ikigai":    [0.9,0.2,0.2,0.2],
             "behavioral":[0.2,0.7,0.4,0.4,0.7,0.2,0.5,0.8]},
            {"label": "Helping & deeply connecting with people",
             "weights":   [0.3,0.2,0.9,0.1,0.5,0.2,0.4,0.1,0.2,0.3,0.9,0.0],
             "ikigai":    [0.9,0.2,0.5,0.2],
             "behavioral":[0.3,0.4,0.2,0.9,0.5,0.3,0.5,0.6]},
            {"label": "Researching & discovering new things",
             "weights":   [0.3,0.9,0.2,0.6,0.2,0.9,0.2,0.9,0.2,0.1,0.4,0.0],
             "ikigai":    [0.9,0.3,0.3,0.1],
             "behavioral":[0.4,0.6,0.8,0.2,0.6,0.8,0.3,0.9]},
            {"label": "Leading teams & making big decisions",
             "weights":   [0.3,0.5,0.7,0.3,0.9,0.5,0.8,0.2,0.2,0.2,0.8,0.0],
             "ikigai":    [0.8,0.4,0.3,0.3],
             "behavioral":[0.6,0.6,0.5,0.7,0.5,0.5,0.7,0.5]},
            {"label": "Solving complex logical puzzles",
             "weights":   [0.3,0.9,0.2,0.8,0.2,0.8,0.3,0.7,0.1,0.1,0.3,0.0],
             "ikigai":    [0.8,0.4,0.2,0.2],
             "behavioral":[0.8,0.3,0.3,0.2,0.3,0.9,0.6,0.7]},
        ]
    },

    # ── Q2: WHAT YOU'RE GOOD AT (Ikigai) ─────────────────────────────────────
    {
        "id": "good_at",
        "quadrant": "good_at",
        "category": "💪 What You're Good At",
        "question": "What do people actually come to you for? (Select all that apply)",
        "options": [
            {"label": "Math, logic, and analytical thinking",
             "weights":   [0.2,1.0,0.2,0.7,0.2,0.8,0.4,0.8,0.1,0.1,0.3,0.0],
             "ikigai":    [0.2,0.9,0.2,0.3],
             "behavioral":[0.8,0.3,0.2,0.2,0.2,0.9,0.7,0.6]},
            {"label": "Writing, storytelling, communication",
             "weights":   [0.7,0.4,0.6,0.2,0.4,0.6,0.4,0.2,0.7,0.1,0.9,0.0],
             "ikigai":    [0.3,0.9,0.2,0.3],
             "behavioral":[0.3,0.5,0.3,0.6,0.6,0.4,0.5,0.7]},
            {"label": "Designing and visualizing ideas",
             "weights":   [0.9,0.4,0.4,0.5,0.2,0.7,0.3,0.1,1.0,0.1,0.5,0.0],
             "ikigai":    [0.3,0.9,0.1,0.2],
             "behavioral":[0.2,0.7,0.4,0.4,0.7,0.3,0.6,0.8]},
            {"label": "Organizing, planning, and executing",
             "weights":   [0.2,0.7,0.5,0.4,0.8,0.9,0.7,0.2,0.2,0.1,0.5,0.0],
             "ikigai":    [0.2,0.9,0.2,0.4],
             "behavioral":[0.9,0.3,0.2,0.4,0.2,0.6,0.9,0.3]},
            {"label": "Building and debugging technical systems",
             "weights":   [0.3,0.8,0.2,1.0,0.3,0.8,0.2,0.6,0.1,0.4,0.3,0.0],
             "ikigai":    [0.2,0.9,0.3,0.4],
             "behavioral":[0.7,0.4,0.4,0.2,0.3,0.8,0.8,0.5]},
            {"label": "Persuading, negotiating, influencing",
             "weights":   [0.5,0.5,0.8,0.2,0.8,0.4,0.8,0.1,0.3,0.2,0.9,0.0],
             "ikigai":    [0.3,0.8,0.2,0.4],
             "behavioral":[0.4,0.7,0.3,0.8,0.5,0.5,0.7,0.5]},
        ]
    },

    # ── Q3: WORLD NEEDS (Ikigai) ──────────────────────────────────────────────
    {
        "id": "world_needs",
        "quadrant": "world_needs",
        "category": "🌍 What the World Needs",
        "question": "Which problems feel most urgent to you? (Select all that apply)",
        "options": [
            {"label": "Healthcare & human wellbeing",
             "weights":   [0.3,0.6,0.8,0.5,0.4,0.7,0.2,0.9,0.2,0.6,0.7,0.0],
             "ikigai":    [0.2,0.2,0.9,0.3],
             "behavioral":[0.4,0.4,0.3,0.7,0.5,0.5,0.5,0.6]},
            {"label": "Technology & digital transformation",
             "weights":   [0.4,0.8,0.3,0.9,0.5,0.6,0.5,0.6,0.2,0.2,0.4,0.0],
             "ikigai":    [0.2,0.3,0.9,0.6],
             "behavioral":[0.5,0.6,0.5,0.3,0.4,0.7,0.6,0.7]},
            {"label": "Education & knowledge access",
             "weights":   [0.5,0.6,0.8,0.3,0.5,0.7,0.2,0.6,0.4,0.2,0.9,0.0],
             "ikigai":    [0.3,0.2,0.9,0.3],
             "behavioral":[0.4,0.4,0.4,0.6,0.5,0.5,0.5,0.7]},
            {"label": "Business, economy & financial systems",
             "weights":   [0.3,0.8,0.5,0.5,0.7,0.7,0.9,0.4,0.2,0.1,0.6,0.0],
             "ikigai":    [0.2,0.2,0.8,0.7],
             "behavioral":[0.6,0.5,0.5,0.4,0.4,0.6,0.7,0.5]},
            {"label": "Environment, infrastructure & cities",
             "weights":   [0.4,0.8,0.4,0.8,0.5,0.8,0.3,0.8,0.3,0.5,0.4,0.0],
             "ikigai":    [0.3,0.2,0.9,0.4],
             "behavioral":[0.5,0.5,0.5,0.4,0.5,0.6,0.6,0.6]},
            {"label": "Culture, arts & human expression",
             "weights":   [0.9,0.3,0.7,0.3,0.3,0.5,0.4,0.2,0.9,0.2,0.8,0.0],
             "ikigai":    [0.4,0.2,0.8,0.2],
             "behavioral":[0.2,0.7,0.3,0.6,0.7,0.3,0.4,0.8]},
        ]
    },

    # ── Q4: PAID FOR (Ikigai) ─────────────────────────────────────────────────
    {
        "id": "paid_for",
        "quadrant": "paid_for",
        "category": "💰 What You Can Be Paid For",
        "question": "Which career realities appeal to you? (Select all that apply)",
        "options": [
            {"label": "High salary & financial security",
             "weights":   [0.1,0.5,0.2,0.6,0.4,0.5,0.7,0.4,0.1,0.1,0.3,1.0],
             "ikigai":    [0.1,0.2,0.1,0.9],
             "behavioral":[0.6,0.3,0.2,0.3,0.2,0.5,0.7,0.3]},
            {"label": "Freelance & entrepreneurial freedom",
             "weights":   [0.6,0.5,0.6,0.5,0.8,0.4,0.9,0.2,0.5,0.3,0.7,0.8],
             "ikigai":    [0.3,0.3,0.2,0.9],
             "behavioral":[0.3,0.8,0.5,0.4,0.7,0.4,0.7,0.7]},
            {"label": "Stable long-term career growth",
             "weights":   [0.2,0.6,0.4,0.6,0.5,0.7,0.5,0.5,0.2,0.3,0.4,0.7],
             "ikigai":    [0.2,0.3,0.3,0.8],
             "behavioral":[0.8,0.2,0.3,0.4,0.2,0.5,0.7,0.3]},
            {"label": "Social impact, even at lower pay",
             "weights":   [0.5,0.4,0.8,0.3,0.5,0.5,0.2,0.5,0.4,0.4,0.7,0.3],
             "ikigai":    [0.4,0.2,0.6,0.5],
             "behavioral":[0.3,0.5,0.6,0.7,0.5,0.4,0.5,0.6]},
            {"label": "Creative work that pays (design, media)",
             "weights":   [0.9,0.3,0.6,0.4,0.3,0.5,0.6,0.1,0.9,0.2,0.7,0.7],
             "ikigai":    [0.5,0.3,0.2,0.7],
             "behavioral":[0.2,0.6,0.4,0.5,0.6,0.3,0.6,0.7]},
            {"label": "In-demand technical expertise",
             "weights":   [0.2,0.8,0.3,0.9,0.4,0.7,0.4,0.6,0.1,0.2,0.3,0.9],
             "ikigai":    [0.2,0.4,0.3,0.9],
             "behavioral":[0.7,0.4,0.3,0.2,0.3,0.8,0.7,0.5]},
        ]
    },

    # ── Q5: BEHAVIORAL — Decision Making (hidden) ─────────────────────────────
    {
        "id": "decision_making",
        "quadrant": None,
        "category": "⚡ When You Face a Big Decision",
        "question": "What best describes how you actually make important decisions? (Select all that apply)",
        "options": [
            {"label": "I research extensively before deciding",
             "weights":   [0.1,0.8,0.1,0.4,0.2,0.9,0.2,0.7,0.1,0.1,0.2,0.0],
             "ikigai":    [0.0,0.0,0.0,0.0],
             "behavioral":[0.8,0.2,0.3,0.2,0.2,0.9,0.6,0.6]},
            {"label": "I go with my gut — instinct is fast",
             "weights":   [0.5,0.3,0.5,0.3,0.7,0.2,0.6,0.2,0.4,0.5,0.5,0.0],
             "ikigai":    [0.0,0.0,0.0,0.0],
             "behavioral":[0.2,0.8,0.6,0.5,0.8,0.3,0.7,0.5]},
            {"label": "I talk it through with people I trust",
             "weights":   [0.3,0.3,0.8,0.2,0.5,0.3,0.4,0.2,0.3,0.2,0.8,0.0],
             "ikigai":    [0.0,0.0,0.0,0.0],
             "behavioral":[0.3,0.5,0.4,0.9,0.6,0.4,0.5,0.5]},
            {"label": "I make a structured pros/cons list",
             "weights":   [0.1,0.8,0.2,0.5,0.4,0.9,0.5,0.5,0.1,0.1,0.3,0.0],
             "ikigai":    [0.0,0.0,0.0,0.0],
             "behavioral":[0.9,0.2,0.2,0.2,0.1,0.9,0.8,0.3]},
            {"label": "I delay — I need more time than most",
             "weights":   [0.2,0.5,0.3,0.3,0.2,0.6,0.2,0.4,0.2,0.1,0.3,0.0],
             "ikigai":    [0.0,0.0,0.0,0.0],
             "behavioral":[0.6,0.2,0.3,0.3,0.3,0.6,0.3,0.5]},
        ]
    },

    # ── Q6: BEHAVIORAL — Work Style (hidden) ──────────────────────────────────
    {
        "id": "work_style",
        "quadrant": None,
        "category": "🧠 Your Natural Work Rhythm",
        "question": "Which of these sounds most like you at your best? (Select all that apply)",
        "options": [
            {"label": "Deep focus alone for hours — flow state",
             "weights":   [0.4,0.8,0.1,0.7,0.2,0.8,0.2,0.6,0.3,0.2,0.2,0.0],
             "ikigai":    [0.0,0.0,0.0,0.0],
             "behavioral":[0.7,0.4,0.2,0.1,0.3,0.8,0.7,0.6]},
            {"label": "Collaborative energy with a team",
             "weights":   [0.5,0.3,0.9,0.3,0.7,0.3,0.6,0.2,0.4,0.3,0.8,0.0],
             "ikigai":    [0.0,0.0,0.0,0.0],
             "behavioral":[0.3,0.5,0.3,0.9,0.6,0.3,0.6,0.5]},
            {"label": "Quick bursts across many tasks",
             "weights":   [0.6,0.4,0.5,0.4,0.5,0.3,0.6,0.3,0.4,0.4,0.5,0.0],
             "ikigai":    [0.0,0.0,0.0,0.0],
             "behavioral":[0.3,0.7,0.6,0.5,0.8,0.4,0.5,0.8]},
            {"label": "Structured routine, consistent output",
             "weights":   [0.2,0.6,0.4,0.5,0.5,0.9,0.5,0.4,0.2,0.3,0.4,0.0],
             "ikigai":    [0.0,0.0,0.0,0.0],
             "behavioral":[0.9,0.2,0.2,0.4,0.1,0.7,0.9,0.2]},
            {"label": "Creative chaos — ideas first, structure later",
             "weights":   [0.9,0.3,0.5,0.3,0.5,0.2,0.5,0.2,0.8,0.3,0.6,0.0],
             "ikigai":    [0.0,0.0,0.0,0.0],
             "behavioral":[0.1,0.9,0.7,0.5,0.9,0.3,0.4,0.9]},
        ]
    },

    # ── Q7: BEHAVIORAL — Stress Response (hidden) ─────────────────────────────
    {
        "id": "stress_response",
        "quadrant": None,
        "category": "🔥 Under Pressure",
        "question": "When a deadline hits and things get hard, you typically... (Select all that apply)",
        "options": [
            {"label": "Break it down into smaller steps and grind",
             "weights":   [0.2,0.7,0.2,0.5,0.4,0.9,0.4,0.4,0.1,0.3,0.3,0.0],
             "ikigai":    [0.0,0.0,0.0,0.0],
             "behavioral":[0.9,0.3,0.2,0.2,0.1,0.7,0.9,0.3]},
            {"label": "Get energized — pressure activates me",
             "weights":   [0.4,0.5,0.4,0.5,0.7,0.4,0.6,0.3,0.3,0.5,0.5,0.0],
             "ikigai":    [0.0,0.0,0.0,0.0],
             "behavioral":[0.4,0.8,0.5,0.4,0.6,0.5,0.8,0.5]},
            {"label": "Reach out for help or collaboration",
             "weights":   [0.3,0.3,0.8,0.2,0.4,0.3,0.3,0.2,0.2,0.2,0.8,0.0],
             "ikigai":    [0.0,0.0,0.0,0.0],
             "behavioral":[0.3,0.4,0.3,0.9,0.5,0.3,0.5,0.4]},
            {"label": "Step back and rethink the whole approach",
             "weights":   [0.5,0.7,0.3,0.4,0.4,0.5,0.4,0.5,0.3,0.2,0.4,0.0],
             "ikigai":    [0.0,0.0,0.0,0.0],
             "behavioral":[0.5,0.6,0.6,0.3,0.6,0.7,0.5,0.7]},
            {"label": "Push through on instinct and momentum",
             "weights":   [0.4,0.3,0.3,0.4,0.6,0.3,0.5,0.2,0.3,0.6,0.4,0.0],
             "ikigai":    [0.0,0.0,0.0,0.0],
             "behavioral":[0.3,0.8,0.5,0.3,0.6,0.3,0.8,0.4]},
        ]
    },

    # ── Q8: BEHAVIORAL — Curiosity Pattern (hidden) ───────────────────────────
    {
        "id": "curiosity",
        "quadrant": None,
        "category": "🔭 When Something Interests You",
        "question": "How far down the rabbit hole do you usually go? (Select all that apply)",
        "options": [
            {"label": "I read everything — papers, books, forums",
             "weights":   [0.3,0.9,0.2,0.5,0.2,0.9,0.2,0.8,0.2,0.1,0.4,0.0],
             "ikigai":    [0.0,0.0,0.0,0.0],
             "behavioral":[0.6,0.3,0.2,0.2,0.3,0.9,0.5,0.9]},
            {"label": "I build or make something to understand it",
             "weights":   [0.5,0.6,0.2,0.8,0.3,0.6,0.3,0.5,0.3,0.4,0.3,0.0],
             "ikigai":    [0.0,0.0,0.0,0.0],
             "behavioral":[0.5,0.6,0.3,0.2,0.5,0.6,0.8,0.7]},
            {"label": "I find people to talk to about it",
             "weights":   [0.4,0.3,0.8,0.2,0.4,0.3,0.3,0.3,0.3,0.2,0.9,0.0],
             "ikigai":    [0.0,0.0,0.0,0.0],
             "behavioral":[0.3,0.5,0.3,0.9,0.6,0.3,0.4,0.7]},
            {"label": "I skim the surface — enough to get the picture",
             "weights":   [0.5,0.4,0.4,0.3,0.4,0.3,0.5,0.3,0.4,0.3,0.5,0.0],
             "ikigai":    [0.0,0.0,0.0,0.0],
             "behavioral":[0.3,0.6,0.6,0.4,0.8,0.3,0.5,0.5]},
            {"label": "I get obsessed and go extremely deep",
             "weights":   [0.4,0.8,0.2,0.6,0.3,0.9,0.2,0.7,0.3,0.2,0.3,0.0],
             "ikigai":    [0.0,0.0,0.0,0.0],
             "behavioral":[0.7,0.4,0.1,0.2,0.2,0.9,0.6,0.9]},
        ]
    },
]


def _display_question(q, index, total):
    print(f"\n{'─'*62}")
    print(f"  Question {index}/{total}  |  {q['category']}")
    print(f"{'─'*62}")
    print(f"\n  {q['question']}\n")
    for j, opt in enumerate(q['options'], 1):
        print(f"    [{j}] {opt['label']}")


def _get_choices(n_options):
    while True:
        raw = input("\n  Your choices (e.g. 1, 3 or just 2): ").strip()
        try:
            choices = [int(x.strip()) for x in raw.split(",") if x.strip()]
            if not choices:
                raise ValueError
            if all(1 <= c <= n_options for c in choices):
                return choices
            print(f"  ⚠  Enter numbers between 1 and {n_options}.")
        except ValueError:
            print("  ⚠  Enter numbers separated by commas, e.g. 1, 3")


def run_assessment():
    """
    Run the full assessment. Returns:
        user_vector    : np.ndarray (12,)  — normalized feature vector
        ikigai_scores  : dict              — {quadrant: score 0-100}
        behavioral_raw : np.ndarray (8,)  — normalized behavioral signal
        user_answers   : list[dict]        — answer log for contradiction detection
    """
    user_vector    = np.zeros(len(FEATURE_NAMES))
    ikigai_acc     = np.zeros(4)
    behavioral_raw = np.zeros(len(BEHAVIORAL_DIMS))
    user_answers   = []
    total          = len(QUESTIONS)

    print("\n" + "="*62)
    print("  🎯  AI CAREER DISCOVERY ENGINE")
    print("  Powered by Ikigai × Behavioral Pattern Analysis")
    print("="*62)
    print("\n  Select ALL options that feel true for you.")
    print("  Enter numbers separated by commas — e.g.  1, 3, 5")
    print("  There are no right or wrong answers.\n")

    for i, q in enumerate(QUESTIONS, 1):
        _display_question(q, i, total)
        choices = _get_choices(len(q['options']))

        selected_weights   = np.array([q['options'][c-1]['weights']   for c in choices])
        selected_ikigai    = np.array([q['options'][c-1]['ikigai']    for c in choices])
        selected_behavioral= np.array([q['options'][c-1]['behavioral']for c in choices])

        avg_weights    = selected_weights.mean(axis=0)
        avg_ikigai     = selected_ikigai.mean(axis=0)
        avg_behavioral = selected_behavioral.mean(axis=0)

        user_vector    += avg_weights
        behavioral_raw += avg_behavioral

        if q['quadrant'] is not None:
            idx = ["love","good_at","world_needs","paid_for"].index(q['quadrant'])
            ikigai_acc[idx] += avg_ikigai[idx]

        labels = [q['options'][c-1]['label'] for c in choices]
        user_answers.append({
            "id":         q['id'],
            "quadrant":   q['quadrant'],
            "category":   q['category'],
            "choices":    choices,
            "labels":     labels,
            "n_selected": len(choices),
            "avg_behavioral": avg_behavioral.tolist(),
        })
        print(f"  ✓  {len(choices)} selected.")

    # Normalize feature vector
    mx = user_vector.max()
    if mx > 0:
        user_vector = user_vector / mx

    # Normalize behavioral vector
    bx = behavioral_raw.max()
    if bx > 0:
        behavioral_raw = behavioral_raw / bx

    # Ikigai scores 0-100
    ikigai_scores = {
        q: round(min(float(ikigai_acc[i]) * 100, 100.0), 1)
        for i, q in enumerate(["love","good_at","world_needs","paid_for"])
    }

    print("\n" + "="*62)
    print("  ✅  Assessment complete. Analysing your profile...")
    print("="*62)

    return user_vector, ikigai_scores, behavioral_raw, user_answers