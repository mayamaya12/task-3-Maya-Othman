"""
personality.py
─────────────────────────────────────────────────────────────────────────────
Adaptive personality profiling engine.

1. Builds an 8D personality embedding from behavioral signals
2. Detects contradictions between stated interests and behavioral patterns
3. Identifies hidden potential — latent traits the user may not have stated
4. Returns a personality profile dict used downstream in ml_pipeline.py
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ── Personality Dimension Labels ──────────────────────────────────────────────
BEHAVIORAL_DIMS = [
    "structure_preference",   # 0: loves chaos ↔ 1: loves order
    "risk_tolerance",         # 0: risk-averse ↔ 1: risk-seeking
    "motivation_type",        # 0: intrinsic    ↔ 1: extrinsic
    "social_orientation",     # 0: solo         ↔ 1: collaborative
    "ambiguity_tolerance",    # 0: needs clarity ↔ 1: thrives in uncertainty
    "analytical_bias",        # 0: feeling-first ↔ 1: logic-first
    "execution_bias",         # 0: ideas-person  ↔ 1: finisher
    "curiosity_breadth",      # 0: depth-seeker  ↔ 1: wide explorer
]

# ── Personality Archetypes ─────────────────────────────────────────────────────
# Each archetype is an 8D vector representing a personality pattern.
# Used for nearest-neighbor archetype labeling.
ARCHETYPES = {
    "The Architect":        [0.9,0.4,0.4,0.4,0.3,0.9,0.8,0.5],  # structured, analytical, executes
    "The Explorer":         [0.2,0.8,0.7,0.5,0.9,0.4,0.3,0.9],  # chaotic, curious, intrinsic
    "The Connector":        [0.4,0.5,0.6,0.9,0.5,0.4,0.5,0.6],  # social, collaborative, people-first
    "The Visionary":        [0.2,0.7,0.8,0.5,0.8,0.5,0.2,0.8],  # big ideas, low structure, intrinsic
    "The Executor":         [0.9,0.3,0.5,0.5,0.2,0.7,0.9,0.3],  # disciplined, delivers, low ambiguity
    "The Scientist":        [0.7,0.4,0.3,0.3,0.4,0.9,0.6,0.7],  # analytical, deep curious, solo
    "The Entrepreneur":     [0.3,0.9,0.7,0.6,0.8,0.6,0.8,0.7],  # risk-taking, extrinsic + intrinsic
    "The Empath":           [0.3,0.5,0.7,0.9,0.6,0.3,0.5,0.6],  # social, feeling-first, people-driven
}

ARCHETYPE_DESCRIPTIONS = {
    "The Architect":    "You think in systems. You build things that last, with precision and intent.",
    "The Explorer":     "You're wired for discovery. Novelty energizes you; routine drains you.",
    "The Connector":    "You thrive through people. Your superpower is bringing others together.",
    "The Visionary":    "You see what doesn't exist yet. You think in futures, not constraints.",
    "The Executor":     "You get things done. Where others plan, you ship.",
    "The Scientist":    "You need to understand deeply. Surface-level answers frustrate you.",
    "The Entrepreneur": "You're built for ownership. Risk is your language; impact is your metric.",
    "The Empath":       "You read people and rooms. Your EQ is your greatest career asset.",
}

# ── Hidden Potential Trait Definitions ────────────────────────────────────────
# Each trait has a detection rule: a threshold vector + required dimensions
HIDDEN_POTENTIAL_TRAITS = {
    "Systems Thinking": {
        "description": "You naturally see how parts connect into wholes — rare and valuable in complex domains.",
        "signal": lambda b, fv: b[0] > 0.55 and b[5] > 0.55 and fv[1] > 0.5,  # structure + analytical + analytical feature
        "careers": ["Data Scientist", "Product Manager", "Architect", "Researcher"],
    },
    "Pattern Recognition": {
        "description": "You detect signals in noise. This is a core ML/research superpower.",
        "signal": lambda b, fv: b[5] > 0.6 and b[7] > 0.55 and fv[1] > 0.55,
        "careers": ["Data Scientist", "Cybersecurity Analyst", "Researcher", "Financial Analyst"],
    },
    "Emotional Intelligence": {
        "description": "You read people with accuracy. This makes you a force multiplier in any team.",
        "signal": lambda b, fv: b[3] > 0.6 and b[2] > 0.5 and fv[2] > 0.5,
        "careers": ["Psychologist", "Product Manager", "Teacher", "Marketing Manager"],
    },
    "Leadership Potential": {
        "description": "Your decisiveness + social orientation creates natural followership.",
        "signal": lambda b, fv: b[3] > 0.55 and b[6] > 0.6 and fv[4] > 0.55,
        "careers": ["Entrepreneur", "Product Manager", "Marketing Manager", "Lawyer"],
    },
    "Entrepreneurial Drive": {
        "description": "High risk tolerance + execution bias + intrinsic motivation = founder DNA.",
        "signal": lambda b, fv: b[1] > 0.55 and b[6] > 0.55 and b[4] > 0.5,
        "careers": ["Entrepreneur", "Product Manager", "Content Creator", "Marketing Manager"],
    },
    "Research Aptitude": {
        "description": "Deep curiosity + analytical bias + structure = the researcher's trifecta.",
        "signal": lambda b, fv: b[7] > 0.6 and b[5] > 0.6 and b[0] > 0.5,
        "careers": ["Researcher", "Data Scientist", "Biomedical Engineer", "Doctor"],
    },
    "Creative Intelligence": {
        "description": "You combine divergent thinking with execution — rarer than pure creativity.",
        "signal": lambda b, fv: b[4] > 0.6 and fv[0] > 0.55 and b[6] > 0.45,
        "careers": ["UX Designer", "Architect", "Content Creator", "Graphic Designer"],
    },
    "Strategic Thinking": {
        "description": "You zoom out naturally. You see 3 moves ahead while others see 1.",
        "signal": lambda b, fv: b[0] > 0.5 and b[5] > 0.55 and fv[4] > 0.5 and b[1] > 0.45,
        "careers": ["Product Manager", "Entrepreneur", "Financial Analyst", "Lawyer"],
    },
}

# ── Contradiction Detection Rules ─────────────────────────────────────────────
# A contradiction is when a stated feature score conflicts with a behavioral signal.
CONTRADICTION_RULES = [
    {
        "name": "Creativity vs Structure",
        "condition": lambda fv, b: fv[0] > 0.6 and b[0] > 0.7,
        "stated":    "You said creativity drives you",
        "detected":  "but your behavioral patterns show high structure preference and low ambiguity tolerance",
        "insight":   "You may be a systematic creative — someone who innovates within constraints. Consider: UX Design, Architecture, Product Management.",
    },
    {
        "name": "Social vs Solo",
        "condition": lambda fv, b: fv[2] > 0.6 and b[3] < 0.35,
        "stated":    "You indicated strong social/communication interest",
        "detected":  "but your work style signals suggest you do your best work alone",
        "insight":   "You may be an introvert who communicates well — a huge advantage in writing, research, and technical leadership.",
    },
    {
        "name": "Leadership vs Risk Aversion",
        "condition": lambda fv, b: fv[4] > 0.6 and b[1] < 0.35,
        "stated":    "You expressed leadership interest",
        "detected":  "but your behavioral signals show low risk tolerance",
        "insight":   "You likely thrive as a structured leader — operations, project management, or institutional leadership over entrepreneurship.",
    },
    {
        "name": "Analytical vs Execution Gap",
        "condition": lambda fv, b: fv[1] > 0.65 and b[6] < 0.35,
        "stated":    "Strong analytical profile detected",
        "detected":  "but behavioral signals suggest you're more of an ideas-generator than a finisher",
        "insight":   "Research, strategy, or consulting roles may suit you better than pure engineering or delivery roles.",
    },
    {
        "name": "Entrepreneurial Interest vs Structure Need",
        "condition": lambda fv, b: fv[6] > 0.6 and b[0] > 0.7 and b[1] < 0.4,
        "stated":    "Strong business/entrepreneurial interest",
        "detected":  "but you show high structure preference and low risk tolerance behaviorally",
        "insight":   "Corporate strategy, intrapreneurship, or MBA-track roles may be a better fit than founding a startup.",
    },
]


def build_personality_embedding(behavioral_vec: np.ndarray) -> dict:
    """
    Build a labeled personality embedding from the 8D behavioral vector.
    Returns personality profile dict.
    """
    profile = {dim: round(float(val), 3) for dim, val in zip(BEHAVIORAL_DIMS, behavioral_vec)}

    # Find nearest archetype via cosine similarity
    archetype_matrix = np.array(list(ARCHETYPES.values()))
    sims = cosine_similarity(behavioral_vec.reshape(1,-1), archetype_matrix)[0]
    best_idx   = int(np.argmax(sims))
    best_name  = list(ARCHETYPES.keys())[best_idx]
    best_score = round(float(sims[best_idx]) * 100, 1)

    # Secondary archetype
    sims_copy = sims.copy()
    sims_copy[best_idx] = -1
    second_idx  = int(np.argmax(sims_copy))
    second_name = list(ARCHETYPES.keys())[second_idx]

    profile['archetype']             = best_name
    profile['archetype_score']       = best_score
    profile['secondary_archetype']   = second_name
    profile['archetype_description'] = ARCHETYPE_DESCRIPTIONS[best_name]

    return profile


def detect_contradictions(feature_vec: np.ndarray, behavioral_vec: np.ndarray) -> list:
    """
    Compare stated feature scores against behavioral signals.
    Returns list of detected contradictions (may be empty).
    """
    found = []
    for rule in CONTRADICTION_RULES:
        if rule['condition'](feature_vec, behavioral_vec):
            found.append({
                "name":    rule['name'],
                "stated":  rule['stated'],
                "detected":rule['detected'],
                "insight": rule['insight'],
            })
    return found


def detect_hidden_potential(feature_vec: np.ndarray, behavioral_vec: np.ndarray) -> list:
    """
    Identify latent strengths the user may not have explicitly stated.
    Returns list of detected hidden potential traits.
    """
    found = []
    for trait, meta in HIDDEN_POTENTIAL_TRAITS.items():
        if meta['signal'](behavioral_vec, feature_vec):
            found.append({
                "trait":       trait,
                "description": meta['description'],
                "careers":     meta['careers'],
            })
    return found


def run_personality_analysis(feature_vec: np.ndarray,
                              behavioral_vec: np.ndarray) -> dict:
    """
    Master function: runs all three personality analysis stages.
    Returns full personality report dict.
    """
    embedding      = build_personality_embedding(behavioral_vec)
    contradictions = detect_contradictions(feature_vec, behavioral_vec)
    hidden         = detect_hidden_potential(feature_vec, behavioral_vec)

    return {
        "embedding":       embedding,
        "behavioral_vec":  behavioral_vec,
        "contradictions":  contradictions,
        "hidden_potential":hidden,
    }