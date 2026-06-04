# AI Career Path Recommendation System
### DecodeLabs Industrial Training Kit | Batch 2026 | Project 3

A hybrid ML recommendation engine that maps a user's personality, skills, and values to ranked career paths — combining content-based filtering, behavioral AI, and GCC/MENA market intelligence.

> **Exceeds spec:** The spec requires 3 user inputs and basic similarity matching. This system takes 8 multi-select questions, builds a 24-dimensional user profile, runs a 4-stage ML pipeline, and outputs a full behavioral intelligence report alongside ranked career recommendations.

---

## Demo Output

```
════════════════════════════════════════════
  AI Career Recommendation System — v3
  Hybrid ML × Ikigai × Behavioral AI
════════════════════════════════════════════

Your Top 5 Career Matches:
  1. Marketing Manager    99.0%
  2. Entrepreneur         97.4%
  3. Product Manager      95.8%
  4. Lawyer               92.3%
  5. Journalist           84.0%

Personality Archetype: The Visionary Leader
Hidden Potential: Systems Thinking, EQ, Leadership

Market Intelligence:
  Top match salary (MENA): $45,000–$90,000
  GCC opportunity score:   High
  Automation risk:         Low
```

*Note: Recommendations are personality-fit based, not background-based.
High social, communication, and leadership scores drive people-facing career matches
regardless of technical background — this is by design.*

---

## Why This Is a Recommendation System

The spec asks for: take user input → match using similarity → display recommendations.

This system does exactly that — at production scale:

| Spec Requirement | This Implementation |
|---|---|
| Take user input (min 3 inputs) | 8 multi-select questions across Ikigai + behavioral dimensions |
| Match preferences using similarity | 4-stage ML pipeline: cosine similarity + KMeans + weighted dot product + Ikigai alignment |
| Display recommended items | Top-N ranked careers with % fit, market data, and full score breakdown |

---

## Architecture

```
modules/
├── assessment.py    → 8 questions: Ikigai multi-select + 8 hidden behavioral signals
├── personality.py   → Archetype assignment, contradiction detection, hidden potential
├── market.py        → GCC/MENA salary, automation risk, demand, Vision 2030 alignment
└── Career_Path_v3.ipynb  → Full pipeline + 3 visualization figures
```

---

## The Full Pipeline

```
User answers 8 questions (multi-select)
          ↓
 ┌─────────────────────────────────────────┐
 │         24-Dimensional User Profile     │
 │  12D feature vector (creativity, social,│
 │  analytical, technical, leadership...)  │
 │  + 4D Ikigai scores                     │
 │  + 8D behavioral profile                │
 └─────────────────────────────────────────┘
          ↓
 ┌─────────────────────────────────────────┐
 │Hybrid Recommendation Pipeline           │
 │                                         │
 │  Stage 1: Cosine Similarity    (35%)    │
 │  Stage 2: K-Means Clustering   (15%)    │
 │  Stage 3: Weighted Dot Product (25%)    │
 │  Stage 4: Ikigai Alignment     (15%)    │
 │           Market Intelligence  (10%)    │
 └─────────────────────────────────────────┘
          ↓
 ┌─────────────────────────────────────────┐
 │              Output Layer               │
 │  Top-5 ranked careers with % fit        │
 │  + Personality archetype                │
 │  + Contradiction detection              │
 │  + Hidden potential traits              │
 │  + Market intelligence per career       │
 └─────────────────────────────────────────┘
```

---

## The Similarity Engine

**Why cosine similarity over Euclidean distance:**
Cosine measures the *angle* between vectors — the direction of preferences, not their magnitude. A user who strongly selects 3 interests and one who mildly selects 10 can still point in the same direction. Euclidean distance would penalize the second user unfairly.

```
cos(θ) = (A · B) / (‖A‖ × ‖B‖)

Score 1.0 → perfect alignment
Score 0.0 → no overlap
Score −1  → opposite profiles
```

**Why 4 stages instead of 1:**
Single cosine similarity treats all features equally. The weighted dot product lets domain-specific tags (high technical score matching a Data Scientist) outweigh generic tags. KMeans adds cluster-level matching — users in the same behavioral cluster get a bonus for careers that cluster members historically prefer. Ikigai alignment ensures the recommendation respects not just what a user is good at, but what they love, what the world needs, and what they can be paid for.

---

## Knowledge Base

**20 careers** each with:
- 12-dimensional feature vector (creativity, analytical, social, technical, leadership, detail-oriented, business, science, arts, physical, communication, market demand)
- 4-dimensional Ikigai vector (love, good_at, world_needs, paid_for)
- Market data: salary range (MENA/GCC), automation risk %, remote work %, GCC opportunity score, Vision 2030 alignment

**Careers covered:**
Software Engineer, Data Scientist, UX Designer, Product Manager, Cybersecurity Analyst, Doctor, Biomedical Engineer, Architect, Civil Engineer, Mechanical Engineer, Financial Analyst, Entrepreneur, Lawyer, Psychologist, Teacher, Researcher, Graphic Designer, Marketing Manager, Journalist, Content Creator

---

## Visualizations Generated

| Figure | What it shows |
|---|---|
| `dashboard_v3.png` | Top-5 bar chart, profile radar, market scatter, score breakdown |
| `ikigai_venn_v3.png` | Ikigai venn diagram with 4 quadrant scores and sweet spot |
| `behavioral_v3.png` | Personality archetype match + hidden potential traits |
| Profile vector chart | 12-feature normalized strength visualization |

---

## The Behavioral Intelligence Layer

**Hidden from the user during assessment. Revealed dramatically in results.**

The 8 questions contain embedded behavioral signals the user doesn't know they're being analyzed on:

- Response pattern consistency
- Contradiction detection (stated preference vs behavioral signal)
- Ambiguity tolerance
- Risk orientation
- Structure vs exploration preference

At the end, the system reveals: *"We detected a contradiction between your stated preference for creativity and your behavioral signal for high structure and low ambiguity tolerance. This suggests you may prefer creative work within defined systems rather than open-ended exploration."*

---

## The Cold Start Solution

The spec discusses the cold start problem — new users with no history. This system solves it through the onboarding survey approach: the 8-question assessment *forces* the ingestion step, bootstrapping a full profile vector from zero history.

---

## How to Run

**Requirements:**
```bash
pip install numpy pandas matplotlib scikit-learn
```

**Run:**
```bash
jupyter notebook Career_Path_v3.ipynb
```
Run cells sequentially. Answer all 8 questions when prompted. Results and visualizations generate automatically.

---

## Project Structure

```
project-3/
├── Career_Path_v3.ipynb    # Main notebook — run this
├
│   assessment.py       # Question engine
│   personality.py      # Behavioral analysis
│   market.py           # Market intelligence
├── README.md               # This file
└── screenshots/
    ├── career_results.png
    ├── profile_vector.png
    └── behavioral.png
```

---

## Key Learning Outcomes

- Content-based filtering vs collaborative filtering — why and when to use each
- TF-IDF weighting logic applied to career feature vectors
- Cosine similarity as the industry-standard text/profile matching metric
- Why Euclidean distance fails for high-dimensional preference matching
- The cold start problem and onboarding survey as a bypass strategy
- Multi-stage weighted scoring for nuanced recommendations
- KMeans clustering as a secondary signal layer

---

## Connection to Projects 1 and 2

| Project 1 | Project 2 | Project 3 |
|---|---|---|
| Rule-based: exact match | Classification: learned patterns | Recommendation: active prediction |
| Dictionary lookup O(1) | KNN + TF-IDF intent classifier | Cosine similarity + KMeans ranker |
| Static responses | Predicts label from input | Predicts what a user *wants* |
| No user modeling | No user modeling | Full 24D user profile |

Three projects. One coherent progression from deterministic logic → learned classification → personalized prediction.

---


---

*Built as part of the DecodeLabs AI Industrial Training Program, Batch 2026.*
