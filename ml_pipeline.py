"""
ml_pipeline.py
─────────────────────────────────────────────────────────────────────────────
Hybrid ML recommendation pipeline.

Stage 1 — Cosine Similarity      (35%): feature vector alignment
Stage 2 — K-Means Clustering     (10%): peer-group career boost
Stage 3 — Weighted Dot Product   (25%): market-demand aware scoring
Stage 4 — Ikigai Alignment       (15%): purpose-framework alignment
Stage 5 — Market Intelligence    (15%): demand, GCC opportunity, growth

Returns ranked career list with per-career score breakdown.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

from market import get_all_market_scores

# ── Career Feature Matrix (12D) ───────────────────────────────────────────────
#    creativity, analytical, social, technical, leadership, detail_oriented,
#    business, science, arts, physical, communication, market_demand

FEATURE_NAMES = [
    "creativity","analytical","social","technical","leadership",
    "detail_oriented","business","science","arts","physical",
    "communication","market_demand"
]

IKIGAI_QUADRANTS = ["love","good_at","world_needs","paid_for"]

CAREER_DATA = {
    #                              cre  ana  soc  tec  lea  det  bus  sci  art  phy  com  mkt
    "Software Engineer":          [0.5, 0.9, 0.4, 1.0, 0.5, 0.8, 0.5, 0.7, 0.2, 0.1, 0.5, 1.0],
    "Data Scientist":             [0.5, 1.0, 0.3, 0.9, 0.4, 0.9, 0.6, 0.9, 0.1, 0.1, 0.5, 1.0],
    "UX Designer":                [0.9, 0.6, 0.7, 0.6, 0.4, 0.8, 0.5, 0.2, 0.9, 0.1, 0.7, 0.8],
    "Product Manager":            [0.6, 0.7, 0.8, 0.6, 0.9, 0.7, 0.8, 0.4, 0.4, 0.1, 0.9, 0.9],
    "Cybersecurity Analyst":      [0.4, 0.9, 0.3, 1.0, 0.5, 0.9, 0.4, 0.7, 0.1, 0.2, 0.4, 1.0],
    "Doctor":                     [0.4, 0.8, 0.8, 0.6, 0.6, 0.9, 0.3, 1.0, 0.2, 0.6, 0.8, 0.9],
    "Biomedical Engineer":        [0.5, 0.9, 0.4, 0.9, 0.4, 0.9, 0.3, 1.0, 0.2, 0.3, 0.5, 0.8],
    "Architect":                  [0.9, 0.7, 0.5, 0.7, 0.6, 0.9, 0.5, 0.6, 0.8, 0.3, 0.6, 0.7],
    "Civil Engineer":             [0.4, 0.9, 0.5, 0.9, 0.6, 0.9, 0.4, 0.8, 0.2, 0.5, 0.5, 0.8],
    "Mechanical Engineer":        [0.4, 0.9, 0.4, 0.9, 0.5, 0.9, 0.3, 0.8, 0.2, 0.5, 0.4, 0.8],
    "Financial Analyst":          [0.3, 0.9, 0.5, 0.6, 0.5, 0.9, 0.9, 0.6, 0.1, 0.1, 0.6, 0.9],
    "Entrepreneur":               [0.8, 0.7, 0.8, 0.5, 1.0, 0.5, 1.0, 0.4, 0.5, 0.3, 0.9, 0.9],
    "Lawyer":                     [0.5, 0.8, 0.7, 0.3, 0.7, 0.9, 0.7, 0.4, 0.3, 0.2, 0.9, 0.8],
    "Psychologist":               [0.6, 0.7, 1.0, 0.3, 0.5, 0.7, 0.3, 0.8, 0.5, 0.2, 0.9, 0.7],
    "Teacher":                    [0.6, 0.6, 0.9, 0.4, 0.6, 0.7, 0.3, 0.6, 0.5, 0.3, 0.9, 0.6],
    "Researcher":                 [0.5, 1.0, 0.3, 0.7, 0.4, 1.0, 0.2, 0.9, 0.3, 0.2, 0.6, 0.7],
    "Graphic Designer":           [1.0, 0.4, 0.5, 0.6, 0.3, 0.7, 0.5, 0.1, 1.0, 0.1, 0.6, 0.7],
    "Marketing Manager":          [0.7, 0.6, 0.8, 0.4, 0.7, 0.6, 0.9, 0.3, 0.6, 0.2, 0.9, 0.8],
    "Journalist":                 [0.7, 0.6, 0.8, 0.3, 0.5, 0.6, 0.5, 0.5, 0.7, 0.3, 1.0, 0.6],
    "Content Creator":            [0.9, 0.4, 0.8, 0.5, 0.6, 0.5, 0.7, 0.2, 0.9, 0.3, 0.9, 0.7],
}

CAREER_IKIGAI = {
    "Software Engineer":      [0.7,0.9,0.9,1.0],
    "Data Scientist":         [0.7,0.9,0.9,1.0],
    "UX Designer":            [0.8,0.8,0.8,0.8],
    "Product Manager":        [0.7,0.8,0.9,0.9],
    "Cybersecurity Analyst":  [0.6,0.8,1.0,1.0],
    "Doctor":                 [0.7,0.9,1.0,0.9],
    "Biomedical Engineer":    [0.6,0.9,0.9,0.8],
    "Architect":              [0.8,0.8,0.8,0.7],
    "Civil Engineer":         [0.5,0.8,0.9,0.8],
    "Mechanical Engineer":    [0.6,0.8,0.8,0.8],
    "Financial Analyst":      [0.5,0.8,0.7,0.9],
    "Entrepreneur":           [0.9,0.8,0.8,0.9],
    "Lawyer":                 [0.6,0.8,0.8,0.8],
    "Psychologist":           [0.8,0.8,0.9,0.7],
    "Teacher":                [0.8,0.7,0.9,0.6],
    "Researcher":             [0.8,0.9,0.8,0.6],
    "Graphic Designer":       [0.9,0.8,0.6,0.7],
    "Marketing Manager":      [0.6,0.7,0.7,0.8],
    "Journalist":             [0.8,0.7,0.8,0.6],
    "Content Creator":        [0.9,0.7,0.7,0.7],
}

CAREERS            = list(CAREER_DATA.keys())
CAREER_MATRIX      = np.array(list(CAREER_DATA.values()))
CAREER_IKIGAI_MATRIX = np.array(list(CAREER_IKIGAI.values()))

DIMENSION_WEIGHTS  = np.array([0.8,0.9,0.8,1.0,0.85,0.85,0.9,0.9,0.75,0.7,0.8,1.0])
DIMENSION_WEIGHTS  = DIMENSION_WEIGHTS / DIMENSION_WEIGHTS.sum()

CLUSTER_BOOST = 0.12
N_CLUSTERS    = 5


def run_pipeline(user_vector:    np.ndarray,
                 ikigai_scores:  dict,
                 personality:    dict) -> list:
    """
    Run all 5 scoring stages. Returns ranked list of dicts.

    Args:
        user_vector   : np.ndarray (12,) normalized
        ikigai_scores : dict {quadrant: score 0-100}
        personality   : output of personality.run_personality_analysis()

    Returns:
        ranked_careers: list of dicts sorted by percent desc
    """

    # ── Stage 1: Cosine Similarity ────────────────────────────────────────────
    cosine_scores = cosine_similarity(user_vector.reshape(1,-1), CAREER_MATRIX)[0]

    # ── Stage 2: K-Means Clustering ───────────────────────────────────────────
    kmeans        = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    career_labels = kmeans.fit_predict(CAREER_MATRIX)
    user_cluster  = int(kmeans.predict(user_vector.reshape(1,-1))[0])

    cluster_map   = {}
    for career, label in zip(CAREERS, career_labels):
        cluster_map.setdefault(int(label), []).append(career)
    user_cluster_careers = cluster_map.get(user_cluster, [])

    # ── Stage 3: Weighted Dot Product ─────────────────────────────────────────
    weighted_user = user_vector * DIMENSION_WEIGHTS

    # ── Stage 4: Ikigai Alignment ─────────────────────────────────────────────
    user_ikigai_vec = np.array([ikigai_scores.get(q, 0) / 100.0
                                for q in IKIGAI_QUADRANTS])

    # ── Stage 5: Market Intelligence ──────────────────────────────────────────
    market_scores = get_all_market_scores()

    # ── Compute Final Scores ───────────────────────────────────────────────────
    raw_scores = []
    career_details = []

    for i, career in enumerate(CAREERS):
        career_vec    = CAREER_MATRIX[i]
        weighted_car  = career_vec * DIMENSION_WEIGHTS

        cos_score     = float(cosine_scores[i])
        cluster_score = CLUSTER_BOOST if career in user_cluster_careers else 0.0
        dot_score     = min(float(np.dot(weighted_user, weighted_car)), 1.0)
        iki_sim       = float(cosine_similarity(
                            user_ikigai_vec.reshape(1,-1),
                            CAREER_IKIGAI_MATRIX[i].reshape(1,-1)
                        )[0][0])
        mkt_score     = market_scores.get(career, 0.5)

        final = (cos_score   * 0.35 +
                 cluster_score* 0.10 +
                 dot_score    * 0.25 +
                 iki_sim      * 0.15 +
                 mkt_score    * 0.15)

        raw_scores.append(final)
        career_details.append({
            "career":        career,
            "cosine":        round(cos_score,   4),
            "cluster":       round(cluster_score,4),
            "dot":           round(dot_score,   4),
            "ikigai_sim":    round(iki_sim,     4),
            "market":        round(mkt_score,   4),
            "raw_final":     round(final,       4),
            "in_cluster":    career in user_cluster_careers,
        })

    # Normalize to 45-98%
    raw_arr = np.array(raw_scores)
    scaler  = MinMaxScaler(feature_range=(45, 98))
    norm    = scaler.fit_transform(raw_arr.reshape(-1,1)).flatten()

    for detail, pct in zip(career_details, norm):
        detail['percent'] = round(float(pct), 1)

    ranked = sorted(career_details, key=lambda x: x['percent'], reverse=True)
    return ranked, user_cluster, cluster_map