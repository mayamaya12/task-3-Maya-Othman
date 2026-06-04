import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# ── TITLE ─────────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
# 🎯 AI Career Path Recommendation System — v3
### Hybrid ML × Ikigai × Behavioral AI × Market Intelligence

---

**Architecture:**
```
modules/
├── assessment.py   → Ikigai + behavioral questions, multi-select, response timing
├── personality.py  → Adaptive archetypes, contradiction detection, hidden potential
└── market.py       → GCC/MENA salary data, automation risk, demand scores
```

**Pipeline:**
```
Assessment (8 questions, multi-select)
    ↓
12D Feature Vector  +  4D Ikigai Scores  +  8D Behavioral Profile  +  Response Times
    ↓
Stage 1: Cosine Similarity       (40%)
Stage 2: K-Means Clustering      (15%)
Stage 3: Weighted Dot Product    (30%)
Stage 4: Ikigai Alignment        (15%)
    ↓
Career Rankings
    ↓
Personality Archetype  +  Contradictions  +  Hidden Potential  +  Market Intelligence
    ↓
Full Results Dashboard (5 visualizations)
```

---
"""))

# ── SETUP ─────────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""\
import sys, os
sys.path.insert(0, os.path.abspath('modules'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, FancyBboxPatch
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from assessment  import (FEATURE_NAMES, IKIGAI_QUADRANTS, BEHAVIORAL_DIMS,
                          QUESTIONS, run_assessment)
from personality import (PERSONALITY_ARCHETYPES, HIDDEN_POTENTIAL_TRAITS,
                          build_personality_profile)
from market      import (MARKET_DATA, enrich_career_results,
                          compute_market_score, format_salary,
                          get_automation_risk_label)

print("✅ All modules loaded.")
print(f"   assessment.py  → {len(QUESTIONS)} questions, {len(FEATURE_NAMES)} features, {len(BEHAVIORAL_DIMS)} behavioral dims")
print(f"   personality.py → {len(PERSONALITY_ARCHETYPES)} archetypes, {len(HIDDEN_POTENTIAL_TRAITS)} hidden potential traits")
print(f"   market.py      → {len(MARKET_DATA)} careers with GCC/MENA market data")
"""))

# ── CAREER MATRIX ─────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("---\n## Section 1 — Career Knowledge Base"))

cells.append(nbf.v4.new_code_cell("""\
#                              cre  ana  soc  tec  lea  det  bus  sci  art  phy  com  mkt
CAREER_DATA = {
    "Software Engineer":      [0.5, 0.9, 0.4, 1.0, 0.5, 0.8, 0.5, 0.7, 0.2, 0.1, 0.5, 1.0],
    "Data Scientist":         [0.5, 1.0, 0.3, 0.9, 0.4, 0.9, 0.6, 0.9, 0.1, 0.1, 0.5, 1.0],
    "UX Designer":            [0.9, 0.6, 0.7, 0.6, 0.4, 0.8, 0.5, 0.2, 0.9, 0.1, 0.7, 0.8],
    "Product Manager":        [0.6, 0.7, 0.8, 0.6, 0.9, 0.7, 0.8, 0.4, 0.4, 0.1, 0.9, 0.9],
    "Cybersecurity Analyst":  [0.4, 0.9, 0.3, 1.0, 0.5, 0.9, 0.4, 0.7, 0.1, 0.2, 0.4, 1.0],
    "Doctor":                 [0.4, 0.8, 0.8, 0.6, 0.6, 0.9, 0.3, 1.0, 0.2, 0.6, 0.8, 0.9],
    "Biomedical Engineer":    [0.5, 0.9, 0.4, 0.9, 0.4, 0.9, 0.3, 1.0, 0.2, 0.3, 0.5, 0.8],
    "Architect":              [0.9, 0.7, 0.5, 0.7, 0.6, 0.9, 0.5, 0.6, 0.8, 0.3, 0.6, 0.7],
    "Civil Engineer":         [0.4, 0.9, 0.5, 0.9, 0.6, 0.9, 0.4, 0.8, 0.2, 0.5, 0.5, 0.8],
    "Mechanical Engineer":    [0.4, 0.9, 0.4, 0.9, 0.5, 0.9, 0.3, 0.8, 0.2, 0.5, 0.4, 0.8],
    "Financial Analyst":      [0.3, 0.9, 0.5, 0.6, 0.5, 0.9, 0.9, 0.6, 0.1, 0.1, 0.6, 0.9],
    "Entrepreneur":           [0.8, 0.7, 0.8, 0.5, 1.0, 0.5, 1.0, 0.4, 0.5, 0.3, 0.9, 0.9],
    "Lawyer":                 [0.5, 0.8, 0.7, 0.3, 0.7, 0.9, 0.7, 0.4, 0.3, 0.2, 0.9, 0.8],
    "Psychologist":           [0.6, 0.7, 1.0, 0.3, 0.5, 0.7, 0.3, 0.8, 0.5, 0.2, 0.9, 0.7],
    "Teacher":                [0.6, 0.6, 0.9, 0.4, 0.6, 0.7, 0.3, 0.6, 0.5, 0.3, 0.9, 0.6],
    "Researcher":             [0.5, 1.0, 0.3, 0.7, 0.4, 1.0, 0.2, 0.9, 0.3, 0.2, 0.6, 0.7],
    "Graphic Designer":       [1.0, 0.4, 0.5, 0.6, 0.3, 0.7, 0.5, 0.1, 1.0, 0.1, 0.6, 0.7],
    "Marketing Manager":      [0.7, 0.6, 0.8, 0.4, 0.7, 0.6, 0.9, 0.3, 0.6, 0.2, 0.9, 0.8],
    "Journalist":             [0.7, 0.6, 0.8, 0.3, 0.5, 0.6, 0.5, 0.5, 0.7, 0.3, 1.0, 0.6],
    "Content Creator":        [0.9, 0.4, 0.8, 0.5, 0.6, 0.5, 0.7, 0.2, 0.9, 0.3, 0.9, 0.7],
}

CAREER_IKIGAI = {
    "Software Engineer":      [0.7, 0.9, 0.9, 1.0],
    "Data Scientist":         [0.7, 0.9, 0.9, 1.0],
    "UX Designer":            [0.8, 0.8, 0.8, 0.8],
    "Product Manager":        [0.7, 0.8, 0.9, 0.9],
    "Cybersecurity Analyst":  [0.6, 0.8, 1.0, 1.0],
    "Doctor":                 [0.7, 0.9, 1.0, 0.9],
    "Biomedical Engineer":    [0.6, 0.9, 0.9, 0.8],
    "Architect":              [0.8, 0.8, 0.8, 0.7],
    "Civil Engineer":         [0.5, 0.8, 0.9, 0.8],
    "Mechanical Engineer":    [0.6, 0.8, 0.8, 0.8],
    "Financial Analyst":      [0.5, 0.8, 0.7, 0.9],
    "Entrepreneur":           [0.9, 0.8, 0.8, 0.9],
    "Lawyer":                 [0.6, 0.8, 0.8, 0.8],
    "Psychologist":           [0.8, 0.8, 0.9, 0.7],
    "Teacher":                [0.8, 0.7, 0.9, 0.6],
    "Researcher":             [0.8, 0.9, 0.8, 0.6],
    "Graphic Designer":       [0.9, 0.8, 0.6, 0.7],
    "Marketing Manager":      [0.6, 0.7, 0.7, 0.8],
    "Journalist":             [0.8, 0.7, 0.8, 0.6],
    "Content Creator":        [0.9, 0.7, 0.7, 0.7],
}

CAREERS             = list(CAREER_DATA.keys())
CAREER_MATRIX       = np.array(list(CAREER_DATA.values()))
CAREER_IKIGAI_MTX   = np.array(list(CAREER_IKIGAI.values()))

print(f"✅ Career matrix ready: {CAREER_MATRIX.shape[0]} × {CAREER_MATRIX.shape[1]}")
"""))

# ── ASSESSMENT ────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("---\n## Section 2 — Assessment\n\nAnswer all 8 questions. Select everything that applies using comma-separated numbers."))

cells.append(nbf.v4.new_code_cell("""\
user_vector, ikigai_scores, behavioral_vec, response_times, user_answers = run_assessment()
"""))

cells.append(nbf.v4.new_code_cell("""\
# Quick profile summary
print("\\n📊 Your Raw Profiles:\\n")
print("Feature Vector:")
for name, val in zip(FEATURE_NAMES, user_vector):
    bar = '█'*int(val*20) + '░'*(20-int(val*20))
    print(f"  {name:<18} [{bar}] {val:.3f}")

print("\\nBehavioral Profile:")
labels_short = ["Structured↔Explore","Intrinsic↔Extrinsic","Solo↔Collab",
                "Risk-Averse↔Seeking","BigPic↔Detail","Reactive↔Proactive",
                "Concrete↔Abstract","Breadth↔Depth"]
for name, val in zip(labels_short, behavioral_vec):
    bar = '█'*int(val*20) + '░'*(20-int(val*20))
    print(f"  {name:<22} [{bar}] {val:.3f}")

print("\\nIkigai Scores:")
for q, score in ikigai_scores.items():
    bar = '█'*int(score/5) + '░'*(20-int(score/5))
    print(f"  {q:<14} [{bar}] {score:.0f}/100")
"""))

# ── ML PIPELINE ───────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("---\n## Section 3 — ML Pipeline (4 Stages)"))

cells.append(nbf.v4.new_code_cell("""\
# Stage 1: Cosine Similarity
cosine_scores = cosine_similarity(user_vector.reshape(1,-1), CAREER_MATRIX)[0]

# Stage 2: K-Means
kmeans        = KMeans(n_clusters=5, random_state=42, n_init=10)
career_labels = kmeans.fit_predict(CAREER_MATRIX)
user_cluster  = kmeans.predict(user_vector.reshape(1,-1))[0]
cluster_map   = {}
for career, lbl in zip(CAREERS, career_labels):
    cluster_map.setdefault(lbl, []).append(career)
user_cluster_careers = cluster_map[user_cluster]
CLUSTER_BOOST = 0.15

# Stage 3: Weighted dot product
DIM_WEIGHTS = np.array([0.8,0.9,0.8,1.0,0.85,0.85,0.9,0.9,0.75,0.7,0.8,1.0])
DIM_WEIGHTS = DIM_WEIGHTS / DIM_WEIGHTS.sum()
weighted_user = user_vector * DIM_WEIGHTS

# Stage 4: Ikigai alignment per career
user_ikigai_vec = np.array([ikigai_scores[q]/100.0 for q in IKIGAI_QUADRANTS])

final_scores = {}
for i, career in enumerate(CAREERS):
    cos_score    = float(cosine_scores[i])
    cluster_sc   = CLUSTER_BOOST if career in user_cluster_careers else 0.0
    dot_score    = min(float(np.dot(weighted_user, CAREER_MATRIX[i]*DIM_WEIGHTS)), 1.0)
    iki_sim      = float(cosine_similarity(
        user_ikigai_vec.reshape(1,-1),
        CAREER_IKIGAI_MTX[i].reshape(1,-1)
    )[0][0])
    final = cos_score*0.40 + cluster_sc*0.15 + dot_score*0.30 + iki_sim*0.15
    final_scores[career] = dict(final=final, cosine=cos_score,
                                 cluster=cluster_sc, dot=dot_score, ikigai_sim=iki_sim)

# Normalize to 45-98%
raw    = np.array([v['final'] for v in final_scores.values()])
scaler = MinMaxScaler(feature_range=(45,98))
norm   = scaler.fit_transform(raw.reshape(-1,1)).flatten()
for career, n in zip(CAREERS, norm):
    final_scores[career]['percent'] = round(float(n),1)

ranked = sorted(final_scores.items(), key=lambda x: x[1]['percent'], reverse=True)
top5   = ranked[:5]

# Enrich with market data
top5_enriched = enrich_career_results(top5)

print("ML Pipeline complete.")
print(f"{'Rank':<4} {'Career':<28} {'%Fit':<7} {'Cos':<7} {'Iki':<7} {'Market'}")
print("─"*65)
for rank,(career,s) in enumerate(ranked,1):
    mkt = compute_market_score(career)
    mark = " ⭐" if rank<=5 else ""
    print(f"  {rank:<3} {career:<28} {s['percent']:>5.1f}%  {s['cosine']:.3f}  {s['ikigai_sim']:.3f}  {mkt:.3f}{mark}")
"""))

# ── PERSONALITY ANALYSIS ──────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("---\n## Section 4 — Personality & Behavioral Analysis\n\n*This ran silently during your assessment. Now we reveal it.*"))

cells.append(nbf.v4.new_code_cell("""\
personality = build_personality_profile(
    user_vector, behavioral_vec, response_times, user_answers, FEATURE_NAMES
)

arch = personality['archetype']
print("╔" + "═"*60 + "╗")
print("║" + "  🧠  YOUR PERSONALITY ARCHETYPE".center(60) + "║")
print("╚" + "═"*60 + "╝\\n")
print(f"  Primary:   {arch['primary']}  (similarity: {arch['primary_sim']:.2f})")
print(f"  Secondary: {arch['secondary']}  (similarity: {arch['secondary_sim']:.2f})")
print(f"\\n  {arch['primary_data']['description']}")
print(f"\\n  Core Strengths:")
for s in arch['primary_data']['strengths']:
    print(f"    ✓ {s}")
print(f"\\n  Blind Spots to Watch:")
for b in arch['primary_data']['blind_spots']:
    print(f"    ⚠ {b}")

ta = personality['time_analysis']
print(f"\\n  Decision Style: {ta['decision_style']}  |  Consistency: {ta['consistency']}")
print(f"  Avg response time: {ta['mean_seconds']}s  |  Std dev: {ta['std_seconds']}s")
for ins in ta['insights']:
    print(f"  → {ins}")
"""))

cells.append(nbf.v4.new_code_cell("""\
# Contradiction Detection
contradictions = personality['contradictions']
print("╔" + "═"*60 + "╗")
print("║" + "  ⚡  BEHAVIORAL CONTRADICTIONS DETECTED".center(60) + "║")
print("╚" + "═"*60 + "╝\\n")
if not contradictions:
    print("  ✓ No significant contradictions detected.")
    print("  Your stated preferences and behavioral patterns are consistent.")
else:
    for c in contradictions:
        print(f"  🔍 {c['type']}")
        print(f"     Signal:  {c['signal']}")
        print(f"     Insight: {c['insight']}")
        print()
"""))

cells.append(nbf.v4.new_code_cell("""\
# Hidden Potential
hp = personality['hidden_potential']
print("╔" + "═"*60 + "╗")
print("║" + "  💎  HIDDEN POTENTIAL DETECTED".center(60) + "║")
print("╚" + "═"*60 + "╝\\n")
if not hp:
    print("  No standout hidden potential signals detected.")
else:
    for i, trait in enumerate(hp[:5], 1):
        bar = '█'*int(trait['percent']/5) + '░'*(20-int(trait['percent']/5))
        print(f"  {i}. {trait['trait']}")
        print(f"     [{bar}] {trait['percent']:.0f}%")
        print(f"     {trait['description']}")
        print(f"     Career boost: {', '.join(trait['career_boost'][:3])}")
        print()
"""))

# ── RESULTS ───────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("---\n## Section 5 — 🎯 Full Results"))

cells.append(nbf.v4.new_code_cell("""\
medals = ["🥇","🥈","🥉","4️⃣ ","5️⃣ "]
IKIGAI_LABELS = ["What You ❤️ Love","What You're 💪 Good At",
                  "What the 🌍 World Needs","What You Can 💰 Be Paid For"]

print("╔" + "═"*62 + "╗")
print("║" + "  🎯  YOUR TOP 5 CAREER MATCHES".center(62) + "║")
print("╚" + "═"*62 + "╝\\n")

for i,(career,s) in enumerate(top5_enriched):
    pct = s['percent']
    bar = '█'*int(pct/5) + '░'*(20-int(pct/5))
    trend_icon = {"growing":"📈","stable":"➡️","declining":"📉"}.get(s['trend'],"➡️")
    ai_tag  = "🤖 AI-Adjacent" if s['ai_adjacent'] else ""
    gcc_tag = "🌙 GCC Aligned" if s['gcc_aligned'] else ""
    tags = "  ".join(filter(None,[ai_tag, gcc_tag]))

    print(f"  {medals[i]}  {career}   {tags}")
    print(f"       Fit Score:      [{bar}] {pct:.1f}%")
    print(f"       Global Salary:  ${s['salary_usd']:,}/yr   GCC: ${s['salary_gcc']:,}/yr   Egypt: EGP {s['salary_egypt']:,}/yr")
    print(f"       Market Demand:  {s['demand_score']*100:.0f}/100   GCC Demand: {s['gcc_demand']*100:.0f}/100   {trend_icon} {s['trend'].capitalize()}")
    print(f"       Automation:     {get_automation_risk_label(career)}   Remote: {s['remote_index']*100:.0f}%")
    print(f"       Ikigai Align:   {s['ikigai_sim']:.0%}   Cluster Match: {'✓' if s['cluster']>0 else '✗'}")
    print(f"       Market Note:    {s['market_notes']}")
    print()
"""))

# ── VISUALIZATIONS ────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("---\n## Section 6 — Visualizations"))

cells.append(nbf.v4.new_code_cell("""\
# ── Fig 1: Main Dashboard (2x3 grid) ─────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
gs  = GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.38)
GREEN = ['#1B4332','#2D6A4F','#40916C','#52B788','#74C69D']

# Plot 1: Top 5 Career Fit Scores
ax1 = fig.add_subplot(gs[0,:2])
names = [c for c,_ in top5_enriched]
pcts  = [s['percent'] for _,s in top5_enriched]
bars  = ax1.barh(names[::-1], pcts[::-1], color=GREEN[::-1],
                 edgecolor='white', linewidth=0.8, height=0.6)
ax1.set_xlim(0,108)
ax1.set_xlabel("% Career Fit", fontsize=11)
ax1.set_title("🎯 Top 5 Career Matches", fontsize=13, fontweight='bold')
for bar, p in zip(bars, pcts[::-1]):
    ax1.text(p+0.8, bar.get_y()+bar.get_height()/2,
             f'{p:.1f}%', va='center', fontsize=11, fontweight='bold', color='#1B4332')
ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
ax1.set_facecolor('#F8F9FA')

# Plot 2: Profile Radar
ax2 = fig.add_subplot(gs[0,2], projection='polar')
angles = np.linspace(0,2*np.pi,len(FEATURE_NAMES),endpoint=False).tolist()
vals   = user_vector.tolist()
angles += angles[:1]; vals += vals[:1]
ax2.plot(angles, vals, 'o-', lw=2, color='#2D6A4F')
ax2.fill(angles, vals, alpha=0.2, color='#52B788')
ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(FEATURE_NAMES, size=6.5)
ax2.set_ylim(0,1)
ax2.set_title("Profile Radar", size=12, fontweight='bold', pad=14)

# Plot 3: Market Intelligence (salary + demand)
ax3 = fig.add_subplot(gs[1,0])
sal  = [s['salary_usd']/1000 for _,s in top5_enriched]
dem  = [s['demand_score']*100 for _,s in top5_enriched]
x    = np.arange(len(names))
ax3b = ax3.twinx()
ax3.bar(x-0.2, sal, 0.4, label='Salary (k USD)', color='#1B4332', alpha=0.85)
ax3b.bar(x+0.2, dem, 0.4, label='Market Demand %', color='#74C69D', alpha=0.85)
ax3.set_xticks(x)
ax3.set_xticklabels([n[:12] for n in names], rotation=20, ha='right', fontsize=8)
ax3.set_ylabel("Salary (USD k/yr)", fontsize=9, color='#1B4332')
ax3b.set_ylabel("Market Demand (0-100)", fontsize=9, color='#40916C')
ax3.set_title("💰 Market Intelligence", fontsize=11, fontweight='bold')
ax3.set_facecolor('#F8F9FA')
lines = [mpatches.Patch(color='#1B4332',label='Salary'), mpatches.Patch(color='#74C69D',label='Demand')]
ax3.legend(handles=lines, fontsize=7, loc='upper right')

# Plot 4: Automation Risk vs Remote Index
ax4 = fig.add_subplot(gs[1,1])
auto = [s['automation_risk']*100 for _,s in top5_enriched]
rem  = [s['remote_index']*100   for _,s in top5_enriched]
colors4 = ['#2D6A4F' if a<40 else '#F4A261' if a<55 else '#E76F51' for a in auto]
sc = ax4.scatter(auto, rem, s=200, c=colors4, edgecolors='white', linewidth=1.5, zorder=3)
for i,(career,_) in enumerate(top5_enriched):
    ax4.annotate(career[:12], (auto[i], rem[i]),
                 textcoords="offset points", xytext=(6,4), fontsize=7)
ax4.set_xlabel("Automation Risk (%)", fontsize=10)
ax4.set_ylabel("Remote Work Index (%)", fontsize=10)
ax4.set_title("🤖 Automation Risk vs Remote", fontsize=11, fontweight='bold')
ax4.axvline(x=45, color='gray', linestyle='--', alpha=0.4)
ax4.set_facecolor('#F8F9FA')
ax4.grid(True, alpha=0.3)

# Plot 5: Score Breakdown (4 components)
ax5 = fig.add_subplot(gs[1,2])
x5  = np.arange(len(top5_enriched))
w   = 0.18
cos_v = [s['cosine']     for _,s in top5_enriched]
dot_v = [s['dot']        for _,s in top5_enriched]
cl_v  = [1 if s['cluster']>0 else 0 for _,s in top5_enriched]
ik_v  = [s['ikigai_sim'] for _,s in top5_enriched]
ax5.bar(x5-1.5*w, cos_v, w, label='Cosine',   color='#1B4332', alpha=0.85)
ax5.bar(x5-0.5*w, dot_v, w, label='Weighted', color='#40916C', alpha=0.85)
ax5.bar(x5+0.5*w, cl_v,  w, label='Cluster',  color='#74C69D', alpha=0.85)
ax5.bar(x5+1.5*w, ik_v,  w, label='Ikigai',   color='#B7E4C7', alpha=0.85)
ax5.set_xticks(x5)
ax5.set_xticklabels([n[:12] for n,_ in top5_enriched], rotation=20, ha='right', fontsize=8)
ax5.set_ylim(0,1.3)
ax5.set_title("Score Components", fontsize=11, fontweight='bold')
ax5.legend(fontsize=7)
ax5.set_facecolor('#F8F9FA')

fig.suptitle("AI Career Recommendation System v3  ×  Behavioral AI  ×  Market Intelligence",
             fontsize=14, fontweight='bold', y=1.01)
plt.savefig('dashboard_v3.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
print("✅ Dashboard saved.")
"""))

cells.append(nbf.v4.new_code_cell("""\
# ── Fig 2: Ikigai Venn Diagram ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10,10))
ax.set_xlim(-3.2,3.2); ax.set_ylim(-3.2,3.2)
ax.set_aspect('equal'); ax.axis('off')
fig.patch.set_facecolor('#FAFAF8')
ax.set_facecolor('#FAFAF8')

R, positions = 1.55, [
    ( 0.00,  0.85, '#FF6B6B', "What You\\n❤️ LOVE",          ikigai_scores['love']),
    ( 0.78, -0.45, '#96CEB4', "What You Can\\n💰 Be PAID For", ikigai_scores['paid_for']),
    (-0.78, -0.45, '#45B7D1', "What the\\n🌍 World NEEDS",   ikigai_scores['world_needs']),
    ( 0.00, -1.50, '#4ECDC4', "What You're\\n💪 GOOD At",    ikigai_scores['good_at']),
]
for cx,cy,color,label,score in positions:
    ax.add_patch(Circle((cx,cy), R, color=color, alpha=0.18,
                        linewidth=2.5, edgecolor=color, fill=True))
    lx = cx*2.15 if abs(cx)>0.2 else 0
    ly = cy*2.1  if abs(cy)>0.3 else cy*2.4
    ax.text(lx, ly, f"{label}\\n({score:.0f}/100)", ha='center', va='center',
            fontsize=11, fontweight='bold', color=color,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor=color))

ikigai_total = np.mean(list(ikigai_scores.values()))
ax.add_patch(Circle((0,0), 0.44, color='#FFD700', alpha=0.92, zorder=5))
ax.text(0, 0.13, "✨ IKIGAI", ha='center', va='center',
        fontsize=12, fontweight='bold', color='#7B6000', zorder=6)
ax.text(0,-0.13, f"Sweet Spot\\n{ikigai_total:.0f}/100", ha='center', va='center',
        fontsize=9, color='#7B6000', zorder=6)

for ix,iy,label,color in [( 0.42, 0.25,"Passion","#CC4444"),
                            (-0.42, 0.25,"Mission","#2980B9"),
                            ( 0.42,-0.62,"Vocation","#1E8449"),
                            (-0.42,-0.62,"Profession","#117A65")]:
    ax.text(ix,iy,label,ha='center',va='center',fontsize=9,style='italic',color=color,alpha=0.85)

top_career = top5_enriched[0][0]
top_pct    = top5_enriched[0][1]['percent']
ax.text(0,-2.85, f"Your #1 Match: {top_career}  ({top_pct:.0f}% fit)",
        ha='center', va='center', fontsize=12, fontweight='bold', color='#1B4332',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#D8F3DC', edgecolor='#2D6A4F', lw=1.5))
ax.set_title("Your Ikigai Map", fontsize=16, fontweight='bold', pad=20, color='#1B4332')
plt.savefig('ikigai_venn_v3.png', dpi=150, bbox_inches='tight', facecolor='#FAFAF8')
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""\
# ── Fig 3: Personality Archetype Radar ───────────────────────────────────────
from personality import PERSONALITY_ARCHETYPES

arch_names  = list(PERSONALITY_ARCHETYPES.keys())
arch_scores = [personality['archetype']['all_scores'].get(n,0) for n in arch_names]

fig, axes = plt.subplots(1,2, figsize=(16,6))

# Left: Archetype similarity bar chart
colors_arch = ['#2D6A4F' if n==personality['archetype']['primary']
               else '#74C69D' if n==personality['archetype']['secondary']
               else '#B7E4C7' for n in arch_names]
axes[0].barh(arch_names[::-1], [arch_scores[i] for i in range(len(arch_names)-1,-1,-1)],
             color=colors_arch[::-1], edgecolor='white', linewidth=0.6, height=0.65)
axes[0].set_xlim(0,1.1)
axes[0].set_xlabel("Similarity Score", fontsize=11)
axes[0].set_title(f"🧠 Personality Archetype Match\\nYours: {personality['archetype']['primary']}",
                  fontsize=12, fontweight='bold')
for i,(name,score) in enumerate(zip(reversed(arch_names), reversed(arch_scores))):
    axes[0].text(score+0.01, i, f'{score:.3f}', va='center', fontsize=9)
axes[0].spines['top'].set_visible(False); axes[0].spines['right'].set_visible(False)
axes[0].set_facecolor('#F8F9FA')

# Right: Hidden Potential bars
hp = personality['hidden_potential'][:6] if personality['hidden_potential'] else []
if hp:
    hp_names  = [t['trait'] for t in hp]
    hp_scores = [t['percent'] for t in hp]
    hp_colors = plt.cm.YlGn(np.linspace(0.4, 0.85, len(hp)))
    bars = axes[1].barh(hp_names[::-1], hp_scores[::-1], color=hp_colors[::-1],
                        edgecolor='white', linewidth=0.6, height=0.65)
    axes[1].set_xlim(0,115)
    axes[1].set_xlabel("Potential Score (%)", fontsize=11)
    axes[1].set_title("💎 Hidden Potential Traits Detected", fontsize=12, fontweight='bold')
    for bar, sc in zip(bars, hp_scores[::-1]):
        axes[1].text(sc+1, bar.get_y()+bar.get_height()/2,
                     f'{sc:.0f}%', va='center', fontsize=9, fontweight='bold', color='#1B4332')
    axes[1].axvline(x=60, color='gray', linestyle='--', alpha=0.4, label='Detection threshold (60%)')
    axes[1].legend(fontsize=8)
    axes[1].spines['top'].set_visible(False); axes[1].spines['right'].set_visible(False)
    axes[1].set_facecolor('#F8F9FA')
else:
    axes[1].text(0.5,0.5,"No strong hidden\\npotential signals detected",
                 ha='center',va='center',fontsize=13,transform=axes[1].transAxes)
    axes[1].axis('off')

plt.suptitle("Behavioral Intelligence Layer", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('behavioral_v3.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
print("✅ All visualizations complete.")
"""))

# ── CLOSING ───────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
---
## ✅ System Complete — v3

### What This System Does (CV description):
> *Designed and built a modular AI career recommendation engine in Python combining a 4-stage ML pipeline (cosine similarity, K-Means clustering, weighted feature scoring, Ikigai alignment) with a behavioral intelligence layer featuring adaptive personality archetype assignment, contradiction detection between stated and behavioral preferences, and hidden potential surfacing across 8 trait dimensions — enriched with a GCC/MENA market intelligence layer covering salary bands, automation risk, and Vision 2030 alignment across 20 careers.*

---

### Module Summary

| File | What it does | Key techniques |
|------|-------------|----------------|
| `assessment.py` | 8-question Ikigai + behavioral assessment, multi-select, response timing | Weighted vector averaging, behavioral signal extraction |
| `personality.py` | Archetype assignment, contradiction detection, hidden potential | Cosine similarity, rule-based pattern matching |
| `market.py` | Salary, demand, automation risk, GCC alignment | Static knowledge base, composite scoring |
| `Career_Path_v3.ipynb` | Full pipeline orchestration + 3 visualization figures | sklearn, matplotlib, GridSpec |

---

### Next: Streamlit App
Wrap this into a live demo → shareable link → put it in your CV.
"""))

# ── Assemble ──────────────────────────────────────────────────────────────────
nb.cells = cells
nb.metadata = {
    "kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
    "language_info": {"name":"python","version":"3.10.0"}
}
out = "/home/claude/career_system/Career_Path_v3.ipynb"
with open(out, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"✅ Notebook → {out}")