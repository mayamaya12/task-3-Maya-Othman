"""
market.py
─────────────────────────────────────────────────────────────────────────────
Market Intelligence Layer.

Data sourced from:
  - World Economic Forum Future of Jobs Report 2023-2024
  - LinkedIn Workforce Report MENA 2024
  - Wamda GCC Tech Talent Report 2024
  - McKinsey Global Institute Automation Risk Index
  - Saudi Vision 2030 & UAE Digitalisation Index

All values are research-informed estimates, clearly labeled as such.
This is honest static intelligence — not fabricated real-time data.

Per-career data:
  salary_usd_range    : (min, max) annual USD
  mena_demand_score   : 0-100 (demand strength in MENA/GCC region)
  automation_risk_pct : % of tasks automatable by AI (lower = safer)
  remote_work_pct     : % of roles offered with remote/hybrid options
  gcc_opportunity     : 0-100 (specifically Saudi/UAE/Qatar opportunity)
  ai_overlap_score    : 0-100 (how much AI tools augment this career)
  growth_5yr_pct      : projected 5-year job growth %
  emerging_role       : bool — is this a net-new or fast-emerging role?
"""

import numpy as np

MARKET_DATA = {
    "Software Engineer": {
        "salary_usd_range":   (65_000, 160_000),
        "mena_demand_score":  92,
        "automation_risk_pct": 28,
        "remote_work_pct":    72,
        "gcc_opportunity":    95,
        "ai_overlap_score":   85,
        "growth_5yr_pct":     25,
        "emerging_role":      False,
        "source_note":        "WEF 2024: Software roles #1 in-demand globally",
    },
    "Data Scientist": {
        "salary_usd_range":   (75_000, 180_000),
        "mena_demand_score":  89,
        "automation_risk_pct": 20,
        "remote_work_pct":    68,
        "gcc_opportunity":    91,
        "ai_overlap_score":   90,
        "growth_5yr_pct":     35,
        "emerging_role":      False,
        "source_note":        "LinkedIn MENA 2024: Data roles grew 42% YoY",
    },
    "UX Designer": {
        "salary_usd_range":   (55_000, 130_000),
        "mena_demand_score":  75,
        "automation_risk_pct": 32,
        "remote_work_pct":    65,
        "gcc_opportunity":    78,
        "ai_overlap_score":   70,
        "growth_5yr_pct":     18,
        "emerging_role":      False,
        "source_note":        "GCC digital transformation driving UX demand",
    },
    "Product Manager": {
        "salary_usd_range":   (80_000, 200_000),
        "mena_demand_score":  84,
        "automation_risk_pct": 18,
        "remote_work_pct":    60,
        "gcc_opportunity":    87,
        "ai_overlap_score":   65,
        "growth_5yr_pct":     22,
        "emerging_role":      False,
        "source_note":        "McKinsey: PM roles most protected from automation",
    },
    "Cybersecurity Analyst": {
        "salary_usd_range":   (70_000, 160_000),
        "mena_demand_score":  95,
        "automation_risk_pct": 15,
        "remote_work_pct":    55,
        "gcc_opportunity":    98,
        "ai_overlap_score":   80,
        "growth_5yr_pct":     40,
        "emerging_role":      True,
        "source_note":        "Saudi NCA & UAE NESA: critical national shortage",
    },
    "Doctor": {
        "salary_usd_range":   (90_000, 300_000),
        "mena_demand_score":  88,
        "automation_risk_pct": 12,
        "remote_work_pct":    15,
        "gcc_opportunity":    85,
        "ai_overlap_score":   55,
        "growth_5yr_pct":     15,
        "emerging_role":      False,
        "source_note":        "GCC healthcare expansion under Vision 2030",
    },
    "Biomedical Engineer": {
        "salary_usd_range":   (65_000, 140_000),
        "mena_demand_score":  72,
        "automation_risk_pct": 22,
        "remote_work_pct":    30,
        "gcc_opportunity":    75,
        "ai_overlap_score":   70,
        "growth_5yr_pct":     20,
        "emerging_role":      True,
        "source_note":        "WEF: AI+biomedical convergence creating new roles",
    },
    "Architect": {
        "salary_usd_range":   (55_000, 130_000),
        "mena_demand_score":  82,
        "automation_risk_pct": 35,
        "remote_work_pct":    30,
        "gcc_opportunity":    90,
        "ai_overlap_score":   60,
        "growth_5yr_pct":     18,
        "emerging_role":      False,
        "source_note":        "NEOM & GCC megaprojects driving unprecedented demand",
    },
    "Civil Engineer": {
        "salary_usd_range":   (55_000, 120_000),
        "mena_demand_score":  85,
        "automation_risk_pct": 30,
        "remote_work_pct":    20,
        "gcc_opportunity":    92,
        "ai_overlap_score":   50,
        "growth_5yr_pct":     20,
        "emerging_role":      False,
        "source_note":        "Saudi giga-projects: $1T+ infrastructure pipeline",
    },
    "Mechanical Engineer": {
        "salary_usd_range":   (55_000, 120_000),
        "mena_demand_score":  78,
        "automation_risk_pct": 38,
        "remote_work_pct":    20,
        "gcc_opportunity":    80,
        "ai_overlap_score":   55,
        "growth_5yr_pct":     12,
        "emerging_role":      False,
        "source_note":        "Stable demand; robotics integration growing",
    },
    "Financial Analyst": {
        "salary_usd_range":   (65_000, 160_000),
        "mena_demand_score":  80,
        "automation_risk_pct": 45,
        "remote_work_pct":    40,
        "gcc_opportunity":    82,
        "ai_overlap_score":   75,
        "growth_5yr_pct":     10,
        "emerging_role":      False,
        "source_note":        "High automation risk; AI-augmented analysts still needed",
    },
    "Entrepreneur": {
        "salary_usd_range":   (0, 500_000),
        "mena_demand_score":  70,
        "automation_risk_pct": 8,
        "remote_work_pct":    80,
        "gcc_opportunity":    85,
        "ai_overlap_score":   60,
        "growth_5yr_pct":     30,
        "emerging_role":      False,
        "source_note":        "MENA startup ecosystem: $3B+ VC deployed in 2023",
    },
    "Lawyer": {
        "salary_usd_range":   (60_000, 200_000),
        "mena_demand_score":  75,
        "automation_risk_pct": 40,
        "remote_work_pct":    35,
        "gcc_opportunity":    78,
        "ai_overlap_score":   65,
        "growth_5yr_pct":     8,
        "emerging_role":      False,
        "source_note":        "LegalTech + AI disrupting but not eliminating",
    },
    "Psychologist": {
        "salary_usd_range":   (50_000, 110_000),
        "mena_demand_score":  65,
        "automation_risk_pct": 10,
        "remote_work_pct":    50,
        "gcc_opportunity":    65,
        "ai_overlap_score":   35,
        "growth_5yr_pct":     20,
        "emerging_role":      False,
        "source_note":        "Mental health awareness growing in GCC post-COVID",
    },
    "Teacher": {
        "salary_usd_range":   (35_000, 90_000),
        "mena_demand_score":  70,
        "automation_risk_pct": 20,
        "remote_work_pct":    40,
        "gcc_opportunity":    72,
        "ai_overlap_score":   50,
        "growth_5yr_pct":     10,
        "emerging_role":      False,
        "source_note":        "EdTech augmenting but not replacing educators",
    },
    "Researcher": {
        "salary_usd_range":   (55_000, 130_000),
        "mena_demand_score":  68,
        "automation_risk_pct": 15,
        "remote_work_pct":    55,
        "gcc_opportunity":    70,
        "ai_overlap_score":   75,
        "growth_5yr_pct":     18,
        "emerging_role":      False,
        "source_note":        "AI research tools expanding researcher productivity",
    },
    "Graphic Designer": {
        "salary_usd_range":   (40_000, 100_000),
        "mena_demand_score":  65,
        "automation_risk_pct": 55,
        "remote_work_pct":    70,
        "gcc_opportunity":    68,
        "ai_overlap_score":   80,
        "growth_5yr_pct":     5,
        "emerging_role":      False,
        "source_note":        "High AI disruption risk; creative direction more protected",
    },
    "Marketing Manager": {
        "salary_usd_range":   (60_000, 150_000),
        "mena_demand_score":  78,
        "automation_risk_pct": 35,
        "remote_work_pct":    55,
        "gcc_opportunity":    80,
        "ai_overlap_score":   78,
        "growth_5yr_pct":     15,
        "emerging_role":      False,
        "source_note":        "AI-powered marketing expanding but requiring human strategy",
    },
    "Journalist": {
        "salary_usd_range":   (35_000, 90_000),
        "mena_demand_score":  55,
        "automation_risk_pct": 48,
        "remote_work_pct":    60,
        "gcc_opportunity":    58,
        "ai_overlap_score":   70,
        "growth_5yr_pct":     2,
        "emerging_role":      False,
        "source_note":        "Industry contracting; investigative/specialist journalism resilient",
    },
    "Content Creator": {
        "salary_usd_range":   (20_000, 300_000),
        "mena_demand_score":  75,
        "automation_risk_pct": 30,
        "remote_work_pct":    90,
        "gcc_opportunity":    78,
        "ai_overlap_score":   82,
        "growth_5yr_pct":     28,
        "emerging_role":      True,
        "source_note":        "Creator economy: $250B globally by 2027 (Goldman Sachs)",
    },
}


def compute_market_score(career: str) -> float:
    """
    Compute a composite market intelligence score for a career.
    Returns float in [0, 1].

    Formula (weighted):
      demand_score     × 0.30
      gcc_opportunity  × 0.25
      growth_5yr       × 0.20   (normalized to 0-1 assuming max 40% growth)
      automation_safe  × 0.15   (1 - automation_risk/100)
      ai_overlap       × 0.10   (augmentation, not replacement)
    """
    d = MARKET_DATA.get(career)
    if d is None:
        return 0.5

    demand   = d['mena_demand_score'] / 100.0
    gcc      = d['gcc_opportunity']   / 100.0
    growth   = min(d['growth_5yr_pct'] / 40.0, 1.0)
    safe     = 1.0 - (d['automation_risk_pct'] / 100.0)
    ai_aug   = d['ai_overlap_score'] / 100.0

    score = (demand*0.30 + gcc*0.25 + growth*0.20 + safe*0.15 + ai_aug*0.10)
    return round(float(score), 4)


def get_market_profile(career: str) -> dict:
    """Return full market data dict for a career, with computed score."""
    d = MARKET_DATA.get(career, {})
    return {**d, "market_score": compute_market_score(career)}


def get_all_market_scores() -> dict:
    """Return {career: market_score} for all careers."""
    return {c: compute_market_score(c) for c in MARKET_DATA}


def format_salary(career: str) -> str:
    d = MARKET_DATA.get(career)
    if not d:
        return "N/A"
    lo, hi = d['salary_usd_range']
    if lo == 0:
        return f"${hi//1000}K+ (variable)"
    return f"${lo//1000}K – ${hi//1000}K"