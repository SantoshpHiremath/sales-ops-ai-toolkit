"""AI-assisted capture-plan generation: turns structured facts about a
target account (and, optionally, free-text notes such as an excerpt from
a public annual report or press release) into a structured capture-plan
document (AI-powered capture planning).

Text generation here uses local template synthesis with extractive
summarization (frequency-based sentence scoring over the input notes),
not a large language model. An LLM-backed version would generate more
fluent, freely-composed prose; this approach selects and assembles real
sentences from the input, and is therefore more constrained but fully
inspectable and reproducible.
"""
import re
from collections import Counter

STOPWORDS = set("""
a an the of and or to in on for with is are was were be been being
this that these those it its as at by from into over under between
we our their they he she his her you your i not no
""".split())


def _sentences(text):
    # naive but adequate sentence splitter for this scope
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if len(s.strip()) > 15]


def _word_freq(sentences):
    freq = Counter()
    for s in sentences:
        for w in re.findall(r"[a-zA-Z]+", s.lower()):
            if w not in STOPWORDS:
                freq[w] += 1
    return freq


def extractive_summary(text, max_sentences=3):
    """Frequency-based extractive summarization: scores each sentence by
    the sum of its (non-stopword) word frequencies, and returns the
    top-scoring sentences in their original order. This is a real,
    well-established classical NLP technique (not a large language
    model) — deterministic and fully explainable.
    """
    sentences = _sentences(text)
    if not sentences:
        return ""
    if len(sentences) <= max_sentences:
        return " ".join(sentences)

    freq = _word_freq(sentences)
    scored = []
    for idx, s in enumerate(sentences):
        words = re.findall(r"[a-zA-Z]+", s.lower())
        score = sum(freq[w] for w in words if w not in STOPWORDS)
        # normalize by length so long sentences don't win purely on word count
        score = score / max(1, len(words))
        scored.append((score, idx, s))

    top = sorted(scored, reverse=True)[:max_sentences]
    top_in_order = [s for _, idx, s in sorted(top, key=lambda t: t[1])]
    return " ".join(top_in_order)


def build_capture_plan(account_name, industry, region, fleet_size_estimate,
                        competitor_incumbent, notes_text, win_probability=None):
    """Assembles a structured capture-plan dict from account facts plus
    an extractive summary of free-text notes (e.g. public report
    excerpts). This is template assembly + extractive NLP, not freeform
    LLM generation — every field is either a direct fact or a sentence
    lifted verbatim from the input notes, so nothing is fabricated.
    """
    summary = extractive_summary(notes_text, max_sentences=3) if notes_text else ""

    incumbent_line = (
        f"Displacing incumbent: {competitor_incumbent}."
        if competitor_incumbent else
        "Greenfield opportunity — no confirmed incumbent on file."
    )

    fleet_line = (
        f"Estimated fleet size: {fleet_size_estimate} aircraft."
        if fleet_size_estimate is not None else
        "Fleet size not yet confirmed — recommend a discovery call to establish this."
    )

    win_line = (
        f"Model-estimated win probability: {win_probability:.0%} "
        "(from the lead-scoring model; a planning input, not a guarantee)."
        if win_probability is not None else ""
    )

    return {
        "account_name": account_name,
        "industry": industry,
        "region": region,
        "fleet_line": fleet_line,
        "incumbent_line": incumbent_line,
        "win_line": win_line,
        "account_summary": summary or "No public notes provided for this account yet.",
        "recommended_next_step": _recommend_next_step(competitor_incumbent, fleet_size_estimate),
    }


def _recommend_next_step(competitor_incumbent, fleet_size_estimate):
    if competitor_incumbent and fleet_size_estimate and fleet_size_estimate >= 15:
        return "Large fleet with an incumbent in place — prioritize a total-cost-of-ownership comparison."
    if competitor_incumbent:
        return "Incumbent in place — focus outreach on differentiation, not just price."
    if fleet_size_estimate is None:
        return "Fleet size unknown — schedule a discovery call before drafting a formal proposal."
    return "Greenfield with known fleet size — move directly to a tailored capability briefing."


def format_capture_plan_text(plan):
    return (
        f"CAPTURE PLAN — {plan['account_name']}\n"
        f"Industry: {plan['industry']}  |  Region: {plan['region']}\n"
        f"{plan['fleet_line']}\n"
        f"{plan['incumbent_line']}\n"
        + (f"{plan['win_line']}\n" if plan["win_line"] else "")
        + f"\nAccount summary (extracted from notes):\n{plan['account_summary']}\n"
        f"\nRecommended next step: {plan['recommended_next_step']}\n"
    )
