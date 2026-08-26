"""Test suite for the sales-ops AI toolkit (leads, scoring, capture
planning, outreach prompt generation)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from leads import generate_leads, Lead
from scoring import train_and_evaluate, score_lead, leads_to_rows, build_pipeline
from capture_plan import extractive_summary, build_capture_plan, format_capture_plan_text
from outreach import build_outreach_prompt, render_template_email, personalization_fields_present, ANGLES


# --- Leads --------------------------------------------------------------

def test_generate_leads_returns_requested_count():
    leads = generate_leads(n=200, seed=1)
    assert len(leads) == 200


def test_leads_have_unique_ids():
    leads = generate_leads(n=200, seed=1)
    ids = [l.lead_id for l in leads]
    assert len(ids) == len(set(ids))


def test_generate_leads_is_deterministic_given_seed():
    a = generate_leads(n=100, seed=7)
    b = generate_leads(n=100, seed=7)
    assert [l.won for l in a] == [l.won for l in b]


def test_generate_leads_different_seeds_differ():
    a = generate_leads(n=200, seed=1)
    b = generate_leads(n=200, seed=2)
    assert [l.won for l in a] != [l.won for l in b]


def test_some_leads_have_missing_fleet_size():
    # models real, imperfect CRM data — not every lead should be complete
    leads = generate_leads(n=900, seed=42)
    missing = [l for l in leads if l.fleet_size_estimate is None]
    assert 0 < len(missing) < len(leads)


def test_label_is_not_degenerate():
    # the label should not be all-0 or all-1 -- that would indicate a
    # broken generator, not a usable classification dataset
    leads = generate_leads(n=900, seed=42)
    won_count = sum(l.won for l in leads)
    assert 0 < won_count < len(leads)


# --- Scoring: rigor checks (this is where the SAP project's memorization
# bug happened — guard against the same class of mistake here) ----------

@pytest.fixture(scope="module")
def eval_result():
    leads = generate_leads(n=900, seed=42)
    return train_and_evaluate(leads)


def test_train_test_split_is_disjoint():
    leads = generate_leads(n=900, seed=42)
    from sklearn.model_selection import train_test_split
    import pandas as pd
    rows = leads_to_rows(leads)
    X_df = pd.DataFrame(rows)
    y = [l.won for l in leads]
    X_train, X_test, _, _ = train_test_split(X_df, y, test_size=0.25, random_state=42, stratify=y)
    assert set(X_train.index).isdisjoint(set(X_test.index))


def test_accuracy_is_not_suspiciously_perfect(eval_result):
    # Regression guard: real, noisy sales-signal data should NOT produce
    # near-100% accuracy. If this ever fires, it means the label leaked
    # into a feature (or a similar bug), not that the model got better.
    assert eval_result["accuracy"] < 0.90


def test_accuracy_beats_majority_baseline(eval_result):
    # the model should be meaningfully better than always predicting the
    # majority class, or it isn't actually learning anything useful
    majority_baseline = max(eval_result["positive_rate_in_test"], 1 - eval_result["positive_rate_in_test"])
    assert eval_result["accuracy"] > majority_baseline


def test_roc_auc_beats_random(eval_result):
    # 0.5 = random; a working model on real signal should clear this
    # with room to spare, but not implausibly (near-1.0 would be
    # suspicious given how noisy the label-generation process is)
    assert 0.55 < eval_result["roc_auc"] < 0.95


def test_precision_and_recall_are_sane(eval_result):
    assert 0.0 < eval_result["precision"] <= 1.0
    assert 0.0 < eval_result["recall"] <= 1.0


def test_score_lead_returns_probability_in_range(eval_result):
    pipeline = eval_result["pipeline"]
    sample = {
        "industry": "Defense", "region": "DACH", "lead_source": "Referral",
        "competitor_incumbent": "Leonardo", "fleet_size_estimate": 10,
        "budget_confirmed": 1, "engaged_last_90_days": 1, "prior_customer": 1,
        "num_touchpoints": 5,
    }
    prob = score_lead(pipeline, sample)
    assert 0.0 <= prob <= 1.0


def test_score_lead_handles_unknown_categorical_gracefully(eval_result):
    # a category never seen in training (e.g. a new competitor name) must
    # not crash the pipeline -- OneHotEncoder(handle_unknown="ignore")
    pipeline = eval_result["pipeline"]
    sample = {
        "industry": "Defense", "region": "DACH", "lead_source": "Referral",
        "competitor_incumbent": "SomeNewCompetitorNeverSeenBefore",
        "fleet_size_estimate": 10, "budget_confirmed": 1,
        "engaged_last_90_days": 1, "prior_customer": 1, "num_touchpoints": 5,
    }
    prob = score_lead(pipeline, sample)
    assert 0.0 <= prob <= 1.0


def test_stronger_signal_leads_score_higher_on_average(eval_result):
    # sanity check that the model learned *something* directionally
    # sensible: leads with strong positive signal should score higher on
    # average than leads with weak/negative signal
    pipeline = eval_result["pipeline"]
    strong = {
        "industry": "EMS/Rescue", "region": "Nordics", "lead_source": "Referral",
        "competitor_incumbent": "none", "fleet_size_estimate": 30,
        "budget_confirmed": 1, "engaged_last_90_days": 1, "prior_customer": 1,
        "num_touchpoints": 8,
    }
    weak = {
        "industry": "Defense", "region": "Middle East", "lead_source": "Cold Outreach",
        "competitor_incumbent": "Sikorsky", "fleet_size_estimate": -1,
        "budget_confirmed": 0, "engaged_last_90_days": 0, "prior_customer": 0,
        "num_touchpoints": 0,
    }
    assert score_lead(pipeline, strong) > score_lead(pipeline, weak)


# --- Capture planning: extractive summarization -------------------------

def test_extractive_summary_returns_sentences_from_input():
    text = ("Alpha sentence about fleet modernization goals. "
            "Beta sentence about unrelated filler content here today. "
            "Gamma sentence about dispatch reliability priorities stated publicly. "
            "Delta sentence about something else entirely unrelated now.")
    summary = extractive_summary(text, max_sentences=2)
    # every word in the summary must come from the original text (nothing
    # fabricated) -- this is the core honesty property of extractive
    # summarization vs. generative summarization
    for sentence in summary.split(". "):
        assert sentence.strip(".") in text or sentence.strip() in text


def test_extractive_summary_short_text_returned_whole():
    text = "Only one short sentence here."
    assert extractive_summary(text, max_sentences=3) == text


def test_extractive_summary_empty_text():
    assert extractive_summary("", max_sentences=3) == ""


def test_extractive_summary_preserves_original_sentence_order():
    text = ("First sentence establishes the topic clearly. "
            "Second sentence adds supporting detail about the topic. "
            "Third sentence is irrelevant filler about nothing. "
            "Fourth sentence concludes with topic-related implications.")
    summary = extractive_summary(text, max_sentences=3)
    # whichever sentences got picked, their relative order in the summary
    # must match their order in the source text
    positions = [text.index(s) for s in summary.split(". ") if s.strip(".") in text]
    assert positions == sorted(positions)


def test_build_capture_plan_includes_all_key_fields():
    plan = build_capture_plan(
        "Test Account", "Defense", "DACH", 12, "Leonardo",
        "Test Account has grown its fleet steadily over the past decade according to reports.",
        win_probability=0.55,
    )
    assert plan["account_name"] == "Test Account"
    assert "Leonardo" in plan["incumbent_line"]
    assert "12" in plan["fleet_line"]
    assert "55%" in plan["win_line"]


def test_build_capture_plan_handles_missing_fleet_and_greenfield():
    plan = build_capture_plan("Test Account", "Defense", "DACH", None, None, "", win_probability=None)
    assert "not yet confirmed" in plan["fleet_line"]
    assert "Greenfield" in plan["incumbent_line"]
    assert plan["win_line"] == ""


def test_format_capture_plan_text_is_nonempty_string():
    plan = build_capture_plan("Test Account", "Defense", "DACH", 5, None, "Some notes here about the account.", 0.3)
    text = format_capture_plan_text(plan)
    assert isinstance(text, str)
    assert "Test Account" in text


# --- Outreach: prompt engineering ---------------------------------------

def test_build_outreach_prompt_rejects_unknown_angle():
    with pytest.raises(ValueError):
        build_outreach_prompt("Acme", "Defense", "not_a_real_angle", None, "notes")


def test_build_outreach_prompt_is_personalized():
    prompt = build_outreach_prompt("Acme Rescue", "EMS/Rescue", "greenfield_capability", None, "some notes")
    assert personalization_fields_present(prompt, "Acme Rescue", "EMS/Rescue")


def test_build_outreach_prompt_mentions_incumbent_when_present():
    prompt = build_outreach_prompt("Acme Defense", "Defense", "cost_displacement", "Leonardo", "some notes")
    assert "Leonardo" in prompt


def test_build_outreach_prompt_does_not_assume_incumbent_when_absent():
    prompt = build_outreach_prompt("Acme Rescue", "EMS/Rescue", "greenfield_capability", None, "some notes")
    assert "greenfield" in prompt.lower()


def test_all_angles_produce_valid_prompts():
    for angle in ANGLES:
        prompt = build_outreach_prompt("Test Co", "Utility", angle, None, "notes")
        assert len(prompt) > 100


def test_render_template_email_is_personalized_and_uses_correct_casing():
    email = render_template_email("Acme Rescue", "Jane Doe", "EMS/Rescue", "greenfield_capability", None, "notes")
    assert "Jane Doe" in email
    assert "EMS/Rescue" in email  # regression guard: industry must not be lowercased
    assert "ems/rescue" not in email


def test_render_template_email_mentions_incumbent_when_present():
    email = render_template_email("Acme Defense", "John Smith", "Defense", "cost_displacement", "Bell", "notes")
    assert "Bell" in email


def test_render_template_email_rejects_unknown_angle():
    with pytest.raises(ValueError):
        render_template_email("Acme", "Contact", "Defense", "invalid_angle", None, "notes")
