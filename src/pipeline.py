"""End-to-end demo: generate a synthetic lead pipeline, train and
evaluate the lead-scoring model, then run a capture plan + outreach draft
for a sample high-scoring account.
"""
from leads import generate_leads
from scoring import train_and_evaluate, score_lead, leads_to_rows
from capture_plan import build_capture_plan, format_capture_plan_text
from outreach import build_outreach_prompt, render_template_email

SAMPLE_NOTES = (
    "Nordic Rescue Services operates a mixed fleet of aging rotorcraft across "
    "three coastal bases and has publicly stated a fleet modernization goal for "
    "the next budget cycle. The organization has cited dispatch reliability and "
    "harsh-weather performance as its top operational priorities in recent "
    "public statements. Maintenance costs on the current fleet have been rising "
    "for the past two years according to their own annual report. "
    "The procurement team has not yet issued a formal tender."
)


def run():
    print("=== Generating synthetic lead pipeline ===")
    leads = generate_leads(n=900, seed=42)
    print(f"  {len(leads)} synthetic leads generated (not real Airbus CRM data)\n")

    print("=== Lead-Scoring Model: Evaluation ===")
    result = train_and_evaluate(leads)
    print(f"  Train/test split: {result['n_train']}/{result['n_test']} (held-out test set)")
    print(f"  Accuracy:  {result['accuracy']}")
    print(f"  ROC-AUC:   {result['roc_auc']}")
    print(f"  Precision: {result['precision']}")
    print(f"  Recall:    {result['recall']}")
    print(f"  Positive rate in test set (base rate): {result['positive_rate_in_test']}\n")

    print("=== Sample Capture Plan (AI-Powered Capture Planning) ===")
    sample_lead_row = {
        "industry": "EMS/Rescue",
        "region": "Nordics",
        "lead_source": "Trade Show",
        "competitor_incumbent": "none",
        "fleet_size_estimate": 22,
        "budget_confirmed": 1,
        "engaged_last_90_days": 1,
        "prior_customer": 0,
        "num_touchpoints": 6,
    }
    win_prob = score_lead(result["pipeline"], sample_lead_row)

    plan = build_capture_plan(
        account_name="Nordic Rescue Services",
        industry="EMS/Rescue",
        region="Nordics",
        fleet_size_estimate=22,
        competitor_incumbent=None,
        notes_text=SAMPLE_NOTES,
        win_probability=win_prob,
    )
    print(format_capture_plan_text(plan))

    print("=== Sample Outreach Prompt (Prompt Engineering für Sales) ===")
    prompt = build_outreach_prompt(
        account_name="Nordic Rescue Services",
        industry="EMS/Rescue",
        angle="greenfield_capability",
        incumbent=None,
        notes_summary=plan["account_summary"],
    )
    print(prompt)
    print()

    print("=== Deterministic Template Fallback (no LLM available) ===")
    email = render_template_email(
        account_name="Nordic Rescue Services",
        contact_name="Ops Director",
        industry="EMS/Rescue",
        angle="greenfield_capability",
        incumbent=None,
        notes_summary=plan["account_summary"],
    )
    print(email)


if __name__ == "__main__":
    run()
