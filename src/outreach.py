"""Prompt-templated outreach generation: prompt engineering for sales —
generating personalized outreach emails from account facts and a chosen
tone/angle.

This module does not call an LLM. It implements the prompt-engineering
side — designing reusable, parameterized prompt templates — and renders
them two ways: (1) as the actual prompt text that would be sent
to an LLM (e.g. Claude/ChatGPT) in production, so the prompt-engineering
artifact itself is real and inspectable, and (2) as a deterministic
template-filled email for cases with no LLM available, so there's a
working, testable output today.
"""

ANGLES = {
    "cost_displacement": {
        "hook": "total cost of ownership",
        "cta": "a short call to walk through a side-by-side TCO comparison",
    },
    "greenfield_capability": {
        "hook": "mission capability and dispatch reliability",
        "cta": "a capability briefing tailored to your current fleet mix",
    },
    "renewal_upsell": {
        "hook": "fleet modernization and support-contract renewal",
        "cta": "a review of your current support agreement ahead of renewal",
    },
}


def build_outreach_prompt(account_name, industry, angle, incumbent, notes_summary):
    """Returns the actual prompt text that would be sent to an LLM to
    generate a personalized outreach email. This is the real prompt-
    engineering deliverable — reusable, parameterized, and testable for
    structure even without calling a live model.
    """
    if angle not in ANGLES:
        raise ValueError(f"Unknown angle: {angle}. Choose from {list(ANGLES)}")

    angle_cfg = ANGLES[angle]
    incumbent_clause = (
        f"They currently operate {incumbent} aircraft, so acknowledge that directly rather than ignoring it."
        if incumbent else
        "This is a greenfield account with no confirmed incumbent, so do not assume prior helicopter experience."
    )

    prompt = f"""You are a sales development assistant writing a short, personalized outreach email on behalf of a helicopter manufacturer's sales representative.

Account: {account_name}
Industry: {industry}
Angle: lead with {angle_cfg['hook']}.
{incumbent_clause}
Context notes: {notes_summary or "No additional notes available."}

Write a 3-paragraph email:
1. A specific, non-generic opening line referencing the account's actual situation (not a template greeting).
2. One paragraph connecting {angle_cfg['hook']} to a concrete benefit for this account's operation.
3. A clear call to action: {angle_cfg['cta']}.

Keep the tone professional and direct, not salesy. Do not use exclamation points. Do not fabricate specific numbers or claims not present in the context notes above."""

    return prompt.strip()


def render_template_email(account_name, contact_name, industry, angle, incumbent, notes_summary):
    """Deterministic fallback: fills a fixed template directly (no LLM
    call), producing a real, sendable-quality draft today. Less fluent
    than what an LLM would produce from the prompt above; it is the
    currently-working, no-LLM path of the feature.
    """
    if angle not in ANGLES:
        raise ValueError(f"Unknown angle: {angle}. Choose from {list(ANGLES)}")

    angle_cfg = ANGLES[angle]
    incumbent_line = (
        f"I know {account_name} currently operates {incumbent} aircraft, so I'll keep this specific rather than generic."
        if incumbent else
        f"I understand {account_name} hasn't finalized a rotorcraft partner yet, so I wanted to reach out early."
    )

    body = (
        f"Hi {contact_name},\n\n"
        f"{incumbent_line}\n\n"
        f"Given {industry} operations like yours, {angle_cfg['hook']} tends to be the deciding "
        f"factor once budget conversations start. I'd like to offer {angle_cfg['cta']} — "
        f"no obligation, just a chance to see whether it's worth a closer look on your side.\n\n"
        f"Would you have 20 minutes in the next two weeks?\n\n"
        f"Best regards,\n"
    )
    return body


def personalization_fields_present(prompt_text, account_name, industry):
    """Used by tests to check a generated prompt is genuinely
    personalized (contains the account-specific facts) rather than a
    generic template with placeholders left unfilled.
    """
    return account_name in prompt_text and industry in prompt_text
