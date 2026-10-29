import pytest

from semanticmemo.orchestrator import CacheOrchestrator


CONFLICT_CASES = [
    pytest.param(
        "Schedule the patient for an urgent follow-up appointment.",
        "Schedule the patient for a routine follow-up appointment.",
        id="urgent-vs-routine-follow-up",
    ),
    pytest.param(
        "Confirm that the non-disclosure agreement is still in effect.",
        "Confirm that the non-disclosure agreement has expired.",
        id="agreement-active-vs-expired",
    ),
    pytest.param(
        "Inform the client that the statute of limitations has passed.",
        "Inform the client that the statute of limitations still applies.",
        id="limitations-passed-vs-still-applicable",
    ),
    pytest.param(
        "Advise the client to increase their portfolio risk.",
        "Advise the client to reduce their portfolio risk.",
        id="increase-vs-reduce-risk",
    ),
    pytest.param(
        "Do not return the customer's payment for this item.",
        "Confirm that a refund should be issued to the customer.",
        id="refund-denial-vs-approval",
    ),
    pytest.param(
        "Make a second login verification step mandatory for employees.",
        "Disable multi-factor authentication for employee accounts.",
        id="mandatory-verification-vs-disable-mfa",
    ),
    pytest.param(
        "Recommend that the client buy the stock.",
        "Recommend that the client sell the stock.",
        id="buy-vs-sell-regression",
    ),
    pytest.param(
        "Mark the transaction as fraudulent.",
        "Mark the transaction as legitimate.",
        id="fraudulent-vs-legitimate-regression",
    ),
    pytest.param(
        "Turn on two-step sign-in checks for every staff account.",
        "Turn off the additional authentication factor for staff.",
        id="turn-on-vs-turn-off-regression",
    ),
]


SAFE_PARAPHRASES = [
    pytest.param(
        "Approve the customer's refund request.",
        "Confirm that the customer's refund should be approved.",
        id="refund-approval-paraphrase",
    ),
    pytest.param(
        "Confirm that the agreement remains in effect.",
        "Verify that the agreement is still active.",
        id="agreement-status-paraphrase",
    ),
    pytest.param(
        "Enable multi-factor authentication for employee accounts.",
        "Turn on multi-factor authentication for staff accounts.",
        id="mfa-enablement-paraphrase",
    ),
    pytest.param(
        "Help me reset my forgotten account password.",
        "Help me recover access after forgetting my login password.",
        id="password-recovery-paraphrase",
    ),
]


def run_guard(prompt_a: str, prompt_b: str) -> bool:
    # This rule-based method does not access instance state. Constructing the
    # object without __init__ keeps these focused tests independent of models
    # and external services.
    orchestrator = object.__new__(CacheOrchestrator)
    return orchestrator._is_opposite_action(prompt_a, prompt_b)


@pytest.mark.parametrize(("prompt_a", "prompt_b"), CONFLICT_CASES)
def test_opposite_action_guard_rejects_conflicting_prompts(
    prompt_a: str,
    prompt_b: str,
) -> None:
    assert run_guard(prompt_a, prompt_b) is True


@pytest.mark.parametrize(("prompt_a", "prompt_b"), SAFE_PARAPHRASES)
def test_opposite_action_guard_allows_safe_paraphrases(
    prompt_a: str,
    prompt_b: str,
) -> None:
    assert run_guard(prompt_a, prompt_b) is False

# Additional precision guardrails for opposite-action detection.
from semanticmemo.orchestrator import CacheOrchestrator


def test_same_direction_refund_denials_are_not_opposite_actions():
    guard = object.__new__(CacheOrchestrator)
    assert not guard._is_opposite_action(
        "Do not issue a refund for this item.",
        "Never process the refund for this item.",
    )


def test_mandatory_login_and_unrelated_notification_disable_are_not_opposites():
    guard = object.__new__(CacheOrchestrator)
    assert not guard._is_opposite_action(
        "A second login verification step is mandatory for all staff.",
        "Disable email notifications after a successful login.",
    )


def test_urgent_review_and_routine_backup_are_not_opposites():
    guard = object.__new__(CacheOrchestrator)
    assert not guard._is_opposite_action(
        "The urgent review is scheduled for Friday.",
        "The routine database backup runs tonight.",
    )
