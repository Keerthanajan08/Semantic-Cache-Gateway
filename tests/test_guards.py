import pytest

from app.guards import (
    action_mismatch,
    action_polarity,
    is_negated,
    negation_mismatch,
    number_mismatch,
    numbers,
    veto_reason,
)


class TestNegation:
    @pytest.mark.parametrize(
        "text",
        [
            "How do I NOT reset my password?",
            "I can't log in",
            "I cannot log in",
            "it doesn't work",
            "why won't it load",
            "never charge me again",
            "login without 2FA",
        ],
    )
    def test_detects_negation(self, text):
        assert is_negated(text)

    @pytest.mark.parametrize(
        "text", ["How do I reset my password?", "notification settings", "another note"]
    )
    def test_ignores_words_that_merely_contain_negations(self, text):
        assert not is_negated(text)

    def test_mismatch_only_when_polarity_differs(self):
        assert negation_mismatch("reset my password", "do not reset my password")
        assert not negation_mismatch("I can't log in", "I cannot sign in")
        assert not negation_mismatch("reset password", "reset my password")


class TestNumbers:
    def test_extracts_standalone_numbers(self):
        assert numbers("order 1234 charged $20.50 twice") == {"1234", "20.50"}

    def test_thousands_separator_is_normalised(self):
        assert numbers("refund 1,000") == numbers("refund 1000")

    @pytest.mark.parametrize("text", ["how to turn on 2FA", "use an mp3 file", "x86 build"])
    def test_numbers_inside_words_are_not_numbers(self, text):
        """Regression: '2FA' was once treated as the number 2 and vetoed 2FA paraphrases."""
        assert numbers(text) == frozenset()

    def test_mismatch(self):
        assert number_mismatch("refund for order 1234", "refund for order 5678")
        assert number_mismatch("10 seat plan", "50 seat plan")
        assert number_mismatch("charged $20", "charged $200")
        assert number_mismatch("plan for 10 seats", "plan for seats")
        assert not number_mismatch("order 1234 refund", "refund order 1234")
        assert not number_mismatch("how to turn on 2FA", "enable two-factor auth")


class TestOpposingActions:
    @pytest.mark.parametrize(
        ("text", "polarity"),
        [
            ("how to turn on 2FA", {"on"}),
            ("enable two-factor", {"on"}),
            ("set up an authenticator", {"on"}),
            ("how to turn off 2FA", {"off"}),
            ("disable 2fa", {"off"}),
            ("remove two step verification", {"off"}),
            ("reset my password", set()),
        ],
    )
    def test_polarity(self, text, polarity):
        assert action_polarity(text) == polarity

    def test_on_vs_off_is_a_mismatch(self):
        assert action_mismatch("how to turn on 2FA", "how to turn off 2FA")
        assert action_mismatch("enable two step verification", "remove two step verification")

    def test_same_direction_is_not(self):
        assert not action_mismatch("enable 2FA", "turn on 2FA")
        assert not action_mismatch("cancel my plan", "stop my subscription renewing")

    def test_one_sided_action_is_not(self):
        """Only veto when both sides state a direction; otherwise trust similarity."""
        assert not action_mismatch("two-factor authentication", "turn on 2FA")


class TestVetoReason:
    @pytest.mark.parametrize(
        ("cached", "incoming", "reason"),
        [
            ("How do I reset my password?", "How do I NOT reset my password?", "negation"),
            ("Refund for order 1234", "Refund for order 5678", "numbers"),
            ("How to turn on 2FA", "How to turn off 2FA", "opposite_action"),
            ("How do I reset my password?", "how can I reset my password", None),
            ("How do I enable 2FA?", "set up two factor authentication", None),
        ],
    )
    def test_reasons(self, cached, incoming, reason):
        assert veto_reason(incoming, cached) == reason

    def test_is_symmetric(self):
        a, b = "turn on 2FA", "don't turn on 2FA"
        assert veto_reason(a, b) == veto_reason(b, a)
