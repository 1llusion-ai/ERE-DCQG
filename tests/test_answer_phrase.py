"""Regression tests for answer phrase completeness validation.

These tests verify that truncated/dangling answer phrases are correctly
detected and marked as partial.
"""
from dcqg.path.answer_extraction import extract_answer_phrase_local, _check_phrase_completeness


# ================================================================
# _check_phrase_completeness unit tests
# ================================================================

def test_dangling_preposition():
    ok, reason = _check_phrase_completeness("drove the Iroquois out of western New York")
    assert ok, f"Expected complete, got: {reason}"


def test_dangling_titled():
    ok, reason = _check_phrase_completeness("was released in VHS titled")
    assert not ok


def test_dangling_preposition_by():
    ok, reason = _check_phrase_completeness("was written by")
    assert not ok


def test_dangling_known_as():
    ok, reason = _check_phrase_completeness("is known as")
    assert not ok


def test_unclosed_bracket():
    ok, reason = _check_phrase_completeness("was released (VHS")
    assert not ok
    assert "unclosed" in reason


def test_unclosed_quote():
    ok, reason = _check_phrase_completeness('was released in VHS titled "')
    assert not ok
    assert "unclosed" in reason


def test_closed_quotes_ok():
    ok, reason = _check_phrase_completeness('was titled "The Movie"')
    assert ok, f"Expected complete, got: {reason}"


def test_normal_complete_phrase():
    ok, reason = _check_phrase_completeness("drove the Iroquois out of western New York")
    assert ok


def test_passive_with_object():
    ok, reason = _check_phrase_completeness("was released in theaters nationwide")
    assert ok, f"Expected complete, got: {reason}"


# ================================================================
# Fragment starter tests
# ================================================================

def test_fragment_making():
    ok, reason = _check_phrase_completeness("making landfalls on Long Island")
    assert not ok
    assert "fragment" in reason


def test_fragment_could():
    ok, reason = _check_phrase_completeness("could operate in environments other than")
    assert not ok
    assert "fragment" in reason


def test_fragment_would():
    ok, reason = _check_phrase_completeness("would have been better")
    assert not ok
    assert "fragment" in reason


def test_fragment_starting():
    ok, reason = _check_phrase_completeness("starting the offensive")
    assert not ok
    assert "fragment" in reason


def test_non_fragment_started():
    """'started X' is a finite verb, not a bare participle — should pass."""
    ok, reason = _check_phrase_completeness("started the December 2014 Sinjar offensive")
    assert ok, f"Expected complete, got: {reason}"


def test_non_fragment_released():
    """'released X' is fine — 'released' is not in fragment starters."""
    ok, reason = _check_phrase_completeness("released three albums")
    assert ok, f"Expected complete, got: {reason}"


# ================================================================
# "and" boundary fix tests
# ================================================================

def test_and_in_noun_phrase():
    """'restore electrical and water services' should NOT truncate at 'and'."""
    sentence = "Hundreds of State Emergency Service (SES) volunteers were deployed to restore electrical and water services, evacuate locals, and assist with clean up."
    phrase, status = extract_answer_phrase_local(sentence, "deployed")
    assert "water" in phrase, f"Expected 'water' in phrase, got: {phrase!r}"
    assert status == "complete", f"Expected complete, got: {status!r}"


def test_and_joins_clauses():
    """'and he actively sought' — should stop before 'he'."""
    sentence = "He was not present at Wyoming and he actively sought to minimize the atrocities."
    phrase, status = extract_answer_phrase_local(sentence, "present")
    # Should NOT include "he actively sought"
    assert "sought" not in phrase, f"Expected no 'sought' in phrase, got: {phrase!r}"


# ================================================================
# extract_answer_phrase_local integration tests
# ================================================================

def test_bad_example_from_pilot():
    """The known bad example: 'was released in VHS titled '
    should NOT pass as complete."""
    sentence = 'Who\'s That Girl was broadcast in a number of international television channels and was released in VHS titled "".'
    phrase, status = extract_answer_phrase_local(sentence, "released")
    assert status != "complete", f"Expected partial/invalid, got status={status!r}, phrase={phrase!r}"


def test_good_example():
    """Normal phrase should be complete."""
    sentence = "The massacre contributed to calls for reprisals, leading to the 1779 Sullivan Expedition which drove the Iroquois out of western New York."
    phrase, status = extract_answer_phrase_local(sentence, "drove")
    assert status == "complete", f"Expected complete, got status={status!r}, phrase={phrase!r}"
    assert "Iroquois" in phrase


def test_described_as_good():
    """'described as X' where X is a full noun phrase should be complete."""
    sentence = "It has been described as one of the most horrific frontier massacres of the war."
    phrase, status = extract_answer_phrase_local(sentence, "described")
    assert status == "complete", f"Expected complete, got status={status!r}, phrase={phrase!r}"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"All {len(tests)} tests passed!")
