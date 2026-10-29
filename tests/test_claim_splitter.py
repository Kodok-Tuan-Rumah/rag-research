from src.utils.claim_splitter import split_claims


def test_empty_returns_empty():
    assert split_claims("") == []
    assert split_claims(None) == []


def test_single_sentence():
    assert split_claims("Hello world.") == ["Hello world."]


def test_deterministic():
    text = "First claim. Second claim! Third claim?"
    a = split_claims(text)
    b = split_claims(text)
    assert a == b


def test_multiple_sentences():
    text = "Claim one. Claim two. Claim three."
    assert split_claims(text) == ["Claim one.", "Claim two.", "Claim three."]


def test_no_terminator_single_claim():
    assert split_claims("no terminator here") == ["no terminator here"]


def test_whitespace_normalized():
    assert split_claims("a.  b.") == ["a.", "b."]
