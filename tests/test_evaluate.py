from app.classifier.fake import FakeClassifier
from app.evaluate import evaluate, format_report, load_labelled
from tests.conftest import PERMANENT, ScriptedClassifier


def test_every_sample_ticket_has_a_label():
    pairs = load_labelled()
    assert len(pairs) == 10
    for ticket, label in pairs:
        assert ticket["id"] == label["id"]
        assert label["category"] in ("billing", "technical", "account", "other")
        assert label["priority"] in ("low", "medium", "high")


async def test_agreement_is_computed_from_labels():
    labelled = [
        ({"id": "a", "subject": "s", "body": "b"}, {"id": "a", "category": "billing", "priority": "high"}),
        ({"id": "b", "subject": "s", "body": "b"}, {"id": "b", "category": "technical", "priority": "low"}),
    ]
    # Same answer for both: matches the first label fully, the second not at all.
    clf = ScriptedClassifier('{"category": "billing", "priority": "high", "summary": "x"}')
    report = await evaluate(clf, labelled=labelled, concurrency=1)
    assert report.category_agreement == 0.5 and report.priority_agreement == 0.5 and report.both_agreement == 0.5
    assert report.unclassified == 0 and report.rejected_outputs == 0
    assert "X" in format_report(report)


async def test_rejected_outputs_and_unclassified_are_counted():
    labelled = [({"id": "a", "subject": "s", "body": "b"}, {"id": "a", "category": "billing", "priority": "high"})]
    clf = ScriptedClassifier("garbage", '{"category": "billing", "priority": "high", "summary": "x"}')
    report = await evaluate(clf, labelled=labelled, max_attempts=3)
    assert report.rows[0].attempts == 2 and report.rejected_outputs == 1 and report.both_agreement == 1.0

    report = await evaluate(ScriptedClassifier("garbage"), labelled=labelled, max_attempts=2)
    assert report.unclassified == 1 and report.rejected_outputs == 2 and report.both_agreement == 0.0


async def test_fake_classifier_mostly_agrees_with_labels():
    """A regression guard on the fake's heuristics, not a claim about real models."""
    report = await evaluate(FakeClassifier(failure_rate=0.0, latency=0.0))
    assert report.category_agreement >= 0.9, format_report(report)
    assert report.priority_agreement >= 0.6, format_report(report)
