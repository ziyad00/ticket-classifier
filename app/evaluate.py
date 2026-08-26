"""Agreement between the classifier and a small hand-labelled set.

Runs the classifier directly (no HTTP, no database) so it can be used as a
check after any prompt change. Only category and priority are scored; the
summary is free text and is shown for eyeballing.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path

from app.classifier import Classifier, InvalidModelOutput, ModelTransientError, parse_classification
from app.prompt import PROMPT_VERSION, build_prompt

DATA = Path(__file__).resolve().parent.parent / "data"


@dataclass
class Row:
    id: str
    expected_category: str
    expected_priority: str
    got_category: str | None = None
    got_priority: str | None = None
    summary: str | None = None
    attempts: int = 0
    error: str | None = None

    @property
    def category_ok(self) -> bool:
        return self.got_category == self.expected_category

    @property
    def priority_ok(self) -> bool:
        return self.got_priority == self.expected_priority


@dataclass
class Report:
    prompt_version: str
    classifier: str
    rows: list[Row] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.rows)

    @property
    def category_agreement(self) -> float:
        return sum(r.category_ok for r in self.rows) / self.n if self.n else 0.0

    @property
    def priority_agreement(self) -> float:
        return sum(r.priority_ok for r in self.rows) / self.n if self.n else 0.0

    @property
    def both_agreement(self) -> float:
        return sum(r.category_ok and r.priority_ok for r in self.rows) / self.n if self.n else 0.0

    @property
    def unclassified(self) -> int:
        return sum(r.error is not None for r in self.rows)

    @property
    def rejected_outputs(self) -> int:
        """Model responses the parser refused (counted across retries)."""
        return sum(r.attempts - 1 for r in self.rows if r.error is None) + sum(
            r.attempts for r in self.rows if r.error is not None
        )


def load_labelled(tickets_path: Path = DATA / "sample_tickets.json", labels_path: Path = DATA / "labelled_tickets.json"):
    tickets = {t["id"]: t for t in json.loads(tickets_path.read_text())}
    labels = json.loads(labels_path.read_text())
    return [(tickets[label["id"]], label) for label in labels]


async def _classify_with_retries(classifier: Classifier, subject: str, body: str, max_attempts: int) -> tuple:
    prompt = build_prompt(subject, body)
    last = None
    for attempt in range(1, max_attempts + 1):
        try:
            return parse_classification(await classifier.complete(prompt)), attempt, None
        except (InvalidModelOutput, ModelTransientError) as e:
            last = e
    return None, max_attempts, f"{type(last).__name__}: {last}"


async def evaluate(classifier: Classifier, max_attempts: int = 3, concurrency: int = 3, labelled=None) -> Report:
    labelled = labelled if labelled is not None else load_labelled()
    report = Report(prompt_version=PROMPT_VERSION, classifier=classifier.name)
    sem = asyncio.Semaphore(concurrency)

    async def one(ticket: dict, label: dict) -> Row:
        row = Row(ticket["id"], label["category"], label["priority"])
        async with sem:
            result, row.attempts, row.error = await _classify_with_retries(
                classifier, ticket["subject"], ticket["body"], max_attempts
            )
        if result is not None:
            row.got_category, row.got_priority, row.summary = result.category.value, result.priority.value, result.summary
        return row

    report.rows = await asyncio.gather(*(one(t, l) for t, l in labelled))
    return report


def format_report(report: Report) -> str:
    lines = [
        f"classifier={report.classifier}  prompt_version={report.prompt_version}  tickets={report.n}",
        "",
        f"{'id':8} {'category':22} {'priority':16} {'att':>3}  summary / error",
    ]
    for r in report.rows:
        cat = f"{r.expected_category} -> {r.got_category or '-'}" + ("" if r.category_ok else "  X")
        pri = f"{r.expected_priority} -> {r.got_priority or '-'}" + ("" if r.priority_ok else "  X")
        lines.append(f"{r.id:8} {cat:22} {pri:16} {r.attempts:>3}  {(r.summary or r.error or '')[:60]}")
    lines += [
        "",
        f"category agreement : {report.category_agreement:6.1%}",
        f"priority agreement : {report.priority_agreement:6.1%}",
        f"both agree         : {report.both_agreement:6.1%}",
        f"unclassified       : {report.unclassified}",
        f"rejected outputs   : {report.rejected_outputs}  (model replies the parser refused, incl. retries)",
    ]
    return "\n".join(lines)
