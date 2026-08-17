"""The console must not suggest questions the engine declines.

A refusal is a feature everywhere else in this system: when the subject of a
question is not measured, saying so is the only honest answer. But a refusal in
response to the product's OWN suggested question is different in kind. Nobody
asked it — the console did — and the operator learns, before asking anything of
their own, that the copilot does not understand its own screen.

The first draft of the chip list was written from intuition. Two of four were
declined:

    "Why is HDF the binding constraint on L-03?"   -> could not interpret
    "When will the next crossing happen?"          -> could not interpret

Which phrasings the grammar supports is a fact about the code, not something to
remember, so it is pinned here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from copilot.engine import Engine
from copilot.session import SessionState

CONSOLE = Path(__file__).resolve().parent.parent / "copilot" / "static" / "console.html"


def _declared_prompts() -> list[str]:
    """Read the chip templates straight out of the page.

    Parsed rather than duplicated: a copy in the test would drift from the page
    silently and assert nothing about what ships.
    """
    source = CONSOLE.read_text()
    block = re.search(r"const FOLLOW_UPS = \[(.*?)\];", source, re.S)
    assert block, "FOLLOW_UPS not found in console.html"
    return re.findall(r'"([^"]+)"', block.group(1))


@pytest.fixture(scope="module")
def engine() -> Engine:
    return Engine.build()


class TestEverySuggestedQuestionIsAnswerable:
    def test_the_page_declares_some(self):
        assert len(_declared_prompts()) >= 3

    @pytest.mark.parametrize("template", _declared_prompts())
    def test_it_is_not_declined(self, engine, template):
        question = template.replace("{machine}", "L-03").replace("{wear}", "150")
        answer = engine.ask(question, SessionState())
        assert not answer.refused, f"the console suggests {question!r} and the engine declines it"
        assert answer.plan is not None

    @pytest.mark.parametrize("template", _declared_prompts())
    def test_it_survives_every_machine_on_the_line(self, engine, template):
        """Chips are rendered per machine, so a phrasing that only parses for
        one id is a chip that breaks on selection."""
        for machine in ("L-01", "M-02", "H-05"):
            question = template.replace("{machine}", machine).replace("{wear}", "8")
            assert not engine.ask(question, SessionState()).refused


class TestTheExplainButton:
    def test_it_uses_a_phrasing_the_engine_answers(self, engine):
        source = CONSOLE.read_text()
        match = re.search(r'\$\("q"\)\.value = `([^`]+)`;\s*\n\s*askNow\(\);', source)
        assert match, "the explain button's question could not be located"
        question = match.group(1).replace("${m.machine}", "L-03").replace("${m.binding}", "OSF")
        assert not engine.ask(question, SessionState()).refused
