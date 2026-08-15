"""Terminal copilot.

    copilot                 interactive chat
    copilot ask "..."       one question
    copilot demo            scripted walkthrough of the acceptance criteria
    copilot replay <hash>   re-execute a previous answer

Runs with zero credentials. With none configured the deterministic planner and
template narrator answer everything the grammar tier covers; setting
COPILOT_PROVIDER adds model-backed planning for the long tail.
"""

from __future__ import annotations

import argparse
import sys
import textwrap

from copilot.engine import Answer, Engine
from copilot.session import SessionState

BANNER = """Industrial Copilot — Margin Engine
Computes distance to the failure boundary. No number is authored by a model.
Commands: /evidence  /plan  /state  /learned  /reset  /tier  /help  /quit"""

DEMO = [
    ("Understand machine behaviour", "What are the typical operating conditions?"),
    ("Analyse historical data", "What's the failure rate by product variant?"),
    ("Premise verification", "Why are we seeing more failures at high rotational speeds?"),
    ("Investigate failures", "Compare the operating conditions of machines that failed versus those that did not."),
    ("Root cause, single cycle", "Why did cycle 9016 fail?"),
    ("Degradation trend", "How does failure rate vary with tool wear?"),
    ("Driver ranking", "What drives failures?"),
    ("Follow-up (filter mutation)", "What about H variants?"),
    ("Counterfactual", "What if we reduce torque by 5 Nm?"),
    ("Data quality", "Can I trust this data?"),
]

_WIDTH = 84


def _rule(char: str = "─") -> str:
    return char * _WIDTH


def _render(answer: Answer, *, timing: bool = True) -> str:
    out = [textwrap.indent(answer.text, "")]
    if timing:
        out.append(
            f"  ·  {answer.tier} tier · plan {answer.plan_ms:.1f} ms · "
            f"exec {answer.exec_ms:.1f} ms · total {answer.elapsed_ms:.1f} ms"
            f"{'' if answer.verified else ' · UNVERIFIED'}"
        )
    return "\n".join(out)


def cmd_ask(args: argparse.Namespace) -> int:
    engine = Engine.build(show_evidence=args.evidence)
    answer = engine.ask(args.question, SessionState())
    print(_render(answer, timing=not args.quiet))
    return 0 if not answer.refused else 1


def cmd_demo(args: argparse.Namespace) -> int:
    engine = Engine.build(show_evidence=args.evidence)
    state = SessionState()
    print(f"{BANNER}\nprovider: {engine.provider_name}\n")

    verified = total = 0
    for label, question in DEMO:
        print(_rule("═"))
        print(f"  {label}")
        print(f"  > {question}")
        print(_rule())
        answer = engine.ask(question, state)
        print(_render(answer))
        print()
        total += 1
        verified += int(answer.verified)

    dist = engine.router.tier_distribution()
    print(_rule("═"))
    print(f"  {verified}/{total} answers passed numeric verification")
    print(
        "  tier distribution: "
        + "  ".join(f"{k} {v:.0%}" for k, v in dist.items())
        + f"   cache hit rate {engine.router.cache.hit_rate:.0%}"
    )
    print(_rule("═"))
    return 0 if verified == total else 1


def cmd_chat(args: argparse.Namespace) -> int:
    engine = Engine.build(show_evidence=args.evidence)
    state = SessionState()
    last: Answer | None = None

    print(f"{BANNER}\nprovider: {engine.provider_name}\n")
    while True:
        try:
            question = input("› ").strip()
        except (EOFError, KeyboardInterrupt):
            engine.router.exemplars.save()
            print()
            return 0
        if not question:
            continue

        if question.startswith("/"):
            command = question[1:].split()[0].lower()
            if command in {"quit", "exit", "q"}:
                engine.router.exemplars.save()
                return 0
            if command == "reset":
                state.clear()
                print("  session cleared\n")
                continue
            if command == "state":
                print(textwrap.indent(state.as_prompt_block(), "  "))
                print(f"  ~{state.token_estimate()} tokens\n")
                continue
            if command == "plan" and last is not None and last.plan is not None:
                print(textwrap.indent(last.plan.model_dump_json(indent=2, exclude_defaults=True), "  "))
                print()
                continue
            if command == "evidence" and last is not None and last.bundle is not None:
                print(textwrap.indent(last.bundle.evidence_table(), "  "))
                print()
                continue
            if command == "learned":
                stats = engine.router.exemplars.stats()
                print(f"  {stats['exemplars']} verified exemplars  "
                      f"{stats['hit_rate']:.0%} hit rate")
                for op, n in stats["by_op"].items():
                    print(f"    {op:<16} {n}")
                for row in stats["most_used"]:
                    if row["uses"]:
                        print(f"    used {row['uses']}x  {row['question'][:52]}")
                print()
                continue
            if command == "tier":
                dist = engine.router.tier_distribution()
                print("  " + "  ".join(f"{k} {v:.0%}" for k, v in dist.items()))
                print(f"  cache: {len(engine.router.cache)} plans, "
                      f"{engine.router.cache.hit_rate:.0%} hit rate\n")
                continue
            if command == "help":
                print(textwrap.indent(BANNER, "  ") + "\n")
                continue
            print(f"  unknown command: /{command}\n")
            continue

        last = engine.ask(question, state)
        print()
        print(_render(last))
        print()


def cmd_replay(args: argparse.Namespace) -> int:
    """Re-execute a previous answer from its plan hash.

    Resolved against the exemplar store, which persists across sessions. The
    cache holds entity-free shapes and cannot answer this: a shape has no hash
    until it is bound to a question.
    """
    from copilot.ir import parse_plan
    from copilot.ops import execute

    engine = Engine.build(show_evidence=True)
    for exemplar in engine.router.exemplars._exemplars:  # noqa: SLF001
        try:
            plan = parse_plan(exemplar.shape)
        except Exception:
            continue
        if plan.hash == args.handle:
            print(f"  {exemplar.question}\n")
            print(execute(plan, engine.ctx).evidence_table())
            return 0
    print(
        f"No plan with handle {args.handle} is stored. Replay resolves against "
        "verified exemplars; a handle from an unstored answer cannot be rebuilt.",
        file=sys.stderr,
    )
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="copilot", description=__doc__)
    parser.add_argument("--evidence", action="store_true", help="show the evidence table")
    sub = parser.add_subparsers(dest="command")

    ask = sub.add_parser("ask", help="answer one question")
    ask.add_argument("question")
    ask.add_argument("--quiet", action="store_true", help="suppress timings")
    ask.set_defaults(func=cmd_ask)

    demo = sub.add_parser("demo", help="scripted walkthrough")
    demo.set_defaults(func=cmd_demo)

    chat = sub.add_parser("chat", help="interactive session")
    chat.set_defaults(func=cmd_chat)

    replay = sub.add_parser("replay", help="re-execute a previous answer")
    replay.add_argument("handle")
    replay.set_defaults(func=cmd_replay)

    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        args.func = cmd_chat
        args.evidence = getattr(args, "evidence", False)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
