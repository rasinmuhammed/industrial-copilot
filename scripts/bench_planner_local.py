#!/usr/bin/env python3
"""Measure the distilled planner on the machine it would actually be deployed on.

The Kaggle notebook printed exact-match and latency to stdout and then wrote a
planner_card.json containing neither, so the numbers died with the session. That
was a real defect: the artifact that survives a training run must carry its own
evidence. This script closes the gap from the other side - it re-derives the
identical held-out split (same seed, same generator) and evaluates the exported
GGUF through Ollama on local hardware.

Local measurement is the stronger claim anyway. A GPU number proves the adapter
trained; an M-series laptop number proves the deployment story.

    python scripts/bench_planner_local.py --model margin-planner
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path

OLLAMA = "http://127.0.0.1:11434/api/chat"

# ── Vocabulary, verbatim from the notebook so the split is reproducible ────────
METRICS = ['air_temp_k', 'process_temp_k', 'temp_delta_k', 'rotational_speed_rpm',
           'torque_nm', 'tool_wear_min', 'power_w', 'overstrain_min_nm', 'failure']
DIMENSIONS = ['udi', 'product_type', 'machine_id', 'shift']
OPS = ['describe', 'rate', 'compare', 'trend', 'drivers', 'root_cause',
       'counterfactual', 'envelope', 'forecast', 'records', 'data_quality',
       'sql_explore']

SYSTEM = (
    'Translate the engineer question into one Analysis Plan line. Output the line only.\n'
    'Format: op|cohorts|metrics|dimensions|bin|extra   (use - for an empty field)\n'
    f"ops: {' '.join(OPS)}\n"
    f"metrics: {' '.join(METRICS)}\n"
    f"dimensions: {' '.join(DIMENSIONS)}\n"
    'cohorts: name:field=value;name:field=value    bin: metric:q5 or metric:w6\n'
    'extra: effect size, grain=day, order=..., filters like udi=9016, changes like torque_nm-5'
)

_NUM = re.compile(r'^-?\d+(?:\.\d+)?$')
_FILT = re.compile(r'^([a-z_]+)(<=|>=|!=|=|<|>)(.+)$')


def _cast(v):
    return float(v) if _NUM.match(v) and '.' in v else (int(v) if _NUM.match(v) else v)


def dsl_to_plan(s: str) -> dict:
    parts = (s.strip().split('|') + ['-'] * 6)[:6]
    op, coh, met, dim, binning, extra = parts
    if op not in OPS:
        raise ValueError(f'unknown op {op!r}')
    plan: dict = {'op': op}
    if coh != '-':
        plan['cohorts'] = []
        for c in coh.split(';'):
            name, _, fs = c.partition(':')
            filters = []
            for f in filter(None, fs.split(',')):
                m = _FILT.match(f)
                if m:
                    filters.append({'field': m[1], 'op': m[2], 'value': _cast(m[3])})
            plan['cohorts'].append({'name': name, 'filters': filters})
    if met != '-':
        plan['metrics'] = [m for m in met.split(',') if m in METRICS]
    if dim != '-':
        plan['group_by'] = [d for d in dim.split(',') if d in DIMENSIONS]
    if binning != '-':
        field, _, spec = binning.partition(':')
        plan['bin'] = {'field': field,
                       'method': 'quantile' if spec[:1] == 'q' else 'width',
                       'bins': int(spec[1:] or 5)}
    filters, params = [], {}
    for e in filter(lambda x: x != '-', extra.split(',')):
        if e in ('cohens_d', 'rate_ratio', 'risk_diff'):
            plan['effect_size'] = e
            continue
        if e.startswith('grain='):
            plan['time_grain'] = e.split('=')[1]
            continue
        if e.startswith('order='):
            params['order'] = e.split('=')[1]
            continue
        m = _FILT.match(e)
        if m and m[1] in METRICS + DIMENSIONS:
            filters.append({'field': m[1], 'op': m[2], 'value': _cast(m[3])})
        elif re.match(r'^[a-z_]+[+-]', e):
            k = re.match(r'^([a-z_]+)', e)[1]
            params.setdefault('changes', {})[k] = float(e[len(k):])
    if filters:
        plan['filters'] = filters
    if params:
        plan['params'] = params
    return plan


# ── Corpus generator, verbatim, so shuffle(seed=11) yields the same split ─────
SYNONYM = {
    'torque_nm': ['torque', 'applied torque', 'load', 'torque setting'],
    'rotational_speed_rpm': ['speed', 'rotational speed', 'rpm', 'spindle speed', 'rotation speed'],
    'tool_wear_min': ['tool wear', 'wear', 'tool age', 'tool life'],
    'temp_delta_k': ['temperature differential', 'delta t', 'thermal gradient', 'temperature difference'],
    'power_w': ['power', 'mechanical power', 'power draw', 'wattage'],
    'overstrain_min_nm': ['overstrain', 'strain', 'strain product', 'accumulated strain'],
    'air_temp_k': ['air temperature', 'ambient temperature', 'ambient temp'],
    'process_temp_k': ['process temperature', 'process temp'],
}
VARIANT = {'L': ['L', 'low quality', 'low-grade'], 'M': ['M', 'medium quality'], 'H': ['H', 'high quality']}
POLITE = ['', 'please ', 'can you ', 'could you ', 'i need to know ', 'tell me ', 'show me ']
TAIL = ['', '?', ' please', '.', ' for me?']
FAILED_HEALTHY = 'failed:failure=1;healthy:failure=0'


def syn(metric):
    return random.choice(SYNONYM.get(metric, [metric]))


def wrap(q):
    q = random.choice(POLITE) + q
    return (q[0].upper() + q[1:] if random.random() < .5 else q) + random.choice(TAIL)


def gen_pairs(n_per_intent=90):
    rows = []

    def add(q, dsl):
        rows.append({'question': wrap(q), 'dsl': dsl})

    for _ in range(n_per_intent):
        m = random.choice(list(SYNONYM))
        add(random.choice([
            f'what are typical {syn(m)} values', f'describe the {syn(m)}',
            f'what is the average {syn(m)}', f'summarise {syn(m)} across the fleet',
            f'what does {syn(m)} normally look like']), f'describe|-|{m}|-|-|-')

        v = random.choice(list(VARIANT))
        add(random.choice([
            f'what are operating conditions for {random.choice(VARIANT[v])} variants',
            f'describe conditions on {random.choice(VARIANT[v])} product']),
            f'describe|-|torque_nm,rotational_speed_rpm,tool_wear_min|-|-|product_type={v}')

        add(random.choice([
            'what is the overall failure rate', 'how often do machines fail',
            'what proportion of cycles fail', 'how many failures are there',
            'what is the breakdown rate']), 'rate|-|-|-|-|-')

        d = random.choice(['product_type', 'shift', 'machine_id'])
        add(random.choice([
            f'failure rate by {d}', f'break failures down by {d}',
            f'how does the failure rate differ by {d}', f'failures grouped by {d}']),
            f'rate|-|-|{d}|-|-')

        m = random.choice(['rotational_speed_rpm', 'torque_nm', 'tool_wear_min', 'power_w'])
        add(random.choice([
            f'why are we seeing more failures at high {syn(m)}',
            f'do failures increase with {syn(m)}', f'is {syn(m)} driving more breakdowns',
            f'failure rate across {syn(m)} bands']), f'rate|-|-|-|{m}:q5|-')

        ms = random.sample(['torque_nm', 'rotational_speed_rpm', 'tool_wear_min',
                            'temp_delta_k', 'power_w'], random.randint(2, 4))
        add(random.choice([
            'compare operating conditions of machines that failed versus those that did not',
            'contrast failed and healthy cycles', 'how do failures differ from normal runs',
            'difference between broken and working machines',
            'set failed against healthy conditions']),
            f"compare|{FAILED_HEALTHY}|{','.join(ms)}|-|-|cohens_d")

        m = random.choice(['tool_wear_min', 'rotational_speed_rpm', 'torque_nm'])
        add(random.choice([
            f'how does failure rate vary with {syn(m)}', f'trend of failures against {syn(m)}',
            f'relationship between failures and {syn(m)}',
            f'does failure rate change as {syn(m)} increases']), f'trend|-|-|-|{m}:q5|-')

        add(random.choice([
            'what drives failures', 'which variables separate failures from healthy operation',
            'what distinguishes broken machines', 'biggest factors behind breakdowns',
            'which parameters predict failure']), 'drivers|-|-|-|-|-')

        u = random.randint(1, 10000)
        add(random.choice([
            f'why did cycle {u} fail', f'what caused cycle {u} to fail',
            f'root cause for cycle {u}', f'diagnose cycle {u}',
            f'what went wrong on record {u}']), f'root_cause|-|-|-|-|udi={u}')

        add(random.choice([
            'what causes failures', 'what are the main failure modes',
            'attribute the failures', 'which modes are firing']), 'root_cause|-|-|-|-|-')

        delta = random.choice([-10, -8, -5, -3, 3, 5])
        m = random.choice(['torque_nm', 'rotational_speed_rpm'])
        verb = 'reduce' if delta < 0 else 'increase'
        unit = 'Nm' if m == 'torque_nm' else 'rpm'
        add(random.choice([
            f'what if we {verb} {syn(m)} by {abs(delta)} {unit}',
            f'suppose we {verb} {syn(m)} {abs(delta)} {unit}',
            f'impact of {verb[:-1]}ing {syn(m)} by {abs(delta)} {unit}']),
            f'counterfactual|-|-|-|-|{m}{delta:+g}')

        w, t = random.randint(80, 240), random.randint(30, 70)
        add(random.choice([
            f'what is the safe torque range at {w} minutes of wear',
            f'operating window at {t} Nm and {w} min wear',
            f'what should i set torque to with {w} min of wear']),
            f'envelope|-|-|-|-|tool_wear_min={w},torque_nm={t}')

        add(random.choice([
            f'when will the tool cross the overstrain limit at {w} min wear',
            f'give me the time to crossing at {w} minutes of wear',
            f'how long until failure at {t} Nm',
            f'estimate the crossing time at {w} minutes of wear',
            f'predict when we cross with {w} min of wear']),
            f'forecast|-|-|-|-|tool_wear_min={w},torque_nm={t}')

        add(random.choice([
            'show me the cycles closest to failing', 'list the riskiest records',
            'which cycles are nearest the boundary', 'give me examples of near misses']),
            'records|-|-|-|-|order=closest_to_failure')

        add(random.choice([
            'can i trust this data', 'are there problems with the dataset',
            'data quality report', 'is the labelling reliable',
            'are the thresholds still accurate']), 'data_quality|-|-|-|-|-')
    return rows


def held_out_split():
    """Replay the notebook's RNG exactly to recover the same 120 test rows."""
    random.seed(11)
    pairs = gen_pairs()
    seen, uniq = set(), []
    for r in pairs:
        k = r['question'].lower().strip()
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    random.shuffle(uniq)
    split = int(len(uniq) * 0.9)
    return uniq[:split], uniq[split:]


def ask(model: str, question: str, timeout: float = 120.0):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": question}],
        "stream": False,
        # Greedy. A planner that must emit one canonical line has no business
        # sampling - two runs of the same question must give the same plan.
        "options": {"temperature": 0, "top_k": 1, "top_p": 1.0,
                    "num_predict": 48, "seed": 0},
    }).encode()
    req = urllib.request.Request(OLLAMA, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read())
    wall = time.perf_counter() - t0
    text = (payload.get("message") or {}).get("content", "")
    # Qwen3 emits a reasoning block; the plan is the last non-empty line.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return (lines[-1] if lines else ""), wall, payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="margin-planner")
    ap.add_argument("--limit", type=int, default=0, help="0 = the whole held-out set")
    ap.add_argument("--out", default="artifacts/planner_card.json")
    args = ap.parse_args()

    train_rows, test_rows = held_out_split()
    if args.limit:
        test_rows = test_rows[:args.limit]
    print(f"train {len(train_rows)}   held-out {len(test_rows)}   model {args.model}")

    try:
        _, _, probe = ask(args.model, "what is the overall failure rate")
    except urllib.error.URLError as e:
        print(f"\ncannot reach Ollama at {OLLAMA}: {e}\n  start it with:  ollama serve")
        return 2

    exact = valid = op_ok = 0
    walls, decode_rates, misses = [], [], []
    for i, r in enumerate(test_rows, 1):
        pred, wall, payload = ask(args.model, r["question"])
        walls.append(wall)
        n_tok = payload.get("eval_count") or 0
        dur = (payload.get("eval_duration") or 0) / 1e9
        if n_tok and dur:
            decode_rates.append(n_tok / dur)
        if pred == r["dsl"]:
            exact += 1
        else:
            misses.append({"question": r["question"], "want": r["dsl"], "got": pred})
        try:
            plan = dsl_to_plan(pred)
            valid += 1
            if plan["op"] == r["dsl"].split("|")[0]:
                op_ok += 1
        except Exception:
            pass
        if i % 20 == 0:
            print(f"  {i}/{len(test_rows)}   exact {exact/i:.3f}")

    n = len(test_rows)
    walls.sort()
    card = {
        "model": args.model,
        "held_out": n,
        "dsl_exact_match": round(exact / n, 4),
        "plan_validity": round(valid / n, 4),
        "op_accuracy": round(op_ok / n, 4),
        "latency_ms_p50": round(walls[len(walls) // 2] * 1000, 1),
        "latency_ms_p95": round(walls[int(len(walls) * 0.95)] * 1000, 1),
        "latency_ms_max": round(walls[-1] * 1000, 1),
        "decode_tok_per_s": round(statistics.mean(decode_rates), 1) if decode_rates else None,
        "decoding": "greedy (temperature 0, top_k 1)",
        "misses": misses[:25],
    }
    print()
    for k in ("held_out", "dsl_exact_match", "plan_validity", "op_accuracy",
              "latency_ms_p50", "latency_ms_p95", "decode_tok_per_s"):
        print(f"  {k:<20} {card[k]}")
    if misses:
        print(f"\n  {len(misses)} misses; first three:")
        for m in misses[:3]:
            print(f"    Q    {m['question'][:70]}")
            print(f"    want {m['want']}")
            print(f"    got  {m['got']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(card, indent=2))
    print(f"\n  card -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
