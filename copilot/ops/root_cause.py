"""`root_cause` — attribute failures by evaluating the documented physics.

This is the operator the whole architecture exists to enable. On this dataset
HDF, PWF and OSF are recovered from their documented rules with zero false
positives and zero false negatives across all 10,000 rows, so attribution is
*arithmetic with a citation*, not a classifier with a confidence.

Five contracts, each of which a conventional classifier violates:

  1. Report EVERY firing mode. 23 failures fire two or more simultaneously; a
     single-label model must pick one and is wrong on the rest by construction.
  2. Report the MARGIN to every boundary, fired or not. "How close were the
     others?" is the question an engineer asks next.
  3. Report the CROSSING POINT where the trajectory is known.
  4. TWF returns a PROBABILITY. It is stochastic (43 of 790 in-window rows fail);
     stating a certainty would be a lie.
  5. RNF is NEVER attributed, and orphan failures return cause_undetermined.
     Both are verified-correct answers, not fallbacks.
"""

from __future__ import annotations

from typing import Any

from copilot.evidence import EvidenceBundle, Quality, Severity
from copilot.ir import AnalysisPlan, OpName
from copilot.knowledge import mode_index
from copilot.ops.registry import (
    TABLE,
    ExecutionContext,
    cohort_where,
    new_bundle,
    register,
)

# Boundary definitions, mirroring copilot/knowledge/failure_modes.yaml. The
# margin column is materialised at ingest; these describe how to read it.
_BOUNDARIES: list[dict[str, Any]] = [
    {
        "mode": "HDF",
        "fired_sql": "(temp_delta_k < 8.6 AND rotational_speed_rpm < 1380)",
        "margins": [
            ("temp_delta_margin_k", "temp_delta_margin_k", "ΔK"),
            ("speed_margin_rpm", "speed_margin_rpm", "Δrpm"),
        ],
        "conjunctive": True,  # BOTH conditions required
    },
    {
        "mode": "PWF",
        "fired_sql": "(power_w < 3500 OR power_w > 9000)",
        "margins": [
            ("power_low_margin_w", "power_low_margin_w", "ΔW"),
            ("power_high_margin_w", "power_high_margin_w", "ΔW"),
        ],
        "conjunctive": False,  # EITHER side fails
    },
    {
        "mode": "OSF",
        "fired_sql": "(overstrain_min_nm > osf_threshold_min_nm)",
        "margins": [("overstrain_margin_min_nm", "overstrain_margin_min_nm", "Δmin·N·m")],
        "conjunctive": True,
    },
]

_ROW_COLUMNS = (
    "udi, product_type, machine_id, machine_failure, twf, hdf, pwf, osf, rnf, "
    "air_temperature_k, process_temperature_k, temp_delta_k, rotational_speed_rpm, "
    "torque_nm, tool_wear_min, power_w, overstrain_min_nm, osf_threshold_min_nm, "
    "temp_delta_margin_k, speed_margin_rpm, power_low_margin_w, power_high_margin_w, "
    "overstrain_margin_min_nm, worst_normalised_margin, "
    "hdf_distance, pwf_distance, osf_distance"
)

TWF_WINDOW = (200.0, 240.0)
TWF_IN_WINDOW_RATE = 43 / 790  # measured; see docs/01-DATASET.md §3


@register(OpName.ROOT_CAUSE)
def root_cause(plan: AnalysisPlan, ctx: ExecutionContext) -> EvidenceBundle:
    bundle = new_bundle(plan, ctx)
    where, params = cohort_where(plan, None)

    total = ctx.cursor.execute(
        f"SELECT count(*) FROM {TABLE} WHERE {where}", params  # noqa: S608
    ).fetchone()[0]

    if total == 0:
        bundle.put("cause.verdict", None, quality=Quality.ABSTAIN)
        bundle.warn("abstained", "No rows match this scope, so there is nothing to attribute.",
                    severity=Severity.CRITICAL)
        bundle.summary = "root cause: no matching rows"
        return bundle

    if total == 1:
        _single_row(bundle, ctx, where, params)
    else:
        _cohort(bundle, ctx, where, params, int(total))

    bundle.provenance = bundle.provenance.model_copy(update={"row_count": int(total)})
    return bundle


# --------------------------------------------------------------------------
# Single observation — the "why did THIS fail?" path
# --------------------------------------------------------------------------


def _single_row(bundle: EvidenceBundle, ctx: ExecutionContext, where: str, params: list) -> None:
    row = ctx.cursor.execute(
        f"SELECT {_ROW_COLUMNS} FROM {TABLE} WHERE {where}", params  # noqa: S608
    ).df().iloc[0].to_dict() if _has_pandas() else _fetch_dict(ctx, where, params)

    bundle.put("cycle.udi", int(row["udi"]), unit="count", sig_figs=8)
    bundle.put("cycle.product_type", str(row["product_type"]), unit="")
    bundle.put("cycle.failed", "yes" if row["machine_failure"] else "no", unit="")

    fired: list[str] = []
    for spec in _BOUNDARIES:
        mode = spec["mode"]
        is_fired = bool(row[f"{mode.lower()}_rule_eval"]) if f"{mode.lower()}_rule_eval" in row \
            else _evaluate(spec, row)
        bundle.put(f"{mode}.fired", "yes" if is_fired else "no", unit="")
        for slot_name, column, unit in spec["margins"]:
            bundle.put(f"{mode}.{slot_name}", float(row[column]), unit=unit, sig_figs=5)
        if is_fired:
            fired.append(mode)

    # TWF — stochastic, so a probability and never a certainty.
    wear = float(row["tool_wear_min"])
    in_window = TWF_WINDOW[0] <= wear <= TWF_WINDOW[1]
    bundle.put("TWF.in_window", "yes" if in_window else "no", unit="")
    bundle.put("TWF.tool_wear", wear, unit="min")
    bundle.put("TWF.wear_to_window", TWF_WINDOW[0] - wear, unit="Δmin")
    if in_window:
        bundle.put("TWF.failure_probability", TWF_IN_WINDOW_RATE * 100.0, unit="%", sig_figs=2)
        bundle.warn(
            "data_quality",
            "Tool wear is inside the documented 200-240 min replacement window. TWF is "
            "stochastic, so this is a per-cycle probability, not a prediction that the "
            "tool will fail.",
            severity=Severity.INFO,
            affects=["TWF.failure_probability"],
        )
        if bool(row["twf"]):
            fired.append("TWF")

    _crossing_point(bundle, row)

    # Verdict.
    if fired:
        bundle.put("cause.verdict", " + ".join(fired), unit="")
        bundle.put("cause.mode_count", len(fired), unit="count", sig_figs=8)
        if len(fired) > 1:
            bundle.warn(
                "data_quality",
                f"{len(fired)} failure modes fired simultaneously ({', '.join(fired)}). "
                "A single-label classifier would report only one of them.",
                severity=Severity.INFO,
            )
    elif row["machine_failure"]:
        _orphan(bundle, row)
    else:
        bundle.put("cause.verdict", "no failure mode triggered", unit="")
        _explain_absence(bundle, row)

    if row["rnf"]:
        bundle.warn(
            "data_quality",
            "This cycle carries the RNF flag. RNF is a documented 0.1% background "
            "failure rate independent of every process parameter; it cannot be "
            "attributed to operating conditions and no root cause is offered for it.",
            severity=Severity.WARNING,
        )

    bundle.summary = f"root cause for UDI {int(row['udi'])}: " + (
        " + ".join(fired) if fired else "none determined"
    )


def _evaluate(spec: dict[str, Any], row: dict[str, Any]) -> bool:
    """Fire the rule from the materialised margins. Negative margin == violated."""
    margins = [float(row[column]) for _, column, _ in spec["margins"]]
    if spec["conjunctive"]:
        return all(m < 0 for m in margins)
    return any(m < 0 for m in margins)


def _crossing_point(bundle: EvidenceBundle, row: dict[str, Any]) -> None:
    """Where along the wear trajectory did OSF become inevitable?

    Overstrain is wear x torque, so at constant torque the crossing wear is
    threshold / torque. This is the quantitative answer a classifier cannot give.
    """
    torque = float(row["torque_nm"])
    threshold = float(row["osf_threshold_min_nm"])
    if torque <= 0:
        return
    crossing_wear = threshold / torque
    bundle.put("OSF.crossing_wear_min", crossing_wear, unit="min", sig_figs=4)
    wear = float(row["tool_wear_min"])
    if wear > crossing_wear:
        bundle.put("OSF.exceeded_by_min", wear - crossing_wear, unit="Δmin", sig_figs=4)


def _orphan(bundle: EvidenceBundle, row: dict[str, Any]) -> None:
    """A labelled failure with no documented mode.

    Nine such rows exist. Tested for hidden structure and found none: their mean
    worst-normalised margin is 0.057 against 0.068 for healthy rows. Reporting
    "undetermined" here is a verified-correct answer.
    """
    bundle.put("cause.verdict", "cause_undetermined", unit="")
    bundle.put(
        "cause.worst_normalised_margin", float(row["worst_normalised_margin"]),
        unit="ratio", sig_figs=3,
    )
    bundle.warn(
        "data_quality",
        "This cycle is labelled a failure but no documented mode fires, and all "
        "margins sit in the normal range. It is one of 9 such rows in the published "
        "dataset. The cause cannot be determined from the available parameters — "
        "this is a limitation of the data, not an inconclusive analysis.",
        severity=Severity.WARNING,
        affects=["cause.verdict"],
    )


def _explain_absence(bundle: EvidenceBundle, row: dict[str, Any]) -> None:
    """Answer "why did nothing fire?" — classifiers cannot explain a non-event.

    Distance is computed per RULE, not per condition. HDF is conjunctive, so a
    row can sit far below 1380 rpm and still be nowhere near an HDF failure if
    the thermal gradient is healthy; its binding constraint is the *larger* of
    the two normalised margins. Reporting the raw speed margin here would tell
    an engineer they were past a boundary they had not approached.
    """
    rules = [
        ("heat dissipation (HDF)", float(row["hdf_distance"])),
        ("power band (PWF)", float(row["pwf_distance"])),
        ("overstrain (OSF)", float(row["osf_distance"])),
    ]
    name, distance = min(rules, key=lambda r: r[1])

    # Report the binding condition in its native units, so the answer is actionable.
    binding: tuple[str, float, str]
    if name.startswith("heat"):
        temp_m, speed_m = float(row["temp_delta_margin_k"]), float(row["speed_margin_rpm"])
        binding = (
            ("thermal gradient", temp_m, "ΔK")
            if temp_m / 8.6 >= speed_m / 1380.0
            else ("rotational speed", speed_m, "Δrpm")
        )
    elif name.startswith("power"):
        low, high = float(row["power_low_margin_w"]), float(row["power_high_margin_w"])
        binding = ("stall floor", low, "ΔW") if low / 3500.0 <= high / 9000.0 else (
            "overload ceiling", high, "ΔW"
        )
    else:
        binding = ("strain budget", float(row["overstrain_margin_min_nm"]), "Δmin·N·m")

    bundle.put("closest.rule", name, unit="")
    bundle.put("closest.normalised_distance", distance, unit="ratio", sig_figs=3)
    bundle.put("closest.binding_condition", binding[0], unit="")
    bundle.put("closest.margin", binding[1], unit=binding[2], sig_figs=5)


# --------------------------------------------------------------------------
# Cohort — the "what causes failures here?" path
# --------------------------------------------------------------------------


def _cohort(
    bundle: EvidenceBundle, ctx: ExecutionContext, where: str, params: list, total: int
) -> None:
    row = ctx.cursor.execute(
        f"""SELECT
              sum(machine_failure)::BIGINT                                  AS failures,
              sum(CASE WHEN {_BOUNDARIES[0]['fired_sql']} THEN 1 ELSE 0 END)::BIGINT AS hdf,
              sum(CASE WHEN {_BOUNDARIES[1]['fired_sql']} THEN 1 ELSE 0 END)::BIGINT AS pwf,
              sum(CASE WHEN {_BOUNDARIES[2]['fired_sql']} THEN 1 ELSE 0 END)::BIGINT AS osf,
              sum(twf)::BIGINT                                              AS twf,
              sum(rnf)::BIGINT                                              AS rnf,
              sum(CASE WHEN machine_failure = 1 AND twf = 0 AND hdf = 0
                        AND pwf = 0 AND osf = 0 THEN 1 ELSE 0 END)::BIGINT  AS orphans,
              sum(CASE WHEN machine_failure = 1 AND
                       (CASE WHEN {_BOUNDARIES[0]['fired_sql']} THEN 1 ELSE 0 END +
                        CASE WHEN {_BOUNDARIES[1]['fired_sql']} THEN 1 ELSE 0 END +
                        CASE WHEN {_BOUNDARIES[2]['fired_sql']} THEN 1 ELSE 0 END +
                        twf) > 1 THEN 1 ELSE 0 END)::BIGINT                 AS multi,
              -- DISTINCT failures explained by any deterministic rule. Summing
              -- the per-mode counts instead would double-count the 21 rows that
              -- fire two modes, and report more explained failures than exist.
              sum(CASE WHEN machine_failure = 1 AND
                       ({_BOUNDARIES[0]['fired_sql']} OR {_BOUNDARIES[1]['fired_sql']}
                        OR {_BOUNDARIES[2]['fired_sql']}) THEN 1 ELSE 0 END)::BIGINT AS explained
            FROM {TABLE} WHERE {where}""",  # noqa: S608
        params,
    ).fetchone()

    failures, hdf, pwf, osf, twf, rnf, orphans, multi, explained = (int(v or 0) for v in row)

    bundle.put("cohort.n", total, unit="count", sig_figs=8)
    bundle.put("cohort.failures", failures, unit="count", sig_figs=8)
    for mode, count in (("HDF", hdf), ("PWF", pwf), ("OSF", osf), ("TWF", twf)):
        bundle.put(f"{mode}.count", count, unit="count", sig_figs=8)
        bundle.put(
            f"{mode}.share_of_failures",
            (count / failures * 100.0) if failures else None,
            unit="%", sig_figs=3,
            quality=Quality.OK if failures else Quality.ABSTAIN,
        )

    bundle.put("orphans.count", orphans, unit="count", sig_figs=8)
    bundle.put("multi_mode.count", multi, unit="count", sig_figs=8)
    bundle.put("RNF.count", rnf, unit="count", sig_figs=8)

    bundle.put("explained.deterministic", explained, unit="count", sig_figs=8)

    # ── Which KIND of claim each remaining failure permits.
    #
    # The unexplained remainder is not one thing. TWF has no crisp boundary, so
    # the honest object is an interval with a coverage guarantee. RNF has no
    # predicate over any measured parameter, so it is not merely hard to predict
    # — it is impossible, and offering a number would be fabricating one.
    #
    # Reporting a single "explained" figure invited the reader to treat the
    # remainder as a shortfall to be closed. It is not. It is the process
    # telling us which of its failures are rule-shaped.
    bundle.put("regime.exact", explained, unit="count", sig_figs=8)
    bundle.put("regime.statistical", twf, unit="count", sig_figs=8)
    bundle.put("regime.irreducible", rnf, unit="count", sig_figs=8)
    if failures:
        bundle.put("regime.exact_share", explained / failures * 100.0,
                   unit="%", sig_figs=3)
        bundle.put("regime.irreducible_share", rnf / failures * 100.0,
                   unit="%", sig_figs=3)
    if rnf:
        bundle.warn(
            "exploratory",
            "Random failures carry no predicate over any measured parameter. "
            "They are not weakly predicted here, they are unpredictable by the "
            "process definition, and no operating change addresses them.",
            affects=["regime.irreducible"],
        )
    bundle.put(
        "explained.share_of_failures",
        (explained / failures * 100.0) if failures else None,
        unit="%", sig_figs=3,
        quality=Quality.OK if failures else Quality.ABSTAIN,
    )

    if orphans:
        bundle.warn(
            "data_quality",
            f"{orphans} failure(s) in this scope have no documented mode and are "
            "excluded from attribution. Their operating parameters sit in the normal "
            "range; the cause is not determinable from the published data.",
            affects=["orphans.count"],
        )
    if multi:
        bundle.warn(
            "data_quality",
            f"{multi} failure(s) fired more than one mode simultaneously. Shares by "
            "mode therefore sum to more than the failure count; this is correct, not "
            "double counting.",
            severity=Severity.INFO,
        )
    if rnf:
        bundle.warn(
            "data_quality",
            f"{rnf} row(s) carry the RNF flag. RNF is parameter-independent by "
            "definition and is reported separately, never attributed.",
            severity=Severity.INFO,
        )

    modes = mode_index()
    dominant = max((("HDF", hdf), ("PWF", pwf), ("OSF", osf), ("TWF", twf)), key=lambda kv: kv[1])
    if dominant[1]:
        bundle.put("dominant.mode", dominant[0], unit="")
        bundle.put("dominant.name", modes[dominant[0]]["name"], unit="")

    bundle.summary = (
        f"root cause across {total} cycles: {failures} failures, "
        f"{explained} explained by deterministic rules"
    )


# --------------------------------------------------------------------------


def _has_pandas() -> bool:
    return False  # avoid a pandas dependency on the hot path


def _fetch_dict(ctx: ExecutionContext, where: str, params: list) -> dict[str, Any]:
    cur = ctx.cursor.execute(
        f"SELECT {_ROW_COLUMNS} FROM {TABLE} WHERE {where}", params  # noqa: S608
    )
    names = [d[0] for d in cur.description]
    return dict(zip(names, cur.fetchone()))


__all__ = ["root_cause"]
