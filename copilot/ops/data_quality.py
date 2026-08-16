"""`data_quality` — what is wrong with this dataset, computed not recited.

A copilot that surfaces its data's defects is more useful than one that silently
averages over them. Three issues in AI4I are not stated on the dataset page and
were found by inspection:

  * 9 rows are labelled failures with no documented mode;
  * the RNF flag does not roll up into the failure label (19 rows, 1 overlap);
  * the prose describes 120 tool events but the TWF column flags 46.

Every finding here is recomputed live against the warehouse rather than read
from the knowledge base, so this op doubles as a regression check: if the data
changes and the KB does not, the two disagree and the disagreement is reported.
"""

from __future__ import annotations

from copilot.evidence import EvidenceBundle, Severity
from copilot.ir import AnalysisPlan, OpName
from copilot.knowledge import failure_modes
from copilot.ops.registry import TABLE, ExecutionContext, new_bundle, register

# Physics invariants — quantities that must hold regardless of operating point.
# A violation means instrumentation, not operations. See docs/06-RELIABILITY.md.
_INVARIANTS = (
    (
        "I1",
        "process temperature exceeds air temperature",
        "SELECT count(*) FROM {t} WHERE process_temperature_k <= air_temperature_k",
        0,
    ),
)


#: Which channels each invariant vouches for. An answer whose margins depend on
#: a channel with a violated invariant is computed on data we already know is
#: physically impossible, and it must say so.
INVARIANT_CHANNELS: dict[str, tuple[str, ...]] = {
    "I1": ("air_temp_k", "process_temp_k", "temp_delta_k"),
}


def check_invariants(con) -> dict[str, int]:
    """Run every invariant once and return the violation counts.

    Cheap enough to do at engine build (three aggregate scans) and far too
    expensive to repeat per question, which is why the result is cached and
    consulted rather than recomputed.
    """
    out: dict[str, int] = {}
    for code, _desc, sql, _expected in _INVARIANTS:
        out[code] = int(con.execute(sql.format(t=TABLE)).fetchone()[0])
    return out


@register(OpName.DATA_QUALITY)
def data_quality(plan: AnalysisPlan, ctx: ExecutionContext) -> EvidenceBundle:
    bundle = new_bundle(plan, ctx)
    q = lambda sql: ctx.cursor.execute(sql.format(t=TABLE)).fetchone()  # noqa: E731

    total, failures = q("SELECT count(*), sum(machine_failure) FROM {t}")
    bundle.put("dataset.rows", int(total), unit="count", sig_figs=8)
    bundle.put("dataset.failures", int(failures), unit="count", sig_figs=8)

    # --- finding 1: orphan failures ---------------------------------------
    orphans, orphan_margin = q(
        "SELECT count(*), avg(worst_normalised_margin) FROM {t} "
        "WHERE machine_failure = 1 AND twf = 0 AND hdf = 0 AND pwf = 0 AND osf = 0"
    )
    healthy_margin = q(
        "SELECT avg(worst_normalised_margin) FROM {t} WHERE machine_failure = 0"
    )[0]
    bundle.put("orphan_failures.count", int(orphans), unit="count", sig_figs=8)
    if orphans:
        bundle.put("orphan_failures.mean_margin", float(orphan_margin), unit="ratio", sig_figs=3)
        bundle.put("healthy.mean_margin", float(healthy_margin), unit="ratio", sig_figs=3)
        bundle.warn(
            "data_quality",
            f"{int(orphans)} rows are labelled failures with no documented mode. Their "
            "operating parameters sit in the normal range, so the cause is not "
            "determinable from the published data. They are excluded from root-cause "
            "attribution rather than assigned a mode.",
            severity=Severity.WARNING,
            affects=["orphan_failures.count"],
        )

    # --- finding 2: RNF does not roll up ----------------------------------
    rnf_total, rnf_counted = q(
        "SELECT sum(rnf), sum(CASE WHEN rnf = 1 AND machine_failure = 1 THEN 1 ELSE 0 END) FROM {t}"
    )
    bundle.put("rnf.flagged", int(rnf_total), unit="count", sig_figs=8)
    bundle.put("rnf.also_machine_failure", int(rnf_counted), unit="count", sig_figs=8)
    if int(rnf_total) != int(rnf_counted):
        bundle.warn(
            "data_quality",
            f"{int(rnf_total)} rows carry the RNF flag but only {int(rnf_counted)} also "
            "set Machine failure. The published RNF flag therefore does not roll up "
            "into the failure label. All rates here are computed against Machine "
            "failure, with RNF reported separately.",
            severity=Severity.WARNING,
            affects=["rnf.flagged"],
        )

    # --- finding 3: documentation vs column on TWF ------------------------
    twf_flagged, twf_in_window, window_rows = q(
        "SELECT sum(twf), sum(CASE WHEN twf = 1 AND twf_window THEN 1 ELSE 0 END), "
        "sum(CASE WHEN twf_window THEN 1 ELSE 0 END) FROM {t}"
    )
    bundle.put("twf.flagged", int(twf_flagged), unit="count", sig_figs=8)
    bundle.put("twf.window_rows", int(window_rows), unit="count", sig_figs=8)
    bundle.put("twf.outside_window", int(twf_flagged) - int(twf_in_window),
               unit="count", sig_figs=8)
    bundle.put(
        "twf.in_window_failure_rate",
        int(twf_in_window) / int(window_rows) * 100.0,
        unit="%", sig_figs=3,
    )
    documented = _documented_twf_events()
    if documented and documented != int(twf_flagged):
        bundle.warn(
            "data_quality",
            f"The dataset documentation describes {documented} tool-wear events but the "
            f"TWF column flags {int(twf_flagged)} rows. Every statement here is computed "
            "from the column, not the prose.",
            severity=Severity.INFO,
            affects=["twf.flagged"],
        )

    # --- rule audit: KB vs data -------------------------------------------
    audit = q(
        "SELECT "
        " sum(CASE WHEN hdf_rule AND hdf = 0 THEN 1 ELSE 0 END),"
        " sum(CASE WHEN NOT hdf_rule AND hdf = 1 THEN 1 ELSE 0 END),"
        " sum(CASE WHEN pwf_rule AND pwf = 0 THEN 1 ELSE 0 END),"
        " sum(CASE WHEN NOT pwf_rule AND pwf = 1 THEN 1 ELSE 0 END),"
        " sum(CASE WHEN osf_rule AND osf = 0 THEN 1 ELSE 0 END),"
        " sum(CASE WHEN NOT osf_rule AND osf = 1 THEN 1 ELSE 0 END) FROM {t}"
    )
    disagreements = 0
    for i, mode in enumerate(("HDF", "PWF", "OSF")):
        fp, fn = int(audit[2 * i] or 0), int(audit[2 * i + 1] or 0)
        bundle.put(f"rule_audit.{mode}.false_positives", fp, unit="count", sig_figs=8)
        bundle.put(f"rule_audit.{mode}.false_negatives", fn, unit="count", sig_figs=8)
        disagreements += fp + fn
    bundle.put("rule_audit.total_disagreements", disagreements, unit="count", sig_figs=8)
    if disagreements:
        bundle.warn(
            "kb_drift",
            f"The knowledge base and the data disagree on {disagreements} rows. Either a "
            "threshold has moved or the data has changed. Investigate before trusting "
            "any root-cause answer.",
            severity=Severity.CRITICAL,
            affects=["rule_audit.total_disagreements"],
        )

    # --- invariants (Gate 2) ----------------------------------------------
    violations = 0
    for code, description, sql, expected in _INVARIANTS:
        actual = int(q(sql)[0])
        bundle.put(f"invariant.{code}.violations", actual, unit="count", sig_figs=8)
        bundle.put(f"invariant.{code}.description", description, unit="")
        violations += abs(actual - expected)
    delta = q("SELECT avg(temp_delta_k), stddev_samp(temp_delta_k) FROM {t}")
    bundle.put("invariant.I2.mean_temp_delta", float(delta[0]), unit="ΔK")
    bundle.put("invariant.I2.sd_temp_delta", float(delta[1]), unit="ΔK")
    corr = q("SELECT corr(rotational_speed_rpm, torque_nm) FROM {t}")[0]
    bundle.put("invariant.I3.rpm_torque_corr", float(corr), unit="", sig_figs=4)

    if violations:
        bundle.warn(
            "sensor_suspect",
            f"{violations} physics invariant violation(s). An invariant break indicates "
            "instrumentation, not a change in operations.",
            severity=Severity.CRITICAL,
        )

    verdict = (
        "trustworthy with documented caveats"
        if not disagreements and not violations
        else "requires investigation"
    )
    bundle.put("verdict", verdict, unit="")
    bundle.provenance = bundle.provenance.model_copy(update={"row_count": int(total)})
    bundle.summary = (
        f"data quality: {verdict}; {int(orphans)} orphan failures, "
        f"{disagreements} rule disagreements"
    )
    return bundle


def _documented_twf_events() -> int | None:
    """The event count the dataset page's prose claims.

    Read from an explicit KB field rather than scraped out of the prose. An
    earlier version parsed the first large number it found in the note and
    picked up the window size instead — a good illustration of why numbers
    belong in fields, not sentences.
    """
    for mode in failure_modes()["modes"]:
        if mode["code"] == "TWF":
            return mode.get("calibration", {}).get("documented_events")
    return None


__all__ = ["data_quality"]
