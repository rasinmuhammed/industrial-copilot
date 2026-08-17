from __future__ import annotations

from typing import Any

from copilot.evidence import EvidenceBundle, Provenance, Quality, Severity, Slot, Warning_
from copilot.ir import AnalysisPlan, OpName
from copilot.ops.registry import ExecutionContext, register
from copilot.reliability.invariants import diagnose_drift

@register(OpName.DRIFT)
def drift_op(plan: AnalysisPlan, ctx: ExecutionContext) -> EvidenceBundle:
    report = diagnose_drift(
        ctx.con,
        window_where="udi > 5000",
        baseline_where="udi <= 5000"
    )
    
    warnings = []
    if report.verdict.value != "baseline":
        warnings.append(
            Warning_(code="DRIFT_DETECTED", severity=Severity.CRITICAL, message=report.explanation)
        )
        
    slots = {
        "drift.explanation": Slot(value=report.explanation, unit="text", n=10000, quality=Quality.OK),
        "drift.verdict": Slot(value=report.verdict.value, unit="text", n=10000, quality=Quality.OK)
    }
    
    return EvidenceBundle(
        provenance=Provenance(plan_hash=plan.hash(), model=None),
        slots=slots,
        rows=[{"verdict": report.verdict.value, "explanation": report.explanation}],
        warnings=warnings
    )
