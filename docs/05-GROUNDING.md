# 05 - Numeric Grounding (Gate 4)

> **No number in any answer is authored by a language model.**
>
> Not "unlikely to be wrong." Structurally impossible.

---

## 1. The problem

Numeric hallucination is the failure mode that destroys trust in an industrial
copilot, and small deviations are the dangerous ones - reporting 6.0 % where the
truth is 5.7 % is worse than an obvious error, because it survives review.

Retrieval and citations improve transparency but **cannot guarantee accuracy**.
The number still passes through the model's output distribution.

There is a second, subtler class that naive verification misses:
**mis-attribution.** The model quotes a *correct* number from the *wrong* cohort.
"Failed machines averaged 40.0 N·m" - a real figure, but it is the *healthy*
cohort's mean. A verifier that only asks *"does this numeral appear in the
evidence?"* passes it.

---

## 2. The protocol: Proof-Carrying Numbers

We implement **PCN** ([arXiv:2509.06902](https://arxiv.org/abs/2509.06902)), which
supplies formal guarantees rather than a heuristic:

> "Numeric values are emitted as **claim-bound tokens** connected to structured
> claims. A verifier checks each token against a declared policy covering exact
> equality, rounding, aliases, or tolerance ranges. **Only claim-checked numbers
> are marked as verified, and all others default to unverified.**"

Key property: verification lives in the **renderer, not the model**. The paper
proves soundness, completeness under honest tokens, fail-closed behaviour, and
monotonicity under policy refinement.

Citing a protocol with proofs is materially stronger than describing a homegrown
regex check.

---

## 3. Mechanism

### 3.1 Slot IDs encode the cohort

Every quantity in the evidence bundle is a `Slot` whose ID is **fully qualified**:

```
failed.torque_nm.mean          = 49.71  [N·m]  n=339
healthy.torque_nm.mean         = 39.63  [N·m]  n=9661
failed.torque_nm.mean.ci_low   = 48.62  [N·m]
osf.margin.min                 = −1433  [min·N·m]
```

The cohort is part of the identifier. **Mis-attribution becomes unrepresentable**
- you cannot write `healthy`'s value while referring to `failed`, because the
reference *is* the cohort.

### 3.2 The narrator is forbidden digits

System instruction, enforced by the verifier rather than trusted:

> Write the finding as prose. Refer to every quantity **only** by its slot ID in
> `{{double braces}}`. You may not write any digit. You may not round, convert
> units, or compute. If a quantity you want is not in the bundle, say so.

Draft output:

```
Failed cycles ran at {{failed.torque_nm.mean}} against {{healthy.torque_nm.mean}}
for healthy cycles - a difference of {{delta.torque_nm}} with a standardised
effect of {{effect.torque_nm.cohens_d}}.
```

### 3.3 The verifier is fail-closed

```
1. Scan draft for any bare numeral not inside {{...}}     → REJECT
2. Resolve every {{slot}} against the bundle
   unknown slot                                            → REJECT
3. Substitute values, formatted with unit and precision
   from the slot, never from the model
4. Attach provenance: n, CI, filters, plan hash
5. Re-scan rendered output; unmarked numeral               → REJECT
```

Rejection triggers **one** regeneration with the error fed back. A second failure
degrades to **template narration** - less fluent, still correct. The system
degrades to *less readable*, never to *wrong*.

### 3.4 Allowed exceptions, explicitly whitelisted

Narrow, enumerated, and checked:

| Allowed | Example |
|---|---|
| Numerals quoted from the user's own question | "you asked about 1380 rpm" |
| Ordinals in enumerated structure | "first", "second" |
| Threshold constants cited **from the KB** | 8.6 K, resolved as `kb.hdf.temp_threshold` |

KB constants are slots too. They are never literals in the prompt.

---

## 4. Measured in evals

| Metric | Definition | Target |
|---|---|---|
| **Unsourced numeral rate** | numerals in final answer with no slot origin | **0.000** |
| **Numeric exactness** | rendered values matching `evals/reference.py` | **1.000** |
| Mis-attribution rate | slot cohort ≠ claimed cohort | **0.000** |
| Verifier rejection rate | drafts rejected before render | tracked (health signal) |
| Degradation rate | answers falling back to template | tracked |

The first three are **hard gates**. A non-zero value fails the build.

Note that exactness is 1.000 *by construction*, not by measurement - the eval
exists to prove the construction holds, and to catch regressions in the plumbing.

---

## 5. Provenance and replay

Every answer carries a replay handle:

```
plan_hash      sha256 of the canonicalised Analysis Plan
kb_version     knowledge base version + content hash
data_version   warehouse fingerprint
elapsed_ms     per stage
tier           cache | grammar | llm
```

Given a handle, `copilot replay <hash>` re-executes and returns either the
identical result or a structured diff showing exactly what changed - the data,
the KB, or the plan.

This is what regulated industries actually require during incident review, and it
is nearly absent from LLM products.

---

## 6. What this does not protect against

Honest boundaries:

- **A correct number in a misleading frame.** PCN guarantees the number is real
  and correctly attributed. It cannot guarantee the *analysis* was the right one
  to run. That is what premise verification (Gate 1) and collinearity warnings
  address, partially.
- **A wrong knowledge base.** If the KB says 11,000 and reality is 10,200, every
  margin is verifiably computed and wrong. Gate 3 exists for this.
- **A broken sensor.** Verified arithmetic on a bad reading is verified garbage.
  Gate 2 and interval margins exist for this.

Four gates, because one is never enough.

---

**Next:** [06-RELIABILITY.md](06-RELIABILITY.md) - drift, bad sensors, and stale rules.
