# Shared brief for every competing design agent

## The ask
Design a system for CONTINUOUS 2-D cursor control on a screen from a g.tec 32-channel DRY hybrid-electrode
EEG cap (g.Nautilus-class, g.SAHARA-style dry/hybrid electrodes), sampled at 500 Hz, standard 10-20 layout
including FP1/FP2, no dedicated EOG pair, no eye tracker today (one could be added), streaming over LSL.
The user wants to move a cursor around continuously ("have a cursor and continuously move it around").
Their stated preference: NO external stimulation (no flicker, no oddball), NO instructed imagery of hands or
feet; the person should "just naturally control the cursor". They asked that ALL paradigms be considered and
compete anyway (SSVEP, P300, MI, and stimulus-free/implicit approaches), with a winner announced.

## The "bitter lesson" framing the user wants applied
Design for scale: general learning methods plus data volume beat hand-crafted features. More hours, more
sessions, more participants should monotonically improve control. Think: self-supervised pretraining on
large EEG corpora, a data engine that turns ordinary use into training data, co-adaptation (decoder and
user learning together), and policies that improve with accumulated interaction. State where scaling
saturates and why (SNR physics, non-stationarity, label noise).

## Hard-won lessons from the user's prior project (RLNF), which you must not repeat
RLNF ran an EEG-gated proposal loop: an agent proposed a cursor move (an arrow), the EEG decoder
(0–0.6 s post-proposal window, Riemannian tangent space + logistic regression, online re-centring)
accepted or rejected it; the ball moved only on an accept. 29 participants, ~11.5k trials.
- Prospective AUC against the label was ≈0.66–0.69 (participant mean); per-loop permutation floor ≈0.595;
  the 95% prediction interval for a NEW participant was [0.53, 0.80]. Online figures of 0.55–0.69 AUC are
  one to two bands below what the literature treats as publishable.
- FATAL DESIGN FLAW: the training label was a deterministic threshold on the DISPLAYED angle of the
  proposal, so every AUC measured agreement with a displayed criterion, never intent/evaluation. A no-EEG
  control reached 0.655. Near the decision boundary the decoder was at chance. Rule: any contrast defined by
  a threshold on a displayed variable IS that variable. Do the estimability audit (positivity violation,
  near-collinearity, deterministic visual mediator, generator-determined sequences) BEFORE recording.
- NOTHING improved the decoder: 24 pre-registered candidates, 209 windows, 6 deep architectures (EEGNet
  etc.), preprocessing factorial, ICA/EOG regression, electrode subsets — all TOST-equivalent to the
  deployed pipeline. Retraining gains were data VOLUME, not closed-loop provenance.
- OCULAR CONFOUND unresolved: no EOG pair, no eye tracker; FP1/FP2 carried the largest effects; an
  EOG-only decoder was above chance. You must make eye vs cortex identifiable by design.
- NO TASK BENEFIT: a rejection cost a 5.25 s trial, so accept-everything beat the decoder on goals reached.
  Payoff structure, not decoder quality. Fix the payoff structure before claiming utility.
- Session-level adaptation: null. "The loop taught the participant": not supported.
- Outcome-locked ErrP: not supported (display-identifiable, ocular, unstable topography).
- Methodological rules to honour: permutation floors at matched n (never assume 0.5); always run a
  NON-NEURAL surrogate matched to the decoder (if a signal with no brain data passes your test, the test
  is wrong); state effect sizes with CIs; hold the test set fixed across any resource ladder; no
  performance-based participant exclusion; gate per-channel amplitude (≈5× session median RMS; FP1/FP2
  exempt) and log impedance; pre-register endpoints and falsifiers; prefer controller utility (Fitts
  throughput, time-to-target) as co-primary endpoint over decoder AUC.
- RLNF's own registered next steps: hide the goal from the proposal generator; record a delayed keypress
  on every proposal; oversample a near-criterion band; add EOG/eye tracking; make controller utility
  co-primary; fix the payoff structure (cheap rejection).

## Rubric the judge will use (score each 0–10; weights in brackets)
1. Naturalness [25]: no external stimulation, no instructed imagery; the person just intends/attends/acts.
2. Achievable continuous-control performance on THIS hardware [20]: literature-backed throughput
   (bits/s, Fitts), accuracy, time-to-target, with a dry-electrode penalty applied honestly.
3. Scaling with data [20]: does more data (hours, sessions, subjects) buy more control, via what mechanism,
   and where does it saturate?
4. Validity / identifiability [15]: labels not confounded by the display; ocular/EMG confounds measurable
   and separable; a non-neural surrogate control exists and is pre-registered.
5. Feasibility & user burden [10]: latency budget, dry-electrode noise, calibration time, software stack.
6. Evaluation rigour & reproducibility [10]: pre-registered endpoints, floors, held-out sessions/subjects,
   deterministic pipeline, explicit hyperparameters.

## Required structure of your proposal (markdown, ≤ 2,800 words, concrete numbers everywhere)
1. Paradigm and signal basis (what physiological signal, why it is or is not "natural").
2. Expected performance on this hardware, citing specific literature numbers (author/year, metric,
   value) and applying a dry-electrode penalty; give a realistic range, not a best case.
3. Scaling behaviour (bitter lesson): data engine, pretraining, cross-subject transfer, co-adaptation,
   expected scaling curve, saturation point.
4. Data-collection protocol: participants, sessions, blocks, trial structure, durations, labels and how
   they are generated, randomisation, LSL streams and markers to log, impedance/amplitude gating,
   controls and surrogates, pre-registration list, power calculation.
5. Decoder and system: preprocessing, features/architecture with explicit hyperparameters, training and
   validation splits, online latency budget, cursor dynamics (velocity vs position control, gain,
   smoothing, assistance), co-adaptation rule.
6. Evaluation protocol: primary and secondary endpoints, permutation floors, non-neural surrogates,
   statistical tests, held-out sessions and participants.
7. Failure modes and debugging strategies.
8. Honest self-score on the rubric (0–10 per criterion) with one sentence of justification each.
9. A five-sentence pitch.
Be a strong advocate for your paradigm, but never overclaim: the judge will check your literature numbers
and will penalise fabricated citations more than modest numbers. If your paradigm cannot meet a criterion,
say so and say what mitigates it.
