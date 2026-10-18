# Tournament brief — pure-EEG "control by thinking" for any cursor or robot

## The goal (from the principal investigator)
Control a computer cursor continuously — and, through the same decoder output, any robot — **by thinking**,
from **pure EEG**. Publish in Nature or Nature Communications. Be a pioneer.
The PI's working hypothesis, which you must take seriously but may argue against: "the bitter lesson —
a person sits and we collect as much data as possible, then try, collect, implement and adapt the model
again and again, and the system becomes more and more robust."

## Hard constraints
- **Control input = EEG only.** No EOG or gaze as a control signal, no EMG, no eye tracker as an input.
- **No external stimulation for control**: no SSVEP/c-VEP flicker, no P300/oddball flashes, no tactile or
  auditory evoked paradigms.
- Instructed motor imagery is allowed only if you argue it honestly; "natural" (the person just intends) is
  strongly preferred.
- **Auditors are allowed and, for the purity claim, required**: an eye tracker, EMG, accelerometer, may be
  *recorded* to prove the EEG decoder is not reading eyes/muscles/the display — never fed to the decoder.
  Say how your design proves purity.
- Hardware of record: g.tec 32-channel dry hybrid-electrode cap, 500 Hz, LSL, 10-20 layout incl. FP1/FP2.
  You may argue for an additional gel high-density arm for the science, with cost.
- Team: one lab, 12–18 months, a few participants available for very long recording (tens to hundreds of
  hours) and/or a larger cohort (tens) for shorter sessions.

## What is already established in this project (do not re-derive; build on it)
- Previous design competition (SSVEP, P300, MI, covert attention, shared autonomy+ErrP, end-to-end scaled
  decoder, EOG hybrid) concluded: stimulus-free cortical signals on a dry 32-ch cap carry roughly
  0.03–0.3 bits/s; EOG pointing reaches ~1 bit/s but is not brain control (now excluded by the PI).
- Simulation in this repo (`neural-cursor/`): a weak continuous velocity decoder (r≈0.4) reaches 8 targets
  at 0.49 success alone and ~0.8 with a Bayesian goal posterior (shared autonomy) that never reads the goal;
  EOG-only and no-neural surrogates stay at chance. An error-evidence (ErrP) channel at AUC≈0.68, one packet
  per second, added nothing measurable over the dense velocity posterior. A frozen decoder collapsed under
  dry-electrode gain drift; online re-centring recovered part of it.
- The PI's previous project (RLNF, 29 participants, 11.5k trials) failed scientifically because: the label
  was a threshold on a DISPLAYED variable (so every AUC was agreement with the screen, a no-EEG control
  reached 0.655); FP1/FP2 carried the largest effects with no way to separate eyes from cortex; 24 decoder
  variants, 209 windows, 6 deep nets were all equivalent — gains came only from data VOLUME, not closed-loop
  provenance or architecture; rejection cost made the loop useless; no session-level user learning was
  detectable. Rules: labels must never be a threshold on what is displayed; non-neural surrogates matched
  to the decoder on the same trials; permutation floors at matched n; fixed test sets on any resource
  ladder; no performance-based exclusion.
- A specific trap for "decode intended direction": if the target is shown on screen, the label (direction
  to target) is display-identifiable — gaze, visual evoked responses and the visible cursor's own motion
  all decode it. In closed loop, the decoder can learn to read the visual consequence of its own output.
- Real public data reachable from this machine: OpenNeuro ds004362 = PhysioNet EEG Motor Movement/Imagery
  (109 subjects, 64-ch gel, 160 Hz; executed and imagined left/right fist, both fists/both feet; targets
  shown on left/right/top/bottom of the screen). Other OpenNeuro datasets on S3 are reachable too.
- Relevant precedents you may cite (verify what you cite; fabricated numbers are penalised hard):
  Wolpaw & McFarland 2004 PNAS (2-D SMR cursor); McFarland 2010 (3-D); Edelman et al. 2019 Science Robotics
  (continuous pursuit, robotic arm); Stieger et al. 2021 Sci Data (62 subjects, ~600 h SMR cursor data);
  Meng et al. 2016 Sci Rep (robotic arm reach-and-grasp by EEG); Kaifosh et al. 2025 Nature (Meta,
  generic non-invasive sEMG interface, scaling across thousands of participants — the clearest
  "bitter lesson" precedent for a non-invasive interface); Sadtler et al. 2014 Nature (intracortical
  within-manifold learnability); Gilja 2012 (ReFIT); Orsborn 2014 (CLDA); EEG foundation models
  (BENDR 2021, LaBraM 2024, CBraMod 2025, EEGPT 2024) with modest downstream gains; Lévy et al. 2025
  Brain2Qwerty (EEG typing CER ≈ 0.67 vs MEG ≈ 0.32); Iturrate 2015 / Chavarriaga 2014 (ErrP);
  Friedrich et al. 2012/2013 (user-chosen mental tasks); Perdikis 2018 (Cybathlon mutual learning).

## Rubric the judges will use (weights)
1. Pure-EEG physiology: does the signal exist in scalp EEG at usable SNR on a dry 32-ch cap? [20]
2. Naturalness — "by thinking", minimal instruction, low user burden [15]
3. Scaling with data — explicit mechanism, evidence, where it saturates [20]
4. Purity / validity — can the design PROVE it is EEG and not eyes, muscles, or the display? [15]
5. Nature-level novelty — the headline figure and claim nobody has shown [15]
6. Feasibility for this team in 12–18 months [10]
7. Generality — the same decoder output drives any cursor or robot [5]

## Required proposal structure (markdown, ≤ 2,500 words, concrete numbers)
1. One-line idea and the signal basis (what neural phenomenon, why it is "thinking").
2. Label source: where every training label comes from and why it is NOT display-identifiable.
3. Data engine and scaling mechanism; expected curve and saturation; cite evidence.
4. Expected performance on this hardware with literature anchors (bits/s, success, time-to-target).
5. Purity proof: auditors, surrogates, fixation/blank blocks, feedback-circularity controls.
6. Protocol: participants, hours, sessions, block structure.
7. Real-time architecture: windows, model, latency, adaptation rule, effector interface (cursor/robot).
8. The Nature claim: headline figure, the one sentence, why it is new.
9. Falsifier and biggest risk.
10. What can be tested NOW on ds004362 (109-subject public data) to support or kill the idea.
