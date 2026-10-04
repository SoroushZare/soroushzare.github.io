# Synaptrix "Portal" cursor demo — what can and cannot be concluded

Source material: the LinkedIn post text and two video stills supplied by the user (a dense dry-pin cap,
a monitor showing a grid with a highlighted cell, a laptop beside it), plus public reporting found by
search. The company site, the LinkedIn post, Forbes, Substack and clinicaltrials.gov were blocked by
this session's network proxy, so everything below is based on the post text, the stills and search
snippets. Nothing here is a verdict on the company; it is a list of what the evidence shown supports.

## What the post claims
- "Full computer cursor control from brain signals", "entirely non-invasive", "no surgery, implants,
  or eye-tracking, at normal computer-use speeds".
- "Treating EEG decoding as a frontier AI problem" with investment in "models, data, and hardware".
- "Approaching the capabilities of implanted BCIs."
- No numbers are given: no throughput, no accuracy, no time-to-target, no number of users or sessions,
  no calibration time, no description of the task on screen.

## What public reporting adds
- Founded 2023 (Aryan Govil, Eric Yao), New York; seed round led by Mark Cuban, reported at
  USD 8–10 M in October 2025 ([startupmap](https://startupmap.iamsterdam.com/news/feed/synaptrix-labs-secures-mark-cuban-funding),
  [Crunchbase](https://www.crunchbase.com/organization/synaptrix-labs)).
- The product line is described as an EEG headset with dry sensors plus "large motor-cortex foundation
  models" that decode "imagined movement"; the first application is a wheelchair
  ([Delphi Digital](https://members.delphidigital.io/reports/synaptrix-i-think-therefore-i-can),
  [hyper.ai](https://hyper.ai/en/stories/fe6d955dd60057f8ba6e060efa2a1d6b)).
- The same hyper.ai report describes the Neuralis wheelchair headset as having "dry sensors on the
  occipital lobe" and says its "key breakthrough lies in [the] capability to extract brain activity
  from subtle eye movements". That is a description of an ocular or visual-cortex route, not motor
  cortex.
- A wheelchair feasibility study is registered on clinicaltrials.gov (NCT06169696, "EMPOWER"), which
  is a reasonable step; no peer-reviewed cursor result was found.

## What the stills show
- A cap with many dry pins over the whole scalp (consistent with 32–64 channels).
- A grid task with a target cell: a discrete target-acquisition task, the easiest cursor task to
  make look continuous. A grid of discrete cells is exactly the setting where a learned target
  prior plus coarse direction evidence suffices (see the shared-autonomy simulations in this
  repository, where a weak decoder plus a posterior over targets reaches 0.78–0.86 success).
- The participant's hands are below the desk edge and the laptop is open. Nothing in a still can
  rule out hand, jaw, neck or forehead electromyography, or eye movements.

## What a neuroscientist would ask before believing "approaching implanted BCIs"
The best peer-reviewed non-invasive continuous control (Wolpaw & McFarland 2004; Edelman et al.
2019; Stieger et al. 2021) sits at roughly 0.2–0.6 bits/s Fitts throughput after many sessions, with
gel electrodes. Implanted cursor control (Pandarinath et al. 2017; Willett-era systems) sits at
2–3 bits/s and above. A claim of closing that gap non-invasively with dry electrodes is a claim of a
roughly 5–10× jump over the entire published field. Extraordinary claims are fine; they need:

1. **Throughput numbers** on a standard task (ISO 9241-9 multi-directional Fitts) with time-to-target
   and error rate, for several users, on held-out sessions.
2. **Ocular controls.** "No eye tracker used for control" is not "no eye contribution". The test is
   an eye tracker *recorded* but not used, then (a) a decoder trained on frontal/EOG channels only,
   (b) fixation-enforced blocks, (c) regression of gaze out of the EEG, all reported beside the
   headline. The RLNF project found an EOG-only decoder above chance without ever intending it.
3. **Myogenic controls.** Facial, jaw and neck EMG reach every dry electrode; a hand-still, face-still
   block with EMG monitoring is the minimum. Dry pins are more EMG-sensitive than gel, not less.
4. **Display controls.** A learned target prior on a grid can produce high success with almost no
   neural information. Report the same system with the EEG replaced by shuffled outputs
   (the no-neural surrogate). If success barely drops, the agent is doing the work.
5. **Calibration and transfer.** Minutes of calibration per user per day? Cross-user zero-shot?
   Which sessions are held out?
6. **The label.** What was the training label in the "frontier AI" data: observed mouse movement,
   gaze, imagined movement, a cue? Each has a different ceiling and a different confound.

## Bottom line for this project
The demo, as published, is a plausibility video, not evidence. It is compatible with any of:
a genuine step forward in motor-cortex decoding; a well-engineered gaze/EOG decoder; a strong
target prior on a discrete grid; or a mix. The one thing it demonstrates for certain is the
*framing* the user wants to adopt: EEG decoding treated as a data-and-models problem. The design in
`DESIGN.md` adopts the framing and adds the controls that would let a reader distinguish the cases.
