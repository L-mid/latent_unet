
"""
What I actually do.

1). Rewriting repo parts from scratch -> Software engineering intuition. 

2). Math & Paper recreation -> Research layer. 

3). Debugging -> Debugging.

4). Training models -> Applided outcome.



Weighting:
70% on building/debugging codebase -> (what I'm doing now almost exactly).
20% on math/paper recreation -> conceptual musle matters, after repo stable.
10% on training runs -> treat as demonstrations, not main practice until base is reliable.

"""

# good sanity check
"""
Signs I'm doing it!

You can refactor something ugly into a clean, reusable abstraction.
You can debug errors by narrowing the cause systemically (not just random trial).
You can implement math from a paper without copying code.
You notice you're reading papers differently: less "this is alien" and more "ok, so they changes X in the loss, added Y in the block".
You can explain design tradeoffs in your repo, not just code them.

"""


# schedule allocation cal (only messy cal here)
"""
Time Split:

Base (repo/debugging): 28h (50%) (20 if day off)
Research (math/papers): 16h (30%)
Demonstrations (training): 8h (15%)
Stretch (exploration): 4h (5%)                               # idk if these %ages are correct


I think it's fair to say "do repo everyday for 4 hrs" so thats that. If not doing 8 hrs per week, use this as baseline. 


Rest of 4 hrs:
And then it's fair to say Reserch is done 50% of the remaining time.

2hs left:

Only 4 hours goes to stretch (exploration) a week. 1X per week. 
8 goes to training and vizusalizations (one day, or split over 2). 2X per week.


So that leaves: 4hrs * 7 = 28 taken (repo/debugging). 

28 left:
28 - 8 (training/viz)
20 - 4 (strech)

16 left: (goes to research) (= 8 * 2) = two days OR four sessions (4hrs)

"""


# Actcual schedule:
"""
So: 1x stretch, 2x training/viz, 4x research/math, 7x repo


Days (started on a wednesday): 

1 - research/math (4 hrs)   - repo (4 hrs)      (day off)
2 - research/math (4 hrs)   - repo (4 hrs) 
3 - training/viz (4 hrs)    - repo (4 hrs)
4 - research/math (4 hrs)   - repo (4 hrs)
5 - strech (4 hrs)          - repo (4 hrs)
6 - research/math (4 hrs)   - repo (4 hrs)
7 - training/viz (4 hrs)    - repo (4 hrs)

"""

# On "look at other repos" vs "look at others":
"""
On "look at other repos" vs "look at others":

Potental allocation split:
Your-repo: 70% (work on your own)
External reading: 30% (hey look what are other people doing?)

(More on this once i see schedule and am like damn wtf do i do i have here)

|
V
"""
import random

print(random.choices(["Look at other Repo.", "Do your Repo."], weights=[30, 70]))   #yay



# This is for Legend marking. You'll find it in planning for some reason.
"""
For this Legend: 

Legend:
🔹: Implimented once - (copied direct)                      - You typed it in, but it's "dead muscle memory".
🔺: Known            - (wrote with some copying)            - Can explain dataflow, run small tests, still peaks for APIs.
🔸: Level 1          - (can write without help)             - Can re-impliment with <3 peaks per function, tests pass.
🔸🔸: Level 2        - (can write and change it a bit)     - Can re-impliment model+train loop cold; contract tests all pass.       
🔸🔸🔸: Level 3     - (can write and refactor)             - Can inject bugs and your asserts/localization catch them quickly; can swap varients easily without breaking.
🌟: Mastered         - (automatic)                          - Can design new abstractions, refactor repo-scale systems, anticipate faliure modes easily.


HOW TO:

=== The progression:

0) First contact (you can't code it yet)

Goal: Build understanding before code.
- One-paragraph summary: What is it? Inputs -> outputs -> invariants.
- Box-and-arrow sketch: tensors + shapes + key ops (e.g., x_t, eps_theta, alpha_t)
- Tiny glossary: 5-10 symbols with plain-English meaning (like ε).
- Golden tests you'd want: What must always hold true? (e.g., shape invarients, monotonic schedules, energy bounds).
Outcomes: a 1-pager you can explain out loud in 90 secs.


1) Guiden scaffold (copying allowed, but only structure)

Goal: understand the skeleton and seams.
- Write a minimal interface: forward(...), expected shapes, return type.
- Create failing tests first (red): shapes, dtype, device, determinism with a fixed seed.
- Paste only function signatures + TODOs, not solution code.
- Add debug prints/asserts for invariants (norms, ranges, NaNs).
Outcomes: a red test suite and a full file full of TODOs with a clear API.


2) Peeking protocol (disciplined looking)

Goal: learn, not memorize blindly. 
- Code attempt -> if stuck, peak at exactly one line/idea -> close it -> continue.
- When you peek, write why you were stuck (missing identity, shape, convention).
- After each peek, generalize the rule (e.g., "DDIM update is determinitic Euler step under reparam with σ=0")
- Keep a Stuck Log: symptom, root cause, the rule you learned.
Outcome: working code with 3-6 short "rules I learned" bullets.


3) Explain-back & invarients

Goal: prove understanding in your own words. 
- Write a comment block at the top, e.g: "We compute ... because...; if β_t->0; limits at t=0 look like ..."
- Encode invariants as asserts (e.g., x0_clip ∈ [-1, 1], schedule len == T).
- Add property tests: sample random shapes/dtypes/devices and ensure your function is total (no errors/NaNs).
Outcome: code that teaches how it workds, plus invarients that catch regressions. 


4) Blind-rebuild drill (no peeking)

Goal: convert understanding into recallable skill. 
- Close everything. From blank file, impliment module using only 1-pager, glossary, and tests.
- If you fail a test, don't open references yet. Use the Stuck Log and derive. 
- Only if blocked > 15 min, do a 60s peek, then immediately re-close and continue. 
Do this 3 times across a few days (spaced repetition). 
Outcome: Faster, fewer peeks.


5) Black-box parity & adversarial tests

Goal: hard validation that impl is correct.
- Compare against a reference impl with fixed seeds; assert max|y_ref-y| < ε .
- Adversarial inputs: extreme timesteps, tiny betas, random masks, mixed precision on/off.
- Gradient checks: finite-difference vs autograd on small tensors (catch sign mistakes).
Outcome: parity report + adversarial test file.


6) Transfer & variation

Goal: prove it's not rote. 
- Implement the variant (e.g., switch σ schedule; change time embedding; swap norm layer).
- Port to a diff framework or different tensor layout (NumPy, JAX, TensorFlow, (B, H, W, C)!). 
- Integrate into a new pipeline location (e.g., use in a different sampler loop).
Outcome: same tests pass with varients; notes on what changed and why.


"""






