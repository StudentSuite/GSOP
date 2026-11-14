# Source documents

Everyone working from issues uses these two files as the single source of truth for numbers, formulas and decisions. They are copies, byte-identical to the originals; do not edit them.

| File | What it is |
|---|---|
| [GSOP_claude_code_setup_prompt.md](GSOP_claude_code_setup_prompt.md) | The setup prompt: decisions D1 to D10, config values (section A3), scoring maths and Monte Carlo (section 3), data plan, tests, handoff. |
| [GSOP_combined_-_plan__weights_and_evidence.md](GSOP_combined_-_plan__weights_and_evidence.md) | The plan file: what we are doing, the weight tables, and the paper evidence behind every number. |

## Which file wins when they disagree

1. **The setup prompt wins.** Its decisions D1 to D10 beat the plan file (also copied in `docs/DECISIONS.md`).
2. The plan file's weight table is kept **for comparison only**. Example: the plan prints a Moon thermal-inertia weight of 0.03; decision D3 says the derived value 0.019 drives the run.
3. Where the setup prompt gives a section, an issue's "section A3" or "section 3" refers to that file.

Anything in neither file is unverified: put it in `docs/UNVERIFIED.md`.
