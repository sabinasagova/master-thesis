"""
Modifications to the GPHH baseline (Yuan Tian, Mei, Zhang — CEC 2024).

Structure
---------
Each modification is a standalone function or class documented with:
  ORIGINAL  – Yuan's exact code
  MODIFIED  – The replacement used in the proposed approach
  RATIONALE – Why the change improves on the baseline

To add a new modification:
  1. Define the modified function/class here.
  2. Register it in ACTIVE_MODIFICATIONS at the bottom of this file.
  3. Import it in gphh_solver.py via:
        from yuantian.modifications import apply_modifications
     and call apply_modifications(pset, params) after the baseline pset is built.
"""

# ── Modification 1: if_then_else_operator ─────────────────────────────────
#
# ORIGINAL (Yuan Tian, 2024):
#
#   def if_then_else(input1, output1, output2):
#       if input1:
#           return output1
#       else:
#           return output2
#
#   Issues with the original:
#   - Eager evaluation: all three branches are evaluated before the call,
#     wasting computation and preventing short-circuit behaviour.
#   - Truthy check ("if input1") is ambiguous for continuous feature values
#     returned by the GP terminals (e.g. a slack of 0.0 would branch to
#     output2 even though the activity has no slack).
#
# MODIFIED:
#   - Lazy evaluation via inner closure — consistent with every other operator
#     in the codebase (add_operator, max_operator, etc.) which all return
#     callables rather than computed values.
#   - Explicit > 0 threshold anchors the branch to a well-defined zero
#     crossing and aligns naturally with IS_ON_CRITICAL_PATH (returns 1 if
#     on the critical path, 0 otherwise), enabling the GP to evolve readable
#     critical-path-aware rules:
#       if_then_else(IS_ON_CRITICAL_PATH, <rush_formula>, <save_resources_formula>)

def if_then_else_operator(cond, out1, out2):
    def if_then_else():
        # Treats values > 0 (like IS_ON_CRITICAL_PATH) as True.
        # If the activity is critical, evaluate it with Formula A to rush it.
        # If it is not critical, evaluate it with Formula B to save resources.
        return out1() if cond() > 0 else out2()
    return if_then_else


# ── Registry ──────────────────────────────────────────────────────────────
# Maps the primitive name (as registered in the pset) to the replacement
# callable. gphh_solver.py reads this dict to swap operators at build time.

ACTIVE_MODIFICATIONS: dict = {
    "if_else": if_then_else_operator,
    # add future modifications here, e.g.:
    # "add": my_improved_add_operator,
}
