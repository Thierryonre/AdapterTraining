"""Language instructions for MimicGen / robomimic environments.

SmolVLA is language-conditioned, so every episode needs a task string. The same string must be
passed at deployment time (`--task` in lerobot-rollout).
"""

# Keyed by environment name without the difficulty/variant suffix (Stack_D1 -> Stack).
MIMICGEN_TASKS = {
    "Stack": "Stack the red cube on top of the green cube.",
    "StackThree": "Stack the red cube on the green cube, then stack the blue cube on top.",
    "Square": "Pick up the square nut and place it on the square peg.",
    "Threading": "Pick up the needle and insert it into the hole of the tripod.",
    "Coffee": "Insert the coffee pod into the coffee machine and close the lid.",
    "CoffeePreparation": "Place the mug under the coffee machine, insert a pod and close the lid.",
    "ThreePieceAssembly": "Assemble the three pieces onto the base.",
    "HammerCleanup": "Open the drawer, put the hammer inside and close the drawer.",
    "MugCleanup": "Open the drawer, put the mug inside and close the drawer.",
    "Kitchen": "Turn on the stove, put the pot on it, put the bread in the pot and serve it.",
    "NutAssembly": "Place the square nut on the square peg and the round nut on the round peg.",
    "PickPlace": "Put each object into its matching bin.",
    # robomimic environments
    "Lift": "Lift the cube.",
    "PickPlaceCan": "Pick up the can and place it in the correct bin.",
    "NutAssemblySquare": "Pick up the square nut and place it on the square peg.",
    "TwoArmTransport": "Transfer the hammer from one bin to the other using both arms.",
    "ToolHang": "Insert the hook into the base and hang the tool on it.",
}


def task_for_env(env_name: str) -> str | None:
    """Look up an instruction for e.g. 'Stack_D1', 'Square_D0', 'Threading_D2_Panda'."""
    base = env_name.split("_")[0]
    return MIMICGEN_TASKS.get(env_name) or MIMICGEN_TASKS.get(base)
