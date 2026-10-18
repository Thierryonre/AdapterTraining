"""Which physical objects does a skill involve? From the skill name, description and first video frame.

Uses Claude (Anthropic API) with structured JSON output. Credentials come from the environment
(ANTHROPIC_API_KEY, or an `ant auth login` profile). Without credentials, or if the call fails, falls
back to a simple noun-phrase heuristic on the description.
"""

import base64
import json
import re
from dataclasses import dataclass

import cv2
import numpy as np

MODEL = "claude-opus-5-5"

SCHEMA = {
    "type": "object",
    "properties": {
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Short lowercase noun phrase, used as a segmentation prompt"},
                    "role": {"type": "string", "enum": ["manipulated", "target", "static"]},
                    "shape_hint": {"type": "string", "enum": ["cylinder", "box", "other"]},
                    "mass_kg": {"type": "number"},
                    "count": {"type": "integer", "description": "How many of these identical objects take part"},
                },
                "required": ["name", "role", "shape_hint", "mass_kg", "count"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["objects"],
    "additionalProperties": False,
}

PROMPT = """A robot will learn a skill from a video of a person demonstrating it on a table.
Skill name: {name}
Description: {description}

The image is the first frame of the demonstration video. List the physical objects involved in the
skill: the ones that get picked up or moved ("manipulated"), and ones they are placed on or in
("target"). Leave out the table, the person, their hands and clothing, and background items the skill
doesn't touch. For each object give:
- name: a short lowercase singular noun phrase that identifies it in the image (e.g. "water bottle",
  "red cup"), distinct between different kinds of objects
- count: how many of this object take part. Identical objects are one entry with a count, e.g.
  stacking one sponge on another is {{"name": "sponge", "count": 2}}
- shape_hint: "cylinder" or "box" if a simple shape describes it well, else "other"
- mass_kg: your best estimate of its mass as it appears (e.g. a full 500 ml bottle is about 0.55 kg)"""


@dataclass
class ObjectInfo:
    name: str
    role: str = "manipulated"
    shape_hint: str = "other"
    mass_kg: float | None = None
    count: int = 1  # identical objects taking part (e.g. 2 sponges to stack); each becomes its own object


def _image_block(image_rgb: np.ndarray, max_width: int = 1024) -> dict:
    h, w = image_rgb.shape[:2]
    if w > max_width:
        image_rgb = cv2.resize(image_rgb, (max_width, round(h * max_width / w)), interpolation=cv2.INTER_AREA)
    ok, png = cv2.imencode(".png", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    data = base64.standard_b64encode(png.tobytes()).decode("utf-8")
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}


def describe_objects_with_claude(name: str, description: str, first_frame: np.ndarray) -> list[ObjectInfo]:
    import anthropic

    client = anthropic.Anthropic()
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        # Server-side fallback on a policy decline (routes by refusal category); see the Claude API docs.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[
            {
                "role": "user",
                "content": [_image_block(first_frame), {"type": "text", "text": PROMPT.format(name=name, description=description)}],
            }
        ],
    )
    if response.stop_reason == "refusal":
        raise RuntimeError(f"declined: {getattr(response.stop_details, 'category', None)}")
    text = next(b.text for b in response.content if b.type == "text")
    objects = json.loads(text)["objects"]
    return [ObjectInfo(o["name"].strip().lower(), o["role"], o["shape_hint"], float(o["mass_kg"]), max(1, int(o.get("count", 1))))
            for o in objects]


STOP = {"in", "on", "into", "onto", "to", "from", "with", "and", "then", "off", "of", "at", "inside", "under", "over", "next", "by"}
# Words that refer to another object or a place on it, not an object of their own ("on top of the other").
REFERENCES = {"other", "others", "one", "ones", "it", "them", "top", "bottom", "side", "middle", "rest", "same", "first",
              "second", "end", "edge", "left", "right", "surface"}
NUMBERS = {"two": 2, "both": 2, "pair": 2, "three": 3, "four": 4, "five": 5}


def singular(phrase: str) -> tuple[str, bool]:
    """'sponges' -> ('sponge', True); 'glass' -> ('glass', False). Only the last word is changed."""
    *head, last = phrase.split()
    if last.endswith("ies") and len(last) > 4:
        last, plural = last[:-3] + "y", True
    elif last.endswith(("ches", "shes", "xes", "sses")):
        last, plural = last[:-2], True
    elif last.endswith("s") and not last.endswith(("ss", "us", "is")) and len(last) > 3:
        last, plural = last[:-1], True
    else:
        plural = False
    return " ".join([*head, last]), plural


def describe_objects_heuristic(name: str, description: str) -> list[ObjectInfo]:
    """Noun phrases after 'the/a/an' or a number (e.g. 'Put the red block in the bowl' -> red block, bowl).
    Plurals become a count ('stack the sponges' -> sponge x2); 'the other', 'the top' etc. are skipped."""
    text = f"{name}. {description}".lower()
    found: dict[str, int] = {}
    words = re.findall(r"[a-z][a-z\-]*|[.,;]", text)
    for i, word in enumerate(words):
        if word not in ("the", "a", "an", "of") and word not in NUMBERS:
            continue
        phrase = []
        for nxt in words[i + 1 : i + 4]:
            if nxt in STOP or nxt in ("the", "a", "an", ".", ",", ";"):
                break
            phrase.append(nxt)
        number = NUMBERS.get(word)
        while phrase and phrase[0] in NUMBERS:  # "the two cups"
            number = NUMBERS[phrase.pop(0)]
        if not phrase or phrase[-1] in REFERENCES or phrase[0] in REFERENCES:
            continue
        noun, plural = singular(" ".join(phrase))
        if noun in {"table", "hand", "left hand", "right hand", "robot", "skill", "surface"}:
            continue
        count = number or (2 if plural else 1)
        found[noun] = max(found.get(noun, 1), count)
    return [ObjectInfo(n, "manipulated" if i == 0 else "target", count=c) for i, (n, c) in enumerate(found.items())]


def describe_objects(name: str, description: str, first_frame: np.ndarray, use_claude: bool = True) -> tuple[list[ObjectInfo], str]:
    """Returns (objects, source) where source is 'claude' or 'heuristic' (with the reason)."""
    if use_claude:
        try:
            objects = describe_objects_with_claude(name, description, first_frame)
            if objects:
                return objects, "claude"
            reason = "Claude listed no objects"
        except Exception as e:  # no credentials, network, refusal...
            reason = f"{type(e).__name__}: {e}"[:200]
    else:
        reason = "disabled"
    return describe_objects_heuristic(name, description), f"heuristic ({reason})"
