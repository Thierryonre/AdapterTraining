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
                },
                "required": ["name", "role", "shape_hint", "mass_kg"],
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
- name: a short lowercase noun phrase that identifies it in the image (e.g. "water bottle", "red cup"),
  distinct between objects
- shape_hint: "cylinder" or "box" if a simple shape describes it well, else "other"
- mass_kg: your best estimate of its mass as it appears (e.g. a full 500 ml bottle is about 0.55 kg)"""


@dataclass
class ObjectInfo:
    name: str
    role: str = "manipulated"
    shape_hint: str = "other"
    mass_kg: float | None = None


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
    return [ObjectInfo(o["name"].strip().lower(), o["role"], o["shape_hint"], float(o["mass_kg"])) for o in objects]


STOP = {"in", "on", "into", "onto", "to", "from", "with", "and", "then", "off", "of", "at", "inside", "under", "over", "next", "by"}


def describe_objects_heuristic(name: str, description: str) -> list[ObjectInfo]:
    """Noun phrases after 'the/a/an' (e.g. 'Put the red block in the bowl' -> red block, bowl)."""
    text = f"{name}. {description}".lower()
    found = []
    words = re.findall(r"[a-z][a-z\-]*|[.,;]", text)
    for i, word in enumerate(words):
        if word not in ("the", "a", "an"):
            continue
        phrase = []
        for nxt in words[i + 1 : i + 4]:
            if nxt in STOP or nxt in ("the", "a", "an", ".", ",", ";"):
                break
            phrase.append(nxt)
        phrase = " ".join(phrase)
        if phrase and phrase not in {"table", "hand", "left hand", "right hand", "robot"} and phrase not in found:
            found.append(phrase)
    return [ObjectInfo(p, "manipulated" if i == 0 else "target") for i, p in enumerate(found)]


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
