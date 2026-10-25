"""Thin client for a local Ollama server (running in WSL) used as the robot's brain.

Gemma 4 12B handles both jobs:
  * language: pull name / drink / interest out of what the guest said
  * vision:   describe a guest's appearance from a camera crop
"""
import base64
import json
import os
import re

import cv2
import requests

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("RECEPTIONIST_MODEL", "gemma4:12b")


def list_models():
    r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
    r.raise_for_status()
    return [m["name"] for m in r.json().get("models", [])]


def resolve_model():
    """Use MODEL if installed, otherwise fall back to any installed gemma4 tag."""
    global MODEL
    models = list_models()
    if MODEL in models:
        return MODEL
    gemma = [m for m in models if m.startswith("gemma4")]
    if gemma:
        MODEL = sorted(gemma)[0]
        return MODEL
    raise RuntimeError(
        f"Model '{MODEL}' not found in Ollama. Installed: {models}. "
        "Set RECEPTIONIST_MODEL to one of them."
    )


def _encode(frame):
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise ValueError("could not encode image")
    return base64.b64encode(buf.tobytes()).decode()


def _parse_json(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            return json.loads(m.group(0))
        raise


def chat(prompt, image=None, json_mode=False, system=None, timeout=180):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    msg = {"role": "user", "content": prompt}
    if image is not None:
        msg["images"] = [_encode(image)]
    messages.append(msg)
    body = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        # Gemma 4 thinks by default; it can burn its whole token budget thinking and
        # return empty content, and is much slower. These short tasks don't need it.
        "think": False,
        "options": {"temperature": 0.2},
    }
    if json_mode:
        body["format"] = "json"
    r = requests.post(f"{OLLAMA_URL}/api/chat", json=body, timeout=timeout)
    r.raise_for_status()
    text = r.json()["message"]["content"].strip()
    return _parse_json(text) if json_mode else text


# ---------------------------------------------------------------- task helpers

def extract_guest_info(utterance):
    """'Hi I'm Sam, I'd love a cola, I'm into football' -> dict."""
    prompt = (
        "A guest arriving at a party said this to a receptionist robot:\n"
        f'"{utterance}"\n\n'
        "Extract their details. Reply with JSON only, using exactly these keys:\n"
        '{"name": string or null, "drink": string or null, "interest": string or null}\n'
        "Use null for anything they did not say. Keep values short (1-3 words)."
    )
    data = chat(prompt, json_mode=True)
    return {k: data.get(k) for k in ("name", "drink", "interest")}


def describe_person(crop):
    """Visible appearance attributes, for introducing this guest to others."""
    prompt = (
        "Describe the visible appearance of the person in this image so someone "
        "else could recognise them in a room. Only describe what is clearly visible; "
        "do not guess identity, ethnicity or age.\n"
        "Reply with JSON only, using exactly these keys:\n"
        '{"top": "colour and type of upper clothing", '
        '"bottom": "colour and type of lower clothing or null", '
        '"hair": "colour and length, or null if not visible", '
        '"glasses": true/false, "hat": true/false, '
        '"other": "one other distinctive visible detail or null", '
        '"summary": "one natural sentence, max 25 words"}'
    )
    return chat(prompt, image=crop, json_mode=True)


def common_ground(host, guest):
    """One sentence introducing two people, mentioning a shared or related interest."""
    prompt = (
        f"Host: {host['name']}, likes {host['drink'] or 'unknown'}, "
        f"interested in {host['interest'] or 'unknown'}.\n"
        f"Guest: {guest['name']}, likes {guest['drink'] or 'unknown'}, "
        f"interested in {guest['interest'] or 'unknown'}.\n"
        "As a friendly receptionist robot, write ONE short sentence (max 30 words) "
        "introducing them to each other, naming both and pointing out what their "
        "interests have in common (or how they relate). Plain text only."
    )
    return chat(prompt) or f"{host['name']}, meet {guest['name']}. {guest['name']}, this is {host['name']}."
