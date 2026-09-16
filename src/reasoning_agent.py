"""VLM reasoning agent for RoadSentinel AI.

Assesses flagged incident frames alongside motion features (avg_speed,
max_deceleration, max_iou, cluster_id) to evaluate severity (Tiers 1-5).

Supports:
1. Anthropic Claude (claude-3-5-sonnet, claude-3-7-sonnet)
2. Google Gemini (gemini-2.0-flash, gemini-1.5-flash)
3. OpenAI (gpt-4o, gpt-4o-mini)
4. Deterministic Local Mock Fallback (when no API key is configured)
"""
from __future__ import annotations

import base64
import json
import os
from typing import Any, Dict

SEVERITY_RUBRIC = """You are a senior emergency traffic-safety AI investigator.
Evaluate the provided collision/incident scene alongside the telemetry motion features.

SEVERITY TIERS RUBRIC:
- Tier 1 (Negligible): Non-collision anomaly (stalled vehicle on shoulder, minor debris). No lane blockage, zero hazard.
- Tier 2 (Minor): Low-speed fender-bender, paint scrape, vehicles operable, no injuries, no traffic obstruction.
- Tier 3 (Moderate): Direct impact between 2 vehicles, moderate structural damage, possible airbag deployment, single lane partially blocked.
- Tier 4 (Severe): High-speed collision, multi-vehicle impact, vehicle rollover, structural intrusion, multi-lane obstruction. High injury risk. Urgent dispatch required.
- Tier 5 (Catastrophic): Critical high-speed pileup, visible fire or heavy smoke, occupant cabin crushed/entrapment, full roadway blockage. Immediate multi-unit trauma dispatch required.

OUTPUT FORMAT REQUIREMENTS:
Respond STRICTLY with a valid JSON object only. No preamble, no markdown backticks, no trailing commentary:
{
  "severity_tier": <integer 1 to 5>,
  "confidence": <float 0.0 to 1.0>,
  "vehicle_damage_assessment": "<concise description of visible damage>",
  "lane_blockage": <boolean true or false>,
  "hazard_indicators": ["<e.g. fire, fluid_leak, airbag_deployed, rollover, none>"],
  "recommended_action": "<actionable directive for dispatch>",
  "explanation": "<detailed strategic rationale correlating visual evidence with telemetry>"
}"""


def _encode_image_b64(image_path: str) -> str:
    """Read an image from disk and return its base64 encoded string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _mock_reasoning(features: Dict[str, Any]) -> Dict[str, Any]:
    """Intelligent deterministic reasoning fallback when no remote VLM keys are provided."""
    prob = float(features.get("accident_probability", 0.8))
    iou = float(features.get("max_iou", 0.4))
    decel = float(features.get("max_deceleration", -45.0))
    cluster = int(features.get("cluster", 1))

    if prob >= 0.85 or iou >= 0.40 or decel <= -60.0:
        tier = 5 if (iou >= 0.55 or decel <= -75.0) else 4
        damage = "Severe frontal deformation with structural cabin compromise and high-energy impact dissipation."
        blockage = True
        hazards = ["high_velocity_impact", "structural_compromise", "debris_scatter"]
        action = "Dispatch Level 1 Trauma EMS, Extrication Fire Unit, and Highway Patrol for total lane closure."
        explanation = (
            f"VLM telemetry correlation indicates a catastrophic collision (P(Accident)={prob*100:.1f}%, "
            f"Max IOU={iou:.2f}, Deceleration={decel:.1f} m/s^2). Vehicle bounding boxes indicate high kinetic energy "
            f"transfer consistent with Tier {tier} critical roadway incident."
        )
    elif prob >= 0.60 or iou >= 0.20 or decel <= -30.0:
        tier = 3
        damage = "Moderate collision damage to front quarter panels and bumper assembly; airbags potentially deployed."
        blockage = True
        hazards = ["lane_obstruction", "minor_fluid_spill"]
        action = "Dispatch municipal emergency response unit, tow vehicle, and traffic routing support."
        explanation = (
            f"Telemetry indicates an active collision with moderate kinetic deceleration ({decel:.1f} m/s^2) "
            f"and bounding-box overlap of {iou:.2f}. Incident requires emergency triage and partial lane diversion."
        )
    elif prob >= 0.35 or iou >= 0.08:
        tier = 2
        damage = "Low-velocity contact or near-miss with minimal cosmetic bumper abrasion."
        blockage = False
        hazards = ["none"]
        action = "Log incident advisory to regional traffic management; no priority siren dispatch necessary."
        explanation = (
            f"Low-severity near-miss/fender incident detected (P(Accident)={prob*100:.1f}%, Max IOU={iou:.2f}). "
            "Telemetry shows safe vehicle deceleration without catastrophic impact signatures."
        )
    else:
        tier = 1
        damage = "No vehicle body damage observed; vehicle operates within normal safety tolerances."
        blockage = False
        hazards = ["none"]
        action = "Monitor sector traffic flow; no intervention required."
        explanation = "Scene analysis confirms normal operational flow or shoulder stop with negligible safety risk."

    return {
        "severity_tier": tier,
        "confidence": round(min(0.98, max(0.75, prob + 0.1)), 2),
        "vehicle_damage_assessment": damage,
        "lane_blockage": blockage,
        "hazard_indicators": hazards,
        "recommended_action": action,
        "explanation": explanation,
        "provider": "Local Strategic Reasoner (Deterministic Fallback)",
    }


def reason_about_clip(frame_path: str, features: Dict[str, Any]) -> Dict[str, Any]:
    """Execute VLM reasoning on an incident frame and its motion features.

    Automatically selects the best available client based on environment variables:
    Anthropic > Google Gemini > OpenAI > Mock Fallback.
    """
    # 1. Check Anthropic
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=anthropic_key)
            img_b64 = _encode_image_b64(frame_path)
            prompt_content = (
                f"Telemetry motion features: {json.dumps(features, indent=2)}\n"
                "Assess this visual frame and return strict JSON as specified."
            )
            msg = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=600,
                system=SEVERITY_RUBRIC,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}},
                        {"type": "text", "text": prompt_content},
                    ],
                }],
            )
            raw_text = msg.content[0].text
            return _clean_and_parse_json(raw_text, "Anthropic Claude")
        except Exception as e:
            print(f"[VLM Warning] Anthropic call failed: {e}. Falling back...")

    # 2. Check Google Gemini
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if gemini_key:
        try:
            from google import genai
            client = genai.Client(api_key=gemini_key)
            with open(frame_path, "rb") as f:
                img_bytes = f.read()
            prompt = f"{SEVERITY_RUBRIC}\n\nTelemetry motion features: {json.dumps(features, indent=2)}"
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[prompt, genai.types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")]
            )
            return _clean_and_parse_json(response.text, "Google Gemini")
        except Exception as e:
            print(f"[VLM Warning] Gemini call failed: {e}. Falling back...")

    # 3. Check OpenAI
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        try:
            import openai
            client = openai.OpenAI(api_key=openai_key)
            img_b64 = _encode_image_b64(frame_path)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SEVERITY_RUBRIC},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"Telemetry features: {json.dumps(features, indent=2)}"},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                        ],
                    },
                ],
                max_tokens=600,
            )
            raw_text = response.choices[0].message.content
            return _clean_and_parse_json(raw_text, "OpenAI GPT-4o")
        except Exception as e:
            print(f"[VLM Warning] OpenAI call failed: {e}. Falling back...")

    # 4. Default: Deterministic Expert Rule-based Mock
    return _mock_reasoning(features)


def _clean_and_parse_json(raw_text: str, provider: str) -> Dict[str, Any]:
    """Robustly extract and validate JSON from model text outputs."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1]) if lines[-1].startswith("```") else "\n".join(lines[1:])
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}") + 1
        if start != -1 and end != 0:
            data = json.loads(cleaned[start:end])
        else:
            raise ValueError(f"Could not parse valid JSON from VLM output: {raw_text}")

    data["provider"] = provider
    # Ensure severity_tier is integer in range 1-5
    if "severity_tier" in data:
        data["severity_tier"] = max(1, min(5, int(data["severity_tier"])))
    return data
