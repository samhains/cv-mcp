from __future__ import annotations

ALT_SYSTEM = (
    "You describe images for accessibility. Be concise and strictly factual. Do not infer unseen details."
)


def alt_user_prompt(max_words: int = 20) -> str:
    return (
        f"Describe this image in <= {max_words} words. Neutral tone. "
        "No brand/species/location guesses. Return one sentence only. If unknown, omit."
    )


CAPTION_SYSTEM = (
    "You carefully describe visual content without guessing. Mention salient text only if clearly readable."
)

CAPTION_USER = (
    "Write a factual, detailed caption (2–6 sentences) for this image. Cover:\n"
    "- Who/what is visible (counts if reliable).\n"
    "- Where/setting if visually indicated.\n"
    "- Salient readable text.\n"
    "- Relationships (e.g., 'person holding red umbrella near taxi').\n"
    "- Lighting/time cues if obvious (e.g., night, golden hour).\n"
    "If uncertain, say 'unclear'. Do not guess brands, species, or locations unless unmistakable. Avoid subjective adjectives."
)


def structured_system() -> str:
    return (
        "You extract only what is visibly supported by the image and caption. "
        "Do not guess. Use null or [] when unknown. Return valid JSON only."
    )


def structured_user(caption: str) -> str:
    return (
        "From this image and caption, return a compact JSON object with exactly these fields: \n"
        "media_type, objects, place, scene, lighting, style, palette, text, people, privacy, tags, notes.\n\n"
        f"CAPTION: '{caption}'\n\n"
        "Rules:\n"
        "- media_type: one of photo | film_still | painting | illustration | render | screenshot | poster | document.\n"
        "- objects: 1–6 salient nouns.\n"
        "- place: null unless clearly evidenced by visible text or filename tokens.\n"
        "- scene: 1–3 tokens (e.g., indoor, corridor, street).\n"
        "- lighting: 1–3 tokens (e.g., soft, dramatic, night).\n"
        "- style: 1–5 aesthetic/genre tokens.\n"
        "- palette: 3–6 plain color words.\n"
        "- text: salient readable words only.\n"
        "- people: {count, faces_visible}.\n"
        "- privacy: only if applicable from content (faces_visible, license_plate_visible, nudity_or_racy, children_visible, sensitive_document).\n"
        "- tags: union of media_type + scene + lighting + style + palette + objects; deduplicate; <=20.\n"
        "- notes: short sentence only if strong evidence (e.g., 'Likely a film still').\n"
        "- Omit fields that would be empty or null, except always include media_type, objects, people, tags.\n"
        "Return JSON only."
    )


AC_SYSTEM = (
    "You describe images accurately and concisely without guessing. Return valid JSON only."
)


def ac_user() -> str:
    return (
        "Return a JSON object with exactly two fields: \n"
        "{\n  \"alt_text\": string,\n  \"caption\": string\n}\n\n"
        "Constraints:\n"
        "- alt_text: one sentence, <= 20 words, strictly factual, neutral tone.\n"
        "- caption: 2–6 factual sentences, include what/where/relationships/lighting.\n"
        "- No brand/species/location guesses unless unmistakable. No subjective adjectives."
    )


def structured_text_system() -> str:
    return (
        "You extract structured metadata from the caption only. Do not guess. "
        "Use null or [] when unknown. Return valid JSON only."
    )


def structured_text_user(caption: str) -> str:
    return (
        "From the caption, return a compact JSON object with exactly these fields: \n"
        "media_type, objects, place, scene, lighting, style, palette, text, people, privacy, tags, notes.\n\n"
        f"CAPTION: '{caption}'\n\n"
        "Rules:\n"
        "- media_type: one of photo | film_still | painting | illustration | render | screenshot | poster | document.\n"
        "- objects: 1–6 salient nouns.\n"
        "- place: null unless clearly evidenced by text or filename tokens.\n"
        "- scene: 1–3 tokens (e.g., indoor, corridor, street).\n"
        "- lighting: 1–3 tokens (e.g., soft, dramatic, night).\n"
        "- style: 1–5 aesthetic/genre tokens.\n"
        "- palette: 3–6 plain color words.\n"
        "- text: salient readable words only.\n"
        "- people: {count, faces_visible}.\n"
        "- privacy: only if applicable from content.\n"
        "- tags: union of media_type + scene + lighting + style + palette + objects; deduplicate; <=20.\n"
        "- notes: short sentence only if strong evidence (e.g., 'Likely a film still').\n"
        "- Omit fields that would be empty or null, except always include media_type, objects, people, tags.\n"
        "Return JSON only."
    )
