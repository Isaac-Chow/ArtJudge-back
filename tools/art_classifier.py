from __future__ import annotations

import base64
import io
import os
import time
from typing import List, Optional

from PIL import Image


_classifier = None  # transformers pipeline instance
_LABEL_MAP: dict[str, str] | None = None


DEFAULT_LABEL_MAP: dict[str, str] = {
    "impressionism": "Impressionism", 
    "expressionism": "Expressionism",
    "cubism": "Cubism",
    "surrealism": "Surrealism",
    "baroque": "Baroque",
    "romanticism": "Romanticism",
    "realism": "Realism",
    "abstract": "Abstract Art", 
    "pop_art": "Pop Art",
    "art_nouveau": "Art Nouveau",
    "art_deco": "Art Deco",
    "renaissance": "Renaissance",
    "gothic": "Gothic",
    "neoclassicism": "Neoclassicism",
    "rococo": "Rococo",
    "mannerism": "Mannerism",
    "fauvism": "Fauvism",
    "dadaism": "Dadaism",
    "minimalism": "Minimalism",
    "post_impressionism": "Post-Impressionism",
    "pointillism": "Pointillism",
    "symbolism": "Symbolism",
    "ukiyo_e": "Ukiyo-e",
    "naive_art": "Naive Art",
    "hyperrealism": "Hyperrealism",
    # advise to ahave an unknown class
}

def _get_label_map() -> dict[str, str]:
    """Return the label map, initialising it once."""
    global _LABEL_MAP
    if _LABEL_MAP is None:
        _LABEL_MAP = DEFAULT_LABEL_MAP.copy()
    return _LABEL_MAP


def _normalise_label(raw_label: str) -> str:
    """Convert a raw model label to a human-readable art style name."""
    label_map = _get_label_map()
    # Normalise: lowercase, strip spaces, replace hyphens/underscores
    key = raw_label.strip().lower().replace(" ", "_").replace("-", "_")
    return label_map.get(key, raw_label.title())

def _load_classifier():
    """Lazy-load the HuggingFace image-classification pipeline."""
    global _classifier
    if _classifier is not None:
        return _classifier

    from transformers import pipeline  # imported here to keep startup fast

    model_name = os.getenv(
        "ART_CLASSIFIER_MODEL",
        "dima806/painting-style-classifier",
    )
    hf_token = os.getenv("HF_TOKEN")  # optional, for gated models

    print(f"[classifier] Loading model '{model_name}'...")
    t0 = time.perf_counter()
    _classifier = pipeline(
        "image-classification",
        model=model_name,
        token=hf_token,
        top_k=3,
    )
    elapsed = time.perf_counter() - t0
    print(f"[classifier] Model loaded in {elapsed:.1f}s")
    return _classifier


# Helper Functions For image prepping
def classify_art_style(image_base64: str) -> dict:
    """Classify the art style of a base64-encoded image.

    Args:
        image_base64: Raw or data-URI base64 string of the image.

    Returns:
        A dict with keys matching ArtStyleClassification fields.
    """

    if "," in image_base64:
        image_base64 = image_base64.split(",")[1]

    image_bytes = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    classifier = _load_classifier()
    t0 = time.perf_counter()
    results: list = classifier(image)
    elapsed = time.perf_counter() - t0
    print(f"Classification took {elapsed:.2f} seconds.")

    if not results:
        return{
            "style_name": "Unknown",
            "confidence": 0.0,
            "period": "Unknown",
            "description": "Could not determine the art style.",
            "key_features": [],
            "notable_artists": [],
        }

    top = results[0]
    style_name = _normalise_label(top["label"])
    confidence = round(float(top["score"]), 3)

    description = _build_description(style_name)
    key_features = _get_features(style_name)
    notable_artists = _get_artists(style_name)
    period = _get_period(style_name)

    return{
        "style_name": style_name,
        "confidence": confidence,
        "period": period,
        "description": description,
        "key_features": key_features,
        "notable_artists": notable_artists,
    }

# Classification Tool: Funtion that calls the huggingface classifer model

_STYLE_METADATA: dict[str, dict] = {
    "Impressionism": {
        "period": "Late 19th century (1860s-1880s)",
        "description": (
            "Impressionism captures the fleeting effects of light and colour "
            "in everyday scenes. Artists painted quickly with visible brushstrokes "
            "to convey the 'impression' of a moment rather than fine detail."
        ),
        "features": [
            "Visible, short brushstrokes",
            "Emphasis on changing light",
            "Ordinary subject matter",
            "Unusual visual angles",
            "No black — shadows painted with complementary colours",
        ],
        "artists": ["Claude Monet", "Pierre-Auguste Renoir", "Edgar Degas", "Camille Pissarro"],
    },
    "Expressionism": {
        "period": "Early 20th century (1905-1930s)",
        "description": (
            "Expressionism distorts reality to convey intense emotion. "
            "Bold colours and exaggerated forms express inner feelings "
            "rather than objective appearances."
        ),
        "features": [
            "Vivid, non-naturalistic colours",
            "Distorted or exaggerated shapes",
            "Emotional intensity over realism",
            "Swirling, dynamic compositions",
        ],
        "artists": ["Edvard Munch", "Wassily Kandinsky", "Ernst Ludwig Kirchner"],
    },
    "Cubism": {
        "period": "Early 20th century (1907-1920s)",
        "description": (
            "Cubism breaks objects into geometric shapes and reassembles "
            "them from multiple viewpoints simultaneously. It challenged "
            "traditional perspective and revolutionised modern art."
        ),
        "features": [
            "Geometric fragmentation",
            "Multiple viewpoints at once",
            "Flat, two-dimensional surface",
            "Monochrome or limited palette",
        ],
        "artists": ["Pablo Picasso", "Georges Braque", "Juan Gris"],
    },
    "Surrealism": {
        "period": "1920s-1950s",
        "description": (
            "Surrealism explores the unconscious mind through dream-like "
            "imagery, unexpected juxtapositions, and illogical scenes. "
            "It seeks to unlock creativity beyond rational thought."
        ),
        "features": [
            "Dream-like, bizarre scenes",
            "Unexpected object combinations",
            "Melting or distorted forms",
            "Automatism and chance techniques",
        ],
        "artists": ["Salvador Dali", "Rene Magritte", "Joan Miro"],
    },
    "Baroque": {
        "period": "17th - early 18th century",
        "description": (
            "Baroque art is dramatic and grand, using strong contrasts "
            "of light and dark (chiaroscuro) to create emotional intensity "
            "and a sense of movement."
        ),
        "features": [
            "Dramatic lighting (chiaroscuro)",
            "Rich, deep colours",
            "Sense of motion and energy",
            "Ornate detail",
        ],
        "artists": ["Caravaggio", "Rembrandt", "Peter Paul Rubens"],
    },
    "Renaissance": {
        "period": "14th - 17th century",
        "description": (
            "The Renaissance revived classical ideals of harmony, proportion, "
            "and realism. Artists developed linear perspective and anatomical "
            "accuracy to create lifelike, balanced compositions."
        ),
        "features": [
            "Linear perspective",
            "Anatomically accurate figures",
            "Balanced, symmetrical composition",
            "Classical themes and motifs",
        ],
        "artists": ["Leonardo da Vinci", "Michelangelo", "Raphael"],
    },
    "Romanticism": {
        "period": "Late 18th - mid 19th century",
        "description": (
            "Romanticism celebrates emotion, nature, and individual imagination. "
            "It often depicts dramatic landscapes, heroic events, and the sublime "
            "power of the natural world."
        ),
        "features": [
            "Emphasis on emotion and individualism",
            "Dramatic, awe-inspiring landscapes",
            "Interest in the exotic and medieval",
            "Bold, expressive brushwork",
        ],
        "artists": ["Caspar David Friedrich", "J.M.W. Turner", "Eugene Delacroix"],
    },
    "Realism": {
        "period": "Mid 19th century (1840s-1880s)",
        "description": (
            "Realism depicts everyday life truthfully, without idealisation. "
            "Artists focused on ordinary people and situations, often highlighting "
            "social conditions."
        ),
        "features": [
            "Accurate, detailed depiction",
            "Everyday subject matter",
            "Natural lighting",
            "Avoidance of romantic idealisation",
        ],
        "artists": ["Gustave Courbet", "Jean-Francois Millet", "Honore Daumier"],
    },
    "Pop Art": {
        "period": "1950s-1970s",
        "description": (
            "Pop Art draws inspiration from popular culture — advertising, "
            "comics, and consumer products. It uses bold colours and mechanical "
            "reproduction techniques to blur the line between high and low art."
        ),
        "features": [
            "Bold, flat colours",
            "Imagery from advertising and media",
            "Repetition and mass-production aesthetic",
            "Irony and cultural commentary",
        ],
        "artists": ["Andy Warhol", "Roy Lichtenstein", "Claes Oldenburg"],
    },
    "Abstract Art": {
        "period": "Early 20th century - present",
        "description": (
            "Abstract art does not attempt to represent visual reality. "
            "Instead it uses shapes, colours, forms, and gestural marks "
            "to achieve its effect, inviting personal interpretation."
        ),
        "features": [
            "Non-representational forms",
            "Emphasis on colour and shape",
            "Gestural or geometric marks",
            "Open to interpretation",
        ],
        "artists": ["Wassily Kandinsky", "Jackson Pollock", "Mark Rothko"],
    },
}


def _build_description(style: str) -> str:
    meta = _STYLE_METADATA.get(style)
    if meta:
        return meta["description"]
    return (
        f"This artwork appears to be in the {style} style. "
        "Use the web_search tool to find more information about this art movement."
    )


def _get_features(style: str) -> List[str]:
    meta = _STYLE_METADATA.get(style)
    return meta["features"] if meta else []


def _get_artists(style: str) -> List[str]:
    meta = _STYLE_METADATA.get(style)
    return meta["artists"] if meta else []


def _get_period(style: str) -> str:
    meta = _STYLE_METADATA.get(style)
    return meta["period"] if meta else "Varies"
