"""Object categories, colors, and delivery vocabulary for recaps.

Colors are OpenCV BGR and match the standalone prototype the recap UI
was designed against: people blue, vehicles orange, deliveries magenta,
animals green, parked-car get-in/out teal.
"""

from __future__ import annotations

# Frigate+ attribute labels that mean "this is a delivery", plus a few
# object labels a custom model might emit directly.
DELIVERY_LABELS = frozenset(
    {
        "amazon",
        "an_post",
        "canada_post",
        "dhl",
        "dpd",
        "fedex",
        "gls",
        "nzpost",
        "package",
        "postnl",
        "postnord",
        "purolator",
        "royal_mail",
        "ups",
        "usps",
    }
)

VEHICLE_LABELS = frozenset({"car", "truck", "bus", "motorcycle", "bicycle"})
ANIMAL_LABELS = frozenset(
    {
        "dog",
        "cat",
        "bird",
        "horse",
        "sheep",
        "cow",
        "bear",
        "deer",
        "rabbit",
        "squirrel",
        "fox",
        "raccoon",
    }
)

CAT_ORDER = ("person", "animal", "delivery", "parked", "vehicle")

CAT_NAME = {
    "person": "People",
    "vehicle": "Vehicles",
    "delivery": "Deliveries",
    "animal": "Animals",
    "parked": "Parked car (got in/out)",
}

# BGR
CAT_COLOR = {
    "person": (255, 185, 60),
    "vehicle": (40, 165, 255),
    "delivery": (215, 80, 235),
    "animal": (80, 215, 90),
    "parked": (185, 175, 0),
}

DELIVERY_QUERIES_VEHICLE = (
    "delivery truck",
    "UPS truck",
    "Amazon delivery van",
    "FedEx truck",
    "USPS mail truck",
)
DELIVERY_QUERIES_PERSON = (
    "person carrying a package",
    "delivery driver carrying a box",
)
DOG_WALKER_QUERY = "a person walking a dog"


def category_of(label: str) -> str:
    """Map a detector label to a recap category."""
    if label in DELIVERY_LABELS:
        return "delivery"
    if label in VEHICLE_LABELS:
        return "vehicle"
    if label in ANIMAL_LABELS:
        return "animal"
    if label == "person":
        return "person"
    return "person"


def bgr_hex(color: tuple[int, int, int]) -> str:
    """Convert an OpenCV BGR triple to a CSS hex color."""
    blue, green, red = color
    return f"#{red:02x}{green:02x}{blue:02x}"
