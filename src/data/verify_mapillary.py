"""
Phase 0 check: can we fetch street-level photos from Mapillary near a real
North Las Vegas intersection?

Mapillary requires a personal access token (Mapillary calls it a "client
token") that you get for free at:
    https://www.mapillary.com/dashboard/developers
Create an app there, copy its client token, and put it in .env as:
    MAPILLARY_TOKEN=your_token_here

This script only checks that the token works and that images exist for a
sample bounding box — it doesn't download or store any photos yet.
"""

import requests

from src.config import MAPILLARY_TOKEN, MAPILLARY_API_URL

# A small box (~200m across) around Las Vegas Blvd N & Lake Mead Blvd,
# a real intersection in North Las Vegas, just to prove images exist there.
SAMPLE_BBOX = (-115.1205, 36.2010, -115.1155, 36.2045)  # (west, south, east, north)


def check_mapillary_token() -> bool:
    """A token is present — doesn't confirm it's *valid*, just that it's set."""
    return bool(MAPILLARY_TOKEN)


def fetch_sample_images(bbox: tuple, limit: int = 5) -> list[dict]:
    """Query the Mapillary Graph API v4 for images inside a bounding box."""
    west, south, east, north = bbox
    response = requests.get(
        f"{MAPILLARY_API_URL}/images",
        params={
            "access_token": MAPILLARY_TOKEN,
            "fields": "id,captured_at,geometry",
            "bbox": f"{west},{south},{east},{north}",
            "limit": limit,
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json().get("data", [])


def verify_mapillary(bbox: tuple = SAMPLE_BBOX) -> dict:
    if not check_mapillary_token():
        return {
            "source": "mapillary",
            "passed": False,
            "reason": (
                "MAPILLARY_TOKEN is not set. Sign up at "
                "https://www.mapillary.com/dashboard/developers, create an app, "
                "and add the client token to .env as MAPILLARY_TOKEN=..."
            ),
        }

    images = fetch_sample_images(bbox)
    return {
        "source": "mapillary",
        "passed": len(images) > 0,
        "sample_image_count": len(images),
    }


if __name__ == "__main__":
    result = verify_mapillary()
    print(result)
