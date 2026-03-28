"""
Goldbach Level Engine - Core ICT/Power of 3 Calculator
"""
from typing import Optional, List
import numpy as np


def calculate_goldbach_levels(high: float, low: float) -> dict:
    """Calculate all Goldbach/PO3 levels for a dealing range."""
    rng = high - low
    eq = (high + low) / 2.0
    levels = []

    powers = [3, 9, 27, 81]
    for p in powers:
        inc = rng / p
        for i in range(1, p):
            price = low + inc * i
            zone = "premium" if price > eq else ("discount" if price < eq else "equilibrium")
            levels.append({
                "price": price,
                "power": p,
                "fraction": f"{i}/{p}",
                "zone": zone,
                "label": f"PO{p} {i}/{p}",
            })

    levels.sort(key=lambda l: l["price"])
    return {
        "high": high,
        "low": low,
        "range": rng,
        "equilibrium": eq,
        "levels": levels,
        "premium_zone": (eq, high),
        "discount_zone": (low, eq),
    }


def get_nearest_goldbach_level(price: float, levels: list) -> Optional[dict]:
    """Find the nearest Goldbach level to a given price."""
    if not levels:
        return None
    return min(levels, key=lambda l: abs(l["price"] - price))


def get_amd_zones(high: float, low: float) -> dict:
    """Calculate AMD (Accumulation/Manipulation/Distribution) zones."""
    rng = high - low
    third = rng / 3.0
    return {
        "accumulation": (low, low + third),
        "manipulation": (low + third, low + 2 * third),
        "distribution": (low + 2 * third, high),
    }


def price_in_zone(price: float, high: float, low: float) -> str:
    """Determine if price is in premium or discount zone."""
    eq = (high + low) / 2.0
    if price > eq:
        return "premium"
    elif price < eq:
        return "discount"
    return "equilibrium"


def get_po3_levels(high: float, low: float, power: int = 3) -> List[float]:
    """Get just the price levels for a specific power."""
    rng = high - low
    inc = rng / power
    return [low + inc * i for i in range(1, power)]
