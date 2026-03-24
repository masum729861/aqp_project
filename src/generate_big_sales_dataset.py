from __future__ import annotations

from pathlib import Path
import random
import csv


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
OUTPUT_FILE = DATASET_DIR / "sales.csv"

NUM_ROWS = 10000
RANDOM_SEED = 42


def weighted_choice(options: list[tuple[str, float]]) -> str:
    values = [item[0] for item in options]
    weights = [item[1] for item in options]
    return random.choices(values, weights=weights, k=1)[0]


def generate_row() -> tuple[str, float, int, float]:
    # Region distribution is intentionally skewed
    region = weighted_choice([
        ("East", 0.35),
        ("West", 0.30),
        ("South", 0.20),
        ("North", 0.15),
    ])

    # Category-like behavior embedded through price/sales patterns by region
    if region == "East":
        base_price = random.uniform(40, 90)
        quantity = random.randint(1, 8)
    elif region == "West":
        base_price = random.uniform(60, 140)
        quantity = random.randint(1, 10)
    elif region == "South":
        base_price = random.uniform(30, 75)
        quantity = random.randint(1, 6)
    else:  # North
        base_price = random.uniform(50, 110)
        quantity = random.randint(1, 7)

    # Add a small amount of skew and noise
    if random.random() < 0.08:
        base_price *= random.uniform(1.5, 2.8)
        quantity += random.randint(1, 4)

    price = round(base_price, 2)
    sales = round(price * quantity + random.uniform(-10, 15), 2)

    # Ensure sales is not negative
    sales = max(sales, 1.0)

    return region, sales, quantity, price


def main() -> None:
    random.seed(RANDOM_SEED)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["region", "sales", "quantity", "price"])

        for _ in range(NUM_ROWS):
            writer.writerow(generate_row())

    print(f"Dataset created successfully: {OUTPUT_FILE}")
    print(f"Total rows: {NUM_ROWS}")


if __name__ == "__main__":
    main()