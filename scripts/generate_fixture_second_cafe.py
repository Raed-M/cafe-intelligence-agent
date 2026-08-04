"""Generates a second, synthetic cafe dataset (`data/sundown_roasters/`) that
matches the exact supplied schemas, to prove generic onboarding (plan section
28.7 / M30): a different cafe name, city/coordinates/region, social handles,
menu and 2 weeks of operational data, produced by this script rather than
committed as opaque fixture files, so the generation logic itself is
auditable. No application source file is touched by running this script or
by pointing the graph at its output.
"""
from __future__ import annotations

import json
import random
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import openpyxl
import pandas as pd

random.seed(42)

OUT_DIR = Path("data/sundown_roasters")
WEEKS = [date(2026, 2, 2), date(2026, 2, 9)]  # 2 complete Monday-start weeks

MENU = [
    {"sku": "ESP-001", "item_en": "Espresso", "item_ar": "إسبريسو", "category": "hot", "price_sar": 12.0, "unit_cost_sar": 3.0, "is_iced": False, "launch_date": "", "retire_date": ""},
    {"sku": "ESP-002", "item_en": "Cappuccino", "item_ar": "كابتشينو", "category": "hot", "price_sar": 16.0, "unit_cost_sar": 4.5, "is_iced": False, "launch_date": "", "retire_date": ""},
    {"sku": "ESP-003", "item_en": "Flat White", "item_ar": "فلات وايت", "category": "hot", "price_sar": 17.0, "unit_cost_sar": 4.8, "is_iced": False, "launch_date": "", "retire_date": ""},
    {"sku": "ICE-101", "item_en": "Iced Americano", "item_ar": "أمريكانو مثلج", "category": "iced", "price_sar": 14.0, "unit_cost_sar": 3.2, "is_iced": True, "launch_date": "", "retire_date": ""},
    {"sku": "ICE-102", "item_en": "Iced Mocha", "item_ar": "موكا مثلج", "category": "iced", "price_sar": 19.0, "unit_cost_sar": 5.5, "is_iced": True, "launch_date": "", "retire_date": ""},
    {"sku": "ICE-103", "item_en": "Coconut Cold Brew", "item_ar": "قهوة باردة بجوز الهند", "category": "iced", "price_sar": 20.0, "unit_cost_sar": 6.0, "is_iced": True, "launch_date": "2026-02-09", "retire_date": ""},
    {"sku": "FOD-101", "item_en": "Almond Croissant", "item_ar": "كرواسون لوز", "category": "food", "price_sar": 15.0, "unit_cost_sar": 5.0, "is_iced": False, "launch_date": "", "retire_date": ""},
    {"sku": "FOD-102", "item_en": "Date Muffin", "item_ar": "مافن تمر", "category": "food", "price_sar": 11.0, "unit_cost_sar": 3.5, "is_iced": False, "launch_date": "", "retire_date": ""},
]

STAFF = [
    {"employee_id": "EMP-01", "name": "Fahad", "role": "barista", "hourly_rate_sar": 22.0},
    {"employee_id": "EMP-02", "name": "Layla", "role": "barista", "hourly_rate_sar": 24.0},
    {"employee_id": "EMP-03", "name": "Omar", "role": "cashier", "hourly_rate_sar": 20.0},
]

PAYMENT_METHODS = ["mada", "apple_pay", "cash", "visa"]
CHANNELS = ["dine_in", "takeaway", "delivery"]


def build_profile() -> dict:
    return {
        "cafe_name": "Sundown Roasters",
        "city": "Jeddah",
        "governorate": "Jeddah",
        "region": "Makkah Region",
        "country": "Saudi Arabia",
        "coordinates": {"lat": 21.5433, "lng": 39.1728},
        "timezone": "Asia/Riyadh",
        "seats": 28,
        "opened": "2024-09-15",
        "opening_hours": {"default": "06:30-22:00", "ramadan": "15:00-00:30"},
        "instagram": "@sundown.roasters",
        "tiktok": "@sundownroasters",
        "currency": "SAR",
        "weekend_days": ["Friday", "Saturday"],
        "notes": "Coastal specialty roastery. Young professional crowd. Bilingual audience skewing English-forward.",
    }


def build_menu_csv(out_dir: Path) -> None:
    pd.DataFrame(MENU).to_csv(out_dir / "menu_items.csv", index=False)


def build_pos_and_traffic_and_staff(out_dir: Path) -> None:
    pos_rows = []
    traffic_rows = []
    staff_rows = []
    tx_counter = 200000

    for week_start in WEEKS:
        for day_offset in range(7):
            d = week_start + timedelta(days=day_offset)
            for emp in STAFF:
                shift_start, shift_end = ("07:00", "15:00") if emp["role"] != "cashier" else ("14:00", "22:00")
                hours = 8.0
                staff_rows.append({
                    "date": d.isoformat(), "employee_id": emp["employee_id"], "name": emp["name"],
                    "role": emp["role"], "shift_start": shift_start, "shift_end": shift_end,
                    "hours": hours, "hourly_rate_sar": emp["hourly_rate_sar"],
                })
            for hour in range(6, 22):
                door_count = random.randint(5, 40) if 7 <= hour <= 20 else random.randint(0, 5)
                traffic_rows.append({"date": d.isoformat(), "hour": hour, "door_count": door_count})
                n_tx = max(0, int(door_count * random.uniform(0.15, 0.35)))
                for _ in range(n_tx):
                    tx_counter += 1
                    tid = f"TXN-{tx_counter}"
                    ts = f"{d.isoformat()} {hour:02d}:{random.randint(0,59):02d}:00"
                    n_items = random.choice([1, 1, 2])
                    for _ in range(n_items):
                        item = random.choice(MENU)
                        if item["launch_date"] and d.isoformat() < item["launch_date"]:
                            item = MENU[0]
                        qty = 1
                        unit_price = item["price_sar"]
                        discount = 0.0
                        line_total = qty * unit_price - discount
                        pos_rows.append({
                            "transaction_id": tid, "timestamp": ts, "sku": item["sku"],
                            "item_name": item["item_en"], "quantity": qty, "unit_price_sar": unit_price,
                            "discount_sar": discount, "line_total_sar": line_total,
                            "payment_method": random.choice(PAYMENT_METHODS),
                            "channel": random.choice(CHANNELS),
                            "cashier_id": random.choice([e["employee_id"] for e in STAFF]),
                        })

    pd.DataFrame(pos_rows).to_csv(out_dir / "pos_transactions.csv", index=False)
    pd.DataFrame(traffic_rows).to_csv(out_dir / "foot_traffic.csv", index=False)
    pd.DataFrame(staff_rows).to_csv(out_dir / "staff_shifts.csv", index=False)


def build_inventory(out_dir: Path) -> None:
    rows = []
    for week_start in WEEKS:
        for item in MENU:
            ordered = random.randint(20, 60)
            sold = random.randint(10, ordered)
            wasted = random.choice([0, 1, 2, None])
            rows.append({
                "week_starting": week_start.isoformat(), "sku": item["sku"], "item": item["item_en"],
                "units_ordered": ordered, "units_sold": sold, "units_wasted": wasted,
                "unit_cost_sar": item["unit_cost_sar"],
            })
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "weekly_counts"
    df = pd.DataFrame(rows)
    ws.append(list(df.columns))
    for row in df.itertuples(index=False):
        ws.append(list(row))
    readme = wb.create_sheet("README")
    readme.append(["note"])
    readme.append(["Counts taken Sunday morning before opening."])
    readme.append(["Blank waste = count not recorded that week."])
    readme.append(["Milk/beans tracked separately by supplier invoice."])
    wb.save(out_dir / "inventory_weekly.xlsx")


def build_emails(out_dir: Path) -> None:
    emails_dir = out_dir / "supplier_emails"
    emails_dir.mkdir(exist_ok=True)
    emails = [
        ("2026-01-20_01.txt", "From: sales@hijazbeans.sa\nDate: 2026-01-20\nSubject: Price notice\n\n"
                                "Roasted blend moves from SAR 82/kg to SAR 90/kg effective 2026-02-02.\n"),
        ("2026-01-28_02.txt", "From: orders@jeddahdairy.com\nDate: 2026-01-28\nSubject: Delivery delay\n\n"
                                "Deliveries on 2026-02-02 will arrive in the afternoon instead of morning due to a vehicle issue.\n"),
        ("2026-02-01_03.txt", "From: events@jeddah-corniche.sa\nDate: 2026-02-01\nSubject: Corniche Winter Market\n\n"
                                "The Jeddah Corniche Winter Market runs 2026-02-05 to 2026-02-14, nightly 17:00-23:00. Vendor slots available.\n"),
    ]
    for name, content in emails:
        (emails_dir / name).write_text(content, encoding="utf-8")


def build_reviews(out_dir: Path) -> None:
    samples_en = ["Great flat white, friendly staff.", "A bit slow during peak hours but worth it.",
                  "Best cold brew in Jeddah.", "Seating is limited on weekends."]
    samples_ar = ["قهوة ممتازة وخدمة سريعة", "المكان جميل لكن مزدحم أحياناً", "أفضل قهوة باردة جربتها", "الأسعار مناسبة"]
    reviews = []
    rid = 1
    for week_start in WEEKS:
        for i in range(15):
            d = week_start + timedelta(days=random.randint(0, 6))
            is_ar = random.random() < 0.4
            reviews.append({
                "review_id": f"REV-{rid:04d}", "date": d.isoformat(),
                "source": random.choice(["google", "instagram", "talabat"]),
                "rating": random.randint(3, 5),
                "text": random.choice(samples_ar if is_ar else samples_en),
            })
            rid += 1
    (out_dir / "customer_reviews.json").write_text(json.dumps(reviews, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "cafe_profile.json").write_text(json.dumps(build_profile(), indent=2), encoding="utf-8")
    build_menu_csv(OUT_DIR)
    build_pos_and_traffic_and_staff(OUT_DIR)
    build_inventory(OUT_DIR)
    build_emails(OUT_DIR)
    build_reviews(OUT_DIR)
    print(f"Second-cafe fixture written to {OUT_DIR}")


if __name__ == "__main__":
    main()
