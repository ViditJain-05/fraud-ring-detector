"""
Synthetic Data Simulation Module for Fraud Ring Detector.
Generates realistic loan application dataset containing injected fraud rings sharing sensitive attributes.
"""

import os
import argparse
import random
import csv
from typing import List, Dict, Any
import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False



def generate_applicant_dataset(
    num_applicants: int = 1000,
    fraud_ratio: float = 0.05,
    min_ring_size: int = 3,
    max_ring_size: int = 6,
    seed: int = 42
) -> Any:

    """
    Generates synthetic loan applicants with injected fraud rings.

    Args:
        num_applicants (int): Total number of applicants to simulate.
        fraud_ratio (float): Approximate fraction of applicants belonging to fraud rings.
        min_ring_size (int): Minimum members in a single fraud ring cluster.
        max_ring_size (int): Maximum members in a single fraud ring cluster.
        seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: Simulated applicants DataFrame with features and is_fraud target label.
    """
    random.seed(seed)
    np.random.seed(seed)

    cities = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia", "San Antonio"]

    num_fraud = int(num_applicants * fraud_ratio)
    num_normal = num_applicants - num_fraud

    applicants: List[Dict[str, Any]] = []

    # 1. Generate Normal (Legitimate) Applicants
    for i in range(num_normal):
        app_id = f"APP_{i+1:04d}"
        applicants.append({
            "applicant_id": app_id,
            "phone": f"555-{random.randint(100, 999)}-{random.randint(1000, 9999)}",
            "bank_account": f"ACCT_{random.randint(10000000, 99999999)}",
            "address": f"{random.randint(100, 9999)} Main St, Apt {random.randint(1, 100)}",
            "income": round(float(np.random.normal(65000, 18000)), 2),
            "loan_amount": round(float(np.random.normal(15000, 5000)), 2),
            "age": random.randint(21, 65),
            "city": random.choice(cities),
            "is_fraud": 0
        })

    # 2. Inject Fraud Rings
    current_fraud_count = 0
    ring_id = 1
    app_counter = num_normal + 1

    while current_fraud_count < num_fraud:
        ring_size = min(random.randint(min_ring_size, max_ring_size), num_fraud - current_fraud_count)
        if ring_size <= 0:
            break

        # Shared ring attributes
        shared_bank = f"SHARED_ACCT_RING_{ring_id}_{random.randint(1000, 9999)}"
        shared_address = f"{random.randint(100, 999)} Fraud Ring Ave, Suite {ring_id}"
        shared_phone = f"555-RING-{random.randint(1000, 9999)}"

        for member_idx in range(ring_size):
            app_id = f"APP_{app_counter:04d}"
            app_counter += 1

            # Fraud members share bank and/or address/phone
            bank = shared_bank if random.random() < 0.85 else f"ACCT_{random.randint(10000000, 99999999)}"
            addr = shared_address if random.random() < 0.85 else f"{random.randint(100, 9999)} Oak St"
            phone = shared_phone if random.random() < 0.50 else f"555-{random.randint(100, 999)}-{random.randint(1000, 9999)}"

            applicants.append({
                "applicant_id": app_id,
                "phone": phone,
                "bank_account": bank,
                "address": addr,
                # Fraud applicants might request higher loans relative to income
                "income": round(float(np.random.normal(45000, 15000)), 2),
                "loan_amount": round(float(np.random.normal(28000, 8000)), 2),
                "age": random.randint(22, 50),
                "city": random.choice(cities),
                "is_fraud": 1
            })

        current_fraud_count += ring_size
        ring_id += 1

    random.shuffle(applicants)

    if HAS_PANDAS:
        df = pd.DataFrame(applicants)
        df["income"] = df["income"].clip(lower=15000)
        df["loan_amount"] = df["loan_amount"].clip(lower=2000)
        return df
    else:
        for app in applicants:
            app["income"] = max(15000.0, float(app["income"]))
            app["loan_amount"] = max(2000.0, float(app["loan_amount"]))
        return applicants


def save_applicants_to_csv(data: Any, output_path: str):
    """Saves applicants data to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if HAS_PANDAS and isinstance(data, pd.DataFrame):
        data.to_csv(output_path, index=False)
    else:
        if HAS_PANDAS and isinstance(data, pd.DataFrame):
            records = data.to_dict(orient="records")
        else:
            records = data
        if records:
            fieldnames = list(records[0].keys())
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(records)




def main():
    parser = argparse.ArgumentParser(description="Generate Synthetic Applicants Dataset")
    parser.add_argument("--num-applicants", type=int, default=1000, help="Total number of synthetic applicants")
    parser.add_argument("--fraud-ratio", type=float, default=0.05, help="Fraud ratio (0.05 = 5%)")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    data = generate_applicant_dataset(num_applicants=args.num_applicants, fraud_ratio=args.fraud_ratio)

    output_path = os.path.join(args.output_dir, "applicants.csv")
    save_applicants_to_csv(data, output_path)

    fraud_count = sum(1 for row in (data.to_dict(orient="records") if HAS_PANDAS else data) if int(row["is_fraud"]) == 1)
    print(f"SUCCESS: Generated dataset with {len(data)} applicants ({fraud_count} fraud).")
    print(f"Saved to '{output_path}'.")



if __name__ == "__main__":
    main()
