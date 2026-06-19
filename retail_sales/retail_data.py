"""Shared retail dataset generator for all algorithm notebooks."""

import numpy as np
import pandas as pd
def generate_retail_dataset(n_samples: int = 2000, random_state: int = 42) -> pd.DataFrame:
    """Generate a synthetic retail customer transaction dataset."""
    rng = np.random.default_rng(random_state)

    regions = ["North", "South", "East", "West"]
    categories = ["Electronics", "Clothing", "Groceries", "Furniture", "Sports"]
    channels = ["Online", "In-Store", "Mobile"]
    segments = ["Budget", "Regular", "Premium", "VIP"]

    age = rng.integers(18, 70, size=n_samples)
    annual_income = rng.integers(20_000, 120_000, size=n_samples)
    spending_score = rng.integers(1, 100, size=n_samples)
    region = rng.choice(regions, size=n_samples)
    product_category = rng.choice(categories, size=n_samples, p=[0.25, 0.22, 0.20, 0.18, 0.15])
    purchase_channel = rng.choice(channels, size=n_samples, p=[0.45, 0.35, 0.20])
    num_purchases = rng.poisson(15, size=n_samples) + 1
    avg_transaction_value = rng.uniform(10, 500, size=n_samples).round(2)
    total_sales = (num_purchases * avg_transaction_value * rng.uniform(0.85, 1.15, size=n_samples)).round(2)

    # Segment is derived from income + spending for realistic classification target
    segment_score = (annual_income / 120_000) * 0.5 + (spending_score / 100) * 0.5
    customer_segment = np.select(
        [
            segment_score < 0.30,
            segment_score < 0.55,
            segment_score < 0.78,
        ],
        ["Budget", "Regular", "Premium"],
        default="VIP",
    )

    df = pd.DataFrame(
        {
            "CustomerID": np.arange(1, n_samples + 1),
            "Age": age,
            "Annual_Income": annual_income,
            "Spending_Score": spending_score,
            "Region": region,
            "Product_Category": product_category,
            "Purchase_Channel": purchase_channel,
            "Num_Purchases": num_purchases,
            "Avg_Transaction_Value": avg_transaction_value,
            "Total_Sales": total_sales,
            "Customer_Segment": customer_segment,
        }
    )
    return df


if __name__ == "__main__":
    data = generate_retail_dataset()
    data.to_csv("retail_dataset.csv", index=False)
    print(f"Saved retail_dataset.csv with {len(data)} rows and {len(data.columns)} columns.")
