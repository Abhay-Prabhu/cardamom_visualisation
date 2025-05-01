import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np

# global styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 100

from sklearn.linear_model import LinearRegression

# --- Data loading ---
def load_data(path="cardamom_price_dataset.csv"):
    df = pd.read_csv(path)
    df["Date of Auction"] = pd.to_datetime(df["Date of Auction"],)
    df["Avg.Price (Rs./Kg)"] = pd.to_numeric(df["Avg.Price (Rs./Kg)"], errors="coerce")
    df["Month-Year"] = df["Date of Auction"].dt.to_period("M")
    df["Quarter"]    = df["Date of Auction"].dt.to_period("Q")
    df["Year"]       = df["Date of Auction"].dt.year
    df["Month"]      = df["Date of Auction"].dt.month
    return df.sort_values("Date of Auction")

# --- Plots ---

def plot_avg_price_over_time(data):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot("Date of Auction", "Avg.Price (Rs./Kg)", data=data, marker="o", lw=1.5)
    ax.set_title("Avg. Price of Cardamom Over Time")
    ax.set_xlabel("")
    ax.set_ylabel("Price (Rs./Kg)")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

def plot_auctioneer_vs_price(data):
    agg = (
        data
        .groupby("Auctioneer")["Avg.Price (Rs./Kg)"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(x="Avg.Price (Rs./Kg)", y="Auctioneer", data=agg, ax=ax)
    ax.set_title("Auctioneer vs Avg. Price")
    ax.set_xlabel("Avg. Price (Rs./Kg)")
    ax.set_ylabel("")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig)

def plot_monthly_trend_per_year(data):
    pivot = (
        data.groupby(["Year", "Month"])["Avg.Price (Rs./Kg)"]
        .mean()
        .unstack(level=0)
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(ax=ax, marker="o")
    ax.set_title("Monthly Avg. Price Trend by Year")
    ax.set_xlabel("Month")
    ax.set_ylabel("Price (Rs./Kg)")
    ax.set_xticks(range(1,13))
    ax.set_xticklabels(
        ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    )
    ax.legend(title="Year", bbox_to_anchor=(1.02, 1))
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig)

def plot_monthly_price_distribution(data):
    fig, ax = plt.subplots(figsize=(10, 5))

    # Create a 12-color palette from Set3
    colors = sns.color_palette("Set3", 12)

    # Pass it into boxplot
    sns.boxplot(
        x="Month",
        y="Avg.Price (Rs./Kg)",
        data=data,
        palette=colors,
        ax=ax
    )

    ax.set_title("Monthly Price Distribution")
    ax.set_xlabel("")
    ax.set_ylabel("Price (Rs./Kg)")
    ax.set_xticklabels(
        ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    )
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)


def plot_auctioneer_qty_sold(data):
    # 1) Clean & aggregate
    df = data.copy()
    df["Qty Sold (Kgs)"] = (
        df["Qty Sold (Kgs)"]
        .astype(str)
        .str.replace(",", "")                     # remove any commas
    )
    df["Qty Sold (Kgs)"] = pd.to_numeric(df["Qty Sold (Kgs)"], errors="coerce").fillna(0)

    qty = (
        df.groupby("Auctioneer")["Qty Sold (Kgs)"]
          .sum()
          .reset_index()
          .sort_values("Qty Sold (Kgs)")
    )

    # 2) Convert to tonnes
    qty["Qty Sold (t)"] = qty["Qty Sold (Kgs)"] / 1000

    # 3) Plot in tonnes
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x="Qty Sold (t)", y="Auctioneer", data=qty, palette="coolwarm", ax=ax)

    # 4) Labels & grid
    ax.set_title("Total Quantity Sold by Auctioneer", fontsize=14)
    ax.set_xlabel("Qty Sold (tonnes-t)", fontsize=12)
    ax.set_ylabel("")
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    # 5) Annotate with one decimal place
    max_t = qty["Qty Sold (t)"].max()
    for i, v in enumerate(qty["Qty Sold (t)"]):
        ax.text(
            v + max_t * 0.01,
            i,
            f"{v:.1f} t",
            va="center",
            fontsize=10
        )

    plt.tight_layout()
    st.pyplot(fig)




# --- Main App ---

def main():
    st.set_page_config(
    page_title="Cardamom Auction Price Analysis",
    layout="wide",              # ← edge-to-edge
    initial_sidebar_state="collapsed",
)
    data = load_data()

    st.sidebar.title("Choose a chart")
    choice = st.sidebar.radio(
        "",
        [
            "Avg Price Over Time",
            "Auctioneer vs Price",
            "Monthly Trend per Year",
            "Monthly Price Distribution",
            "Auctioneer-wise Qty Sold",
        ]
    )

    if choice == "Avg Price Over Time":
        plot_avg_price_over_time(data)
    elif choice == "Auctioneer vs Price":
        plot_auctioneer_vs_price(data)
    elif choice == "Monthly Trend per Year":
        plot_monthly_trend_per_year(data)
    elif choice == "Monthly Price Distribution":
        plot_monthly_price_distribution(data)
    elif choice == "Auctioneer-wise Qty Sold":
        plot_auctioneer_qty_sold(data)

if __name__ == "__main__":
    main()
