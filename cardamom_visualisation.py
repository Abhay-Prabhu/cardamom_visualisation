# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import streamlit as st
# import numpy as np

# # global styling
# sns.set_theme(style="whitegrid", palette="muted")
# plt.rcParams["figure.dpi"] = 200

# from sklearn.linear_model import LinearRegression

# # --- Data loading ---
# def load_data(path="cardamom_price_dataset.csv"):
#     df = pd.read_csv(path)
#     df["Date of Auction"] = pd.to_datetime(df["Date of Auction"],)
#     df["Avg.Price (Rs./Kg)"] = pd.to_numeric(df["Avg.Price (Rs./Kg)"], errors="coerce")
#     df["Month-Year"] = df["Date of Auction"].dt.to_period("M")
#     df["Quarter"]    = df["Date of Auction"].dt.to_period("Q")
#     df["Year"]       = df["Date of Auction"].dt.year
#     df["Month"]      = df["Date of Auction"].dt.month
#     return df.sort_values("Date of Auction")

# # --- Plots ---

# def plot_avg_price_over_time(data):
#     fig, ax = plt.subplots(figsize=(14, 10))
#     ax.plot("Date of Auction", "Avg.Price (Rs./Kg)", data=data, marker="o", lw=1.5)
#     ax.set_title("Avg. Price of Cardamom Over Time")
#     ax.set_xlabel("")
#     ax.set_ylabel("Price (Rs./Kg)")
#     ax.grid(True, linestyle="--", alpha=0.5)
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     st.pyplot(fig)

# def plot_auctioneer_vs_price(data):
#     agg = (
#         data
#         .groupby("Auctioneer")["Avg.Price (Rs./Kg)"]
#         .mean()
#         .sort_values(ascending=False)
#         .reset_index()
#     )
#     fig, ax = plt.subplots(figsize=(12, 5))
#     sns.barplot(x="Avg.Price (Rs./Kg)", y="Auctioneer", data=agg, ax=ax)
#     ax.set_title("Auctioneer vs Avg. Price")
#     ax.set_xlabel("Avg. Price (Rs./Kg)")
#     ax.set_ylabel("")
#     ax.grid(axis="x", linestyle="--", alpha=0.5)
#     plt.tight_layout()
#     st.pyplot(fig)

# def plot_monthly_trend_per_year(data):
#     pivot = (
#         data.groupby(["Year", "Month"])["Avg.Price (Rs./Kg)"]
#         .mean()
#         .unstack(level=0)
#     )
#     fig, ax = plt.subplots(figsize=(10, 5))
#     pivot.plot(ax=ax, marker="o")
#     ax.set_title("Monthly Avg. Price Trend by Year")
#     ax.set_xlabel("Month")
#     ax.set_ylabel("Price (Rs./Kg)")
#     ax.set_xticks(range(1,13))
#     ax.set_xticklabels(
#         ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
#     )
#     ax.legend(title="Year", bbox_to_anchor=(1.02, 1))
#     ax.grid(True, linestyle="--", alpha=0.5)
#     plt.tight_layout()
#     st.pyplot(fig)

# def plot_monthly_price_distribution(data):
#     fig, ax = plt.subplots(figsize=(10, 5))

#     # Create a 12-color palette from Set3
#     colors = sns.color_palette("Set3", 12)

#     # Pass it into boxplot
#     sns.boxplot(
#         x="Month",
#         y="Avg.Price (Rs./Kg)",
#         data=data,
#         palette=colors,
#         ax=ax
#     )

#     ax.set_title("Monthly Price Distribution")
#     ax.set_xlabel("")
#     ax.set_ylabel("Price (Rs./Kg)")
#     ax.set_xticklabels(
#         ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
#     )
#     ax.grid(True, linestyle="--", alpha=0.3)
#     plt.tight_layout()
#     st.pyplot(fig)


# def plot_auctioneer_qty_sold(data):
#     # 1) Clean & aggregate
#     df = data.copy()
#     df["Qty Sold (Kgs)"] = (
#         df["Qty Sold (Kgs)"]
#         .astype(str)
#         .str.replace(",", "")                     # aremove any commas
#     )
#     df["Qty Sold (Kgs)"] = pd.to_numeric(df["Qty Sold (Kgs)"], errors="coerce").fillna(0)

#     qty = (
#         df.groupby("Auctioneer")["Qty Sold (Kgs)"]
#           .sum()
#           .reset_index()
#           .sort_values("Qty Sold (Kgs)")
#     )

#     # 2) Convert to tonnes
#     qty["Qty Sold (t)"] = qty["Qty Sold (Kgs)"] / 1000

#     # 3) Plot in tonnes
#     fig, ax = plt.subplots(figsize=(10, 5))
#     sns.barplot(x="Qty Sold (t)", y="Auctioneer", data=qty, palette="coolwarm", ax=ax)

#     # 4) Labels & grid
#     ax.set_title("Total Quantity Sold by Auctioneer", fontsize=14)
#     ax.set_xlabel("Qty Sold (tonnes-t)", fontsize=12)
#     ax.set_ylabel("")
#     ax.grid(axis="x", linestyle="--", alpha=0.5)

#     # 5) Annotate with one decimal place
#     max_t = qty["Qty Sold (t)"].max()
#     for i, v in enumerate(qty["Qty Sold (t)"]):
#         ax.text(
#             v + max_t * 0.01,
#             i,
#             f"{v:.1f} t",
#             va="center",
#             fontsize=10
#         )

#     plt.tight_layout()
#     st.pyplot(fig)




# # --- Main App ---

# def main():
#     st.set_page_config(
#     page_title="Cardamom Auction Price Analysis",
#     layout="wide",              # ← edge-to-edge
#     initial_sidebar_state="collapsed",
# )
#     data = load_data()

#     st.sidebar.title("Choose a chart")
#     choice = st.sidebar.radio(
#         "",
#         [
#             "Avg Price Over Time",
#             "Auctioneer vs Price",
#             "Monthly Trend per Year",
#             "Monthly Price Distribution",
#             "Auctioneer-wise Qty Sold",
#         ]
#     )

#     if choice == "Avg Price Over Time":
#         plot_avg_price_over_time(data)
#     elif choice == "Auctioneer vs Price":
#         plot_auctioneer_vs_price(data)
#     elif choice == "Monthly Trend per Year":
#         plot_monthly_trend_per_year(data)
#     elif choice == "Monthly Price Distribution":
#         plot_monthly_price_distribution(data)
#     elif choice == "Auctioneer-wise Qty Sold":
#         plot_auctioneer_qty_sold(data)

# if __name__ == "__main__":
#     main()





import pandas as pd
import streamlit as st
import plotly.express as px

# --- Page config ---
st.set_page_config(
    page_title="Cardamom Auction Price Analysis",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Inject CSS to push content below the Streamlit header and hide the default header/menu
st.markdown(
    '''
    <style>
    /* Add top padding so the appbar doesn’t overlap */
    .reportview-container .main .block-container {
        padding-top: 70px;
    }
    /* Hide Streamlit's header and menu for a cleaner view */
    header {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    </style>
    '''
    , unsafe_allow_html=True
)

MONTH_ORDER = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# --- Data loading ---
# def load_data(path="cardamom_price_dataset.csv"):
#     df = pd.read_csv(path)
#     df["Date of Auction"] = pd.to_datetime(df["Date of Auction"])
#     df["Avg.Price (Rs./Kg)"] = pd.to_numeric(df["Avg.Price (Rs./Kg)"], errors="coerce")
#     df["Year"] = df["Date of Auction"].dt.year
#     df["Month"] = df["Date of Auction"].dt.month_name().str.slice(stop=3)
#     return df.sort_values("Date of Auction")

# --- Data loading ---
def load_data(path="cardamom_price_dataset.csv"):
    df = pd.read_csv(path)
    df["Date of Auction"] = pd.to_datetime(df["Date of Auction"])
    df["Avg.Price (Rs./Kg)"] = pd.to_numeric(df["Avg.Price (Rs./Kg)"], errors="coerce")
    df["Year"] = df["Date of Auction"].dt.year

    # Abbreviate month names to three letters
    df["Month"] = df["Date of Auction"].dt.month_name().str.slice(stop=3)

    # Ensure Month column is Categorical with correct ordering
    month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    df["Month"] = pd.Categorical(df["Month"], categories=month_order, ordered=True)

    # Sort by year then month
    return df.sort_values(["Year", "Month"])

# common plotly config for interactivity
def get_plotly_config():
    return {
        'scrollZoom': True,
        'displayModeBar': 'hover',
        'modeBarButtonsToAdd': [
            'zoomIn2d', 'zoomOut2d', 'zoom2d',
            'pan2d', 'select2d', 'lasso2d'
        ],
    }

# --- Main App ---
def main():
    st.title("Cardamom Auction Price Analysis")
    # st.markdown("<br>", unsafe_allow_html=True)  #its already declared at top
    data = load_data()

    st.sidebar.title("Choose a chart")
    choice = st.sidebar.radio(
        "", [
            "Avg Price Over Time",
            "Auctioneer vs Price",
            "Monthly Trend per Year",
            "Monthly Price Distribution",
            "Auctioneer-wise Qty Sold",
        ]
    )

    config = get_plotly_config()

    # 1️⃣ Avg Price Over Time
    if choice == "Avg Price Over Time":
        fig = px.line(
            data,
            x="Date of Auction",
            y="Avg.Price (Rs./Kg)",
            title="Avg. Price of Cardamom Over Time",
            labels={"Avg.Price (Rs./Kg)": "Price (Rs./Kg)", "Date of Auction": ""},
            template="plotly_white",
        )
        fig.update_layout(
            xaxis=dict(tickangle=-45),
            dragmode='zoom',
            clickmode='event+select',
            legend=dict(itemclick='toggle', itemdoubleclick='toggleothers'),
            title_font_color="#333333",
            font_color="#333333",
        )
        st.plotly_chart(fig, use_container_width=True, config=config)

    # 2️⃣ Auctioneer vs Price
    elif choice == "Auctioneer vs Price":
        agg = (
            data.groupby("Auctioneer")["Avg.Price (Rs./Kg)"].mean()
            .sort_values(ascending=False)
            .reset_index()
        )
        # Shorten long auctioneer names
        agg["Auctioneer_short"] = agg["Auctioneer"].apply(
            lambda x: x if len(x) <= 10 else x[: 10] + '...'
        )

        fig = px.bar(
            agg,
            x="Avg.Price (Rs./Kg)",
            y="Auctioneer_short",
            orientation='h',
            text_auto='.0f',            # show values on bars
            title="Auctioneer\n vs Avg. Price",
            labels={"Avg.Price (Rs./Kg)": "Avg. Price (Rs./Kg)", "Auctioneer_short": "Auctioneer"},
            template="plotly_white",
            height=600,
        )

        # Extend x-axis to fill width and push values outside
        max_val = agg["Avg.Price (Rs./Kg)"].max() * 1.1
        fig.update_xaxes(range=[0, max_val], automargin=True)
        fig.update_traces(textposition='outside', textfont=dict(size=10), cliponaxis=False)

        fig.update_layout(
            yaxis=dict(
                autorange="reversed",
                automargin=True,
                tickfont=dict(size=10)
            ),
            xaxis=dict(
                tickformat=',.0f',
                automargin=True,
                title_standoff=15
            ),
            margin=dict(l=20, r=5, t=50, b=20),
            dragmode='zoom', clickmode='event+select',
            title_font_color="#333333",
            font_color="#333333"
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config=config,
            key="auctioneer_vs_price"
        )

    # 3️⃣ Monthly Trend per Year
    elif choice == "Monthly Trend per Year":
        # Pre-aggregate to get one mean per Year-Month
        monthly_mean = (
            data.groupby(["Year", "Month"], as_index=False)["Avg.Price (Rs./Kg)"].mean()
        )
        # Replace categorical Month with numeric month index for plotting
        monthly_mean["Month_Num"] = monthly_mean["Month"].apply(lambda m: MONTH_ORDER.index(m) + 1)

        fig = px.line(
            monthly_mean,
            x="Month_Num",
            y="Avg.Price (Rs./Kg)",
            color="Year",
            markers=True,
            title="Monthly Avg. Price Trend by Year",
            labels={"Avg.Price (Rs./Kg)": "Price (Rs./Kg)", "Month_Num": ""},
            template="plotly_white",
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        fig.update_traces(mode='lines+markers', hovertemplate='%{y:.0f}')
        fig.update_layout(
            xaxis=dict(
                tickmode='array',
                tickvals=list(range(1,13)),
                ticktext=MONTH_ORDER,
                gridcolor='LightGray',
                showgrid=True
            ),
            yaxis=dict(gridcolor='LightGray', showgrid=True),
            dragmode='zoom', clickmode='event+select',
            legend=dict(title='Year', orientation='v', y=0.5, x=1.02,
                        itemclick='toggle', itemdoubleclick='toggleothers'),
            title_font_color="#333333", font_color="#333333"
        )
        st.plotly_chart(fig, use_container_width=True, config=config)

    # 4️⃣ Monthly Price Distribution
    elif choice == "Monthly Price Distribution":
        # Box plot with one box per month (no vertical scatter)
        fig = px.box(
            data,
            x="Month",
            y="Avg.Price (Rs./Kg)",
            title="Monthly Price Distribution",
            labels={"Avg.Price (Rs./Kg)": "Price (Rs./Kg)", "Month": ""},
            template="plotly_white",
            points=False,
            category_orders={"Month": MONTH_ORDER},
        )
        # Style: hide x-grid, show y-grid, single color, no legend
        fig.update_traces(marker_color='#4682B4')
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=True, gridcolor='LightGray', zeroline=False),
            dragmode='zoom',
            clickmode='event+select',
            showlegend=False,
            title_font_color="#333333",
            font_color="#333333",
        )
        st.plotly_chart(fig, use_container_width=True, config=config,  key="monthly_price_distribution")

    # 5️⃣ Auctioneer-wise Qty Sold
    elif choice == "Auctioneer-wise Qty Sold":
        tmp = data.copy()
        # Clean and convert quantity column
        tmp["Qty Sold (Kgs)"] = (
            tmp["Qty Sold (Kgs)"].astype(str)
            .str.replace(",", "")
        )
        tmp["Qty Sold (Kgs)"] = pd.to_numeric(tmp["Qty Sold (Kgs)"], errors="coerce").fillna(0)

        # Aggregate to total per auctioneer
        qty = (
            tmp.groupby("Auctioneer")["Qty Sold (Kgs)"].sum()
            .reset_index()
        )
        qty["Qty Sold (t)"] = qty["Qty Sold (Kgs)"] / 1000

        # Shorten long auctioneer names
        qty["Auctioneer_short"] = qty["Auctioneer"].apply(
            lambda x: x if len(x) <= 10 else x[:10] + '...'
        )

        # Draw horizontal bar with values outside
        fig = px.bar(
            qty,
            x="Qty Sold (t)",
            y="Auctioneer_short",
            orientation='h',
            text_auto='.1f',
            title="\nTotal Quantity Sold by Auctioneer (tonnes)",
            labels={"Qty Sold (t)": "Qty Sold (t)", "Auctioneer_short": "Auctioneer"},
            template="plotly_white",
            height=600,
        )
        # Stretch x-axis and push text out
        max_val = qty["Qty Sold (t)"].max() * 1.1
        fig.update_xaxes(range=[0, max_val], automargin=True)
        fig.update_traces(textposition='outside', textfont=dict(size=10), cliponaxis=False)

        # Layout adjustments
        fig.update_layout(
            yaxis=dict(
                autorange="reversed",
                automargin=True,
                tickfont=dict(size=10)
            ),
            xaxis=dict(
                tickformat=',.1f',
                automargin=True,
                title_standoff=15
            ),
            margin=dict(l=20, r=20, t=50, b=20),
            dragmode='zoom', clickmode='event+select',
            title_font_color="#333333",
            font_color="#333333"
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config=config,
            key="auctioneer_qty_sold"
        )


if __name__ == "__main__":
    main()
