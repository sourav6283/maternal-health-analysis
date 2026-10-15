import pandas as pd
import plotly.express as px
import streamlit as st

# label shown in the app -> (text that identifies the CSV column, higher_is_better)
# The CSV column names are very long and carry footnote digits, so columns
# are found by matching a unique piece of the name.
INDICATORS = {
    "Institutional births (%)": ("Institutional births (in the 5 years", True),
    "Births attended by skilled health personnel (%)": ("Births attended by skilled health personnel", True),
    "Mothers with 4+ antenatal visits (%)": ("at least 4 antenatal care visits", True),
    "Antenatal check-up in first trimester (%)": ("antenatal check-up in the first trimester", True),
    "Iron-folic acid for 180+ days (%)": ("iron folic acid for 180 days", True),
    "Postnatal care within 2 days (%)": ("Mothers who received postnatal care from a doctor", True),
    "Caesarean births (%)": ("Births delivered by caesarean section (in the 5 years", False),
    "Pregnant women who are anaemic (%)": ("Pregnant women age 15-49 years who are anaemic", False),
    "Girls 15-19 already mothers or pregnant (%)": ("already mothers or pregnant", False),
    "Neonatal mortality rate (per 1000 births)": ("Neonatal mortality rate", False),
    "Infant mortality rate (per 1000 births)": ("Infant mortality rate", False),
    "Total fertility rate (children per woman)": ("Total Fertility Rate", False),
    "Women with high blood sugar (%)": ("Women age 15 years and above wih high or very high", False),
    "Women with elevated blood pressure (%)": ("Women age 15 years and above wih Elevated blood pressure", False),
}


class MaternalHealthDashboard:
    def __init__(self, csv_path):
        self.df = self.load_data(csv_path)

    # ---------- data ----------
    @staticmethod
    def load_data(csv_path):
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            st.error(f"Data file not found: {csv_path}")
            return None
        except Exception as e:
            st.error(f"Could not read the data file: {e}")
            return None

        # drop empty "Unnamed" columns and tidy column names
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
        df.columns = [" ".join(str(c).split()) for c in df.columns]
        df = df.rename(columns={"States/UTs": "State"})

        df["State"] = df["State"].str.strip().replace({"Maharastra": "Maharashtra"})
        df["Area"] = df["Area"].str.strip()

        # Values like "(12.3)" mean few cases in the survey; "*" or blank = not available.
        # Turn everything into plain numbers, with NaN where there is no value.
        for c in df.columns:
            if c in ("State", "Area"):
                continue
            cleaned = df[c].astype(str).str.replace(r"[()]", "", regex=True).str.strip()
            df[c] = pd.to_numeric(cleaned, errors="coerce")
        return df

    def _col(self, label):
        key = INDICATORS[label][0]
        return next(c for c in self.df.columns if key in c)

    def _states(self, area):
        d = self.df[(self.df["Area"] == area) & (self.df["State"] != "India")]
        return d

    def _india(self, area, label):
        row = self.df[(self.df["State"] == "India") & (self.df["Area"] == area)]
        if row.empty:
            return None
        v = row[self._col(label)].iloc[0]
        return None if pd.isna(v) else float(v)

    # ---------- widgets / charts ----------
    def show_national_summary(self):
        st.subheader("India at a glance")
        labels = [
            "Institutional births (%)",
            "Mothers with 4+ antenatal visits (%)",
            "Pregnant women who are anaemic (%)",
            "Infant mortality rate (per 1000 births)",
        ]
        cols = st.columns(len(labels))
        for col, label in zip(cols, labels):
            v = self._india("Total", label)
            col.metric(label, "n/a" if v is None else f"{v:g}")

    def create_bubble_chart(self):
        st.subheader("Antenatal care vs institutional births, by state/UT")
        x = self._col("Mothers with 4+ antenatal visits (%)")
        y = self._col("Institutional births (%)")
        size = self._col("Total fertility rate (children per woman)")
        d = self._states("Total").dropna(subset=[x, y, size])
        fig = px.scatter(
            d,
            x=x,
            y=y,
            size=size,
            color="State",
            hover_name="State",
            size_max=35,
            labels={x: "Mothers with 4+ antenatal visits (%)", y: "Institutional births (%)", size: "Fertility rate"},
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    def select_indicator(self):
        return st.selectbox("Choose an indicator", list(INDICATORS.keys()), key="dash_indicator")

    def create_ranking_chart(self, label):
        st.subheader("State/UT ranking")
        area = st.radio("Area", ["Total", "Urban", "Rural"], horizontal=True, key="dash_area")
        col = self._col(label)
        d = self._states(area).dropna(subset=[col]).sort_values(col)
        if d.empty:
            st.info("No data available for this selection.")
            return
        higher_is_better = INDICATORS[label][1]
        fig = px.bar(
            d,
            x=col,
            y="State",
            orientation="h",
            color=col,
            color_continuous_scale="Blues" if higher_is_better else "Reds",
            labels={col: label, "State": ""},
            height=max(450, 24 * len(d)),
        )
        india = self._india(area, label)
        if india is not None:
            fig.add_vline(x=india, line_dash="dash", annotation_text=f"India {india:g}")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        missing = self._states(area)[col].isna().sum()
        if missing:
            st.caption(f"{missing} state(s)/UT(s) have no value for this indicator and are not shown.")

    def create_urban_rural_chart(self, label):
        st.subheader("Urban vs rural")
        col = self._col(label)
        d = self.df[self.df["Area"].isin(["Urban", "Rural"]) & (self.df["State"] != "India")]
        d = d.dropna(subset=[col])
        if d.empty:
            st.info("No data available for this selection.")
            return
        order = d.groupby("State")[col].mean().sort_values(ascending=False).index.tolist()
        fig = px.bar(
            d,
            x="State",
            y=col,
            color="Area",
            barmode="group",
            category_orders={"State": order},
            labels={col: label, "State": ""},
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ---------- explanation texts ----------
    def get_bubble_chart_data(self):
        return """
Each bubble is one state or union territory.

- X-axis: share of mothers who had at least 4 antenatal care visits during their last pregnancy.
- Y-axis: share of births that took place in a health facility.
- Bubble size: total fertility rate (average number of children per woman).
- Hover over a bubble to see the state name and exact values.

States in the top-right corner have both good antenatal care coverage and high institutional births.
"""

    def get_ranking_data(self):
        return """
The bar chart ranks states and union territories for the indicator chosen above.
Blue bars mean a higher value is better (for example institutional births); red bars mean a lower value is better
(for example anaemia or infant mortality). The dashed line marks the all-India value.
Use the Area buttons to switch between total, urban and rural figures.
"""

    def get_urban_rural_data(self):
        return """
Compares urban and rural values of the chosen indicator for every state and union territory,
sorted by the average of the two. A large gap shows where rural women have less access to care than urban women.
"""