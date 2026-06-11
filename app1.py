
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(layout='wide', page_title='Patient Health Records EDA')

# ── Title ──────────────────────────────────────────────────────────────────────
st.title('Patient Health Records Project')

html_title = """<h1 style="color:white;text-align:center;"> Patient Health Records Data Analysis </h1>"""
st.markdown(html_title, unsafe_allow_html=True)

# ── Load Data ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv('Patient_Health_Records_Cleaned_1.csv')

df = load_data()

# ── Sidebar Navigation ─────────────────────────────────────────────────────────
page = st.sidebar.radio('Page', ["Home", "Univariate Analysis", "Multivariate Dashboard", "Health Report"])

# ══════════════════════════════════════════════════════════════════════════════
if page == "Home":

    st.subheader('Dataset Overview')
    st.dataframe(df)

    column_descriptions = {
        "Age":               "عمر المريض.",
        "Gender":            "جنس المريض (MALE / FEMALE).",
        "City":              "المدينة اللي المريض ساكن فيها.",
        "BMI":               "مؤشر كتلة الجسم.",
        "Heart_Rate":        "معدل ضربات القلب.",
        "Cholesterol_Level": "مستوى الكوليسترول (normal / high / low).",
        "Diabetic":          "هل المريض مصاب بالسكري (Yes / No).",
        "Smoker":            "حالة التدخين (Never / Former / Current).",
        "Medications":       "الأدوية اللي بياخدها المريض.",
        "Follow_Up":         "موعد المتابعة القادم.",
        "Notes":             "ملاحظات الطبيب.",
        "Has_Disease":       "هل عنده مرض (0 = لا، 1 = نعم).",
        "Systolic":          "ضغط الدم الانقباضي.",
        "Diastolic":         "ضغط الدم الانبساطي.",
        "Last_Visit_Date":   "تاريخ آخر زيارة.",
    }

    desc_df = pd.DataFrame(list(column_descriptions.items()), columns=["Column Name", "Description"])
    st.subheader("📝 Column Descriptions")
    st.table(desc_df)

# ══════════════════════════════════════════════════════════════════════════════
elif page == "Univariate Analysis":

    html_title = """<h1 style="color:white;text-align:center;"> Univariate Analysis </h1>"""
    st.markdown(html_title, unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric('Total Patients',   f"{len(df):,}")
    col2.metric('Avg Age',          f"{df['Age'].mean():.1f}")
    col3.metric('Avg BMI',          f"{df['BMI'].mean():.1f}")
    col4.metric('Avg Heart Rate',   f"{df['Heart_Rate'].mean():.0f}")
    col5.metric('Disease Rate',     f"{pd.to_numeric(df['Has_Disease'], errors='coerce').mean()*100:.1f}%")

    st.markdown("---")

    # Numeric distributions
    st.subheader("📊 Numeric Feature Distribution")
    num_col = st.selectbox("Select numeric column", df.select_dtypes("number").columns.tolist())
    st.plotly_chart(px.histogram(df, x=num_col, marginal="box",
                                  title=f"Distribution of {num_col}",
                                  color_discrete_sequence=["#636EFA"]),
                    use_container_width=True)

    # Categorical distributions
    st.subheader("📊 Categorical Feature Distribution")
    cat_col = st.selectbox("Select categorical column", df.select_dtypes("object").columns.tolist())
    st.plotly_chart(px.histogram(df, x=cat_col, color=cat_col,
                                  title=f"Distribution of {cat_col}"),
                    use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
elif page == "Multivariate Dashboard":

    st.subheader("🔗 Correlation Heatmap")
    tmp = df.copy()
    tmp["Has_Disease"] = pd.to_numeric(tmp["Has_Disease"], errors="coerce")
    corr = tmp.corr(numeric_only=True)
    st.plotly_chart(px.imshow(corr, text_auto=True, aspect="auto",
                               color_continuous_scale="RdBu_r",
                               title="Correlation Heatmap"),
                    use_container_width=True)

    st.subheader("🏙️ Disease Count by City & Smoker")
    d2 = df.groupby(["City", "Smoker"])[["Has_Disease"]].count().reset_index()
    st.plotly_chart(px.bar(d2, x="City", y="Has_Disease", color="Smoker",
                            barmode="group", title="Disease Count by City & Smoker"),
                    use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🚬 Disease by Smoker")
        st.plotly_chart(px.histogram(df, x="Smoker", y="Has_Disease", color="Smoker"),
                        use_container_width=True)
    with col2:
        st.subheader("👥 Disease by Gender")
        st.plotly_chart(px.histogram(df, x="Gender", y="Has_Disease", color="Gender"),
                        use_container_width=True)

    st.subheader("📈 Heart Rate vs Age by Gender")
    st.plotly_chart(px.histogram(df, x="Age", y="Heart_Rate", color="Gender",
                                  barmode="group", title="Heart Rate vs Age"),
                    use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
elif page == "Health Report":

    st.header("📊 Health Report")

    city_list = sorted(df["City"].dropna().unique().tolist())
    city_list.append("All Cities")
    selected_city = st.sidebar.selectbox("Select City", city_list)

    if selected_city != "All Cities":
        df_filtered = df[df["City"] == selected_city]
    else:
        df_filtered = df

    st.write(f"🔎 Patients: {len(df_filtered):,}")

    if not df_filtered.empty:

        st.subheader("🔥 Disease Rate by Cholesterol Level")
        d4 = df_filtered.groupby("Cholesterol_Level")["Has_Disease"].count().reset_index()
        st.plotly_chart(px.bar(d4, x="Cholesterol_Level", y="Has_Disease",
                                color="Cholesterol_Level",
                                title="Disease Count by Cholesterol Level"),
                        use_container_width=True)

        st.subheader("💊 Disease Factors (Cholesterol × Diabetic × Smoker)")
        d6 = df_filtered.groupby(["Cholesterol_Level", "Diabetic", "Smoker"])["Has_Disease"].count().reset_index(name="count")
        st.plotly_chart(px.bar(d6, x="Cholesterol_Level", y="count",
                                color="Smoker", facet_col="Diabetic",
                                barmode="group", height=500,
                                title="Disease Distribution by Risk Factors"),
                        use_container_width=True)

        st.subheader("📅 Follow-Up by Month")
        tmp2 = df_filtered.copy()
        tmp2["Last_Visit_Date"] = pd.to_datetime(tmp2["Last_Visit_Date"], errors="coerce")
        d5 = tmp2.groupby(tmp2["Last_Visit_Date"].dt.month)["Follow_Up"].value_counts().reset_index(name="count")
        d5.columns = ["Month", "Follow_Up", "count"]
        st.plotly_chart(px.bar(d5, x="Month", y="count", color="Follow_Up",
                                title="Follow-Up Frequency by Month"),
                        use_container_width=True)
    else:
        st.warning("⚠️ لا توجد بيانات لهذا الاختيار.")
