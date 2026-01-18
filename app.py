import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

#  Dark theme style
st.markdown("""
    <style>
        .main {
            background-color: #1E1E1E;
            color: white;
        }
        .stButton>button {
            background-color: #009688;
            color: white;
            border-radius: 8px;
            padding: 8px 20px;
        }
        .stTextInput input, .stNumberInput input {
            background-color: #333333;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🌱 Dry Bean Classification & Analysis")
st.write("Upload dataset → Explore data → Train model → Classify bean variety")
st.divider()

# Upload Dataset
file = st.file_uploader("📂 Upload Dry Bean Dataset (.csv or .xlsx)", type=["csv", "xlsx"])

if file:
    df = pd.read_csv(file) if file.name.endswith('csv') else pd.read_excel(file)

    st.subheader("📌 Data Preview")
    st.dataframe(df.head())

    num_cols = df.select_dtypes(include=np.number).columns

    # EDA plots
    st.subheader("📊 Feature Distribution")
    selected_feature = st.selectbox("Select numeric feature:", num_cols)
    
    fig, ax = plt.subplots()
    ax.set_facecolor("#ffffff")
    fig.patch.set_facecolor("#ffffff")
    sns.histplot(df[selected_feature], ax=ax, bins=20)
    st.pyplot(fig)

    # Correlation Matrix
    st.subheader("🔗 Correlation Matrix")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.set_facecolor("#fffcfc")
    
    sns.heatmap(df[num_cols].corr(),
                annot=False, cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)

    st.divider()
    st.subheader("🤖 Model Training & Evaluation")

    if 'Class' in df.columns:
        X = df.drop('Class', axis=1)
        y = df['Class']

        with st.spinner("🔄 Encoding & Scaling Data..."):
            le = LabelEncoder()
            y_enc = le.fit_transform(y)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_enc, test_size=0.2, stratify=y_enc, random_state=42
        )

        # Training with spinner
        with st.spinner("🚀 Training Model... Please wait!"):
            clf = RandomForestClassifier(n_estimators=200, random_state=42)
            clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        st.success(f"✅ Model Accuracy: **{acc:.4f}**")

        st.write("📌 Classification Report:")
        st.text(classification_report(y_test, y_pred, target_names=le.classes_))

        # Confusion Matrix
        st.subheader("🧮 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)

        fig3, ax3 = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d',
                    xticklabels=le.classes_,
                    yticklabels=le.classes_,
                    cmap="viridis", ax=ax3)
        st.pyplot(fig3)

        st.divider()
        st.subheader("🔍 Predict Single Bean Sample")
        st.write("Enter numeric values for each feature:")

        input_values = []
        for col in X.columns:
            input_values.append(st.number_input(f"{col}", step=0.01, value=float(df[col].mean())))

        if st.button("🔮 Predict Bean Type"):
            with st.spinner("🧠 Predicting bean class..."):
                input_np = scaler.transform([input_values])
                pred = clf.predict(input_np)
                predicted_label = le.inverse_transform(pred)
                st.success(f"🥳 Predicted Bean Type: **{predicted_label[0]}**")

    else:
        st.error("❌ 'Class' Column Missing! Cannot train classifier.")
else:
    st.info("⏳ Upload dataset to begin...")
