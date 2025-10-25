import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import matplotlib.pyplot as plt

# =============================
# 1️⃣ Load the model
# =============================
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
sentiment_model = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, return_all_scores=True)

# =============================
# 2️⃣ App Configuration
# =============================
st.set_page_config(page_title="Sentiment Analysis", page_icon="🌸", layout="centered")

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Poppins', sans-serif;
        }

        body {
            background: linear-gradient(135deg, #f0f4ff 0%, #f9f9ff 100%);
        }

        .main-title {
            text-align: center;
            font-size: 38px;
            color: #1e293b;
            font-weight: 700;
            margin-bottom: 0.4em;
            letter-spacing: 0.5px;
        }

        .subtitle {
            text-align: center;
            font-size: 20px;
            color: #475569;
            margin-bottom: 1.8em;
        }

        .result-box {
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .result-box:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 22px rgba(0,0,0,0.12);
        }

        .stars {
            font-size: 30px;
            color: #FFD700;
            letter-spacing: 3px;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        }

        .stButton>button {
            background-color: #6366f1 !important;
            color: white !important;
            font-weight: 600 !important;
            border-radius: 12px !important;
            padding: 0.7rem 1.2rem !important;
            box-shadow: 0 4px 12px rgba(99,102,241,0.3) !important;
            transition: 0.2s ease-in-out;
        }

        .stButton>button:hover {
            background-color: #4f46e5 !important;
            transform: scale(1.02);
        }
    </style>
    """,
    unsafe_allow_html=True
)

# =============================
# 3️⃣ UI
# =============================
st.markdown("<h1 class='main-title'>🌸 Sentiment Analysis Application</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Enter a movie review, product feedback, or comment for the system to analyze. 🎬</p>", unsafe_allow_html=True)

text = st.text_area("✍️ Type a sentence or paragraph:", "")

# =============================
# 4️⃣ Analyze Sentiment
# =============================
if st.button("🔍 Analyze Sentiment", key="analyze", help="Click to analyze text"):
    if text.strip():
        # Get model results
        results = sentiment_model(text)[0]
        top = max(results, key=lambda x: x["score"])
        label = top["label"].lower()
        score = top["score"]

        # Map to sentiment + colors
        if "negative" in label:
            sentiment = "Negative 😞"
            bg = "#fdecea"
            color = "#611a15"
            stars = 1 + int(score * 2)
        elif "neutral" in label:
            sentiment = "Neutral 😐"
            bg = "#fff8e1"
            color = "#856404"
            stars = 3
        else:
            sentiment = "Positive 😊"
            bg = "#e6f4ea"
            color = "#0f5132"
            stars = 4 if score < 0.95 else 5  # 95%+ = 5 stars

        # Confidence label
        if score > 0.9:
            confidence_label = "Very High 🔥"
        elif score > 0.75:

            confidence_label = "High 👍"
        elif score > 0.5:
            confidence_label = "Medium 🤔"
        else:
            confidence_label = "Low ⚠️"

        # Generate star string
        star_display = "⭐" * stars + "☆" * (5 - stars)

        # 🧾 Display result
        st.markdown(
            f"""
            <div class='result-box' 
                 style='background-color:{bg};
                        color:{color};
                        border-left: 6px solid {color};
                        padding: 25px;
                        border-radius: 16px;
                        margin-top: 25px;'>
                <h3 style='margin-bottom:10px;'>📊 Analysis Result</h3>
                <p style='font-size:18px; line-height:1.6;'>
                    <b>Sentiment:</b> <span style='font-weight:600;'>{sentiment}</span><br>
                    <b>Confidence:</b> {score:.2%} <i>({confidence_label})</i><br>
                    <b>Rating:</b> <span class='stars'>{star_display}</span> <b>({stars}/5)</b>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # =============================
        # 5️⃣ Confidence Chart
        # =============================
        labels = [r["label"].capitalize() for r in results]
        scores = [r["score"] for r in results]

        fig, ax = plt.subplots()
        bars = ax.barh(labels, scores, color=["#dc3545", "#ffc107", "#28a745"])
        ax.set_xlabel("Confidence Level")
        ax.set_title("Sentiment Confidence Distribution")
        ax.set_xlim(0, 1)

        # Add score labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                    f"{width:.2%}", va='center', fontsize=10, color='#333')

        st.pyplot(fig)