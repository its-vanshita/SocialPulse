import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import altair as alt
import os
import google.generativeai as genai

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from langdetect import detect
from googletrans import Translator
from functools import lru_cache
import re
import numpy as np

# -------------------------
# Configurable Category Profiles
# -------------------------
# Each profile defines: aspects, aspect_display, and optional intent/emotion rules
CATEGORY_PROFILES = {
    'Electronics': {
        'aspects': [
            'delivery', 'service', 'price', 'quality', 'packaging', 'usability',
            'performance', 'warranty', 'return_policy', 'durability', 'availability',
            'support', 'features', 'design', 'charging'
        ],
        'aspect_display': {
            'delivery': 'Delivery',
            'service': 'Customer Service',
            'price': 'Price',
            'quality': 'Product Quality',
            'packaging': 'Packaging',
            'usability': 'Ease of Use',
            'performance': 'Performance',
            'warranty': 'Warranty',
            'return_policy': 'Return/Refund Policy',
            'durability': 'Durability',
            'availability': 'Availability',
            'support': 'Technical Support',
            'features': 'Features',
            'design': 'Design & Look',
            'charging': 'Charging/Battery'
        }
    },
    'Fashion': {
        'aspects': [
            'delivery', 'service', 'price', 'quality', 'packaging', 'availability',
            'return_policy', 'design', 'comfort', 'fit', 'material', 'color',
            'stitching', 'size', 'washing', 'durability'
        ],
        'aspect_display': {
            'delivery': 'Delivery',
            'service': 'Customer Service',
            'price': 'Price',
            'quality': 'Fabric Quality',
            'packaging': 'Packaging',
            'availability': 'Availability/Stock',
            'return_policy': 'Return/Exchange Policy',
            'design': 'Design/Style',
            'comfort': 'Comfort',
            'fit': 'Fit',
            'material': 'Material',
            'color': 'Color',
            'stitching': 'Stitching',
            'size': 'Size Accuracy',
            'washing': 'Washing/Care',
            'durability': 'Durability'
        }
    }
}

# Generic, lightweight rules for intents and emotions (can be overridden per profile)
INTENT_RULES_DEFAULT = {
    'praise': [r"\blove\b", r"\bamazing\b", r"\bgreat\b", r"\bexcellent\b", r"\bhappy\b", r"\bsatisfied\b"],
    'complaint': [r"\bnot working\b", r"\bworst\b", r"\bbroken\b", r"\blate\b", r"\bdelay(ed)?\b", r"\brefund\b", r"\breturn\b", r"\bdisappoint(ed)?\b"],
    'inquiry': [r"\bhow\b", r"\bwhen\b", r"\bwhere\b", r"\bwhat\b", r"\bdoes it\b", r"\bcan i\b", r"\bis it\b"],
    'feature_request': [r"\bplease add\b", r"\bwish it had\b", r"\bfeature request\b"],
    'purchase_intent': [r"\bwill buy\b", r"\border(ing)?\b", r"\badd(ed)? to cart\b", r"\bthinking to buy\b"],
    'return_refund': [r"\brefund\b", r"\breturn\b", r"\breplacement\b"],
    'support_needed': [r"\bhelp\b", r"\bsupport\b", r"\bassist\b", r"\bcontact\b"],
    'comparison': [r"\bbetter than\b", r"\bworse than\b", r"\bvs\b", r"\bcompared to\b"]
}

EMOTION_RULES_DEFAULT = {
    'joy': [r"\blove\b", r"\bhappy\b", r"\bdelight(ed)?\b", r"\bpleased\b"],
    'anger': [r"\bangry\b", r"\bfurious\b", r"\bwaste\b", r"\bterrible\b"],
    'sadness': [r"\bsad\b", r"\bdisappoint(ed)?\b", r"\bupset\b"],
    'surprise': [r"\bsurpris(ed|ing)\b", r"\bdidn'?t expect\b"]
}

# Optional: profile-specific overrides can be added under CATEGORY_PROFILES[profile]['intent_rules'/'emotion_rules']

# -------------------------
# Rule Helpers
# -------------------------
def _compile_rules(rules_dict):
    compiled = {}
    for tag, patterns in rules_dict.items():
        compiled[tag] = [re.compile(p, flags=re.IGNORECASE) for p in patterns]
    return compiled

def detect_tags_from_text(text, compiled_rules):
    text = str(text)
    detected = []
    for tag, patterns in compiled_rules.items():
        if any(p.search(text) for p in patterns):
            detected.append(tag)
    return detected

# Initialize analyzers
analyzer = SentimentIntensityAnalyzer()
translator = Translator()
aspects = CATEGORY_PROFILES['Electronics']['aspects']
aspect_display = CATEGORY_PROFILES['Electronics']['aspect_display']


# -------------------------
# Helper Functions
# -------------------------
@lru_cache(maxsize=10000)
def cached_translate_to_english(text):
    try:
        lang = detect(text)
        if lang == 'en':
            return text
        else:
            return translator.translate(text, src=lang, dest='en').text
    except Exception:
        return text

def get_vader_sentiment_score(text):
    return analyzer.polarity_scores(str(text))['compound']

def analyze_vader_sentiment(score):
    if score >= 0.5:
        return 'Positive'
    elif score <= -0.2:
        return 'Negative'
    else:
        return 'Neutral'

def final_sentiment(row):
    rating = row['Rating']
    text_sent = row['text_sentiment']
    if rating >= 4:
        return 'Positive'
    elif rating == 3:
        return text_sent
    elif rating <= 2:
        return text_sent
    else:
        return 'Neutral'

def simple_clean(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"[^a-z\s]", "", text)  # remove punctuation and numbers
    text = re.sub(r"\s+", " ", text)  # remove extra spaces
    return text.strip()

@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

# -------------------------
# Gemini Helpers
# -------------------------
def _get_gemini_api_key():
    try:
        # Prefer Streamlit secrets if available
        key = st.secrets.get("GEMINI_API_KEY", None)  # type: ignore[attr-defined]
    except Exception:
        key = None
    return key or os.environ.get("GEMINI_API_KEY")

def generate_final_report_with_gemini(summary_df, reviews_list, category_name: str):
    api_key = _get_gemini_api_key()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found. Set it in Streamlit secrets or environment.")
    genai.configure(api_key=api_key)
    # Limit reviews to a reasonable number to keep prompt size manageable
    reviews_list = [str(r) for r in reviews_list if isinstance(r, str) or r is not None]
    if len(reviews_list) > 200:
        reviews_list = reviews_list[:200]
    reviews_blob = "\n".join(f"- {r}" for r in reviews_list)

    aspect_table_csv = summary_df.to_csv(index=False)

    prompt = (
        "You are a senior product insights analyst creating an executive-ready report for a company client.\n"
        f"Category context: {category_name}\n\n"
        "Use the aspect-based sentiment table and representative customer reviews below to synthesize a crisp, actionable report.\n"
        "Focus on business relevance. Avoid repeating raw data. Be concise.\n\n"
        "Deliverables:\n"
        "1) Strengths (bulleted)\n"
        "2) Weaknesses (bulleted)\n"
        "3) Opportunities (market/product opportunities; bulleted)\n"
        "4) Risks (bulleted)\n"
        "5) Recommended Actions (prioritized, with rationale)\n"
        "6) Top 5 Short Customer Quotes (verbatim, diverse angles)\n"
        "7) Closing Summary (3-4 lines)\n\n"
        "Aspect Sentiment Table (CSV):\n"
        f"{aspect_table_csv}\n\n"
        "Representative Customer Reviews:\n"
        f"{reviews_blob}\n\n"
        "Constraints: Keep it structured with clear headings, no code fences, no markdown tables."
    )

    # Determine an available model at runtime (prefer latest 1.5 variants)
    preferred_models = [
        "gemini-1.5-pro-latest",
        "gemini-1.5-flash-latest",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gemini-pro",  # legacy text-only fallback
    ]
    available_models = []
    try:
        for m in genai.list_models():
            methods = getattr(m, "supported_generation_methods", []) or []
            if "generateContent" in methods:
                available_models.append(m.name)
    except Exception:
        # If listing fails, we'll rely on preferred set
        available_models = []

    candidate_models = [m for m in preferred_models if m in available_models] or available_models or preferred_models
    last_error = None
    for model_name in candidate_models:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return (getattr(response, "text", "") or "").strip()
        except Exception as e:
            last_error = e
            continue
    raise RuntimeError(f"All Gemini models failed. Last error: {last_error}")

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Sentiment Analyzer", layout="wide")
st.title("🧠 Product Review Sentiment Analyzer")

# -------------------------
# Category/Profile Selection
# -------------------------
st.sidebar.markdown("### Category Profile")
selected_profile_name = st.sidebar.selectbox(
    "Select a category profile",
    list(CATEGORY_PROFILES.keys()),
    index=0
)

# Apply selected profile
profile = CATEGORY_PROFILES[selected_profile_name]
aspects = profile['aspects']
aspect_display = profile['aspect_display']

# Optional: quick custom aspect override per client (comma-separated)
custom_aspects_csv = st.sidebar.text_input("Custom aspects (comma-separated)", value="")
if custom_aspects_csv.strip():
    custom_aspect_list = [a.strip().lower() for a in custom_aspects_csv.split(',') if a.strip()]
    if custom_aspect_list:
        aspects = custom_aspect_list
        # Try to preserve display names from profile when possible
        aspect_display = {a: profile['aspect_display'].get(a, a.title()) for a in aspects}

# -------------------------
# 1. Single Text Analysis
# -------------------------
with st.expander('🔍 Single Text Analysis'):
    text = st.text_input('Enter a review:')
    if text:
        lang = detect(text)
        if lang != 'en':
            english_text = cached_translate_to_english(text)
            st.write('📝 English Translation:', english_text)
        else:
            english_text = text
        score = get_vader_sentiment_score(english_text)
        sentiment = analyze_vader_sentiment(score)
        st.write('📊 Sentiment:', sentiment)
        st.write('📊 VADER Score:', round(score, 2))
        # Compile rules based on selected profile
        intent_rules = _compile_rules(profile.get('intent_rules', INTENT_RULES_DEFAULT))
        emotion_rules = _compile_rules(profile.get('emotion_rules', EMOTION_RULES_DEFAULT))

        aspect_mentions = {aspect_display.get(aspect, aspect.title()): (aspect in english_text.lower()) for aspect in aspects}
        intents = detect_tags_from_text(english_text, intent_rules)
        emotions = detect_tags_from_text(english_text, emotion_rules)
        st.write('🔎 Aspect Mentions:', aspect_mentions)
        st.write('🎯 Intents:', intents or ['none'])
        st.write('💬 Emotions:', emotions or ['none'])

    pre = st.text_input('Clean the text:')
    if pre:
        cleaned = simple_clean(pre)
        st.write('🧼 Cleaned Text:', cleaned)

# -------------------------
# 2. File Upload
# -------------------------
st.subheader("📂 Upload a Review CSV File")
uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])

# -------------------------
# 3. Main Analysis
# -------------------------
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df = df.drop(columns=['Unnamed: 0'], errors='ignore')

    if 'Review_Summary' not in df.columns or 'Rating' not in df.columns:
        st.error("❌ CSV must contain both 'Review_Summary' and 'Rating' columns.")
    else:
        df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
        df = df.dropna(subset=['Rating'])

        # User option for translation
        st.sidebar.markdown("### Translation Options")
        translate_reviews = st.sidebar.checkbox("Translate non-English reviews to English", value=True)

        # Translation with progress bar and caching
        if translate_reviews:
            st.info("Translating reviews to English (if needed)...")
            english_reviews = []
            progress_bar = st.progress(0)
            review_list = df['Review_Summary'].astype(str).tolist()
            for i, review in enumerate(review_list):
                english_reviews.append(cached_translate_to_english(review))
                if len(review_list) > 1:
                    progress_bar.progress((i+1)/len(review_list))
            df['Review_Summary_English'] = english_reviews
            progress_bar.empty()
        else:
            df['Review_Summary_English'] = df['Review_Summary'].astype(str)

        # VADER sentiment analysis
        df['score'] = df['Review_Summary_English'].apply(get_vader_sentiment_score)
        df['text_sentiment'] = df['score'].apply(analyze_vader_sentiment)
        df['final_sentiment'] = df.apply(final_sentiment, axis=1)
        df['mismatch'] = df['text_sentiment'] != df['final_sentiment']

        # Aspect extraction
        for aspect in aspects:
            col_name = aspect + '_mention'
            df[col_name] = df['Review_Summary_English'].str.lower().str.contains(aspect)

        # Intent & Emotion tagging
        intent_rules = _compile_rules(profile.get('intent_rules', INTENT_RULES_DEFAULT))
        emotion_rules = _compile_rules(profile.get('emotion_rules', EMOTION_RULES_DEFAULT))
        df['intents'] = df['Review_Summary_English'].apply(lambda t: detect_tags_from_text(t, intent_rules))
        df['emotions'] = df['Review_Summary_English'].apply(lambda t: detect_tags_from_text(t, emotion_rules))

        st.sidebar.title("📊 Filters")
        min_rating, max_rating = int(df['Rating'].min()), int(df['Rating'].max())
        selected_range = st.sidebar.slider("Select Rating Range", min_rating, max_rating, (min_rating, max_rating))
        df = df[(df['Rating'] >= selected_range[0]) & (df['Rating'] <= selected_range[1])]

        st.success(f"✅ Total reviews analyzed: {len(df)}")
        st.write(df.head())

              # -------------------------
        # 🔹 Sentiment Distribution
        # -------------------------
        st.subheader("📊 Sentiment Distribution")
        col1, col2 = st.columns(2)

        with col1:
                sentiment_counts = df['final_sentiment'].value_counts().reset_index()
                sentiment_counts.columns = ['Sentiment', 'Count']
                color_scale = alt.Scale(domain=['Positive', 'Neutral', 'Negative'],
                                        range=['#21ba45', '#a0a0a0', '#db2828'])  # green, gray, red

                bar_chart = alt.Chart(sentiment_counts).mark_bar().encode(
                    x=alt.X('Sentiment', sort=['Positive', 'Neutral', 'Negative']),
                    y='Count',
                    color=alt.Color('Sentiment', scale=color_scale),
                    tooltip=['Sentiment', 'Count']
                ).properties(
                    width='container',
                    height=350
                )
                st.altair_chart(bar_chart, use_container_width=True)


        with col2:
                sentiment_counts = df['final_sentiment'].value_counts()
                # Ensure the order matches: Positive, Neutral, Negative
                order = ['Positive', 'Neutral', 'Negative']
                sentiment_counts = sentiment_counts.reindex(order).fillna(0)
                colors = ['green', 'grey', 'red']  # green for positive, grey for neutral, red for negative
                fig, ax = plt.subplots()
                ax.pie(sentiment_counts, labels=sentiment_counts.index,
                    autopct="%1.1f%%", startangle=90, colors=colors)
                ax.axis('equal')
                st.pyplot(fig)
      
        # -------------------------
        # Intent & Emotion Summaries
        # -------------------------
        st.subheader("🎯 Intent & 💬 Emotion Summaries")
        col3, col4 = st.columns(2)
        with col3:
            # explode intents
            intents_exploded = df.explode('intents')
            intents_exploded['intents'] = intents_exploded['intents'].fillna('none')
            intent_counts = intents_exploded['intents'].value_counts().reset_index()
            intent_counts.columns = ['Intent', 'Count']
            chart_i = alt.Chart(intent_counts).mark_bar().encode(
                x=alt.X('Intent', sort='-y'),
                y='Count',
                tooltip=['Intent', 'Count']
            ).properties(width='container', height=300)
            st.altair_chart(chart_i, use_container_width=True)
        with col4:
            emotions_exploded = df.explode('emotions')
            emotions_exploded['emotions'] = emotions_exploded['emotions'].fillna('none')
            emotion_counts = emotions_exploded['emotions'].value_counts().reset_index()
            emotion_counts.columns = ['Emotion', 'Count']
            chart_e = alt.Chart(emotion_counts).mark_bar().encode(
                x=alt.X('Emotion', sort='-y'),
                y='Count',
                tooltip=['Emotion', 'Count']
            ).properties(width='container', height=300)
            st.altair_chart(chart_e, use_container_width=True)

        # -------------------------
        # Aspect Breakdown (Counts)
        # -------------------------
        aspect_breakdown = []
        for aspect in aspects:
            aspect_col = aspect + '_mention'
            pos = np.sum((df[aspect_col]) & (df['text_sentiment'] == 'Positive'))
            neg = np.sum((df[aspect_col]) & (df['text_sentiment'] == 'Negative'))
            neu = np.sum((df[aspect_col]) & (df['text_sentiment'] == 'Neutral'))
            total = pos + neg + neu
            aspect_breakdown.append({
                'Aspect': aspect_display.get(aspect, aspect.title()),
                'Positive Count': pos,
                'Negative Count': neg,
                'Neutral Count': neu,
                'Total Mentions': total
            })
        aspect_breakdown_df = pd.DataFrame(aspect_breakdown)
        st.subheader("📊 Aspect Breakdown (Counts)")
        st.dataframe(aspect_breakdown_df)

        # -------------------------
        # Aspect-Based Sentiment Summary Table (Percentages)
        # -------------------------
        aspect_sentiment_summary = []
        for aspect in aspects:
            aspect_col = aspect + '_mention'
            pos = np.sum((df[aspect_col]) & (df['text_sentiment'] == 'Positive'))
            neg = np.sum((df[aspect_col]) & (df['text_sentiment'] == 'Negative'))
            neu = np.sum((df[aspect_col]) & (df['text_sentiment'] == 'Neutral'))
            total = pos + neg + neu
            pos_pct = round(100 * pos / total, 1) if total else 0
            neg_pct = round(100 * neg / total, 1) if total else 0
            neu_pct = round(100 * neu / total, 1) if total else 0
            aspect_sentiment_summary.append({
                "Aspect": aspect_display.get(aspect, aspect.title()),
                "Positive (%)": pos_pct,
                "Negative (%)": neg_pct,
                "Neutral (%)": neu_pct
            })
        st.subheader("📋 Aspect-Based Sentiment Summary (%)")
        st.table(pd.DataFrame(aspect_sentiment_summary))

                # Find the aspect with the lowest positive and highest negative sentiment
        summary_df = pd.DataFrame(aspect_sentiment_summary)
        # Convert percentages to numbers for sorting
        summary_df['Positive (%)'] = pd.to_numeric(summary_df['Positive (%)'])
        summary_df['Negative (%)'] = pd.to_numeric(summary_df['Negative (%)'])

        # Identify the aspect needing most improvement
        weakest_aspect = summary_df.sort_values('Negative (%)', ascending=False).iloc[0]
        strongest_aspect = summary_df.sort_values('Positive (%)', ascending=False).iloc[0]

        # Display AI section with only final report generation
        st.markdown("## 🤖 AI Analysis & Recommendations")
        if st.button("Final report generation"):
            with st.spinner("Generating final report with Gemini..."):
                try:
                    reviews_for_llm = df['Review_Summary_English'].dropna().astype(str).tolist()
                    final_report = generate_final_report_with_gemini(summary_df, reviews_for_llm, selected_profile_name)
                    st.session_state['final_report_text'] = final_report
                except Exception as e:
                    st.error(f"Failed to generate report: {e}")
        if 'final_report_text' in st.session_state and st.session_state['final_report_text']:
            st.markdown(st.session_state['final_report_text'])
            st.download_button(
                "Download Final Report",
                st.session_state['final_report_text'],
                file_name="final_report.txt",
                mime="text/plain"
            )

       

      
        # -------------------------
        # Download Results
        # -------------------------
        st.subheader("⬇ Download Sentiment CSV")
        csv = convert_df(df)
        st.download_button("Download CSV", csv, "sentiment_results.csv", "text/csv")

        # -------------------------
        # Show Full Data
        # -------------------------
        with st.expander("🗂 Show All Reviews"):
            st.dataframe(df)
