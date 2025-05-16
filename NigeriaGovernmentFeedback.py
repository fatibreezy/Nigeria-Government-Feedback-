import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from textblob import TextBlob
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans

# -- FILE
FEEDBACK_FILE = "feedback.csv"
if not os.path.exists(FEEDBACK_FILE):
    df_init = pd.DataFrame(columns=["Post", "Region", "Response Type", "Feedback", "Name", "Sentiment"])
    df_init.to_csv(FEEDBACK_FILE, index=False)

# -- POSTS
government_posts = [
    "🧾 Tax Reform Bill 2025",
    "📢 3MTT Cohort 3 Registration Opens",
    "📝 JAMB Announces Rewriting of Exams for Some Candidates",
    "💼 Nestlé Nigeria Graduate Trainee Program 2025",
    "⚡ Power Supply Reform",
    "💰 New Minimum Wage Proposal",
    "🛡️ Security Enhancement Initiative",
    "📚 Education Budget Update",
    "🏥 Healthcare Improvement Plan",
    "🎓 Youth Empowerment Program",
]

regions = ["South West", "South East", "South South", "North West", "North East", "North Central", "FCT"]

# -- STREAMLIT SETUP
st.set_page_config(page_title="NaijaGov AI Feedback Portal", layout="wide")
st.title("🇳🇬 NaijaGov AI-Powered Feedback Platform")
st.write("Voice your opinion on government updates anonymously or with your name. Powered by AI to extract sentiment, trends and public mood.")

# -- FORM
post = st.selectbox("Choose an update", government_posts)
region = st.selectbox("Select region (optional)", [""] + regions)
response_type = st.radio("Feedback type", ["Suggestion", "Concern", "Note to Government"])
feedback = st.text_area("Write your feedback")
name = st.text_input("Your name (optional)")

def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    return "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"

def is_fake_feedback(text):
    return len(text.split()) < 3 or bool(re.search(r"(buy now|click here|visit)", text.lower()))

if st.button("Submit Feedback"):
    if feedback.strip() == "":
        st.warning("Feedback cannot be empty.")
    elif is_fake_feedback(feedback):
        st.error("🚫 This feedback appears spammy. Please revise.")
    else:
        sentiment = get_sentiment(feedback)
        df_new = pd.DataFrame([{
            "Post": post,
            "Region": region if region else "Unknown",
            "Response Type": response_type,
            "Feedback": feedback,
            "Name": name if name else "Anonymous",
            "Sentiment": sentiment
        }])
        df_new.to_csv(FEEDBACK_FILE, mode='a', header=False, index=False)
        st.success("✅ Feedback submitted successfully!")

# -- VISUALIZATION
st.subheader("📊 See Reactions Per Post")
if os.path.exists(FEEDBACK_FILE):
    df = pd.read_csv(FEEDBACK_FILE)
    selected_post = st.selectbox("Select post to view data", government_posts)
    filtered = df[df["Post"] == selected_post]
    
    if not filtered.empty:
        st.write(f"📄 {len(filtered)} responses recorded for this post.")
        st.dataframe(filtered)

        st.write("📈 Sentiment Analysis")
        st.bar_chart(filtered["Sentiment"].value_counts())

        st.write("🧩 Response Type Breakdown")
        st.bar_chart(filtered["Response Type"].value_counts())

        # Topic Modeling
        st.write("🔍 Topic Modeling (Beta)")
        try:
            vectorizer = TfidfVectorizer(stop_words="english")
            X = vectorizer.fit_transform(filtered["Feedback"])
            lda = LatentDirichletAllocation(n_components=2, random_state=42)
            lda.fit(X)
            terms = vectorizer.get_feature_names_out()
            for idx, topic in enumerate(lda.components_):
                topic_terms = [terms[i] for i in topic.argsort()[-5:]]
                st.write(f"**Topic {idx+1}:** {', '.join(topic_terms)}")
        except:
            st.warning("Not enough feedback for topic modeling.")

        # Clustering
        st.write("📌 Feedback Clustering")
        try:
            kmeans = KMeans(n_clusters=2, random_state=42)
            labels = kmeans.fit_predict(X)
            filtered["Cluster"] = labels
            st.dataframe(filtered[["Feedback", "Cluster"]])
        except:
            st.info("Need more data to cluster effectively.")
    else:
        st.info("No feedback yet for this post.")

# -- Smart Suggestions
st.subheader("🔁 Similar Posts You Might Be Interested In")
import random
suggestions = [x for x in government_posts if x != post]
st.write(random.sample(suggestions, 3))

# -- Chatbot Assistant
st.subheader("🤖 Ask the GovBot")
question = st.text_input("Ask a question about updates or feedback")
if question:
    if "tax" in question.lower():
        st.info("🧾 The Tax Reform Bill aims to simplify the system and improve compliance.")
    elif "3mtt" in question.lower():
        st.info("📢 3MTT Cohort 3 is open. Visit https://3mtt.nitda.gov.ng")
    elif "jamb" in question.lower():
        st.info("📝 JAMB is organizing a resit for affected candidates in June.")
    elif "nestle" in question.lower():
        st.info("💼 Nestlé Nigeria’s Graduate Trainee Program is accepting applications at nestle-cwa.com.")
    else:
        st.info("🤖 I'm still learning! Please ask about known updates.")
