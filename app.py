import streamlit as st
import spacy
import heapq
import torch
import pdfplumber
from newspaper import Article
from transformers import BartTokenizer, BartForConditionalGeneration

# Load spaCy model ONCE
nlp = spacy.load("en_core_web_sm")

class TextSummarizer:
    def __init__(self, model_name="facebook/bart-large-cnn"):
        """Initialize Summarizer with BART transformer model."""
        self.model_name = model_name
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)

    def extractive_summary(self, text, num_sentences=5):
        """Extractive summarization with small text handling."""
        doc = nlp(text)
        sentences = [sent.text for sent in doc.sents if sent.text.strip()]

        # Handle short texts
        if len(sentences) <= num_sentences:
            return " ".join(sentences)

        # Frequency-based sentence scoring
        word_frequencies = {}
        for token in doc:
            if not token.is_stop and not token.is_punct:
                lemma = token.lemma_.lower()
                word_frequencies[lemma] = word_frequencies.get(lemma, 0) + 1

        max_freq = max(word_frequencies.values()) if word_frequencies else 1
        for word in word_frequencies:
            word_frequencies[word] /= max_freq

        sentence_scores = {}
        for sent in sentences:
            sent_doc = nlp(sent.lower())
            for token in sent_doc:
                if token.lemma_.lower() in word_frequencies:
                    sentence_scores[sent] = sentence_scores.get(sent, 0) + word_frequencies[token.lemma_.lower()]

        summary_sentences = heapq.nlargest(min(num_sentences, len(sentences)), sentence_scores, key=sentence_scores.get)
        return " ".join(summary_sentences)

    def abstractive_summary(self, text, max_length=512, min_length=100):
        """Abstractive summarization with small text handling."""
        # Handle short texts by dynamically adjusting length parameters
        text_length = len(text.split())

        if text_length < 50:
            min_length = 30
            max_length = 100
        elif text_length < 100:
            min_length = 50
            max_length = 200
        else:
            min_length = 100
            max_length = 512

        inputs = self.tokenizer.encode(
            "summarize: " + text, 
            return_tensors="pt", 
            max_length=1024, 
            truncation=True
        )
        summary_ids = self.model.generate(
            inputs,
            max_length=max_length,
            min_length=min_length,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    def summarize_pdf(self, pdf_path):
        """Extract and summarize PDF text."""
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

        if not text.strip():
            return "No text found in PDF."

        max_input_length = 1024
        chunks = [text[i:i+max_input_length] for i in range(0, len(text), max_input_length)]

        summaries = [self.abstractive_summary(chunk) for chunk in chunks]
        return " ".join(summaries)

    def summarize_url(self, url):
        """Extract and summarize URL content."""
        article = Article(url)
        try:
            article.download()
            article.parse()
        except Exception as e:
            return f"Failed to download/parse the URL: {e}"

        text = article.text.strip()
        if not text:
            return "No main article content found."

        return self.abstractive_summary(text)

# Streamlit UI
st.title("NLP Text Summarizer")

option = st.selectbox("Choose an option", ["Text", "PDF", "URL"])
summarizer = TextSummarizer()

if option == "Text":
    user_input = st.text_area("Enter text to summarize:")
    if st.button("Summarize Text"):
        if len(user_input.split()) < 10:
            st.warning("Text is too short for meaningful summarization.")
        else:
            extractive = summarizer.extractive_summary(user_input)
            abstractive = summarizer.abstractive_summary(user_input)
            
            st.subheader("Extractive Summary")
            st.write(extractive)
            
            st.subheader("Abstractive Summary")
            st.write(abstractive)

elif option == "PDF":
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file and st.button("Summarize PDF"):
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())
        pdf_summary = summarizer.summarize_pdf("temp.pdf")
        st.subheader("PDF Summary")
        st.write(pdf_summary)

elif option == "URL":
    url_input = st.text_input("Enter a URL to summarize:")
    if st.button("Summarize URL"):
        url_summary = summarizer.summarize_url(url_input)
        st.subheader("URL Summary")
        st.write(url_summary)
