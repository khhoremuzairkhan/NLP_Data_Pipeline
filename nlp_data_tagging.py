import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
import pickle
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import re
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from rouge import Rouge
from textstat import flesch_reading_ease, flesch_kincaid_grade
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

class PDFProcessor:
    """Handle PDF text extraction and preprocessing"""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def extract_text_from_pdf(self, pdf_file):
        """Extract text from uploaded PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
            return None
    
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        if not text:
            return ""
        
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep periods for sentence boundary
        text = re.sub(r'[^\w\s\.]', ' ', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra spaces
        text = ' '.join(text.split())
        
        return text
    
    def tokenize_and_lemmatize(self, text):
        """Tokenize and lemmatize text"""
        tokens = word_tokenize(text)
        
        # Remove stopwords and short tokens
        tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in self.stop_words and len(token) > 2
        ]
        
        return ' '.join(tokens)

class TextSummarizer:
    """Handle text summarization using transformers"""
    
    def __init__(self):
        try:
            # Use a lightweight summarization model
            self.summarizer = pipeline(
                "summarization", 
                model="facebook/bart-large-cnn",
                device=0 if torch.cuda.is_available() else -1
            )
        except Exception as e:
            st.error(f"Error loading summarization model: {str(e)}")
            self.summarizer = None
    
    def chunk_text(self, text, max_length=1000):
        """Split text into chunks for summarization"""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) < max_length:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def summarize_text(self, text, max_length=150, min_length=50):
        """Generate summary of text"""
        if not self.summarizer or not text:
            return "Summary unavailable"
        
        try:
            # Handle long texts by chunking
            if len(text) > 1000:
                chunks = self.chunk_text(text, 1000)
                summaries = []
                
                for chunk in chunks[:3]:  # Limit to first 3 chunks
                    if len(chunk) > 100:  # Only summarize substantial chunks
                        result = self.summarizer(
                            chunk, 
                            max_length=max_length//len(chunks[:3]), 
                            min_length=min_length//len(chunks[:3]),
                            do_sample=False
                        )
                        summaries.append(result[0]['summary_text'])
                
                return ' '.join(summaries)
            else:
                if len(text) < 100:
                    return text  # Return original if too short
                
                result = self.summarizer(
                    text, 
                    max_length=max_length, 
                    min_length=min_length,
                    do_sample=False
                )
                return result[0]['summary_text']
                
        except Exception as e:
            st.error(f"Error during summarization: {str(e)}")
            # Fallback to extractive summary
            return self.extractive_summary(text, num_sentences=3)
    
    def extractive_summary(self, text, num_sentences=3):
        """Fallback extractive summarization"""
        sentences = sent_tokenize(text)
        if len(sentences) <= num_sentences:
            return text
        
        # Simple extractive approach - take first, middle, and last sentences
        indices = [0, len(sentences)//2, -1]
        summary_sentences = [sentences[i] for i in indices[:num_sentences]]
        return ' '.join(summary_sentences)

class TopicModeler:
    """Handle topic modeling and tag generation"""
    
    def __init__(self, n_topics=5):
        self.n_topics = n_topics
        self.lda_model = None
        self.vectorizer = None
        self.feature_names = None
        
    def fit_topic_model(self, texts):
        """Train LDA topic model"""
        if not texts or len(texts) == 0:
            return None
        
        try:
            # Use CountVectorizer for LDA
            self.vectorizer = CountVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.8
            )
            
            text_vectors = self.vectorizer.fit_transform(texts)
            self.feature_names = self.vectorizer.get_feature_names_out()
            
            # Fit LDA model
            self.lda_model = LatentDirichletAllocation(
                n_components=min(self.n_topics, len(texts)),
                random_state=42,
                max_iter=10
            )
            
            self.lda_model.fit(text_vectors)
            return text_vectors
            
        except Exception as e:
            st.error(f"Error training topic model: {str(e)}")
            return None
    
    def get_topic_terms(self, n_words=5):
        """Get top terms for each topic"""
        if not self.lda_model or self.feature_names is None:
            return {}
        
        topics = {}
        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_words_idx = topic.argsort()[-n_words:][::-1]
            top_words = [self.feature_names[i] for i in top_words_idx]
            topics[f"Topic {topic_idx + 1}"] = top_words
        
        return topics
    
    def generate_tags(self, text, n_tags=5):
        """Generate tags for a single document"""
        if not self.vectorizer or not self.lda_model:
            return self.fallback_tags(text, n_tags)
        
        try:
            # Transform text
            text_vector = self.vectorizer.transform([text])
            
            # Get topic probabilities
            topic_probs = self.lda_model.transform(text_vector)[0]
            
            # Get dominant topics
            dominant_topics = np.argsort(topic_probs)[-2:][::-1]
            
            # Generate tags from dominant topics
            tags = []
            topics = self.get_topic_terms(n_words=10)
            
            for topic_idx in dominant_topics:
                topic_key = f"Topic {topic_idx + 1}"
                if topic_key in topics:
                    tags.extend(topics[topic_key][:3])
            
            # Remove duplicates and return top n_tags
            unique_tags = list(dict.fromkeys(tags))
            return unique_tags[:n_tags]
            
        except Exception as e:
            st.error(f"Error generating tags: {str(e)}")
            return self.fallback_tags(text, n_tags)
    
    def fallback_tags(self, text, n_tags=5):
        """Fallback tag generation using TF-IDF"""
        try:
            tfidf = TfidfVectorizer(
                max_features=50,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            tfidf_matrix = tfidf.fit_transform([text])
            feature_names = tfidf.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            
            # Get top scoring terms
            top_indices = scores.argsort()[-n_tags:][::-1]
            tags = [feature_names[i] for i in top_indices if scores[i] > 0]
            
            return tags
            
        except Exception as e:
            return ["document", "text", "content"]

class EvaluationMetrics:
    """Evaluate summaries and topic models"""
    
    def __init__(self):
        self.rouge = Rouge()
    
    def evaluate_summary(self, original_text, summary):
        """Evaluate summary quality"""
        metrics = {}
        
        try:
            # ROUGE scores (using first 1000 chars of original as reference)
            reference = original_text[:1000] if len(original_text) > 1000 else original_text
            
            rouge_scores = self.rouge.get_scores(summary, reference, avg=True)
            metrics['rouge_1_f'] = rouge_scores['rouge-1']['f']
            metrics['rouge_2_f'] = rouge_scores['rouge-2']['f']
            metrics['rouge_l_f'] = rouge_scores['rouge-l']['f']
            
        except Exception as e:
            metrics['rouge_1_f'] = 0
            metrics['rouge_2_f'] = 0
            metrics['rouge_l_f'] = 0
        
        # Compression ratio
        metrics['compression_ratio'] = len(summary) / len(original_text) if original_text else 0
        
        # Readability scores
        try:
            metrics['flesch_reading_ease'] = flesch_reading_ease(summary)
            metrics['flesch_kincaid_grade'] = flesch_kincaid_grade(summary)
        except:
            metrics['flesch_reading_ease'] = 0
            metrics['flesch_kincaid_grade'] = 0
        
        return metrics
    
    def evaluate_topics(self, topic_model, texts):
        """Evaluate topic model quality"""
        if not topic_model.lda_model or not texts:
            return {}
        
        try:
            # Coherence score (simplified version)
            topics = topic_model.get_topic_terms(n_words=10)
            
            # Topic diversity
            all_words = []
            for topic_words in topics.values():
                all_words.extend(topic_words)
            
            unique_words = len(set(all_words))
            total_words = len(all_words)
            diversity = unique_words / total_words if total_words > 0 else 0
            
            return {
                'n_topics': len(topics),
                'topic_diversity': diversity,
                'avg_topic_size': total_words / len(topics) if topics else 0
            }
            
        except Exception as e:
            return {'error': str(e)}

def create_visualizations(topic_model, summaries, tags_list):
    """Create visualizations for the results"""
    
    # Topic terms visualization
    if topic_model.lda_model:
        topics = topic_model.get_topic_terms(n_words=8)
        
        # Create topic terms bar chart
        fig_topics = go.Figure()
        
        for i, (topic_name, words) in enumerate(topics.items()):
            # Get word weights (simplified)
            weights = [1.0 - j*0.1 for j in range(len(words))]
            
            fig_topics.add_trace(go.Bar(
                name=topic_name,
                x=words,
                y=weights,
                visible=True if i == 0 else 'legendonly'
            ))
        
        fig_topics.update_layout(
            title="Topic Terms Distribution",
            xaxis_title="Terms",
            yaxis_title="Relevance Score",
            showlegend=True
        )
        
        st.plotly_chart(fig_topics)
    
    # Tag frequency
    if tags_list:
        all_tags = [tag for tags in tags_list for tag in tags]
        tag_counts = pd.Series(all_tags).value_counts().head(10)
        
        fig_tags = px.bar(
            x=tag_counts.index,
            y=tag_counts.values,
            title="Most Frequent Tags",
            labels={'x': 'Tags', 'y': 'Frequency'}
        )
        st.plotly_chart(fig_tags)
    
    # Word cloud of summaries
    if summaries:
        all_summaries = ' '.join(summaries)
        if all_summaries.strip():
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_summaries)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)

def main():
    st.set_page_config(
        page_title="PDF Auto-Tagging System",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("📄 Automated PDF Tagging System")
    st.markdown("Upload PDF files to generate summaries and meaningful tags using NLP and topic modeling.")
    
    # Initialize components
    pdf_processor = PDFProcessor()
    text_summarizer = TextSummarizer()
    evaluator = EvaluationMetrics()
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    n_topics = st.sidebar.slider("Number of Topics", 3, 10, 5)
    max_summary_length = st.sidebar.slider("Max Summary Length", 50, 300, 150)
    n_tags = st.sidebar.slider("Number of Tags", 3, 10, 5)
    
    topic_model = TopicModeler(n_topics=n_topics)
    
    # File upload
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type="pdf",
        accept_multiple_files=True,
        help="Select one or more PDF files to process"
    )
    
    if uploaded_files:
        # Process files
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        all_texts = []
        all_summaries = []
        all_tags = []
        
        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing {uploaded_file.name}...")
            progress_bar.progress((i + 1) / len(uploaded_files))
            
            # Extract text
            raw_text = pdf_processor.extract_text_from_pdf(uploaded_file)
            
            if raw_text:
                # Preprocess
                cleaned_text = pdf_processor.preprocess_text(raw_text)
                processed_text = pdf_processor.tokenize_and_lemmatize(cleaned_text)
                
                # Summarize
                summary = text_summarizer.summarize_text(
                    cleaned_text, 
                    max_length=max_summary_length
                )
                
                # Store for topic modeling
                all_texts.append(processed_text)
                all_summaries.append(summary)
                
                # Store results
                results.append({
                    'filename': uploaded_file.name,
                    'raw_text': raw_text,
                    'cleaned_text': cleaned_text,
                    'processed_text': processed_text,
                    'summary': summary,
                    'text_length': len(raw_text),
                    'summary_length': len(summary)
                })
        
        # Train topic model
        status_text.text("Training topic model...")
        topic_vectors = topic_model.fit_topic_model(all_texts)
        
        # Generate tags for each document
        status_text.text("Generating tags...")
        for i, result in enumerate(results):
            tags = topic_model.generate_tags(result['processed_text'], n_tags)
            result['tags'] = tags
            all_tags.append(tags)
        
        progress_bar.progress(1.0)
        status_text.text("Processing complete!")
        
        # Display results
        st.header("📊 Results Overview")
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Documents Processed", len(results))
        
        with col2:
            avg_length = np.mean([r['text_length'] for r in results])
            st.metric("Avg Document Length", f"{avg_length:.0f} chars")
        
        with col3:
            avg_summary_length = np.mean([r['summary_length'] for r in results])
            st.metric("Avg Summary Length", f"{avg_summary_length:.0f} chars")
        
        with col4:
            avg_compression = np.mean([
                r['summary_length'] / r['text_length'] 
                for r in results if r['text_length'] > 0
            ])
            st.metric("Avg Compression Ratio", f"{avg_compression:.2f}")
        
        # Visualizations
        st.header("📈 Analysis Visualizations")
        create_visualizations(topic_model, all_summaries, all_tags)
        
        # Detailed results
        st.header("📋 Detailed Results")
        
        for i, result in enumerate(results):
            with st.expander(f"📄 {result['filename']}", expanded=(i == 0)):
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📝 Summary")
                    st.write(result['summary'])
                    
                    st.subheader("🔍 Document Stats")
                    st.write(f"**Original Length:** {result['text_length']:,} characters")
                    st.write(f"**Summary Length:** {result['summary_length']:,} characters")
                    st.write(f"**Compression:** {result['summary_length']/result['text_length']:.1%}")
                
                with col2:
                    st.subheader("🏷️ Generated Tags")
                    for tag in result['tags']:
                        st.badge(tag)
                    
                    # Evaluation metrics
                    st.subheader("📊 Quality Metrics")
                    metrics = evaluator.evaluate_summary(result['raw_text'], result['summary'])
                    
                    metric_col1, metric_col2 = st.columns(2)
                    with metric_col1:
                        st.write(f"**ROUGE-1:** {metrics['rouge_1_f']:.3f}")
                        st.write(f"**ROUGE-L:** {metrics['rouge_l_f']:.3f}")
                    with metric_col2:
                        st.write(f"**Readability:** {metrics['flesch_reading_ease']:.1f}")
                        st.write(f"**Grade Level:** {metrics['flesch_kincaid_grade']:.1f}")
                
                # Show original text (truncated)
                if st.checkbox(f"Show original text for {result['filename']}", key=f"show_text_{i}"):
                    st.text_area(
                        "Original Text (first 1000 characters)",
                        result['raw_text'][:1000] + "..." if len(result['raw_text']) > 1000 else result['raw_text'],
                        height=200,
                        key=f"text_area_{i}"
                    )
        
        # Topic Analysis
        if topic_model.lda_model:
            st.header("🎯 Topic Analysis")
            topics = topic_model.get_topic_terms(n_words=10)
            
            for topic_name, words in topics.items():
                st.subheader(topic_name)
                st.write(" • ".join(words))
        
        # Export results
        st.header("💾 Export Results")
        
        if st.button("Generate Export Data"):
            # Create export DataFrame
            export_data = []
            for result in results:
                export_data.append({
                    'Filename': result['filename'],
                    'Summary': result['summary'],
                    'Tags': ', '.join(result['tags']),
                    'Original_Length': result['text_length'],
                    'Summary_Length': result['summary_length'],
                    'Compression_Ratio': result['summary_length'] / result['text_length']
                })
            
            df = pd.DataFrame(export_data)
            
            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name=f"pdf_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            st.success("Export data generated! Click the download button above.")
    
    else:
        st.info("👆 Please upload one or more PDF files to begin processing.")
        
        # Show example/demo section
        st.header("🔍 About This System")
        
        st.markdown("""
        This automated PDF tagging system processes your documents through several stages:
        
        1. **PDF Text Extraction** - Extracts raw text from uploaded PDF files
        2. **Text Preprocessing** - Cleans and normalizes the text data
        3. **Summarization** - Creates concise summaries using transformer models
        4. **Topic Modeling** - Identifies key themes using Latent Dirichlet Allocation
        5. **Tag Generation** - Produces relevant tags based on topic analysis
        6. **Evaluation** - Provides quality metrics for summaries and topics
        
        **Features:**
        - ✅ Multiple PDF upload support
        - ✅ Automatic text summarization
        - ✅ AI-powered tag generation
        - ✅ Topic modeling and visualization
        - ✅ Quality evaluation metrics
        - ✅ Export results to CSV
        - ✅ Interactive visualizations
        """)

if __name__ == "__main__":
    main()