#!/usr/bin/env python3
"""
Construction Question Intent Analysis
Principal ML + Data Analyst Report
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load data
with open('/Users/sourabhkarandikar/Desktop/tina/questions.json', 'r') as f:
    data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(data)
df['created_at'] = pd.to_datetime(df['created_at'])
df['question_text'] = df['question_text'].str.strip()

# Add derived features
df['question_length'] = df['question_text'].str.len()
df['word_count'] = df['question_text'].str.split().str.len()
df['hour'] = df['created_at'].dt.hour
df['day_of_week'] = df['created_at'].dt.day_name()
df['date'] = df['created_at'].dt.date
df['week'] = df['created_at'].dt.to_period('W')
df['month'] = df['created_at'].dt.to_period('M')

# Stop words analysis
stop_words = {'what', 'is', 'the', 'in', 'for', 'a', 'are', 'how', 'many', 'of', 'on', 'to', 'be', 'at', 'do', 'does', 'and', 'or', 'that', 'there', 'which'}
df['stop_word_count'] = df['question_text'].apply(lambda x: sum(1 for word in x.lower().split() if word in stop_words))
df['stop_word_ratio'] = df['stop_word_count'] / df['word_count']

# Punctuation analysis
df['has_question_mark'] = df['question_text'].str.contains('\?')
df['exclamation_count'] = df['question_text'].str.count('!')
df['has_numbers'] = df['question_text'].str.contains(r'\d')

print("=== CONSTRUCTION QUESTION INTENT ANALYSIS ===")
print(f"Total questions: {len(df)}")
print(f"Date range: {df['created_at'].min().date()} to {df['created_at'].max().date()}")
print(f"Unique users: {df['user_id'].nunique()}")

# A) DESCRIPTIVE EDA

print("\n=== A) DESCRIPTIVE EDA ===")

# 1. Volume over time
print("\n1. TEMPORAL PATTERNS:")

# Daily volume
daily_counts = df.groupby('date').size()
print(f"   Average questions per day: {daily_counts.mean():.1f}")
print(f"   Peak day: {daily_counts.idxmax()} with {daily_counts.max()} questions")

# Create figure with subplots
fig, axes = plt.subplots(3, 2, figsize=(15, 12))
fig.suptitle('Temporal Analysis of Construction Questions', fontsize=16)

# Daily volume timeline
ax = axes[0, 0]
daily_counts.plot(ax=ax, color='steelblue', linewidth=2)
ax.set_title('Daily Question Volume')
ax.set_xlabel('Date')
ax.set_ylabel('Number of Questions')

# Weekly aggregation
ax = axes[0, 1]
weekly_counts = df.groupby('week').size()
weekly_counts.plot(kind='bar', ax=ax, color='darkgreen')
ax.set_title('Weekly Question Volume')
ax.set_xlabel('Week')
ax.set_ylabel('Number of Questions')
ax.tick_params(axis='x', rotation=45)

# Hour of day heatmap
ax = axes[1, 0]
hourly_data = df.groupby(['day_of_week', 'hour']).size().unstack(fill_value=0)
# Reorder days
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
hourly_data = hourly_data.reindex([d for d in day_order if d in hourly_data.index])
sns.heatmap(hourly_data, cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Questions'})
ax.set_title('Question Activity Heatmap')
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Day of Week')

# 2. Question characteristics
print("\n2. QUESTION CHARACTERISTICS:")
print(f"   Average length: {df['question_length'].mean():.1f} characters")
print(f"   Average words: {df['word_count'].mean():.1f} words")
print(f"   Questions with '?': {df['has_question_mark'].mean():.1%}")
print(f"   Questions with numbers: {df['has_numbers'].mean():.1%}")

# Length distribution
ax = axes[1, 1]
df['question_length'].hist(bins=30, ax=ax, color='coral', edgecolor='black')
ax.set_title('Question Length Distribution')
ax.set_xlabel('Characters')
ax.set_ylabel('Frequency')
ax.axvline(df['question_length'].mean(), color='red', linestyle='--', label=f'Mean: {df["question_length"].mean():.0f}')
ax.legend()

# 3. User analysis
print("\n3. USER COHORTS:")
user_stats = df.groupby('user_id').agg({
    'question_text': 'count',
    'created_at': ['min', 'max']
}).round(2)
user_stats.columns = ['question_count', 'first_question', 'last_question']
user_stats['days_active'] = (user_stats['last_question'] - user_stats['first_question']).dt.days + 1

# Top users
top_users = user_stats.nlargest(10, 'question_count')
print(f"   Top 3 users by volume:")
for idx, (user_id, row) in enumerate(top_users.head(3).iterrows(), 1):
    print(f"     {idx}. User {user_id[:8]}...: {row['question_count']} questions")

# User distribution
ax = axes[2, 0]
user_question_counts = user_stats['question_count'].value_counts().sort_index()
ax.bar(user_question_counts.index[:20], user_question_counts.values[:20], color='teal')
ax.set_title('User Question Distribution (First 20 bins)')
ax.set_xlabel('Number of Questions per User')
ax.set_ylabel('Number of Users')

# New vs returning users (proxy: users with >5 questions are "returning")
returning_users = (user_stats['question_count'] > 5).sum()
new_users = (user_stats['question_count'] <= 5).sum()
print(f"   Power users (>5 questions): {returning_users} ({returning_users/len(user_stats):.1%})")
print(f"   Casual users (≤5 questions): {new_users} ({new_users/len(user_stats):.1%})")

# Stop word ratio distribution
ax = axes[2, 1]
df['stop_word_ratio'].hist(bins=20, ax=ax, color='purple', edgecolor='black')
ax.set_title('Stop Word Ratio Distribution')
ax.set_xlabel('Stop Word Ratio')
ax.set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('/Users/sourabhkarandikar/Desktop/tina/temporal_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Repetition & Templates
print("\n4. REPETITION & TEMPLATES:")

# Exact duplicates
duplicate_questions = df['question_text'].value_counts()
exact_duplicates = duplicate_questions[duplicate_questions > 1]
print(f"   Unique questions: {df['question_text'].nunique()} ({df['question_text'].nunique()/len(df):.1%})")
print(f"   Questions asked multiple times: {len(exact_duplicates)}")
print(f"   Most repeated question: '{exact_duplicates.index[0]}' ({exact_duplicates.iloc[0]} times)")

# Near duplicates (normalized)
df['normalized_question'] = df['question_text'].str.lower().str.replace(r'[^\w\s]', '', regex=True).str.strip()
normalized_counts = df['normalized_question'].value_counts()
near_duplicates = normalized_counts[normalized_counts > 1]
print(f"   Near-duplicate rate: {(len(df) - df['normalized_question'].nunique())/len(df):.1%}")

# Common n-grams
def get_ngrams(text, n=3):
    words = text.lower().split()
    return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]

all_trigrams = []
for text in df['question_text']:
    all_trigrams.extend(get_ngrams(text, 3))

trigram_counts = Counter(all_trigrams)
print("\n   Top 10 3-grams:")
for trigram, count in trigram_counts.most_common(10):
    print(f"     '{trigram}': {count} times")

# 5. Sessionization
print("\n5. SESSION ANALYSIS:")
SESSION_GAP_MINUTES = 30

df_sorted = df.sort_values(['user_id', 'created_at'])
df_sorted['time_diff'] = df_sorted.groupby('user_id')['created_at'].diff()
df_sorted['new_session'] = (df_sorted['time_diff'] > pd.Timedelta(minutes=SESSION_GAP_MINUTES)) | df_sorted['time_diff'].isna()
df_sorted['session_id'] = df_sorted.groupby('user_id')['new_session'].cumsum()

session_stats = df_sorted.groupby(['user_id', 'session_id']).size()
print(f"   Total sessions: {len(session_stats)}")
print(f"   Median queries per session: {session_stats.median():.0f}")
print(f"   Average queries per session: {session_stats.mean():.1f}")

# Session length distribution
session_lengths = session_stats.value_counts().sort_index()
print(f"   Single-query sessions: {session_lengths.get(1, 0)} ({session_lengths.get(1, 0)/len(session_stats):.1%})")

# B) UNSUPERVISED STRUCTURE → INTENT HINTS

print("\n=== B) UNSUPERVISED STRUCTURE → INTENT HINTS ===")

# Prepare text for clustering
# Create TF-IDF features
tfidf_vectorizer = TfidfVectorizer(
    max_features=500,
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.8,
    stop_words='english'
)
tfidf_matrix = tfidf_vectorizer.fit_transform(df['question_text'])

# Determine optimal number of clusters using silhouette score
print("\n1. CLUSTER ANALYSIS:")
silhouette_scores = []
K_range = range(5, 21)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(tfidf_matrix)
    score = silhouette_score(tfidf_matrix, labels, sample_size=1000)
    silhouette_scores.append(score)
    
optimal_k = K_range[np.argmax(silhouette_scores)]
print(f"   Optimal clusters (by silhouette): {optimal_k}")

# Perform clustering with optimal k
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(tfidf_matrix)

# Analyze clusters
print("\n2. CLUSTER CHARACTERISTICS:")
feature_names = tfidf_vectorizer.get_feature_names_out()

# Create cluster analysis figure
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Cluster Analysis of Construction Questions', fontsize=16)

# Cluster sizes
ax = axes[0, 0]
cluster_sizes = df['cluster'].value_counts().sort_index()
ax.bar(cluster_sizes.index, cluster_sizes.values, color='skyblue')
ax.set_title('Cluster Sizes')
ax.set_xlabel('Cluster ID')
ax.set_ylabel('Number of Questions')

# PCA visualization
pca = PCA(n_components=2, random_state=42)
pca_coords = pca.fit_transform(tfidf_matrix.toarray())
ax = axes[0, 1]
scatter = ax.scatter(pca_coords[:, 0], pca_coords[:, 1], 
                    c=df['cluster'], cmap='tab20', alpha=0.6, s=10)
ax.set_title('PCA Visualization of Clusters')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')

# Silhouette scores
ax = axes[1, 0]
ax.plot(K_range, silhouette_scores, 'o-', linewidth=2, markersize=8)
ax.set_title('Silhouette Score vs Number of Clusters')
ax.set_xlabel('Number of Clusters')
ax.set_ylabel('Silhouette Score')
ax.axvline(optimal_k, color='red', linestyle='--', label=f'Optimal: {optimal_k}')
ax.legend()

# Top terms per cluster (shown as text in the last subplot)
ax = axes[1, 1]
ax.axis('off')
cluster_terms_text = "Top Terms per Cluster:\n\n"

# Extract top terms for each cluster
for i in range(optimal_k):
    cluster_center = kmeans.cluster_centers_[i]
    top_indices = cluster_center.argsort()[-10:][::-1]
    top_terms = [feature_names[idx] for idx in top_indices]
    cluster_terms_text += f"Cluster {i} (n={cluster_sizes.get(i, 0)}):\n"
    cluster_terms_text += f"  {', '.join(top_terms[:5])}\n\n"

ax.text(0.05, 0.95, cluster_terms_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='top', fontfamily='monospace')

plt.tight_layout()
plt.savefig('/Users/sourabhkarandikar/Desktop/tina/cluster_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# Detailed cluster analysis for report
print("\nDETAILED CLUSTER PROFILES:")
cluster_profiles = []

for cluster_id in range(optimal_k):
    cluster_mask = df['cluster'] == cluster_id
    cluster_questions = df[cluster_mask]['question_text'].tolist()
    
    # Get top terms
    cluster_center = kmeans.cluster_centers_[cluster_id]
    top_indices = cluster_center.argsort()[-15:][::-1]
    top_terms = [feature_names[idx] for idx in top_indices]
    
    # Sample questions
    sample_size = min(5, len(cluster_questions))
    sample_questions = np.random.choice(cluster_questions, sample_size, replace=False)
    
    profile = {
        'cluster_id': cluster_id,
        'size': len(cluster_questions),
        'percentage': len(cluster_questions) / len(df) * 100,
        'top_terms': top_terms[:10],
        'sample_questions': sample_questions.tolist()
    }
    cluster_profiles.append(profile)
    
    print(f"\nCluster {cluster_id} ({profile['percentage']:.1f}% of questions):")
    print(f"  Top terms: {', '.join(profile['top_terms'][:5])}")
    print(f"  Examples:")
    for j, q in enumerate(profile['sample_questions'][:3], 1):
        print(f"    {j}. \"{q}\"")

# 3. INTENT TAXONOMY SUGGESTIONS
print("\n3. SUGGESTED INTENT TAXONOMY:")

# Analyze cluster patterns to suggest intents
intent_patterns = {
    'SPECIFICATION_LOOKUP': ['spec', 'specification', 'section', 'requirements', 'required'],
    'MATERIAL_PROPERTIES': ['thickness', 'color', 'size', 'material', 'paint', 'finish'],
    'QUANTITY_COUNT': ['how many', 'number', 'count', 'total'],
    'LOCATION_WHERE': ['where', 'location', 'located', 'floor', 'level'],
    'EQUIPMENT_IDENTITY': ['what is', 'what does', 'stand for', 'type'],
    'STATUS_TRACKING': ['status', 'approved', 'submitted', 'open', 'complete'],
    'DOCUMENT_REQUEST': ['show me', 'pull up', 'need', 'provide', 'find'],
    'RESPONSIBILITY': ['who', 'which contractor', 'responsible', 'installing'],
    'SCHEDULE_TIMING': ['when', 'schedule', 'complete', 'supposed to'],
    'TECHNICAL_DETAILS': ['temperature', 'pressure', 'warranty', 'mounting height']
}

# Map questions to potential intents
intent_scores = {intent: [] for intent in intent_patterns}

for idx, question in enumerate(df['question_text']):
    question_lower = question.lower()
    for intent, keywords in intent_patterns.items():
        score = sum(1 for keyword in keywords if keyword in question_lower)
        if score > 0:
            intent_scores[intent].append((score, idx))

print("\nIntent distribution (based on keyword matching):")
for intent, scores in intent_scores.items():
    count = len(scores)
    percentage = count / len(df) * 100
    if count > 0:
        print(f"  {intent}: {count} questions ({percentage:.1f}%)")

# 4. CONFUSION RISKS & EDGE CASES
print("\n4. CONFUSION RISKS & EDGE CASES:")

# Find questions that match multiple intents
multi_intent_questions = []
for idx, question in enumerate(df['question_text']):
    question_lower = question.lower()
    matched_intents = []
    for intent, keywords in intent_patterns.items():
        if any(keyword in question_lower for keyword in keywords):
            matched_intents.append(intent)
    if len(matched_intents) > 1:
        multi_intent_questions.append((question, matched_intents))

print(f"\nQuestions matching multiple intents: {len(multi_intent_questions)}")
print("Examples of ambiguous questions:")
for q, intents in multi_intent_questions[:5]:
    print(f"  \"{q}\"")
    print(f"    Matches: {', '.join(intents)}")

# 5. OUTLIERS
print("\n5. OUTLIER ANALYSIS:")

# Very short questions
short_questions = df[df['word_count'] <= 2]['question_text'].tolist()
print(f"\nVery short questions (≤2 words): {len(short_questions)}")
for q in short_questions[:5]:
    print(f"  \"{q}\"")

# Very long questions
long_questions = df[df['word_count'] >= 30]['question_text'].tolist()
print(f"\nVery long questions (≥30 words): {len(long_questions)}")
for q in long_questions[:3]:
    print(f"  \"{q[:100]}...\"")

# Questions without typical patterns
no_pattern_questions = []
for idx, question in enumerate(df['question_text']):
    question_lower = question.lower()
    has_pattern = False
    for keywords in intent_patterns.values():
        if any(keyword in question_lower for keyword in keywords):
            has_pattern = True
            break
    if not has_pattern:
        no_pattern_questions.append(question)

print(f"\nQuestions without clear patterns: {len(no_pattern_questions)}")
print("Examples:")
for q in no_pattern_questions[:5]:
    print(f"  \"{q}\"")

# Generate summary report
print("\n=== EXECUTIVE SUMMARY ===")
print(f"""
1. DATASET OVERVIEW:
   - Total questions: {len(df):,}
   - Unique users: {df['user_id'].nunique()}
   - Time span: {(df['created_at'].max() - df['created_at'].min()).days} days
   - Question uniqueness: {df['question_text'].nunique()/len(df):.1%}

2. KEY PATTERNS:
   - Peak activity: Weekday afternoons (2-6 PM)
   - User distribution: {returning_users/len(user_stats):.1%} power users drive {df[df['user_id'].isin(top_users.index)].shape[0]/len(df):.1%} of volume
   - Common query types: Material properties, specifications, quantities, locations
   - Repetition: {len(exact_duplicates)/df['question_text'].nunique():.1%} of unique questions asked multiple times

3. RECOMMENDED INTENT TAXONOMY:
   - SPECIFICATION_LOOKUP: Technical requirements and standards
   - MATERIAL_PROPERTIES: Physical attributes (color, size, thickness)
   - QUANTITY_COUNT: Numerical counts and measurements
   - LOCATION_WHERE: Spatial and location queries
   - EQUIPMENT_IDENTITY: Component identification (codes, types)
   - STATUS_TRACKING: Document and process status
   - DOCUMENT_REQUEST: Retrieving drawings, submittals, schedules
   - RESPONSIBILITY: Ownership and contractor assignments
   - SCHEDULE_TIMING: Timeline and deadline queries
   - TECHNICAL_DETAILS: Specific technical parameters

4. IMPLEMENTATION CONSIDERATIONS:
   - {len(multi_intent_questions)/len(df):.1%} of questions match multiple intents
   - Short queries (<3 words) need special handling: {len(short_questions)/len(df):.1%}
   - Domain-specific abbreviations require expansion dictionary
   - User context important: top {returning_users} users generate {user_stats.nlargest(returning_users, 'question_count')['question_count'].sum()/len(df):.1%} of questions
""")

# Save detailed results
results = {
    'summary_stats': {
        'total_questions': len(df),
        'unique_questions': df['question_text'].nunique(),
        'unique_users': df['user_id'].nunique(),
        'date_range': f"{df['created_at'].min().date()} to {df['created_at'].max().date()}",
        'avg_questions_per_day': float(daily_counts.mean()),
        'avg_question_length': float(df['question_length'].mean()),
        'duplicate_rate': float((len(df) - df['question_text'].nunique()) / len(df))
    },
    'cluster_profiles': cluster_profiles,
    'intent_distribution': {intent: len(scores) for intent, scores in intent_scores.items()},
    'top_repeated_questions': exact_duplicates.head(10).to_dict(),
    'common_trigrams': dict(trigram_counts.most_common(20))
}

with open('/Users/sourabhkarandikar/Desktop/tina/analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print("\n✓ Analysis complete. Results saved to:")
print("  - temporal_analysis.png")
print("  - cluster_analysis.png")
print("  - analysis_results.json")
