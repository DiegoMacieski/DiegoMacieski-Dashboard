
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
import time

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

st.set_page_config(page_title="CA2 Dashboard", layout="wide")

# === Navegação ===
menu = st.sidebar.radio("📌 Select Task", ["🎬 Task 1: MovieLens", "🧺 Task 2: Bakery Market Basket"])

# === TAREFA 1 ===
if menu == "🎬 Task 1: MovieLens":
    st.markdown(
        """
        <div style='text-align: center; font-size: 32px; font-weight: bold;'>
            Machine Learning and Data Visualisation – Diego Macieski
        </div>
        <h1 style='text-align: center;'>🎬 MovieLens Recommender System</h1>
        """,
        unsafe_allow_html=True
    )
    df = pd.read_csv("movielens_merged.csv")

    # 📊 Gender
    st.subheader("📊 Gender Distribution")
    gender_count = df['Gender'].value_counts().reset_index()
    gender_count.columns = ['Gender', 'Count']

    fig1 = px.bar(
        gender_count,
        x='Gender',
        y='Count',
        title="Gender Distribution",
        color='Gender',
        color_discrete_sequence=["#3A86FF", "#FF006E", "#FFBE0B", "#06D6A0", "#8338EC"]
    )

    fig1.update_layout(
        font_color='black',
        title_font=dict(size=20)
    )

    st.plotly_chart(fig1)

 # Age
    st.subheader("📊 Age Distribution")
    fig2 = px.histogram(df, x='Age', nbins=10, title='User Age Distribution')
    st.plotly_chart(fig2)

    # ⭐ Ratings
    st.subheader("⭐ Rating Distribution")
    fig3 = px.histogram(df, x='Rating', nbins=5, title='Movie Ratings')
    st.plotly_chart(fig3)

    # Top 10 Most Rated Movies
    st.subheader("🎥 Top 10 Most Rated Movies")
    top_movies = df['Title'].value_counts().head(10).sort_values()
    fig4 = px.bar(x=top_movies.values, y=top_movies.index, orientation='h',
                  labels={'x': 'Number of Ratings', 'y': 'Movie Title'},
                  title='Top 10 Most Rated Movies')
    st.plotly_chart(fig4)

    # Top 10 by Rating and Genre
    st.subheader("🎯 Top 10 Movies by Rating and Genre")
    rating_options = sorted(df['Rating'].unique(), reverse=True)
    selected_rating = st.selectbox("Select Rating:", rating_options)

    genres = set(g for sublist in df['Genres'].dropna().str.split('|') for g in sublist)
    selected_genre = st.selectbox("Select Genre:", sorted(genres))

    df_filtered = df[(df['Rating'] == selected_rating) & (df['Genres'].str.contains(selected_genre, na=False))]
    top_filtered = df_filtered['Title'].value_counts().head(10).sort_values()

    if not top_filtered.empty:
        fig5 = px.bar(x=top_filtered.values, y=top_filtered.index, orientation='h',
                      labels={'x': 'Number of Ratings', 'y': 'Movie Title'},
                      title=f"Top 10 Movies with Rating {selected_rating} in Genre '{selected_genre}'")
        st.plotly_chart(fig5)
    else:
        st.warning("No movies found with this combination of rating and genre.")

    # 🎬 Content-Based Filtering
    st.subheader("🎬 Content-Based Filtering (Genres + TF-IDF)")
    st.markdown("Use the dropdown or type to get 5 similar recommendations.")

    movies = df[['MovieID', 'Title', 'Genres']].drop_duplicates().reset_index(drop=True)

    @st.cache_resource
    def compute_cosine_similarity(movies_df):
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(movies_df['Genres'])
        return linear_kernel(tfidf_matrix, tfidf_matrix)

    cosine_sim = compute_cosine_similarity(movies)
    indices = pd.Series(movies.index, index=movies['Title']).drop_duplicates()

    def recommend(title):
        if title not in indices:
            return pd.DataFrame([{"Title": "Movie not found", "Genres": "-"}])
        idx = indices[title]
        sim_scores = sorted(enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True)[1:6]
        movie_indices = [i[0] for i in sim_scores]
        return movies[['Title', 'Genres']].iloc[movie_indices]

    movie_list = sorted(movies['Title'].unique())
    selected_title = st.selectbox("Select a movie:", movie_list, key="dropdown_movie")
    typed = st.text_input("Or type a movie title:", "", key="typed_input")
    suggestions = [title for title in movie_list if typed.lower() in title.lower()]
    selected_typed = st.selectbox("Suggestions:", suggestions, key="typed_suggestions") if typed else None

    if st.button("Recommend", key="recommend_button"):
        title_to_use = selected_typed if selected_typed else selected_title
        recs = recommend(title_to_use)
        st.markdown(f"### 🍿 Top 5 Similar Movies to **{title_to_use}**")
        st.dataframe(recs.reset_index(drop=True))

# === TAREFA 2 ===
elif menu == "🧺 Task 2: Bakery Market Basket":
    st.markdown(
        """
        <div style='text-align: center; font-size: 32px; font-weight: bold;'>
            Machine Learning and Data Visualisation – Diego Macieski
        </div>
        <h1 style='text-align: center;'>🧺 Market Basket Analysis</h1>
        """,
        unsafe_allow_html=True
    )

    # Leitura e limpeza dos dados
    df = pd.read_csv("Bakery_sales_clean.csv")
    df['unit_price'] = df['unit_price'].str.replace('€', '', regex=False).str.replace(',', '.', regex=False).str.strip().astype(float)
    df['ticket_number'] = df['ticket_number'].astype(str)
    df['Revenue'] = df['Quantity'] * df['unit_price']

    # Transações
    transactions = df.groupby('ticket_number')['article'].apply(list).tolist()
    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_array, columns=te.columns_)

    #  Apriori
    start_apriori = time.time()
    frequent_itemsets_apriori = apriori(df_encoded, min_support=0.01, use_colnames=True)
    rules_apriori = association_rules(frequent_itemsets_apriori, metric="confidence", min_threshold=0.4)
    apriori_time = round(time.time() - start_apriori, 2)

    #  FP-Growth
    start_fpgrowth = time.time()
    frequent_itemsets_fpgrowth = fpgrowth(df_encoded, min_support=0.01, use_colnames=True)
    rules_fpgrowth = association_rules(frequent_itemsets_fpgrowth, metric="confidence", min_threshold=0.4)
    fpgrowth_time = round(time.time() - start_fpgrowth, 2)

    # 📊 Itens mais vendidos
    st.subheader("🍞 Top 10 Most Sold Items")
    most_sold = df.groupby('article')['Quantity'].sum().sort_values(ascending=False).head(10)
    fig1 = px.bar(most_sold, x=most_sold.values, y=most_sold.index, orientation='h',
                  labels={"x": "Quantity Sold", "y": "Product"})
    st.plotly_chart(fig1, use_container_width=True)

    #  Receita por produto
    st.subheader("💰 Top 10 Revenue Generating Items")
    revenue = df.groupby('article')['Revenue'].sum().sort_values(ascending=False).head(10)
    fig2 = px.bar(revenue, x=revenue.values, y=revenue.index, orientation='h',
                  labels={"x": "Total Revenue (€)", "y": "Product"})
    st.plotly_chart(fig2, use_container_width=True)

    #  Produtos por preço médio com seletor
    st.subheader("💸 Product Prices – Cheapest vs Most Expensive")
    price_option = st.selectbox("Select view:", ["Cheapest Products", "Most Expensive Products"])
    avg_price = df.groupby('article')['unit_price'].mean().sort_values()

    if price_option == "Cheapest Products":
        selected_data = avg_price.head(10)
        title = "🧂 Cheapest Products (Avg. Unit Price)"
    else:
        selected_data = avg_price.tail(10)
        title = "🧁 Most Expensive Products (Avg. Unit Price)"

    fig3 = px.bar(
        selected_data,
        x=selected_data.values,
        y=selected_data.index,
        orientation='h',
        labels={"x": "Average Price (€)", "y": "Product"},
        title=title
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Regras de associação por produto e algoritmo
    st.subheader("🔍 Explore Rules by Product and Algorithm")
    algorithm_option = st.selectbox("Select algorithm:", ["Apriori", "FP-Growth"])
    rules = rules_apriori if algorithm_option == "Apriori" else rules_fpgrowth

    all_items = sorted(set(item for s in rules['antecedents'] for item in s).union(
                       item for s in rules['consequents'] for item in s))
    selected_item = st.selectbox("Select a product to view rules:", all_items)

    filtered = rules[rules['antecedents'].apply(lambda x: selected_item in x) |
                     rules['consequents'].apply(lambda x: selected_item in x)].copy()

    filtered["antecedents"] = filtered["antecedents"].apply(lambda x: ', '.join(list(x)))
    filtered["consequents"] = filtered["consequents"].apply(lambda x: ', '.join(list(x)))

    # Adiciona interpretação automática
    def interpret_rule(row):
        return f"Customers who buy {row['antecedents']} are likely to also buy {row['consequents']}"

    def explain_metrics(row):
        support_pct = f"{row['support']*100:.2f}%"
        confidence_pct = f"{row['confidence']*100:.2f}%"
        lift_val = f"{row['lift']:.2f}"
        return (
            f"This rule applies to {support_pct} of all transactions; "
            f"{confidence_pct} of customers who bought {row['antecedents']} also bought {row['consequents']}; "
            f"and the likelihood of buying {row['consequents']} increases by {lift_val}× compared to chance."
        )

    filtered["Interpretation"] = filtered.apply(interpret_rule, axis=1)
    filtered["Metric Meaning"] = filtered.apply(explain_metrics, axis=1)

    if not filtered.empty:
        filtered.reset_index(drop=True, inplace=True)
        filtered.index.name = "Rule ID"
        st.dataframe(filtered[['antecedents', 'consequents', 'support', 'confidence', 'lift', 'Interpretation', 'Metric Meaning']])
    else:
        st.info("No rules found for this product using this algorithm.")

    # Comparação do tempo de execução
    st.subheader("⏱️ Execution Time Comparison")
    time_df = pd.DataFrame({
        "Algorithm": ["Apriori", "FP-Growth"],
        "Time (seconds)": [apriori_time, fpgrowth_time]
    })
    fig_time = px.bar(
        time_df,
        x="Algorithm",
        y="Time (seconds)",
        title="⏱️ Execution Time: Apriori vs FP-Growth",
        text_auto=True
    )
    st.plotly_chart(fig_time, use_container_width=True)