import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import io
import re
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from groq import Groq
import arabic_reshaper
from bidi.algorithm import get_display
import time

# ════════════════════════════════════════════════════════════════
#  ARABIC NLP
# ════════════════════════════════════════════════════════════════
ARABIC_STOP_WORDS = {
    "في", "من", "على", "إلى", "عن", "مع", "هذا", "هذه", "ذلك", "تلك",
    "التي", "الذي", "الذين", "اللتين", "اللذين", "اللاتي",
    "هو", "هي", "هم", "هن", "نحن", "أنت", "أنا", "أنتم",
    "كان", "كانت", "يكون", "تكون", "ليس", "ليست",
    "أن", "إن", "لأن", "حيث", "كما", "بين", "حتى", "عند", "بعد", "قبل",
    "كل", "بعض", "غير", "أي", "ما", "لا", "لم", "لن", "قد",
    "ثم", "أو", "و", "ف", "ب", "ل", "ك",
    "ذات", "ذو", "ذي", "أول", "آخر", "خلال", "ضد", "حول", "فوق", "تحت",
    "وفق", "نحو", "عبر", "دون", "منذ", "ضمن", "لدى",
    "دراسة", "بحث", "رسالة", "أطروحة",
}


def normalize_arabic(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[\u0617-\u061A\u064B-\u0652\u0670]", "", text)
    text = re.sub(r"[إأآٱ]", "ا", text)
    text = text.replace("ة", "ه")
    text = text.replace("ى", "ي")
    text = text.replace("ـ", "")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_keywords(text):
    normalized = normalize_arabic(text)
    words = normalized.split()
    return [w for w in words if len(w) > 2 and w not in ARABIC_STOP_WORDS]


# ════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="المنصة الأكاديمية الذكية",
    page_icon="page_facing_up",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════════
#  CSS
# ════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@300;400;500;700;800;900&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Amiri:wght@400;700&display=swap');

:root {
    --primary: #1B2A4A;
    --primary-light: #2C4066;
    --accent: #C8A951;
    --accent-light: #E8D590;
    --success: #1a936f;
    --bg: #F5F3EE;
    --card-bg: #FFFFFF;
    --text: #2D2D2D;
    --text-light: #6B7280;
    --border: #E5E2DB;
}
html, body, [class*="css"] {
    direction: rtl;
    font-family: 'Tajawal', 'Amiri', serif;
}
.stApp { background-color: var(--bg); }

.academic-header {
    background: linear-gradient(135deg, var(--primary) 0%, #0D1B2A 50%, var(--primary-light) 100%);
    color: white; padding: 2.5rem 2rem; text-align: center;
    border-radius: 0 0 25px 25px; margin: -1rem -1rem 2rem -1rem;
    box-shadow: 0 8px 32px rgba(27,42,74,0.3);
}
.academic-header h1 {
    font-family: 'Amiri', serif; font-size: 2.4rem; font-weight: 700; margin: 0;
}
.academic-header .subtitle {
    font-size: 1rem; opacity: 0.8; margin-top: 0.5rem; font-weight: 300;
}
.academic-header .gold-line {
    width: 80px; height: 3px; background: var(--accent);
    margin: 1rem auto 0; border-radius: 2px;
}

.stats-grid {
    display: grid; grid-template-columns: repeat(4, 1fr);
    gap: 15px; margin: 0 0 2rem;
}
.stat-box {
    background: var(--card-bg); border-radius: 12px; padding: 1.2rem;
    text-align: center; box-shadow: 0 2px 12px rgba(0,0,0,0.04);
    border-bottom: 3px solid var(--accent);
}
.stat-box .num { font-size: 2rem; font-weight: 900; color: var(--primary); }
.stat-box .lbl { font-size: 0.82rem; color: var(--text-light); margin-top: 2px; }

.search-section {
    background: var(--card-bg); padding: 2rem; border-radius: 16px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.06); margin-bottom: 1.5rem;
    border: 1px solid var(--border);
}

.thesis-card {
    background: var(--card-bg); padding: 1.8rem; margin-bottom: 1rem;
    border-radius: 14px; box-shadow: 0 2px 15px rgba(0,0,0,0.05);
    border-right: 4px solid var(--primary);
    border: 1px solid var(--border); transition: all 0.3s;
}
.thesis-card:hover {
    transform: translateX(-4px);
    box-shadow: 0 6px 25px rgba(0,0,0,0.1);
    border-right-color: var(--accent);
}
.thesis-card .citation {
    color: var(--text); font-size: 1.05rem; line-height: 1.9; margin-bottom: 0.8rem;
}
.thesis-card .author { color: var(--primary); font-weight: 700; }
.thesis-card .year {
    color: var(--accent); font-weight: 700;
    background: #FFF8E7; padding: 1px 8px; border-radius: 4px;
}
.thesis-card .title {
    font-family: 'Amiri', serif; font-weight: 700;
    color: var(--primary); font-size: 1.1rem;
}
.thesis-card .meta-row {
    display: flex; flex-wrap: wrap; gap: 12px; margin-top: 0.6rem;
}

.relevance-bar {
    display: inline-flex; align-items: center; gap: 8px; float: left; direction: ltr;
}
.relevance-bar .bar-bg {
    width: 80px; height: 6px; background: #E5E2DB; border-radius: 3px; overflow: hidden;
}
.relevance-bar .bar-fill {
    height: 100%; border-radius: 3px;
    background: linear-gradient(90deg, var(--accent), var(--success));
}
.relevance-bar .score-text { font-size: 0.78rem; color: var(--text-light); font-weight: 600; }

.field-tag {
    display: inline-block; padding: 3px 12px; border-radius: 20px;
    font-size: 0.78rem; font-weight: 500; margin: 2px;
}
.field-tag.univ { background: #EEF0F7; color: var(--primary); }
.field-tag.college { background: #E8F5EE; color: var(--success); }
.field-tag.dept { background: #FFF8E7; color: #B8941F; }
.field-tag.degree { background: #F0E8F5; color: #7C3AED; }

.section-title {
    font-family: 'Amiri', serif; color: var(--primary);
    font-size: 1.4rem; font-weight: 700;
    border-bottom: 2px solid var(--accent);
    padding-bottom: 0.5rem; margin-bottom: 1.5rem;
}

.ai-engine {
    background: linear-gradient(135deg, var(--primary) 0%, #2C4066 100%);
    color: var(--accent-light); padding: 8px 18px; border-radius: 25px;
    font-size: 0.85rem; display: inline-block; font-weight: 600; margin-bottom: 1rem;
}

.cite-box {
    background: #F8F7F4; border: 1px solid var(--border);
    border-radius: 8px; padding: 0.8rem; font-size: 0.85rem;
    color: var(--text); direction: rtl; line-height: 1.8;
    font-family: 'Amiri', serif;
}

.academic-footer {
    background: var(--primary); color: rgba(255,255,255,0.8);
    text-align: center; padding: 1.5rem; border-radius: 15px;
    margin-top: 3rem; font-size: 0.85rem;
}
.academic-footer .gold { color: var(--accent); font-weight: 700; }

.page-nav { text-align: center; color: var(--text-light); padding: 0.5rem; font-size: 0.9rem; }

section[data-testid="stSidebar"] { direction: rtl; background: #FAFAF8; }
section[data-testid="stSidebar"] h2 { font-family: 'Amiri', serif; color: var(--primary); }

.login-card {
    max-width: 420px; margin: 2rem auto; background: white; padding: 2.5rem;
    border-radius: 16px; box-shadow: 0 4px 24px rgba(27,42,74,0.12);
    border-top: 4px solid var(--accent); text-align: center;
}
.login-card h3 { color: var(--primary); font-family: Amiri, serif; margin: 0 0 0.5rem; }
.login-card p { color: var(--text-light); font-size: 0.9rem; }

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header[data-testid="stHeader"] { background: transparent; }

@media (max-width: 768px) {
    .stats-grid { grid-template-columns: repeat(2, 1fr); }
    .academic-header h1 { font-size: 1.8rem; }
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
#  DATA LOADING
# ════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    file_path = "فهرس الرسائل الجامعية.xlsx"
    sheets = ["المستنصرية", "بغداد", "التكنولوجية + المعاهد المستقلة", "محافظات"]
    frames = []
    for sheet_name in sheets:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        col_map = {}
        for c in df.columns:
            cl = str(c).strip()
            if "عنوان" in cl:
                col_map[c] = "العنوان"
            elif "اسم" in cl and "باحث" in cl:
                col_map[c] = "اسم الباحث"
            elif cl == "الجامعة":
                col_map[c] = "الجامعة"
            elif cl == "الكلية":
                col_map[c] = "الكلية"
            elif cl == "القسم":
                col_map[c] = "القسم"
            elif cl == "السنة":
                col_map[c] = "السنة"
            elif cl == "الشهادة":
                col_map[c] = "الشهادة"
        df = df.rename(columns=col_map)
        keep = ["العنوان", "اسم الباحث", "الجامعة", "الكلية", "القسم", "السنة", "الشهادة"]
        existing = [c for c in keep if c in df.columns]
        df = df[existing].copy()
        for c in keep:
            if c not in df.columns:
                df[c] = ""
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    data = data.dropna(subset=["العنوان"])
    data["العنوان"] = data["العنوان"].astype(str).str.strip()
    data = data[data["العنوان"].str.len() > 3].reset_index(drop=True)
    for col in ["اسم الباحث", "الجامعة", "الكلية", "القسم", "الشهادة"]:
        data[col] = data[col].fillna("").astype(str).str.strip().replace("nan", "")
    data["السنة"] = pd.to_numeric(data["السنة"], errors="coerce")
    data["عنوان_معالج"] = data["العنوان"].apply(normalize_arabic)
    data["باحث_معالج"] = data["اسم الباحث"].apply(normalize_arabic)
    data["قسم_معالج"] = data["القسم"].apply(normalize_arabic)
    data["نص_بحث"] = (
        data["عنوان_معالج"] + " " + data["عنوان_معالج"] + " " +
        data["باحث_معالج"] + " " + data["قسم_معالج"]
    ).str.strip().replace("", "فارغ")
    return data


@st.cache_resource
def build_search_engine(texts):
    vectorizer = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), max_features=100000,
        sublinear_tf=True, min_df=2, max_df=0.95,
        stop_words=list(ARABIC_STOP_WORDS),
    )
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


@st.cache_data
def extract_all_keywords(titles):
    all_words = []
    for t in titles:
        all_words.extend(extract_keywords(t))
    return Counter(all_words)


@st.cache_data
def run_topic_modeling(texts, n_topics=8):
    cv = CountVectorizer(max_features=5000, min_df=3, max_df=0.9, stop_words=list(ARABIC_STOP_WORDS))
    dtm = cv.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=15, learning_method="online")
    lda.fit(dtm)
    feature_names = cv.get_feature_names_out()
    topics = []
    for idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-8:-1]]
        topics.append({"id": idx, "words": top_words})
    dominant = lda.transform(dtm).argmax(axis=1)
    return topics, dominant


def smart_search(query, vectorizer, matrix, data, top_n=300):
    normalized_query = normalize_arabic(query)
    query_vec = vectorizer.transform([normalized_query])
    tfidf_scores = cosine_similarity(query_vec, matrix).flatten()
    query_lower = query.strip()
    exact_bonus = data["العنوان"].str.contains(query_lower, case=False, na=False, regex=False).astype(float) * 0.3
    author_bonus = data["اسم الباحث"].str.contains(query_lower, case=False, na=False, regex=False).astype(float) * 0.25
    final_scores = np.clip(tfidf_scores + exact_bonus.values + author_bonus.values, 0, 1)
    top_idx = final_scores.argsort()[::-1][:top_n]
    top_idx = top_idx[final_scores[top_idx] > 0.02]
    return top_idx, final_scores[top_idx]


def find_similar(thesis_idx, matrix, top_n=5):
    sim = cosine_similarity(matrix[thesis_idx:thesis_idx + 1], matrix).flatten()
    sim[thesis_idx] = 0
    top = sim.argsort()[::-1][:top_n]
    return top, sim[top]


def generate_citation_apa(row):
    year = int(row["السنة"]) if pd.notna(row["السنة"]) else "بدون تاريخ"
    author = row["اسم الباحث"] if row["اسم الباحث"] else "مؤلف غير معروف"
    return f'{author}. ({year}). {row["العنوان"]}. [{row["الشهادة"]}]. {row["الجامعة"]}، {row["الكلية"]}.'


# ════════════════════════════════════════════════════════════════
#  GROQ AI
# ════════════════════════════════════════════════════════════════
GROQ_MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it"]


def ai_analyze(prompt, max_tokens=1024):
    if "groq_key" not in st.session_state:
        return None
    selected = st.session_state.get("groq_model", GROQ_MODELS[0])
    models_to_try = [selected] + [m for m in GROQ_MODELS if m != selected]
    for model_name in models_to_try:
        try:
            client = Groq(api_key=st.session_state["groq_key"])
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7,
            )
            st.session_state["groq_model"] = model_name
            return response.choices[0].message.content
        except Exception as e:
            err = str(e).lower()
            if "rate" in err or "limit" in err or "quota" in err:
                time.sleep(2)
                continue
            return f"خطأ: {str(e)}"
    return "تم تجاوز حد الاستخدام. حاول مرة أخرى بعد دقيقة."


def ai_analyze_results(query, results_df, top_n=10):
    sample = results_df.head(top_n)
    titles = "\n".join(
        f"- {r['العنوان']} ({r['الجامعة']}, {int(r['السنة']) if pd.notna(r['السنة']) else 'غ.م'})"
        for _, r in sample.iterrows()
    )
    prompt = f"""أنت خبير أكاديمي متخصص في تحليل الأبحاث العلمية العراقية.
المستخدم بحث عن: "{query}"
وهذه أبرز النتائج:
{titles}

قدم تحليلاً أكاديمياً مختصراً يتضمن:
1. ملخص الاتجاه البحثي
2. الفجوات البحثية
3. مقترحات لعناوين رسائل جديدة (3 عناوين)

أجب بالعربية بشكل مختصر واحترافي."""
    return ai_analyze(prompt)


def ai_suggest_topics(department, university=None):
    context = f"قسم {department}"
    if university:
        context += f" في {university}"
    prompt = f"""أنت مستشار أكاديمي متخصص.
اقترح 5 مواضيع بحثية حديثة ومبتكرة لطلبة الدراسات العليا في {context} في العراق.
لكل موضوع اذكر: عنوان الرسالة المقترح، سبب أهمية الموضوع، المنهج البحثي المقترح.
أجب بالعربية بشكل مختصر."""
    return ai_analyze(prompt)


# ════════════════════════════════════════════════════════════════
#  LOAD DATA
# ════════════════════════════════════════════════════════════════
data = load_data()
vectorizer, tfidf_matrix = build_search_engine(data["نص_بحث"].tolist())
keyword_counts = extract_all_keywords(data["العنوان"].tolist())

# ════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## أدوات البحث الأكاديمي")
    st.markdown("---")
    st.markdown("### محرك الذكاء الاصطناعي")

    groq_key = st.text_input("مفتاح Groq API", type="password", placeholder="أدخل المفتاح...")
    if groq_key:
        st.session_state["groq_key"] = groq_key
        sel_model = st.selectbox("نموذج Groq:", GROQ_MODELS, index=0)
        st.session_state["groq_model"] = sel_model
        st.success(f"Groq متصل — {sel_model}")
    else:
        st.caption("أضف مفتاح [Groq API](https://console.groq.com/keys) المجاني")
        st.info("المحرك المحلي النشط:\n\n- NLP عربي + TF-IDF دلالي\n- نمذجة المواضيع LDA\n- اكتشاف الرسائل المشابهة")

    st.markdown("---")
    st.markdown("### فلاتر متقدمة")

    universities = sorted([u for u in data["الجامعة"].unique() if u and u != "nan"])
    sel_univ = st.multiselect("الجامعة", universities)
    if sel_univ:
        colleges_pool = data[data["الجامعة"].isin(sel_univ)]
    else:
        colleges_pool = data
    colleges = sorted([c for c in colleges_pool["الكلية"].unique() if c and c != "nan"])
    sel_college = st.multiselect("الكلية", colleges)
    if sel_college:
        depts_pool = data[data["الكلية"].isin(sel_college)]
    else:
        depts_pool = colleges_pool
    depts = sorted([d for d in depts_pool["القسم"].unique() if d and d != "nan"])
    sel_dept = st.multiselect("القسم", depts)
    degrees = sorted([d for d in data["الشهادة"].unique() if d and d != "nan"])
    sel_degree = st.multiselect("نوع الشهادة", degrees)
    valid_years = data["السنة"].dropna()
    if len(valid_years) > 0:
        min_y, max_y = int(valid_years.min()), int(valid_years.max())
        year_range = st.slider("نطاق السنوات", min_y, max_y, (min_y, max_y))
    else:
        year_range = (1990, 2025)
    per_page = st.selectbox("نتائج لكل صفحة", [10, 20, 50], index=1)

# ════════════════════════════════════════════════════════════════
#  FILTERS
# ════════════════════════════════════════════════════════════════
filtered = data.copy()
if sel_univ:
    filtered = filtered[filtered["الجامعة"].isin(sel_univ)]
if sel_college:
    filtered = filtered[filtered["الكلية"].isin(sel_college)]
if sel_dept:
    filtered = filtered[filtered["القسم"].isin(sel_dept)]
if sel_degree:
    filtered = filtered[filtered["الشهادة"].isin(sel_degree)]
filtered = filtered[
    (filtered["السنة"].isna()) | (filtered["السنة"].between(year_range[0], year_range[1]))
]

# ════════════════════════════════════════════════════════════════
#  HEADER
# ════════════════════════════════════════════════════════════════
st.markdown("""
<div class="academic-header">
    <h1>المنصة الأكاديمية الذكية</h1>
    <div class="subtitle">Iraqi Academic Theses and Dissertations — AI-Powered Search Engine</div>
    <div class="gold-line"></div>
</div>
""", unsafe_allow_html=True)

total = len(data)
univ_count = data["الجامعة"].nunique()
college_count = data["الكلية"].nunique()
filtered_count = len(filtered)

st.markdown(f"""
<div class="stats-grid">
    <div class="stat-box"><div class="num">{total:,}</div><div class="lbl">رسالة وأطروحة</div></div>
    <div class="stat-box"><div class="num">{univ_count}</div><div class="lbl">جامعة ومعهد</div></div>
    <div class="stat-box"><div class="num">{college_count}</div><div class="lbl">كلية وقسم علمي</div></div>
    <div class="stat-box"><div class="num">{filtered_count:,}</div><div class="lbl">نتيجة متاحة</div></div>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
#  TABS
# ════════════════════════════════════════════════════════════════
tab_search, tab_add, tab_ai, tab_discover, tab_stats, tab_compare, tab_export = st.tabs([
    "البحث الأكاديمي",
    "إضافة رسالة",
    "المساعد الذكي",
    "اكتشاف المعرفة",
    "لوحة الإحصائيات",
    "مقارنة الجامعات",
    "التصدير والاقتباس",
])


# ════════════════════════════════════════════════════════════════
#  TAB: SEARCH
# ════════════════════════════════════════════════════════════════
with tab_search:
    st.markdown('<div class="search-section">', unsafe_allow_html=True)
    st.markdown('<h2 style="font-family:Amiri,serif;color:#1B2A4A;margin:0 0 1rem">البحث الذكي في الرسائل والأطروحات</h2>', unsafe_allow_html=True)
    top_keywords = [w for w, c in keyword_counts.most_common(20) if len(w) > 3]
    st.caption(f"كلمات مقترحة: {' | '.join(top_keywords[:10])}")
    col_s, col_b = st.columns([5, 1])
    with col_s:
        query = st.text_input("بحث", placeholder="ابحث بالعنوان، اسم الباحث، أو التخصص...", label_visibility="collapsed")
    with col_b:
        search_clicked = st.button("بحث", use_container_width=True, type="primary")
    st.markdown('<span class="ai-engine">NLP عربي متقدم + TF-IDF دلالي + تطابق متعدد الحقول</span>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if query:
        filtered_indices = set(filtered.index)
        indices, scores = smart_search(query, vectorizer, tfidf_matrix, data, top_n=500)
        result_pairs = [(i, s) for i, s in zip(indices, scores) if i in filtered_indices]
        if result_pairs:
            result_idx, result_scores = zip(*result_pairs)
            results = data.loc[list(result_idx)].copy()
            results["score"] = list(result_scores)
        else:
            results = pd.DataFrame()

        total_results = len(results)
        st.markdown(f'<div class="section-title">نتائج البحث — {total_results} رسالة</div>', unsafe_allow_html=True)

        if total_results > 0:
            st.session_state["search_results"] = results
            total_pages = max(1, (total_results + per_page - 1) // per_page)
            if "page" not in st.session_state:
                st.session_state.page = 1
            if st.session_state.page > total_pages:
                st.session_state.page = 1
            c1, c2, c3 = st.columns([1, 3, 1])
            with c1:
                if st.button("السابق", disabled=st.session_state.page <= 1, key="prev"):
                    st.session_state.page -= 1
                    st.rerun()
            with c2:
                st.markdown(f'<div class="page-nav">صفحة {st.session_state.page} من {total_pages}</div>', unsafe_allow_html=True)
            with c3:
                if st.button("التالي", disabled=st.session_state.page >= total_pages, key="next"):
                    st.session_state.page += 1
                    st.rerun()

            start = (st.session_state.page - 1) * per_page
            page_results = results.iloc[start:start + per_page]

            for _, row in page_results.iterrows():
                score_pct = min(100, int(row["score"] * 100))
                year = int(row["السنة"]) if pd.notna(row["السنة"]) else "غ.م"
                author = row["اسم الباحث"] if row["اسم الباحث"] else "غير محدد"
                title_html = str(row["العنوان"])
                for word in query.split():
                    if len(word) > 1:
                        title_html = re.sub(
                            f"({re.escape(word)})",
                            r'<mark style="background:#FFF3CD;padding:1px 4px;border-radius:3px;font-weight:700">\1</mark>',
                            title_html, flags=re.IGNORECASE,
                        )

                similar_list = []
                actual_idx = row.name
                if actual_idx < tfidf_matrix.shape[0]:
                    sim_idx, sim_scores = find_similar(actual_idx, tfidf_matrix, top_n=3)
                    for si, ss in zip(sim_idx, sim_scores):
                        if ss > 0.05:
                            similar_list.append((data.loc[si, "العنوان"][:80], data.loc[si, "اسم الباحث"]))

                similar_section = ""
                if similar_list:
                    items = "".join(
                        f'<p style="font-size:0.85rem;color:#2D2D2D;padding:4px 0;border-bottom:1px solid #F0EDE8;margin:0">{t}... — <em>{a}</em></p>'
                        for t, a in similar_list
                    )
                    similar_section = (
                        '<div style="background:#FAFAF8;border-radius:10px;padding:1rem;margin-top:0.8rem;border:1px dashed #E5E2DB">'
                        '<h4 style="color:#1B2A4A;margin:0 0 0.5rem;font-size:0.9rem">رسائل مشابهة</h4>'
                        f'{items}</div>'
                    )

                st.markdown(
                    f'<div class="thesis-card">'
                    f'<div class="relevance-bar"><div class="bar-bg"><div class="bar-fill" style="width:{score_pct}%"></div></div>'
                    f'<span class="score-text">{score_pct}%</span></div>'
                    f'<div class="citation"><span class="author">{author}</span> '
                    f'<span class="year">({year})</span><br>'
                    f'<span class="title">{title_html}</span></div>'
                    f'<div class="meta-row">'
                    f'<span class="field-tag univ">{row["الجامعة"]}</span>'
                    f'<span class="field-tag college">{row["الكلية"]}</span>'
                    f'<span class="field-tag dept">{row["القسم"]}</span>'
                    f'<span class="field-tag degree">{row["الشهادة"]}</span></div>'
                    f'{similar_section}</div>',
                    unsafe_allow_html=True,
                )

            if "groq_key" in st.session_state and total_results > 0:
                st.markdown("---")
                if st.button("تحليل النتائج بالذكاء الاصطناعي", type="primary", use_container_width=True, key="analyze_results"):
                    with st.spinner("جاري التحليل..."):
                        analysis = ai_analyze_results(query, results)
                    if analysis:
                        st.markdown(
                            f'<div style="background:linear-gradient(135deg,#1B2A4A,#2C4066);color:white;padding:1.5rem;border-radius:14px;line-height:2;margin-top:1rem">'
                            f'<h3 style="color:#C8A951;margin:0 0 1rem">تحليل Groq AI</h3>'
                            f'<div style="white-space:pre-wrap">{analysis}</div></div>',
                            unsafe_allow_html=True,
                        )
        else:
            st.warning("لم يتم العثور على نتائج. جرّب كلمات مختلفة أو قلّل الفلاتر.")
    else:
        st.markdown('<div class="section-title">مرحباً بك في المنصة الأكاديمية الذكية</div>', unsafe_allow_html=True)
        st.markdown("""
**كيف تستخدم البحث الذكي:**
- ابحث بالعنوان: `الذكاء الاصطناعي في التعليم`
- ابحث باسم الباحث: `محمد علي`
- ابحث بالتخصص: `علوم الحاسوب`
- المحرك يفهم الاختلافات في الهمزات والتشكيل تلقائياً
        """)
        st.markdown("### عينة عشوائية من الرسائل")
        sample = filtered.sample(min(5, len(filtered)), random_state=42)
        for _, row in sample.iterrows():
            year = int(row["السنة"]) if pd.notna(row["السنة"]) else "غ.م"
            st.markdown(
                f'<div class="thesis-card"><div class="citation">'
                f'<span class="author">{row["اسم الباحث"]}</span> '
                f'<span class="year">({year})</span><br>'
                f'<span class="title">{row["العنوان"]}</span></div>'
                f'<div class="meta-row">'
                f'<span class="field-tag univ">{row["الجامعة"]}</span>'
                f'<span class="field-tag college">{row["الكلية"]}</span>'
                f'<span class="field-tag dept">{row["القسم"]}</span>'
                f'<span class="field-tag degree">{row["الشهادة"]}</span></div></div>',
                unsafe_allow_html=True,
            )


# ════════════════════════════════════════════════════════════════
#  TAB: ADD THESIS (with admin login)
# ════════════════════════════════════════════════════════════════
with tab_add:
    st.markdown('<div class="section-title">إضافة رسالة جامعية جديدة</div>', unsafe_allow_html=True)

    ADMIN_USER = "admin"
    ADMIN_PASS = "admin2024"

    if "admin_logged_in" not in st.session_state:
        st.session_state["admin_logged_in"] = False

    if not st.session_state["admin_logged_in"]:
        st.markdown(
            '<div class="login-card"><h3>لوحة تحكم المشرف</h3>'
            '<p>سجّل الدخول لإضافة أو تعديل الرسائل الجامعية</p></div>',
            unsafe_allow_html=True,
        )
        with st.form("login_form"):
            col_login = st.columns([1, 2, 1])[1]
            with col_login:
                login_user = st.text_input("اسم المستخدم", placeholder="أدخل اسم المستخدم")
                login_pass = st.text_input("كلمة المرور", type="password", placeholder="أدخل كلمة المرور")
                login_btn = st.form_submit_button("تسجيل الدخول", type="primary", use_container_width=True)
            if login_btn:
                if login_user == ADMIN_USER and login_pass == ADMIN_PASS:
                    st.session_state["admin_logged_in"] = True
                    st.rerun()
                else:
                    st.error("اسم المستخدم أو كلمة المرور غير صحيحة")
    else:
        col_logout = st.columns([4, 1])
        with col_logout[1]:
            if st.button("تسجيل الخروج", key="logout_btn"):
                st.session_state["admin_logged_in"] = False
                st.rerun()
        st.success("مرحباً بك، مشرف النظام")

        st.markdown("#### تحميل ملف Excel جديد")
        uploaded = st.file_uploader("استبدال قاعدة البيانات بملف Excel جديد:", type=["xlsx"], key="upload_xl")
        if uploaded:
            if st.button("رفع واستبدال القاعدة", type="primary"):
                with open("فهرس الرسائل الجامعية.xlsx", "wb") as f:
                    f.write(uploaded.getvalue())
                st.cache_data.clear()
                st.success("تم تحديث قاعدة البيانات بنجاح! أعد تحميل الصفحة.")

        st.markdown("---")

        with st.form("add_thesis_form", clear_on_submit=True):
            st.markdown("#### بيانات الرسالة")
            add_title = st.text_input("عنوان الرسالة *", placeholder="أدخل عنوان الرسالة كاملاً...")
            add_author = st.text_input("اسم الباحث *", placeholder="الاسم الثلاثي للباحث...")
            col_u, col_c = st.columns(2)
            with col_u:
                existing_univs = sorted([u for u in data["الجامعة"].unique() if u and u != "nan"])
                add_univ_select = st.selectbox("الجامعة *", ["— اختر —"] + existing_univs + ["إدخال يدوي"])
            with col_c:
                if add_univ_select == "إدخال يدوي":
                    add_univ = st.text_input("اسم الجامعة:", placeholder="أدخل اسم الجامعة...")
                elif add_univ_select != "— اختر —":
                    add_univ = add_univ_select
                else:
                    add_univ = ""
            col_cl, col_d = st.columns(2)
            with col_cl:
                add_college = st.text_input("الكلية *", placeholder="مثال: التربية، الآداب، العلوم...")
            with col_d:
                add_dept = st.text_input("القسم *", placeholder="مثال: اللغة العربية، الرياضيات...")
            col_y, col_deg = st.columns(2)
            with col_y:
                add_year = st.number_input("سنة التخرج *", min_value=1980, max_value=2030, value=2024)
            with col_deg:
                add_degree = st.selectbox("نوع الشهادة *", ["ماجستير", "دكتوراه"])
            st.markdown("---")
            submitted = st.form_submit_button("حفظ الرسالة", type="primary", use_container_width=True)
            if submitted:
                if not add_title.strip():
                    st.error("يرجى إدخال عنوان الرسالة")
                elif not add_author.strip():
                    st.error("يرجى إدخال اسم الباحث")
                elif not add_univ.strip():
                    st.error("يرجى اختيار أو إدخال الجامعة")
                elif not add_college.strip():
                    st.error("يرجى إدخال الكلية")
                elif not add_dept.strip():
                    st.error("يرجى إدخال القسم")
                else:
                    try:
                        from openpyxl import load_workbook
                        wb = load_workbook("فهرس الرسائل الجامعية.xlsx")
                        ws = wb["محافظات"]
                        ws.append([add_title.strip(), add_author.strip(), add_univ.strip(),
                                   add_college.strip(), add_dept.strip(), add_year, add_degree, None, None])
                        wb.save("فهرس الرسائل الجامعية.xlsx")
                        wb.close()
                        st.cache_data.clear()
                        st.success("تم حفظ الرسالة بنجاح!")
                        st.markdown(
                            f'<div class="thesis-card"><div class="citation">'
                            f'<span class="author">{add_author.strip()}</span> '
                            f'<span class="year">({add_year})</span><br>'
                            f'<span class="title">{add_title.strip()}</span></div>'
                            f'<div class="meta-row">'
                            f'<span class="field-tag univ">{add_univ.strip()}</span>'
                            f'<span class="field-tag college">{add_college.strip()}</span>'
                            f'<span class="field-tag dept">{add_dept.strip()}</span>'
                            f'<span class="field-tag degree">{add_degree}</span></div></div>',
                            unsafe_allow_html=True,
                        )
                        st.info("أعد تحميل الصفحة لرؤية الرسالة في نتائج البحث")
                    except Exception as e:
                        st.error(f"حدث خطأ أثناء الحفظ: {str(e)}")


# ════════════════════════════════════════════════════════════════
#  TAB: AI ASSISTANT
# ════════════════════════════════════════════════════════════════
with tab_ai:
    st.markdown('<div class="section-title">المساعد الأكاديمي الذكي — Groq</div>', unsafe_allow_html=True)

    if "groq_key" not in st.session_state:
        st.warning("أدخل مفتاح Groq API في الشريط الجانبي لتفعيل هذه الميزة.")
        st.markdown("""
**كيف تحصل على مفتاح مجاني:**
1. اذهب إلى [Google AI Studio](https://aistudio.google.com/apikey)
2. سجّل الدخول بحساب Google
3. انقر على Create API Key
4. انسخ المفتاح وألصقه في الشريط الجانبي
        """)
    else:
        ai_col1, ai_col2 = st.columns(2)
        with ai_col1:
            st.markdown("#### اقتراح مواضيع بحثية")
            st.caption("اختر القسم والجامعة وسيقترح المساعد مواضيع بحثية مبتكرة")
            all_depts_ai = sorted([d for d in data["القسم"].unique() if d and d != "nan" and len(d) > 2])
            sel_dept_ai = st.selectbox("اختر القسم:", all_depts_ai, key="ai_dept")
            all_univs_ai = sorted([u for u in data["الجامعة"].unique() if u and u != "nan"])
            sel_univ_ai = st.selectbox("اختر الجامعة (اختياري):", ["— الكل —"] + all_univs_ai, key="ai_univ")
            univ_val = sel_univ_ai if sel_univ_ai != "— الكل —" else None
            if st.button("اقترح مواضيع", type="primary", key="suggest_btn"):
                with st.spinner("جاري التحليل..."):
                    suggestions = ai_suggest_topics(sel_dept_ai, univ_val)
                if suggestions:
                    st.markdown(suggestions)

        with ai_col2:
            st.markdown("#### المستشار الأكاديمي")
            st.caption("اسأل المساعد أي سؤال أكاديمي عن الرسائل الجامعية العراقية")
            user_question = st.text_area("اكتب سؤالك:", placeholder="مثال: ما أفضل منهج بحثي لدراسة تأثير التعليم الإلكتروني؟", height=120, key="ai_question")
            if st.button("اسأل المساعد", type="primary", key="ask_btn"):
                if user_question.strip():
                    top_depts = data["القسم"].value_counts().head(10).to_dict()
                    top_univs = data["الجامعة"].value_counts().head(10).to_dict()
                    prompt = f"""أنت مستشار أكاديمي خبير في الجامعات العراقية. لديك قاعدة بيانات تضم {len(data)} رسالة جامعية.
الأقسام الأكثر نشاطاً: {top_depts}
الجامعات الأكبر: {top_univs}

سؤال الباحث: {user_question}

أجب بالعربية بشكل أكاديمي احترافي ومفصّل."""
                    with st.spinner("جاري التحليل..."):
                        answer = ai_analyze(prompt, max_tokens=2048)
                    if answer:
                        st.markdown(answer)

        st.markdown("---")
        st.markdown("#### تحليل الفجوات البحثية بالذكاء الاصطناعي")
        gap_dept = st.selectbox("اختر قسماً لتحليل فجواته:", all_depts_ai, key="gap_dept")
        if st.button("حلّل الفجوات", key="gap_btn"):
            dept_theses = data[data["القسم"] == gap_dept].head(30)
            titles_list = "\n".join(f"- {r['العنوان']}" for _, r in dept_theses.iterrows())
            prompt = f"""أنت خبير في تحليل الفجوات البحثية.
قسم: {gap_dept}
عدد الرسائل: {len(data[data['القسم'] == gap_dept])}
عينة من العناوين:
{titles_list}

حلّل الفجوات البحثية:
1. المواضيع المكررة كثيراً
2. المواضيع الحديثة الغائبة عالمياً ولم تُدرس محلياً
3. اقترح 5 عناوين لرسائل تسد هذه الفجوات
أجب بالعربية."""
            with st.spinner("جاري تحليل الفجوات..."):
                gap_analysis = ai_analyze(prompt, max_tokens=2048)
            if gap_analysis:
                st.markdown(gap_analysis)


# ════════════════════════════════════════════════════════════════
#  TAB: KNOWLEDGE DISCOVERY
# ════════════════════════════════════════════════════════════════
with tab_discover:
    st.markdown('<div class="section-title">اكتشاف المعرفة والاتجاهات البحثية</div>', unsafe_allow_html=True)

    disc_col1, disc_col2 = st.columns(2)

    with disc_col1:
        st.markdown("#### سحابة الكلمات المفتاحية")
        top_words_wc = dict(keyword_counts.most_common(150))
        if top_words_wc:
            reshaped_words = {}
            for word, freq in top_words_wc.items():
                reshaped = arabic_reshaper.reshape(word)
                bidi_text = get_display(reshaped)
                reshaped_words[bidi_text] = freq
            wc = WordCloud(
                width=700, height=400, background_color="white",
                font_path="C:/Windows/Fonts/tahoma.ttf",
                max_words=120, colormap="viridis", prefer_horizontal=0.7, min_font_size=10,
            )
            wc.generate_from_frequencies(reshaped_words)
            fig_wc, ax_wc = plt.subplots(figsize=(7, 4))
            ax_wc.imshow(wc, interpolation="bilinear")
            ax_wc.axis("off")
            plt.tight_layout(pad=0)
            st.pyplot(fig_wc)
            plt.close(fig_wc)

    with disc_col2:
        st.markdown("#### التصنيف التلقائي للمواضيع (LDA)")
        with st.spinner("جاري تحليل المواضيع..."):
            topics, dominant_topics = run_topic_modeling(data["عنوان_معالج"].tolist(), n_topics=8)
        for t in topics:
            st.markdown(f"**موضوع {t['id'] + 1}:** {' ، '.join(t['words'])}")

    st.markdown("---")
    st.markdown("#### الاتجاهات البحثية عبر الزمن")
    trend_keyword = st.text_input("أدخل كلمة لتتبع اتجاهها البحثي:", value="الذكاء", key="trend_kw")
    if trend_keyword:
        mask = data["العنوان"].str.contains(trend_keyword, case=False, na=False, regex=False)
        trend_data = data[mask].dropna(subset=["السنة"]).groupby("السنة").size().reset_index(name="العدد").sort_values("السنة")
        if len(trend_data) > 0:
            fig = px.line(trend_data, x="السنة", y="العدد", title=f'اتجاه البحث في "{trend_keyword}" عبر السنوات', markers=True, color_discrete_sequence=["#1B2A4A"])
            fig.update_layout(font=dict(family="Tajawal", size=13), height=350, plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f'لا توجد نتائج تحتوي على "{trend_keyword}"')

    st.markdown("---")
    st.markdown("#### تحليل الفجوات البحثية")
    st.caption("الأقسام الأقل إنتاجاً بحثياً — فرص بحثية محتملة")
    dept_counts = data["القسم"].value_counts()
    low_depts = dept_counts[dept_counts.between(5, 30)].head(15).reset_index()
    low_depts.columns = ["القسم", "عدد الرسائل"]
    fig = px.bar(low_depts, x="القسم", y="عدد الرسائل", title="أقسام بحاجة لمزيد من الدراسات", color="عدد الرسائل", color_continuous_scale=["#FFF3CD", "#E67E22"])
    fig.update_layout(font=dict(family="Tajawal", size=13), height=400, showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════
#  TAB: STATISTICS
# ════════════════════════════════════════════════════════════════
with tab_stats:
    st.markdown('<div class="section-title">لوحة الإحصائيات الأكاديمية</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        uc = filtered["الجامعة"].value_counts().head(15).reset_index()
        uc.columns = ["الجامعة", "العدد"]
        fig = px.bar(uc, x="العدد", y="الجامعة", orientation="h", title="إنتاج الرسائل حسب الجامعة (أعلى 15)", color="العدد", color_continuous_scale=["#EEF0F7", "#1B2A4A"])
        fig.update_layout(font=dict(family="Tajawal", size=13), height=480, yaxis=dict(autorange="reversed"), showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        dc = filtered["الشهادة"].value_counts().reset_index()
        dc.columns = ["الشهادة", "العدد"]
        fig = px.pie(dc, values="العدد", names="الشهادة", title="توزيع الرسائل حسب نوع الشهادة", color_discrete_sequence=["#1B2A4A", "#C8A951", "#1a936f", "#81c3d7"], hole=0.45)
        fig.update_layout(font=dict(family="Tajawal", size=13), height=480)
        st.plotly_chart(fig, use_container_width=True)

    yc = filtered.dropna(subset=["السنة"]).groupby("السنة").size().reset_index(name="العدد").sort_values("السنة")
    fig = px.area(yc, x="السنة", y="العدد", title="تطور الإنتاج البحثي عبر السنوات", color_discrete_sequence=["#1B2A4A"])
    fig.update_layout(font=dict(family="Tajawal", size=13), height=380, plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        cc = filtered["الكلية"].value_counts().head(12).reset_index()
        cc.columns = ["الكلية", "العدد"]
        fig = px.bar(cc, x="الكلية", y="العدد", title="أكثر الكليات إنتاجاً (أعلى 12)", color="العدد", color_continuous_scale=["#E8F5EE", "#1a936f"])
        fig.update_layout(font=dict(family="Tajawal", size=13), height=420, showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    with col_d:
        dpc = filtered["القسم"].value_counts().head(12).reset_index()
        dpc.columns = ["القسم", "العدد"]
        fig = px.bar(dpc, x="القسم", y="العدد", title="أكثر الأقسام إنتاجاً (أعلى 12)", color="العدد", color_continuous_scale=["#FFF8E7", "#C8A951"])
        fig.update_layout(font=dict(family="Tajawal", size=13), height=420, showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### خريطة حرارية: السنة × الشهادة")
    hm = filtered.dropna(subset=["السنة"]).groupby(["السنة", "الشهادة"]).size().reset_index(name="العدد")
    if len(hm) > 0:
        pivot = hm.pivot_table(index="الشهادة", columns="السنة", values="العدد", fill_value=0)
        fig = px.imshow(pivot, title="توزيع الرسائل: السنة × الشهادة", color_continuous_scale="Blues", aspect="auto")
        fig.update_layout(font=dict(family="Tajawal", size=13), height=300)
        st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════
#  TAB: UNIVERSITY COMPARISON
# ════════════════════════════════════════════════════════════════
with tab_compare:
    st.markdown('<div class="section-title">مقارنة أكاديمية بين الجامعات</div>', unsafe_allow_html=True)
    all_univs = sorted([u for u in data["الجامعة"].unique() if u and u != "nan"])
    compare_univs = st.multiselect("اختر الجامعات للمقارنة:", all_univs, default=all_univs[:3] if len(all_univs) >= 3 else all_univs[:2])

    if len(compare_univs) >= 2:
        comp_data = data[data["الجامعة"].isin(compare_univs)]
        comp_totals = comp_data.groupby("الجامعة").size().reset_index(name="العدد")
        fig = px.bar(comp_totals, x="الجامعة", y="العدد", title="إجمالي الرسائل لكل جامعة", color="الجامعة", color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(font=dict(family="Tajawal", size=13), height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        comp_yearly = comp_data.dropna(subset=["السنة"]).groupby(["السنة", "الجامعة"]).size().reset_index(name="العدد")
        fig = px.line(comp_yearly, x="السنة", y="العدد", color="الجامعة", title="مقارنة الإنتاج البحثي عبر السنوات", markers=True)
        fig.update_layout(font=dict(family="Tajawal", size=13), height=400)
        st.plotly_chart(fig, use_container_width=True)

        comp_deg = comp_data.groupby(["الجامعة", "الشهادة"]).size().reset_index(name="العدد")
        fig = px.bar(comp_deg, x="الجامعة", y="العدد", color="الشهادة", title="مقارنة توزيع الشهادات", barmode="group", color_discrete_sequence=["#1B2A4A", "#C8A951", "#1a936f"])
        fig.update_layout(font=dict(family="Tajawal", size=13), height=400)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### ملخص المقارنة")
        summary_rows = []
        for u in compare_univs:
            u_data = comp_data[comp_data["الجامعة"] == u]
            summary_rows.append({
                "الجامعة": u,
                "إجمالي الرسائل": len(u_data),
                "ماجستير": len(u_data[u_data["الشهادة"].str.contains("ماجستير", na=False)]),
                "دكتوراه": len(u_data[u_data["الشهادة"].str.contains("دكتوراه", na=False)]),
                "عدد الكليات": u_data["الكلية"].nunique(),
                "عدد الأقسام": u_data["القسم"].nunique(),
            })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
    else:
        st.info("اختر جامعتين على الأقل للمقارنة")


# ════════════════════════════════════════════════════════════════
#  TAB: EXPORT
# ════════════════════════════════════════════════════════════════
with tab_export:
    st.markdown('<div class="section-title">التصدير والاقتباس الأكاديمي</div>', unsafe_allow_html=True)

    export_col1, export_col2 = st.columns(2)

    with export_col1:
        st.markdown("#### تصدير البيانات")
        export_choice = st.radio("اختر مصدر التصدير:", ["نتائج البحث الحالية", "جميع البيانات (مع الفلاتر)"])
        if export_choice == "نتائج البحث الحالية":
            export_df = st.session_state.get("search_results", pd.DataFrame())
            if export_df.empty:
                st.warning("قم بالبحث أولاً من تبويب البحث الأكاديمي.")
        else:
            export_df = filtered.copy()

        if not export_df.empty:
            st.success(f"جاهز لتصدير {len(export_df)} سجل")
            export_cols = ["العنوان", "اسم الباحث", "الجامعة", "الكلية", "القسم", "السنة", "الشهادة"]
            export_data = export_df[[c for c in export_cols if c in export_df.columns]]

            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                export_data.to_excel(writer, index=False, sheet_name="النتائج")
                wb = writer.book
                ws = writer.sheets["النتائج"]
                hfmt = wb.add_format({"bold": True, "bg_color": "#1B2A4A", "font_color": "#FFFFFF", "border": 1, "align": "center", "font_name": "Arial", "font_size": 12})
                for i, cn in enumerate(export_data.columns):
                    ws.write(0, i, cn, hfmt)
                    col_lens = export_data[cn].astype(str).str.len()
                    ml = max(col_lens.max() if len(col_lens) > 0 else 0, len(cn)) + 2
                    ws.set_column(i, i, min(ml, 60))
                ws.right_to_left()

            st.download_button("تحميل Excel", buffer.getvalue(), "نتائج_أكاديمية.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.document",
                               type="primary", use_container_width=True)
            csv = export_data.to_csv(index=False).encode("utf-8-sig")
            st.download_button("تحميل CSV", csv, "نتائج_أكاديمية.csv", "text/csv", use_container_width=True)

    with export_col2:
        st.markdown("#### مولد الاقتباسات الأكاديمية (APA)")
        st.caption("ابحث عن رسالة لتوليد اقتباسها بنمط APA")
        cite_query = st.text_input("ابحث عن رسالة للاقتباس:", key="cite_q")
        if cite_query:
            norm_q = normalize_arabic(cite_query)
            mask = data["عنوان_معالج"].str.contains(norm_q, case=False, na=False, regex=False)
            cite_results = data[mask].head(10)
            if len(cite_results) > 0:
                for _, row in cite_results.iterrows():
                    citation = generate_citation_apa(row)
                    st.markdown(f'<div class="cite-box">{citation}</div>', unsafe_allow_html=True)
                    st.markdown("")
            else:
                st.info("لا توجد نتائج. جرّب كلمات أخرى.")


# ════════════════════════════════════════════════════════════════
#  FOOTER
# ════════════════════════════════════════════════════════════════
st.markdown("""
<div class="academic-footer">
    <span class="gold">إعداد وتصميم:</span>
    أ.د علي عبد الصمد خضير / كلية الآداب / جامعة البصرة
    &nbsp;|&nbsp;
    أ.د علي الحر لازم / كلية الآداب / الجامعة المستنصرية
    <br><br>
    <small>المنصة الأكاديمية الذكية — Iraqi Academic AI Platform</small>
</div>
""", unsafe_allow_html=True)
