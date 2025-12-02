import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# Настройки страницы
st.set_page_config(
    page_title="K-Factor Analytics",
    page_icon="📊",
    layout="wide"
)

# Заголовок
st.title("📊 K-Factor Analytics Dashboard")
st.markdown("Анализ влияния трафовых пользователей (UA) на органику")

# Загрузка данных
@st.cache_data
def load_data():
    df = pd.read_csv('test_data.csv')
    df['cohort'] = pd.to_datetime(df['cohort'])
    df['month'] = df['cohort'].dt.to_period('M')
    df['day_of_week'] = df['cohort'].dt.dayofweek
    df['year'] = df['cohort'].dt.year
    return df

df = load_data()

# Pivot таблица
pivot = df.pivot_table(index='cohort', columns='ms', values=['user_cnt', 'gross'], aggfunc='sum').reset_index()
pivot.columns = ['cohort', 'gross_ORGANIC', 'gross_UA', 'users_ORGANIC', 'users_UA']
pivot['k_factor'] = pivot['users_ORGANIC'] / pivot['users_UA']
pivot['total_users'] = pivot['users_ORGANIC'] + pivot['users_UA']
pivot['arpu_organic'] = pivot['gross_ORGANIC'] / pivot['users_ORGANIC']
pivot['arpu_ua'] = pivot['gross_UA'] / pivot['users_UA']

# Месячные данные
monthly = df.groupby(['month', 'ms']).agg({'user_cnt': 'sum', 'gross': 'sum'}).unstack()
ua_users = monthly['user_cnt']['UA']
organic_users = monthly['user_cnt']['ORGANIC']
k_factor_monthly = organic_users / ua_users

# =============================================================================
# SIDEBAR
# =============================================================================
st.sidebar.header("⚙️ Настройки")

# Фильтр по дате
date_range = st.sidebar.date_input(
    "Период анализа",
    value=(df['cohort'].min(), df['cohort'].max()),
    min_value=df['cohort'].min(),
    max_value=df['cohort'].max()
)

# Фильтр данных
if len(date_range) == 2:
    mask = (pivot['cohort'] >= pd.Timestamp(date_range[0])) & (pivot['cohort'] <= pd.Timestamp(date_range[1]))
    pivot_filtered = pivot[mask]
else:
    pivot_filtered = pivot

# =============================================================================
# ОСНОВНЫЕ МЕТРИКИ
# =============================================================================
st.header("📈 Основные метрики")

col1, col2, col3, col4 = st.columns(4)

total_ua = pivot_filtered['users_UA'].sum()
total_organic = pivot_filtered['users_ORGANIC'].sum()
avg_k_factor = total_organic / total_ua
total_gross = pivot_filtered['gross_UA'].sum() + pivot_filtered['gross_ORGANIC'].sum()

col1.metric("Всего UA", f"{total_ua:,.0f}")
col2.metric("Всего Organic", f"{total_organic:,.0f}")
col3.metric("K-Factor", f"{avg_k_factor:.3f}")
col4.metric("Общий Gross", f"{total_gross:,.0f}")

# =============================================================================
# ЗАДАЧА 1: K-FACTOR
# =============================================================================
st.header("🎯 Задача 1: K-Factor пользователей")

st.markdown(f"""
**K-Factor = Organic / UA = {avg_k_factor:.4f}**

Это означает: на каждого **1 трафового пользователя** приходит **{avg_k_factor:.2f} органических** пользователей.
""")

# График K-Factor по месяцам
fig_k = go.Figure()
fig_k.add_trace(go.Scatter(
    x=k_factor_monthly.index.to_timestamp(),
    y=k_factor_monthly.values,
    mode='lines+markers',
    name='K-Factor',
    line=dict(color='blue', width=2)
))
fig_k.add_hline(y=k_factor_monthly.mean(), line_dash="dash", line_color="red",
                annotation_text=f"Среднее: {k_factor_monthly.mean():.2f}")
fig_k.add_hline(y=1.0, line_dash="dot", line_color="green",
                annotation_text="K=1 (паритет)")
fig_k.update_layout(title="K-Factor по месяцам", xaxis_title="Месяц", yaxis_title="K-Factor")
st.plotly_chart(fig_k, use_container_width=True)

# =============================================================================
# ЗАДАЧА 2: K-FACTOR ДЛЯ ДЕНЕГ
# =============================================================================
st.header("💰 Задача 2: K-Factor для денег")

# Расчёты
arpu_ua_total = pivot_filtered['gross_UA'].sum() / pivot_filtered['users_UA'].sum()
arpu_organic_total = pivot_filtered['gross_ORGANIC'].sum() / pivot_filtered['users_ORGANIC'].sum()
organic_from_ua = total_ua * avg_k_factor
gross_from_organic_via_ua = organic_from_ua * arpu_organic_total
k_factor_money = gross_from_organic_via_ua / pivot_filtered['gross_UA'].sum()

col1, col2, col3 = st.columns(3)
col1.metric("ARPU UA", f"{arpu_ua_total:.4f}")
col2.metric("ARPU Organic", f"{arpu_organic_total:.4f}")
col3.metric("K-Factor Money", f"{k_factor_money:.4f}")

st.markdown(f"""
**Интерпретация:** На каждый **1 рубль** выручки от UA пользователей,
дополнительно приходит **{k_factor_money:.2f} рубля** от органики, которую этот траф привёл.
""")

# =============================================================================
# ГРАФИКИ
# =============================================================================
st.header("📊 Визуализация данных")

tab1, tab2, tab3, tab4 = st.tabs(["📈 Динамика", "📅 По дням недели", "🔗 Корреляция", "💵 ARPU"])

with tab1:
    # Динамика пользователей
    fig_users = go.Figure()
    pivot_filtered['users_UA_ma'] = pivot_filtered['users_UA'].rolling(7).mean()
    pivot_filtered['users_ORGANIC_ma'] = pivot_filtered['users_ORGANIC'].rolling(7).mean()

    fig_users.add_trace(go.Scatter(
        x=pivot_filtered['cohort'], y=pivot_filtered['users_UA_ma'],
        name='UA (7-дн. среднее)', line=dict(color='steelblue')
    ))
    fig_users.add_trace(go.Scatter(
        x=pivot_filtered['cohort'], y=pivot_filtered['users_ORGANIC_ma'],
        name='Organic (7-дн. среднее)', line=dict(color='forestgreen')
    ))
    fig_users.update_layout(title="Динамика пользователей", xaxis_title="Дата", yaxis_title="Пользователи")
    st.plotly_chart(fig_users, use_container_width=True)

with tab2:
    # K-Factor по дням недели
    days = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс']
    k_by_dow = []
    for d in range(7):
        ua = df[(df['day_of_week']==d) & (df['ms']=='UA')]['user_cnt'].sum()
        org = df[(df['day_of_week']==d) & (df['ms']=='ORGANIC')]['user_cnt'].sum()
        k_by_dow.append(org/ua)

    colors = ['steelblue']*5 + ['forestgreen']*2
    fig_dow = go.Figure(data=[go.Bar(x=days, y=k_by_dow, marker_color=colors)])
    fig_dow.add_hline(y=np.mean(k_by_dow), line_dash="dash", line_color="red",
                      annotation_text=f"Среднее: {np.mean(k_by_dow):.3f}")
    fig_dow.update_layout(title="K-Factor по дням недели", xaxis_title="День", yaxis_title="K-Factor")
    st.plotly_chart(fig_dow, use_container_width=True)

with tab3:
    # Scatter: UA vs Organic
    fig_scatter = px.scatter(
        pivot_filtered, x='users_UA', y='users_ORGANIC',
        trendline='ols', title='Корреляция: UA → Organic',
        labels={'users_UA': 'UA users', 'users_ORGANIC': 'Organic users'}
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    corr = pivot_filtered['users_UA'].corr(pivot_filtered['users_ORGANIC'])
    st.metric("Корреляция", f"{corr:.3f}")

with tab4:
    # ARPU по месяцам
    arpu_ua_monthly = monthly['gross']['UA'] / monthly['user_cnt']['UA']
    arpu_org_monthly = monthly['gross']['ORGANIC'] / monthly['user_cnt']['ORGANIC']

    fig_arpu = go.Figure()
    fig_arpu.add_trace(go.Scatter(
        x=arpu_ua_monthly.index.to_timestamp(), y=arpu_ua_monthly.values,
        name='ARPU UA', mode='lines+markers', line=dict(color='steelblue')
    ))
    fig_arpu.add_trace(go.Scatter(
        x=arpu_org_monthly.index.to_timestamp(), y=arpu_org_monthly.values,
        name='ARPU Organic', mode='lines+markers', line=dict(color='forestgreen')
    ))
    fig_arpu.update_layout(title="ARPU по месяцам", xaxis_title="Месяц", yaxis_title="ARPU")
    st.plotly_chart(fig_arpu, use_container_width=True)

# =============================================================================
# РЕГРЕССИЯ
# =============================================================================
st.header("🔬 Регрессионный анализ")

# Подготовка данных
data = pd.DataFrame({'UA': pivot['users_UA'].values, 'Organic': pivot['users_ORGANIC'].values})
for lag in range(1, 8):
    data[f'UA_lag_{lag}'] = data['UA'].shift(lag)
data_clean = data.dropna()

# Модель 1: Простая регрессия
X_simple = data_clean[['UA']]
y = data_clean['Organic']
model_simple = LinearRegression().fit(X_simple, y)

# Модель 2: С лагами
X_lags = data_clean[['UA', 'UA_lag_1', 'UA_lag_2', 'UA_lag_3', 'UA_lag_7']]
model_lags = LinearRegression().fit(X_lags, y)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Модель 1: Простая")
    st.markdown(f"""
    **Organic = {model_simple.coef_[0]:.4f} × UA + {model_simple.intercept_:.0f}**

    - K-factor (коэф.): **{model_simple.coef_[0]:.4f}**
    - Базовая органика: **{model_simple.intercept_:.0f}** чел/день
    - R²: **{r2_score(y, model_simple.predict(X_simple)):.4f}**
    """)

with col2:
    st.subheader("Модель 2: С лагами")
    coefs = dict(zip(X_lags.columns, model_lags.coef_))
    st.markdown(f"""
    **Коэффициенты:**
    - UA (сегодня): {coefs['UA']:.4f}
    - UA (вчера): {coefs['UA_lag_1']:.4f}
    - UA (2 дня): {coefs['UA_lag_2']:.4f}
    - UA (3 дня): {coefs['UA_lag_3']:.4f}
    - UA (7 дней): {coefs['UA_lag_7']:.4f}

    **Суммарный K-factor: {sum(model_lags.coef_):.4f}**

    R²: **{r2_score(y, model_lags.predict(X_lags)):.4f}**
    """)

# =============================================================================
# ПРОГНОЗ
# =============================================================================
st.header("🔮 Прогноз")

planned_ua = st.slider("Планируемый UA трафик", min_value=1000, max_value=50000, value=10000, step=1000)

predicted_organic = model_simple.coef_[0] * planned_ua + model_simple.intercept_
predicted_gross_ua = planned_ua * arpu_ua_total
predicted_gross_organic = predicted_organic * arpu_organic_total

col1, col2, col3 = st.columns(3)
col1.metric("Ожидаемая органика", f"{predicted_organic:,.0f}")
col2.metric("Gross от UA", f"{predicted_gross_ua:,.0f}")
col3.metric("Gross от органики", f"{predicted_gross_organic:,.0f}")

# =============================================================================
# АНОМАЛИИ
# =============================================================================
st.header("⚠️ Аномалии")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Топ-5 высокий K-Factor")
    top_k = pivot.nlargest(5, 'k_factor')[['cohort', 'users_UA', 'users_ORGANIC', 'k_factor']]
    top_k['cohort'] = top_k['cohort'].dt.strftime('%Y-%m-%d')
    st.dataframe(top_k, hide_index=True)

with col2:
    st.subheader("Топ-5 низкий K-Factor")
    low_k = pivot.nsmallest(5, 'k_factor')[['cohort', 'users_UA', 'users_ORGANIC', 'k_factor']]
    low_k['cohort'] = low_k['cohort'].dt.strftime('%Y-%m-%d')
    st.dataframe(low_k, hide_index=True)

# =============================================================================
# ДАННЫЕ
# =============================================================================
st.header("📋 Данные")

if st.checkbox("Показать исходные данные"):
    st.dataframe(df)

if st.checkbox("Показать K-Factor по месяцам"):
    monthly_summary = pd.DataFrame({
        'UA users': ua_users.values,
        'Organic users': organic_users.values,
        'K-Factor': k_factor_monthly.values
    }, index=k_factor_monthly.index.astype(str))
    st.dataframe(monthly_summary)

# Footer
st.markdown("---")
st.markdown("📊 **K-Factor Analytics Dashboard** | Создано с помощью Streamlit")
