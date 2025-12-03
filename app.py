import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Настройки страницы
st.set_page_config(
    page_title="K-Factor Analytics",
    page_icon="📊",
    layout="wide"
)

# Заголовок
st.title("📊 K-Factor Analytics Dashboard")
st.markdown("Анализ влияния трафовых пользователей (UA) на органику")

# =============================================================================
# ОПИСАНИЕ ДАННЫХ
# =============================================================================
st.header("📋 Описание данных")
st.markdown("""
**Структура данных:**
- **ms** — тип источника трафика: `ORGANIC` (органические) или `UA` (трафовые/платные)
- **cohort** — дата когорты (день)
- **user_cnt** — количество пользователей
- **gross** — выручка (гросс)
""")

# Загрузка данных
@st.cache_data
def load_data():
    df = pd.read_csv('test_data.csv')
    df['cohort'] = pd.to_datetime(df['cohort'])
    df['month'] = df['cohort'].dt.to_period('M')
    df['day_of_week'] = df['cohort'].dt.dayofweek
    return df

df = load_data()

# Показать пример данных
if st.checkbox("Показать пример данных"):
    st.dataframe(df.head(10))

# Pivot таблица
pivot = df.pivot_table(index='cohort', columns='ms', values=['user_cnt', 'gross'], aggfunc='sum').reset_index()
pivot.columns = ['cohort', 'gross_ORGANIC', 'gross_UA', 'users_ORGANIC', 'users_UA']
pivot['k_factor'] = pivot['users_ORGANIC'] / pivot['users_UA']

# Месячные данные
monthly = df.groupby(['month', 'ms']).agg({'user_cnt': 'sum', 'gross': 'sum'}).unstack()
ua_monthly = monthly['user_cnt']['UA']
organic_monthly = monthly['user_cnt']['ORGANIC']
k_factor_monthly = organic_monthly / ua_monthly

# Расчёт общих метрик
total_ua = pivot['users_UA'].sum()
total_organic = pivot['users_ORGANIC'].sum()
total_gross_ua = pivot['gross_UA'].sum()
total_gross_organic = pivot['gross_ORGANIC'].sum()

# =============================================================================
# ЗАДАЧА 1: K-FACTOR ПОЛЬЗОВАТЕЛЕЙ
# =============================================================================
st.header("🎯 Задача 1: K-Factor пользователей")

st.markdown("""
**Что такое K-Factor?**

K-Factor показывает, сколько органических пользователей приходит на каждого трафового (платного) пользователя.
Это метрика виральности и эффективности UA-кампаний.
""")

# Расчёт K-Factor
k_factor = total_organic / total_ua

st.markdown(f"""
### Формула:
```
K-Factor = Organic_users / UA_users
K-Factor = {total_organic:,.0f} / {total_ua:,.0f} = {k_factor:.4f}
```
""")

col1, col2, col3 = st.columns(3)
col1.metric("Всего UA пользователей", f"{total_ua:,.0f}")
col2.metric("Всего Organic пользователей", f"{total_organic:,.0f}")
col3.metric("K-Factor", f"{k_factor:.4f}")

st.info(f"""
**Интерпретация:** На каждого **1 трафового пользователя** приходит **{k_factor:.2f} органических** пользователей.

Это означает, что UA-кампании приводят дополнительно ~{k_factor*100:.0f}% органических пользователей за счёт виральности,
сарафанного радио и других эффектов.
""")

# =============================================================================
# ЗАДАЧА 2: K-FACTOR ДЛЯ ДЕНЕГ
# =============================================================================
st.header("💰 Задача 2: K-Factor для денег")

st.markdown("""
**Что такое K-Factor для денег?**

Это метрика показывает, сколько дополнительной выручки от органики приносит каждый рубль,
потраченный на привлечение UA-пользователей.
""")

# Расчёт ARPU
arpu_ua = total_gross_ua / total_ua
arpu_organic = total_gross_organic / total_organic

# Органика, приведённая UA
organic_from_ua = total_ua * k_factor  # = total_organic

# Выручка от органики, приведённой UA
gross_from_organic_via_ua = organic_from_ua * arpu_organic

# K-Factor для денег
k_factor_money = gross_from_organic_via_ua / total_gross_ua

st.markdown(f"""
### Расчёт:

**Шаг 1: ARPU (средняя выручка на пользователя)**
```
ARPU_UA = Gross_UA / Users_UA = {total_gross_ua:,.0f} / {total_ua:,.0f} = {arpu_ua:.4f}
ARPU_Organic = Gross_Organic / Users_Organic = {total_gross_organic:,.0f} / {total_organic:,.0f} = {arpu_organic:.4f}
```

**Шаг 2: Органика, приведённая UA**
```
Organic_from_UA = UA_users × K-Factor = {total_ua:,.0f} × {k_factor:.4f} = {organic_from_ua:,.0f}
```

**Шаг 3: Выручка от органики**
```
Gross_from_Organic = Organic_from_UA × ARPU_Organic = {organic_from_ua:,.0f} × {arpu_organic:.4f} = {gross_from_organic_via_ua:,.0f}
```

**Шаг 4: K-Factor Money**
```
K-Factor_Money = Gross_from_Organic / Gross_UA = {gross_from_organic_via_ua:,.0f} / {total_gross_ua:,.0f} = {k_factor_money:.4f}
```
""")

col1, col2, col3, col4 = st.columns(4)
col1.metric("ARPU UA", f"{arpu_ua:.4f}")
col2.metric("ARPU Organic", f"{arpu_organic:.4f}")
col3.metric("Gross от органики", f"{gross_from_organic_via_ua:,.0f}")
col4.metric("K-Factor Money", f"{k_factor_money:.4f}")

st.info(f"""
**Интерпретация:** На каждый **1 рубль** выручки от UA пользователей,
дополнительно приходит **{k_factor_money:.2f} рубля** от органики, которую этот траф привёл.

Общий мультипликатор выручки: **{1 + k_factor_money:.2f}x** (1 рубль UA + {k_factor_money:.2f} рубля органики)
""")

# =============================================================================
# ЗАДАЧА 3: ГРАФИКИ И АНАЛИЗ
# =============================================================================
st.header("📊 Задача 3: Графики и анализ")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 K-Factor по месяцам",
    "📅 По дням недели",
    "🔗 Корреляция",
    "💵 ARPU",
    "🔬 Регрессия",
    "⚠️ Аномалии"
])

# --- TAB 1: K-Factor по месяцам ---
with tab1:
    st.subheader("K-Factor по месяцам")

    fig_k = go.Figure()
    fig_k.add_trace(go.Scatter(
        x=k_factor_monthly.index.to_timestamp(),
        y=k_factor_monthly.values,
        mode='lines+markers',
        name='K-Factor',
        line=dict(color='blue', width=2)
    ))
    fig_k.add_hline(y=k_factor_monthly.mean(), line_dash="dash", line_color="red",
                    annotation_text=f"Среднее: {k_factor_monthly.mean():.3f}")
    fig_k.add_hline(y=1.0, line_dash="dot", line_color="green",
                    annotation_text="K=1 (паритет)")
    fig_k.update_layout(
        title="K-Factor по месяцам",
        xaxis_title="Месяц",
        yaxis_title="K-Factor"
    )
    st.plotly_chart(fig_k, use_container_width=True)

    # Динамика пользователей
    st.subheader("Динамика пользователей")
    pivot['users_UA_ma'] = pivot['users_UA'].rolling(7).mean()
    pivot['users_ORGANIC_ma'] = pivot['users_ORGANIC'].rolling(7).mean()

    fig_users = go.Figure()
    fig_users.add_trace(go.Scatter(
        x=pivot['cohort'], y=pivot['users_UA_ma'],
        name='UA (7-дн. среднее)', line=dict(color='steelblue')
    ))
    fig_users.add_trace(go.Scatter(
        x=pivot['cohort'], y=pivot['users_ORGANIC_ma'],
        name='Organic (7-дн. среднее)', line=dict(color='forestgreen')
    ))
    fig_users.update_layout(title="Динамика пользователей", xaxis_title="Дата", yaxis_title="Пользователи")
    st.plotly_chart(fig_users, use_container_width=True)

    st.markdown(f"""
    **Выводы:**
    - Среднемесячный K-Factor: **{k_factor_monthly.mean():.3f}**
    - Максимум: **{k_factor_monthly.max():.3f}** ({k_factor_monthly.idxmax()})
    - Минимум: **{k_factor_monthly.min():.3f}** ({k_factor_monthly.idxmin()})
    - Наблюдается сезонность: K-Factor выше в осенне-зимний период
    """)

# --- TAB 2: По дням недели ---
with tab2:
    st.subheader("K-Factor по дням недели")

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

    weekend_k = np.mean(k_by_dow[5:])
    weekday_k = np.mean(k_by_dow[:5])

    st.markdown(f"""
    **Выводы:**
    - K-Factor в будни: **{weekday_k:.3f}**
    - K-Factor в выходные: **{weekend_k:.3f}**
    - Разница: **{((weekend_k/weekday_k)-1)*100:.1f}%** {'выше' if weekend_k > weekday_k else 'ниже'} в выходные
    """)

# --- TAB 3: Корреляция ---
with tab3:
    st.subheader("Корреляция UA → Organic")

    fig_scatter = px.scatter(
        pivot, x='users_UA', y='users_ORGANIC',
        trendline='ols', title='Корреляция: UA → Organic',
        labels={'users_UA': 'UA users', 'users_ORGANIC': 'Organic users'}
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Расчёт корреляций
    corr_pearson = pivot['users_UA'].corr(pivot['users_ORGANIC'])
    corr_spearman = pivot['users_UA'].corr(pivot['users_ORGANIC'], method='spearman')
    corr_k_ua = pivot['users_UA'].corr(pivot['k_factor'])

    col1, col2, col3 = st.columns(3)
    col1.metric("Корреляция Пирсона", f"{corr_pearson:.3f}")
    col2.metric("Корреляция Спирмена", f"{corr_spearman:.3f}")
    col3.metric("Корр. UA ↔ K-Factor", f"{corr_k_ua:.3f}")

    st.markdown(f"""
    **Выводы:**
    - **Корреляция Пирсона ({corr_pearson:.3f})**: {'Сильная' if abs(corr_pearson) > 0.7 else 'Умеренная' if abs(corr_pearson) > 0.4 else 'Слабая'} положительная связь между UA и Organic
    - **Корреляция Спирмена ({corr_spearman:.3f})**: Подтверждает монотонную связь (устойчива к выбросам)
    - **Корреляция UA ↔ K-Factor ({corr_k_ua:.3f})**: {'Отрицательная' if corr_k_ua < 0 else 'Положительная'} —
      {'при росте UA, K-Factor снижается (насыщение рынка)' if corr_k_ua < 0 else 'при росте UA, K-Factor растёт'}
    """)

# --- TAB 4: ARPU ---
with tab4:
    st.subheader("ARPU по месяцам")

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

    col1, col2 = st.columns(2)
    col1.metric("Средний ARPU UA", f"{arpu_ua_monthly.mean():.4f}")
    col2.metric("Средний ARPU Organic", f"{arpu_org_monthly.mean():.4f}")

    st.markdown(f"""
    **Что такое ARPU?**

    ARPU (Average Revenue Per User) — средняя выручка на одного пользователя.
    ```
    ARPU = Gross / Users
    ```

    **Выводы:**
    - ARPU UA: **{arpu_ua:.4f}** (в среднем за весь период)
    - ARPU Organic: **{arpu_organic:.4f}**
    - Органические пользователи приносят {'больше' if arpu_organic > arpu_ua else 'меньше'} выручки на человека
    """)

# --- TAB 5: Регрессия ---
with tab5:
    st.subheader("Регрессионный анализ: UA → Organic")

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
        st.markdown("### Модель 1: Простая")
        st.markdown(f"""
        **Organic = {model_simple.coef_[0]:.4f} × UA + {model_simple.intercept_:.0f}**

        - K-factor (коэф.): **{model_simple.coef_[0]:.4f}**
        - Базовая органика: **{model_simple.intercept_:.0f}** чел/день
        - R²: **{r2_score(y, model_simple.predict(X_simple)):.4f}**
        """)

    with col2:
        st.markdown("### Модель 2: С лагами")
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

    # Прогноз
    st.subheader("🔮 Прогноз")
    planned_ua = st.slider("Планируемый UA трафик", min_value=1000, max_value=50000, value=10000, step=1000)

    predicted_organic = model_simple.coef_[0] * planned_ua + model_simple.intercept_
    predicted_gross_ua = planned_ua * arpu_ua
    predicted_gross_organic = predicted_organic * arpu_organic

    col1, col2, col3 = st.columns(3)
    col1.metric("Ожидаемая органика", f"{predicted_organic:,.0f}")
    col2.metric("Gross от UA", f"{predicted_gross_ua:,.0f}")
    col3.metric("Gross от органики", f"{predicted_gross_organic:,.0f}")

# --- TAB 6: Аномалии ---
with tab6:
    st.subheader("Аномалии в данных")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Топ-5 высокий K-Factor")
        top_k = pivot.nlargest(5, 'k_factor')[['cohort', 'users_UA', 'users_ORGANIC', 'k_factor']].copy()
        top_k['cohort'] = top_k['cohort'].dt.strftime('%Y-%m-%d')
        st.dataframe(top_k, hide_index=True)

    with col2:
        st.markdown("### Топ-5 низкий K-Factor")
        low_k = pivot.nsmallest(5, 'k_factor')[['cohort', 'users_UA', 'users_ORGANIC', 'k_factor']].copy()
        low_k['cohort'] = low_k['cohort'].dt.strftime('%Y-%m-%d')
        st.dataframe(low_k, hide_index=True)

    # Стандартное отклонение для выявления аномалий
    k_mean = pivot['k_factor'].mean()
    k_std = pivot['k_factor'].std()
    anomalies_high = pivot[pivot['k_factor'] > k_mean + 2*k_std]
    anomalies_low = pivot[pivot['k_factor'] < k_mean - 2*k_std]

    st.markdown(f"""
    **Статистика K-Factor:**
    - Среднее: **{k_mean:.3f}**
    - Стд. отклонение: **{k_std:.3f}**
    - Аномально высокий (>2σ): **{len(anomalies_high)}** дней
    - Аномально низкий (<2σ): **{len(anomalies_low)}** дней

    **Возможные причины аномалий:**
    - Высокий K-Factor: вирусные кампании, праздники, акции
    - Низкий K-Factor: технические проблемы, массовый закуп низкокачественного трафика
    """)

# =============================================================================
# ВЫВОДЫ
# =============================================================================
st.header("📝 Общие выводы")

st.success(f"""
### Основные результаты анализа:

**1. K-Factor пользователей = {k_factor:.4f}**
- На каждого платного пользователя приходит {k_factor:.2f} органических
- Всего за период: {total_ua:,.0f} UA → {total_organic:,.0f} Organic
- Виральность продукта: **{k_factor*100:.0f}%** дополнительных пользователей

**2. K-Factor для денег = {k_factor_money:.4f}**
- Каждый рубль UA-выручки приносит дополнительно {k_factor_money:.2f} рубля от органики
- ARPU органики ({arpu_organic:.4f}) {'выше' if arpu_organic > arpu_ua else 'ниже'} ARPU UA ({arpu_ua:.4f})
- Мультипликатор выручки: **{1 + k_factor_money:.2f}x**

**3. Корреляция и регрессия:**
- Корреляция UA ↔ Organic: **{corr_pearson:.3f}** (сильная связь)
- Регрессионный K-factor: **{model_simple.coef_[0]:.4f}**
- Базовая органика без трафа: **{model_simple.intercept_:.0f}** чел/день

**4. Сезонность:**
- K-Factor выше в осенне-зимний период
- В выходные K-Factor {'выше' if weekend_k > weekday_k else 'ниже'} на {abs((weekend_k/weekday_k)-1)*100:.1f}%
""")


# =============================================================================
# ДАННЫЕ ПО МЕСЯЦАМ
# =============================================================================
st.header("📅 Данные по месяцам")

if st.checkbox("Показать таблицу K-Factor по месяцам"):
    monthly_summary = pd.DataFrame({
        'UA users': ua_monthly.values,
        'Organic users': organic_monthly.values,
        'K-Factor': k_factor_monthly.values,
        'Gross UA': monthly['gross']['UA'].values,
        'Gross Organic': monthly['gross']['ORGANIC'].values
    }, index=k_factor_monthly.index.astype(str))
    st.dataframe(monthly_summary)

# Footer
st.markdown("---")
st.markdown("📊 **K-Factor Analytics Dashboard** | Создано с помощью Streamlit")
