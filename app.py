import streamlit as st
import FinanceDataReader as fdr
import plotly.graph_objects as go
import pandas as pd
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import google.generativeai as genai

st.set_page_config(layout="wide", page_title="국내주식 실시간 뉴스/차트 분석기")
st.title("📈 실시간 뉴스 & 차트 AI 통합 분석기")

# 구글 AI 세팅 (비밀 금고에서 키 꺼내오기)
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    ai_ready = True
except:
    ai_ready = False

# 사이드바 설정
st.sidebar.header("🔍 종목 검색")
stock_dict = {"삼성전자": "005930", "SK하이닉스": "000660", "현대차": "005380", "NAVER": "035420", "카카오": "035720"}
selected_name = st.sidebar.selectbox("추천 종목 선택", list(stock_dict.keys()))
custom_code = st.sidebar.text_input("또는 다른 종목코드 6자리 입력 (예: 005930)", value="")

if custom_code.strip():
    stock_code = custom_code.strip()
    stock_name = f"종목코드 [{stock_code}]"
else:
    stock_code = stock_dict[selected_name]
    stock_name = selected_name

days_to_look = st.sidebar.slider("조회 기간 (일)", 10, 100, 45)

# 데이터 수집 함수
@st.cache_data(ttl=300)
def load_stock_data(code, days):
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=days)).strftime('%Y-%m-%d')
    df = fdr.DataReader(code, start=start_date, end=end_date).reset_index()
    df['Date_str'] = df['Date'].dt.strftime('%Y-%m-%d')
    return df

@st.cache_data(ttl=300)
def load_news_data(query):
    url = f"https://news.google.com/rss/search?q={query}&hl=ko&gl=KR&ceid=KR:ko"
    response = requests.get(url)
    root = ET.fromstring(response.content)
    
    news_list = []
    pos_words = ['상승', '돌파', '호재', '흑자', '최고', '성장', '매수', '급등', '실적개선', '수주']
    neg_words = ['하락', '쇼크', '악재', '적자', '최저', '감소', '매도', '급락', '우려', '소송']

    for item in root.findall('.//item')[:20]:
        title = item.find('title').text
        link = item.find('link').text
        pub_date = item.find('pubDate').text
        
        try:
            date_str = datetime.strptime(pub_date, '%a, %d %b %Y %H:%M:%S %Z').strftime('%Y-%m-%d')
        except:
            date_str = datetime.today().strftime('%Y-%m-%d')
            
        score = sum([1 for w in pos_words if w in title]) - sum([1 for w in neg_words if w in title])
        sentiment = "🟢 호재" if score > 0 else "🔴 악재" if score < 0 else "⚪ 중립"
            
        news_list.append({
            'Date_str': date_str, '제목': title.split(' - ')[0],
            '언론사': title.split(' - ')[1] if ' - ' in title else '뉴스',
            'AI 감성판단': sentiment, '링크': link
        })
    return pd.DataFrame(news_list)

# 화면 출력
try:
    with st.spinner('데이터를 실시간으로 불러오는 중입니다...'):
        stock_df = load_stock_data(stock_code, days_to_look)
        news_df = load_news_data(stock_name)
        
    if not stock_df.empty:
        col1, col2 = st.columns([6, 4])
        with col1:
            st.subheader(f"📊 {stock_name} ({stock_code}) 주가 차트")
            fig = go.Figure(data=[go.Candlestick(
                x=stock_df['Date'], open=stock_df['Open'], high=stock_df['High'],
                low=stock_df['Low'], close=stock_df['Close'], name='주가'
            )])
            
            if not news_df.empty:
                news_days = news_df.groupby('Date_str').first().reset_index()
                chart_news = pd.merge(stock_df, news_days, on='Date_str', how='inner')
                fig.add_trace(go.Scatter(
                    x=chart_news['Date'], y=chart_news['High'] * 1.02,
                    mode='markers+text', marker=dict(symbol='balloon', size=12, color='Gold'),
                    text=['📰 뉴스'] * len(chart_news), textposition='top center',
                    hovertemplate="<b>주요 뉴스:</b> %{customdata}<br><extra></extra>",
                    customdata=chart_news['제목'], name='관련 뉴스'
                ))
            fig.update_layout(xaxis_rangeslider_visible=False, template='plotly_white', height=600)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("🤖 전문 AI의 뉴스 종합 요약")
            if ai_ready and not news_df.empty:
                with st.spinner("구글 AI가 뉴스를 읽고 요약 중입니다..."):
                    prompt = f"다음은 '{stock_name}'와 관련된 최근 20개의 뉴스 기사 제목들입니다.\n"
                    for title in news_df['제목'].tolist():
                        prompt += f"- {title}\n"
                    prompt += "\n위 뉴스들을 종합해서 다음 내용을 작성해줘.\n1. 현재 시장의 전반적인 분위기 (호재/악재 여부 판단)\n2. 핵심 이슈 3줄 요약 (초보자도 이해하기 쉽게)"
                    
                    try:
                        response = model.generate_content(prompt)
                        st.info(response.text)
                    except Exception as e:
                        st.warning("AI 요약 생성 중 오류가 발생했습니다.")
            elif not ai_ready:
                st.warning("⚠️ Streamlit Cloud에 API 키가 설정되지 않아 AI 요약을 할 수 없습니다.")
                
            st.subheader("📰 실시간 뉴스 목록 (키워드 분석)")
            if not news_df.empty:
                for _, row in news_df.iterrows():
                    st.markdown(f"**[{row['AI 감성판단']}]** {row['Date_str']} | {row['언론사']}")
                    st.markdown(f"🔗 [{row['제목']}]({row['링크']})")
                    st.markdown("---")
            else:
                st.write("관련 뉴스가 없습니다.")
except Exception as e:
    st.error("데이터를 불러오는 중 오류가 발생했습니다.")