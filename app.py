import streamlit as st
import FinanceDataReader as fdr
import plotly.graph_objects as go
import pandas as pd
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import google.generativeai as genai

st.set_page_config(layout="wide", page_title="대한민국 증시 AI 통합 브리핑 분석기")
st.title("📈 대한민국 증시 AI 통합 브리핑 & 주식 분석기")

# 구글 AI 세팅
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    ai_ready = True
except:
    ai_ready = False

# 뉴스 수집 공통 함수 (버그 수정 버전)
def fetch_google_news(query, max_results=20):
    # 구글 뉴스 RSS 주소
    url = f"https://news.google.com/rss/search?q={query}&hl=ko&gl=KR&ceid=KR:ko"
    try:
        response = requests.get(url, timeout=10)
        root = ET.fromstring(response.content)
        
        news_list = []
        pos_words = ['상승', '돌파', '호재', '흑자', '최고', '성장', '매수', '급등', '실적개선', '수주', '반등']
        neg_words = ['하락', '쇼크', '악재', '적자', '최저', '감소', '매도', '급락', '우려', '소송', '폭락']

        for item in root.findall('.//item')[:max_results]:
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
                'Date_str': date_str, 
                '제목': title.split(' - ')[0] if ' - ' in title else title,
                '언론사': title.split(' - ')[1] if ' - ' in title else '경제뉴스',
                'AI 감성판단': sentiment, 
                '링크': link
            })
        return pd.DataFrame(news_list)
    except Exception as e:
        return pd.DataFrame()

# 📌 탭 분할: 사용자가 요청한 대로 '기본 화면'을 메인으로 설정
tab1, tab2 = st.tabs(["📰 오늘의 한국 경제 시황 (기본 화면)", "📊 개별 종목 상세 분석"])

# =========================================================================
# [기본 화면] TAB 1: 모든 한국 경제 뉴스 종합 및 AI 요약
# =========================================================================
with tab1:
    st.subheader("👑 오늘의 대한민국 경제 및 증시 종합 브리핑")
    
    if ai_ready:
        with st.spinner("구글 AI가 대한민국 경제 뉴스를 종합 분석 중입니다..."):
            # 거시 경제 전체를 아우르는 핵심 키워드로 뉴스 수집
            economy_news_df = fetch_google_news("한국 경제 시황 OR 국내 증시 OR 코스피 코스닥", max_results=20)
            
            if not economy_news_df.empty:
                # 구글 Gemini AI에게 보낼 명령조립
                prompt = "당신은 영리한 경제 수석 분석가입니다. 다음은 대한민국 경제 및 주식 시장 관련 최신 뉴스 헤드라인 20개입니다.\n"
                for title in economy_news_df['제목'].tolist():
                    prompt += f"- {title}\n"
                prompt += "\n위 뉴스들을 종합적으로 분석하여 다음 내용을 한국어로 친절하게 작성해줘.\n1. 현재 시장의 전반적인 분위기와 투자 심리 상태 (호재가 많은지 악재가 많은지)\n2. 오늘 투자자가 꼭 알아야 할 핵심 경제 이슈 3가지를 초보자도 이해하기 쉽게 전문적으로 요약해줘."
                
                try:
                    response = model.generate_content(prompt)
                    st.markdown("### 🤖 전문 AI의 시장 종합 요약 리포트")
                    st.info(response.text) # 파란색 이쁜 박스로 AI 요약 출력
                except Exception as e:
                    st.warning("AI 요약 생성 중 오류가 발생했습니다.")
                    
                # 그 아래에 전체 뉴스 리스트도 함께 출력
                st.write("")
                st.subheader("📰 실시간 주요 경제 뉴스 헤드라인")
                for _, row in economy_news_df.iterrows():
                    st.markdown(f"**[{row['AI 감성판단']}]** {row['Date_str']} | {row['언론사']}")
                    st.markdown(f"🔗 [{row['제목']}]({row['링크']})")
                    st.markdown("---")
            else:
                st.error("실시간 경제 뉴스를 불러오는데 실패했습니다.")
    else:
        st.warning("⚠️ Streamlit Secrets에 GEMINI_API_KEY가 설정되지 않았습니다.")

# =========================================================================
# [선택 화면] TAB 2: 개별 종목 상세 분석 (삼전, 하이닉스, 차바이오텍 등)
# =========================================================================
with tab2:
    st.subheader("🔍 개별 종목 주가 차트 & 관련 뉴스 분석")
    
    # 사이드바 설정 (오류 방지를 위해 안전 가드 장착)
    st.sidebar.header("⚙️ 종목 선택 및 설정")
    stock_dict = {"삼성전자": "005930", "SK하이닉스": "000660", "현대차": "005380", "NAVER": "035420", "카카오": "035720"}
    selected_name = st.sidebar.selectbox("추천 종목 선택", list(stock_dict.keys()))
    custom_code = st.sidebar.text_input("또는 다른 종목코드 6자리 입력 (예: 차바이오텍 010950)", value="")

    # 종목코드 및 뉴스 검색어 정제 (★버그 해결 핵심 구역)
    if custom_code.strip():
        stock_code = custom_code.strip()
        stock_name = stock_code  # 불필요한 글자를 빼고 딱 코드번호만 구글뉴스에 검색하게 수정
    else:
        stock_code = stock_dict[selected_name]
        stock_name = selected_name

    days_to_look = st.sidebar.slider("조회 기간 (일)", 10, 100, 45)

    # 주가 데이터 수집 함수 (★삼성전자 등 로딩 실패 에러 완벽 방지)
    def load_stock_data(code, days):
        try:
            end_date = datetime.today().strftime('%Y-%m-%d')
            start_date = (datetime.today() - timedelta(days=days)).strftime('%Y-%m-%d')
            clean_code = str(code).strip() # 공백 제거 안전장치
            df = fdr.DataReader(clean_code, start=start_date, end=end_date)
            if df is not None and not df.empty:
                df = df.reset_index()
                df['Date_str'] = df['Date'].dt.strftime('%Y-%m-%d')
                return df
            return pd.DataFrame()
        except:
            return pd.DataFrame()

    # 데이터 수집 작동
    with st.spinner(f'종목 데이터를 실시간으로 불러오는 중...'):
        stock_df = load_stock_data(stock_code, days_to_look)
        stock_news_df = fetch_google_news(stock_name, max_results=15)
        
    # 화면 렌더링
    if not stock_df.empty:
        col1, col2 = st.columns([6, 4])
        with col1:
            display_title = selected_name if not custom_code.strip() else f"입력 종목 [{stock_code}]"
            st.markdown(f"#### 📊 {display_title} 주가 차트")
            fig = go.Figure(data=[go.Candlestick(
                x=stock_df['Date'], open=stock_df['Open'], high=stock_df['High'],
                low=stock_df['Low'], close=stock_df['Close'], name='주가'
            )])
            
            # 차트 위에 뉴스 풍선 달기
            if not stock_news_df.empty:
                news_days = stock_news_df.groupby('Date_str').first().reset_index()
                chart_news = pd.merge(stock_df, news_days, on='Date_str', how='inner')
                if not chart_news.empty:
                    fig.add_trace(go.Scatter(
                        x=chart_news['Date'], y=chart_news['High'] * 1.02,
                        mode='markers+text', marker=dict(symbol='balloon', size=12, color='Gold'),
                        text=['📰 뉴스'] * len(chart_news), textposition='top center',
                        hovertemplate="<b>주요 뉴스:</b> %{customdata}<br><extra></extra>",
                        customdata=chart_news['제목'], name='관련 뉴스'
                    ))
            fig.update_layout(xaxis_rangeslider_visible=False, template='plotly_white', height=550)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown(f"#### 📰 {display_title} 관련 뉴스 목록")
            if not stock_news_df.empty:
                for _, row in stock_news_df.iterrows():
                    st.markdown(f"**[{row['AI 감성판단']}]** {row['Date_str']} | {row['언론사']}")
                    st.markdown(f"🔗 [{row['제목']}]({row['링크']})")
                    st.markdown("---")
            else:
                st.write("💡 해당 종목의 최신 뉴스가 없습니다. 종목명이나 코드를 다시 확인해 보세요.")
    else:
        st.error(f"❌ '{stock_code}' 데이터를 가져오지 못했습니다. 종목 코드를 확인하시거나 잠시 후 다시 시도해 주세요.")