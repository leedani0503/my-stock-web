import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import io

st.set_page_config(layout="wide", page_title="대한민국 증시 AI 통합 브리핑 분석기")
st.title("📈 대한민국 증시 AI 통합 브리핑 & 실시간 차트 분석기")

# 🛠️ 구글 AI 세팅
try:
    import google.generativeai as genai
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')
    ai_ready = True
except Exception as e:
    ai_ready = False
    ai_error_msg = str(e)

# 📡 0. 한국거래소(KRX) 전체 종목코드 불러오기 (차단 방지 로직 추가)
@st.cache_data(show_spinner=False)
def load_krx_stock_list():
    try:
        # searchType=13 을 붙여 코스피, 코스닥, 코넥스 전 종목을 명시적으로 요청합니다.
        url = 'http://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13'
        
        # 봇으로 인식되지 않도록 일반 크롬 브라우저인 것처럼 위장합니다.
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        
        # 한글 깨짐 방지 및 데이터프레임 변환
        df = pd.read_html(io.StringIO(response.text), header=0)[0]
        
        # 종목코드를 6자리 문자열로 맞춤 (예: 5930 -> 005930)
        df['종목코드'] = df['종목코드'].astype(str).str.zfill(6)
        
        # 딕셔너리로 변환
        stock_dict = dict(zip(df['회사명'], df['종목코드']))
        
        if not stock_dict:
            raise ValueError("데이터를 파싱했지만 비어있습니다.")
            
        return stock_dict
        
    except Exception as e:
        # 혹시라도 접속이 안될 경우를 대비한 기본 종목들
        return {"삼성전자": "005930", "SK하이닉스": "000660", "현대차": "005380", "NAVER": "035420", "카카오": "035720", "에코프로": "086520", "차바이오텍": "010950"}

# 📡 1. 뉴스 수집 및 시간/분 파싱 (최신순 정렬)
def fetch_google_news(query, max_results=40):
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
            
            # 뉴스 발행 시간 파싱 (KST 보정)
            try:
                dt = datetime.strptime(pub_date, '%a, %d %b %Y %H:%M:%S %Z')
                dt = dt + timedelta(hours=9) 
            except:
                try:
                    dt = datetime.strptime(pub_date, '%a, %d %b %Y %H:%M:%S %z')
                except:
                    dt = datetime.today()
            
            date_str = dt.strftime('%Y-%m-%d %H:%M')
            score = sum([1 for w in pos_words if w in title]) - sum([1 for w in neg_words if w in title])
            sentiment = "🟢 호재" if score > 0 else "🔴 악재" if score < 0 else "⚪ 중립"
                
            news_list.append({
                'Datetime': dt,
                'Date_str': date_str, 
                '제목': title.split(' - ')[0] if ' - ' in title else title,
                '언론사': title.split(' - ')[1] if ' - ' in title else '경제뉴스',
                'AI 감성판단': sentiment, 
                '링크': link
            })
            
        df = pd.DataFrame(news_list)
        if not df.empty:
            df = df.sort_values(by='Datetime', ascending=False).reset_index(drop=True)
        return df
    except Exception as e:
        return pd.DataFrame()

# 📡 2. 주가 데이터 수집 함수 (yfinance 엔진)
def load_stock_data_yf(code, period, interval):
    try:
        clean_code = str(code).strip()
        if clean_code.isdigit():
            for suffix in ['.KS', '.KQ']:
                ticker = clean_code + suffix
                df = yf.download(ticker, period=period, interval=interval, progress=False)
                if df is not None and not df.empty:
                    break
        else:
            df = yf.download(clean_code, period=period, interval=interval, progress=False)
            
        if df is not None and not df.empty:
            df = df.reset_index()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]
            
            if 'Datetime' in df.columns:
                df = df.rename(columns={'Datetime': 'Date'})
                
            df['Date_str'] = df['Date'].dt.strftime('%Y-%m-%d %H:%M')
            return df
        return pd.DataFrame()
    except:
        return pd.DataFrame()

# 📌 탭 설정
tab1, tab2 = st.tabs(["📰 오늘의 한국 경제 시황 (기본 화면)", "📊 개별 종목 상세 분석 (일봉/분봉)"])

# =========================================================================
# [기본 화면] TAB 1: 한국 전체 경제 뉴스 종합 및 AI 요약
# =========================================================================
with tab1:
    st.subheader("👑 오늘의 대한민국 경제 및 증시 종합 브리핑")
    
    with st.spinner("구글 AI가 실시간 경제 뉴스를 분석 중입니다..."):
        economy_news_df = fetch_google_news("한국 경제 시황 OR 국내 증시 OR 코스피 코스닥", max_results=40)
        
        if not economy_news_df.empty:
            if ai_ready:
                prompt = "당신은 영리한 경제 수석 분석가입니다. 다음은 대한민국 경제 및 주식 시장 관련 최신 뉴스 헤드라인입니다.\n"
                for title in economy_news_df['제목'].head(20).tolist():
                    prompt += f"- {title}\n"
                prompt += "\n위 뉴스들을 종합적으로 분석하여 다음 내용을 한국어로 친절하게 작성해줘.\n1. 현재 시장의 전반적인 분위기와 투자 심리 상태\n2. 오늘 투자자가 꼭 알아야 할 핵심 경제 이슈 3가지."
                
                try:
                    response = model.generate_content(prompt)
                    st.markdown("### 🤖 전문 AI의 시장 종합 요약 리포트")
                    st.info(response.text) 
                except Exception as ai_err:
                    err_msg = str(ai_err).lower()
                    if "quota" in err_msg or "429" in err_msg:
                        st.warning("⚠️ **구글 AI 무료 버전의 호출 제한에 도달했습니다.**\n\n잠시 후 새로고침 해주세요. (아래 실시간 뉴스는 정상적으로 보실 수 있습니다.)")
                    else:
                        st.warning(f"⚠️ AI 요약 생성 중 오류가 발생했습니다.")
            else:
                st.warning(f"⚠️ AI 엔진이 준비되지 않았습니다. Secrets 설정을 확인하세요.")
                
            st.write("")
            st.subheader("📰 실시간 주요 경제 뉴스 헤드라인 (최신순)")
            for _, row in economy_news_df.iterrows():
                st.markdown(f"**[{row['AI 감성판단']}]** {row['Date_str']} | {row['언론사']}")
                st.markdown(f"🔗 [{row['제목']}]({row['링크']})")
                st.markdown("---")
        else:
            st.error("실시간 경제 뉴스를 불러오는데 실패했습니다. 잠시 후 새로고침 해주세요.")

# =========================================================================
# [선택 화면] TAB 2: 개별 종목 상세 분석 (일봉 / 분봉 차트)
# =========================================================================
with tab2:
    st.subheader("🔍 개별 종목 주가 차트 & 관련 뉴스 분석")
    
    st.sidebar.header("⚙️ 종목 및 차트 조건 설정")
    
    # 📌 종목명 자동완성 로직
    with st.spinner("한국거래소(KRX) 전체 상장사 2,600여 개를 불러오는 중..."):
        krx_dict = load_krx_stock_list()
    
    stock_names = list(krx_dict.keys())
    default_index = stock_names.index("삼성전자") if "삼성전자" in stock_names else 0
    stock_name = st.sidebar.selectbox("🔎 종목명 검색 (이름을 입력하세요)", stock_names, index=default_index)
    
    stock_code = krx_dict[stock_name]

    st.sidebar.markdown("---")
    view_type = st.sidebar.radio("⏱️ 차트 조회 기준 선택", ["일 단위 (Daily)", "분 단위 (Minute)"])
    
    if view_type == "일 단위 (Daily)":
        days_to_look = st.sidebar.slider("조회 기간 (일)", 10, 100, 45)
        period = f"{days_to_look}d"
        interval = "1d"
    else:
        minute_option = st.sidebar.selectbox("분봉 선택 (최대 5~7일까지만 제공됨)", ["1분봉 (최근 1일)", "5분봉 (최근 5일)", "15분봉 (최근 5일)"])
        if "1분봉" in minute_option:
            period = "1d"
            interval = "1m"
        elif "5분봉" in minute_option:
            period = "5d"
            interval = "5m"
        else:
            period = "5d"
            interval = "15m"

    with st.spinner(f'[{stock_name}] 데이터를 실시간으로 수집하는 중...'):
        stock_df = load_stock_data_yf(stock_code, period, interval)
        stock_news_df = fetch_google_news(stock_name, max_results=20)
        
    if not stock_df.empty:
        col1, col2 = st.columns([6, 4])
        with col1:
            st.markdown(f"#### 📊 {stock_name}({stock_code}) 실시간 차트 ({interval})")
            
            fig = go.Figure(data=[go.Candlestick(
                x=stock_df['Date'], open=stock_df['Open'], high=stock_df['High'],
                low=stock_df['Low'], close=stock_df['Close'], name='주가'
            )])
            
            if not stock_news_df.empty:
                chart_news_list = []
                for _, news_row in stock_news_df.iterrows():
                    try:
                        news_dt = pd.to_datetime(news_row['Date_str'])
                        time_diffs = (stock_df['Date'] - news_dt).abs()
                        closest_idx = time_diffs.idxmin()
                        if time_diffs[closest_idx] < pd.Timedelta(days=1):
                            match_row = stock_df.iloc[closest_idx].copy()
                            match_row['뉴스제목'] = news_row['제목']
                            chart_news_list.append(match_row)
                    except:
                        continue
                
                if chart_news_list:
                    chart_news_df = pd.DataFrame(chart_news_list)
                    fig.add_trace(go.Scatter(
                        x=chart_news_df['Date'], y=chart_news_df['High'] * 1.01,
                        mode='markers', marker=dict(symbol='triangle-up', size=11, color='Gold'),
                        hovertemplate="<b>매칭 뉴스:</b> %{customdata}<br><extra></extra>",
                        customdata=chart_news_df['뉴스제목'], name='관련 뉴스 마커'
                    ))
                    
            fig.update_layout(xaxis_rangeslider_visible=False, template='plotly_white', height=550)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown(f"#### 📰 {stock_name} 관련 최신 뉴스")
            if not stock_news_df.empty:
                for _, row in stock_news_df.iterrows():
                    st.markdown(f"**[{row['AI 감성판단']}]** {row['Date_str']} | {row['언론사']}")
                    st.markdown(f"🔗 [{row['제목']}]({row['링크']})")
                    st.markdown("---")
            else:
                st.write("💡 해당 종목의 검색된 최신 뉴스가 없습니다.")
    else:
        st.error(f"❌ '{stock_name}' 데이터를 가져오지 못했습니다. 신규 상장주이거나 조회 기간 설정 문제일 수 있습니다.")