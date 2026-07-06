import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import io

st.set_page_config(layout="wide", page_title="대한민국 증시 AI 통합 브리핑 분석기")
st.title("📈 대한민국 증시 AI 통합 브리핑 & 실시간 차트 분석기")

# =========================================================================
# 🛠️ 구글 AI 세팅
# =========================================================================
try:
    import google.generativeai as genai
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')
    ai_ready = True
except Exception as e:
    ai_ready = False
    ai_error_msg = str(e)


def ask_gemini(prompt):
    """Gemini 호출 공통 함수. 성공 시 (True, 텍스트), 실패 시 (False, 에러메시지) 반환."""
    if not ai_ready:
        return False, "AI 엔진이 준비되지 않았습니다. Secrets 설정을 확인하세요."
    try:
        response = model.generate_content(prompt)
        return True, response.text
    except Exception as ai_err:
        err_msg = str(ai_err).lower()
        if "quota" in err_msg or "429" in err_msg:
            return False, "⚠️ 구글 AI 무료 버전의 호출 제한(쿼터)에 도달했습니다. 잠시 후 다시 시도해주세요."
        return False, f"⚠️ AI 분석 중 오류가 발생했습니다: {ai_err}"


# =========================================================================
# 📡 0. 한국거래소(KRX) 전체 종목코드 불러오기
# =========================================================================
@st.cache_data(show_spinner=False, ttl=60 * 60 * 12)
def load_krx_stock_list():
    try:
        url = 'http://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        df = pd.read_html(io.StringIO(response.text), header=0)[0]
        df['종목코드'] = df['종목코드'].astype(str).str.zfill(6)
        stock_dict = dict(zip(df['회사명'], df['종목코드']))
        if not stock_dict:
            raise ValueError("데이터 비어있음")
        return stock_dict
    except Exception:
        return {
            "삼성전자": "005930", "SK하이닉스": "000660", "현대차": "005380",
            "NAVER": "035420", "카카오": "035720", "에코프로": "086520", "차바이오텍": "010950"
        }


# =========================================================================
# 📡 1. 뉴스 수집 (최신순 정렬 + 감성 태깅)
# =========================================================================
POS_WORDS = ['상승', '돌파', '호재', '흑자', '최고', '성장', '매수', '급등', '실적개선', '수주', '반등', '신고가', '기대감']
NEG_WORDS = ['하락', '쇼크', '악재', '적자', '최저', '감소', '매도', '급락', '우려', '소송', '폭락', '신저가', '리스크']


@st.cache_data(show_spinner=False, ttl=60 * 10)
def fetch_google_news(query, max_results=40):
    url = f"https://news.google.com/rss/search?q={query}&hl=ko&gl=KR&ceid=KR:ko"
    try:
        response = requests.get(url, timeout=10)
        root = ET.fromstring(response.content)

        news_list = []
        for item in root.findall('.//item')[:max_results]:
            title = item.find('title').text
            link = item.find('link').text
            pub_date = item.find('pubDate').text

            try:
                dt = datetime.strptime(pub_date, '%a, %d %b %Y %H:%M:%S %Z')
                dt = dt + timedelta(hours=9)
            except Exception:
                try:
                    dt = datetime.strptime(pub_date, '%a, %d %b %Y %H:%M:%S %z')
                except Exception:
                    dt = datetime.today()

            date_str = dt.strftime('%Y-%m-%d %H:%M')
            score = sum(1 for w in POS_WORDS if w in title) - sum(1 for w in NEG_WORDS if w in title)
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
    except Exception:
        return pd.DataFrame()


# =========================================================================
# 📡 2. 주가 데이터 수집 (yfinance)
# =========================================================================
@st.cache_data(show_spinner=False, ttl=60 * 5)
def load_stock_data_yf(code, period, interval):
    try:
        clean_code = str(code).strip()
        df = pd.DataFrame()
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

            # 이동평균선 계산 (분봉/일봉 공통 적용, 데이터 부족 시 자동으로 NaN 처리됨)
            df['MA5'] = df['Close'].rolling(window=5).mean()
            df['MA20'] = df['Close'].rolling(window=20).mean()
            df['MA60'] = df['Close'].rolling(window=60).mean()
            return df
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def render_news_cards(news_df, key_prefix=""):
    """뉴스 리스트를 카드(expander) 형태로 출력"""
    for idx, row in news_df.iterrows():
        with st.container(border=True):
            st.markdown(f"**{row['AI 감성판단']}**  ·  {row['Date_str']}  ·  {row['언론사']}")
            st.markdown(f"🔗 [{row['제목']}]({row['링크']})")


def build_price_chart(stock_df, stock_name, stock_code, interval):
    """가독성을 높인 캔들스틱 + 이동평균선 + 거래량 차트 생성"""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.75, 0.25], vertical_spacing=0.03,
        subplot_titles=(f"{stock_name}({stock_code}) 가격 ({interval})", "거래량")
    )

    fig.add_trace(go.Candlestick(
        x=stock_df['Date'], open=stock_df['Open'], high=stock_df['High'],
        low=stock_df['Low'], close=stock_df['Close'], name='주가',
        increasing_line_color='#d64550', decreasing_line_color='#3b82f6'
    ), row=1, col=1)

    for ma_col, color in [('MA5', '#f5a623'), ('MA20', '#8e44ad'), ('MA60', '#2c3e50')]:
        if ma_col in stock_df.columns and stock_df[ma_col].notna().any():
            fig.add_trace(go.Scatter(
                x=stock_df['Date'], y=stock_df[ma_col], mode='lines',
                line=dict(width=1.4, color=color), name=ma_col
            ), row=1, col=1)

    if 'Volume' in stock_df.columns:
        vol_colors = ['#d64550' if c >= o else '#3b82f6' for o, c in zip(stock_df['Open'], stock_df['Close'])]
        fig.add_trace(go.Bar(
            x=stock_df['Date'], y=stock_df['Volume'], name='거래량',
            marker_color=vol_colors, showlegend=False
        ), row=2, col=1)

    fig.update_layout(
        template='plotly_white', height=650,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(t=60, b=20, l=10, r=10),
        hovermode='x unified'
    )
    fig.update_xaxes(rangeslider_visible=False, row=2, col=1)
    return fig


# 📌 탭 설정
tab1, tab2 = st.tabs(["📰 오늘의 한국 경제 시황 (기본 화면)", "📊 개별 종목 상세 분석 (일봉/분봉 + AI 뉴스 분석)"])

# =========================================================================
# [기본 화면] TAB 1
# =========================================================================
with tab1:
    st.subheader("👑 오늘의 대한민국 경제 및 증시 종합 브리핑")

    with st.spinner("구글 AI가 실시간 경제 뉴스를 분석 중입니다..."):
        economy_news_df = fetch_google_news("한국 경제 시황 OR 국내 증시 OR 코스피 코스닥", max_results=40)

        if not economy_news_df.empty:
            prompt = (
                "당신은 영리한 경제 수석 분석가입니다. 다음은 대한민국 경제 및 주식 시장 관련 최신 뉴스 헤드라인입니다.\n"
                + "\n".join(f"- {t}" for t in economy_news_df['제목'].head(20).tolist())
                + "\n\n위 뉴스들을 종합적으로 분석하여 다음 내용을 한국어로 친절하게 작성해줘."
                  "\n1. 현재 시장의 전반적인 분위기와 투자 심리 상태"
                  "\n2. 오늘 투자자가 꼭 알아야 할 핵심 경제 이슈 3가지."
            )
            ok, text = ask_gemini(prompt)
            st.markdown("### 🤖 전문 AI의 시장 종합 요약 리포트")
            if ok:
                st.info(text)
            else:
                st.warning(f"{text}\n\n(아래 실시간 뉴스는 정상적으로 보실 수 있습니다.)")

            st.write("")
            pos_cnt = (economy_news_df['AI 감성판단'] == "🟢 호재").sum()
            neg_cnt = (economy_news_df['AI 감성판단'] == "🔴 악재").sum()
            neu_cnt = (economy_news_df['AI 감성판단'] == "⚪ 중립").sum()
            m1, m2, m3 = st.columns(3)
            m1.metric("🟢 호재성 뉴스", f"{pos_cnt}건")
            m2.metric("🔴 악재성 뉴스", f"{neg_cnt}건")
            m3.metric("⚪ 중립 뉴스", f"{neu_cnt}건")

            st.subheader("📰 실시간 주요 경제 뉴스 헤드라인 (최신순)")
            render_news_cards(economy_news_df)
        else:
            st.error("실시간 경제 뉴스를 불러오는데 실패했습니다. 잠시 후 새로고침 해주세요.")

# =========================================================================
# [선택 화면] TAB 2: 개별 종목 상세 분석 + 종목별 뉴스 AI 분석
# =========================================================================
with tab2:
    st.subheader("🔍 개별 종목 주가 차트 & 종목 전용 AI 뉴스 분석")
    st.sidebar.header("⚙️ 종목 및 차트 조건 설정")

    with st.spinner("한국거래소(KRX) 전체 상장사 목록을 불러오는 중..."):
        krx_dict = load_krx_stock_list()

    stock_names = list(krx_dict.keys())
    default_index = stock_names.index("삼성전자") if "삼성전자" in stock_names else 0
    stock_name = st.sidebar.selectbox("🔎 종목명 검색 (이름을 입력하세요)", stock_names, index=default_index)
    stock_code = krx_dict[stock_name]

    st.sidebar.markdown("---")
    view_type = st.sidebar.radio("⏱️ 차트 조회 기준 선택", ["일 단위 (Daily)", "분 단위 (Minute)"])

    if view_type == "일 단위 (Daily)":
        days_to_look = st.sidebar.slider("조회 기간 (일)", 10, 200, 60)
        period = f"{days_to_look}d"
        interval = "1d"
    else:
        minute_option = st.sidebar.selectbox(
            "분봉 선택 (최대 5~7일까지만 제공됨)",
            ["1분봉 (최근 1일)", "5분봉 (최근 5일)", "15분봉 (최근 5일)"]
        )
        if "1분봉" in minute_option:
            period, interval = "1d", "1m"
        elif "5분봉" in minute_option:
            period, interval = "5d", "5m"
        else:
            period, interval = "5d", "15m"

    with st.spinner(f'[{stock_name}] 데이터를 실시간으로 수집하는 중...'):
        stock_df = load_stock_data_yf(stock_code, period, interval)
        stock_news_df = fetch_google_news(stock_name, max_results=25)

    if not stock_df.empty:
        # --- 상단 요약 메트릭 ---
        last_close = stock_df['Close'].iloc[-1]
        prev_close = stock_df['Close'].iloc[-2] if len(stock_df) > 1 else last_close
        change = last_close - prev_close
        change_pct = (change / prev_close * 100) if prev_close else 0
        c1, c2, c3 = st.columns(3)
        c1.metric(f"{stock_name} 현재가", f"{last_close:,.0f}원", f"{change:,.0f}원 ({change_pct:.2f}%)")
        c2.metric("최고가 (조회기간)", f"{stock_df['High'].max():,.0f}원")
        c3.metric("최저가 (조회기간)", f"{stock_df['Low'].min():,.0f}원")

        st.markdown(f"#### 📊 {stock_name}({stock_code}) 실시간 차트 ({interval})")
        fig = build_price_chart(stock_df, stock_name, stock_code, interval)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # --- ⭐ 종목별 뉴스 + AI 분석 (핵심 신규 기능) ---
        news_col, ai_col = st.columns([5, 5])

        with news_col:
            st.markdown(f"#### 📰 {stock_name} 관련 최신 뉴스")
            if not stock_news_df.empty:
                pos_cnt = (stock_news_df['AI 감성판단'] == "🟢 호재").sum()
                neg_cnt = (stock_news_df['AI 감성판단'] == "🔴 악재").sum()
                neu_cnt = (stock_news_df['AI 감성판단'] == "⚪ 중립").sum()
                st.caption(f"🟢 호재 {pos_cnt}건 · 🔴 악재 {neg_cnt}건 · ⚪ 중립 {neu_cnt}건 (총 {len(stock_news_df)}건)")
                render_news_cards(stock_news_df)
            else:
                st.write("💡 해당 종목의 검색된 최신 뉴스가 없습니다.")

        with ai_col:
            st.markdown(f"#### 🤖 {stock_name} 전용 AI 뉴스 분석")
            st.caption("이 종목과 관련된 최신 뉴스만 모아서 AI가 직접 분석합니다.")

            if stock_news_df.empty:
                st.info("분석할 뉴스가 없습니다.")
            else:
                if st.button(f"🔍 {stock_name} 뉴스 AI 분석 실행", key="analyze_stock_news", use_container_width=True):
                    with st.spinner(f"AI가 {stock_name} 관련 뉴스 {len(stock_news_df)}건을 분석 중입니다..."):
                        headlines = "\n".join(
                            f"- ({row['Date_str']}) {row['제목']}"
                            for _, row in stock_news_df.head(25).iterrows()
                        )
                        stock_prompt = (
                            f"당신은 대한민국 주식 시장을 전문적으로 분석하는 애널리스트입니다.\n"
                            f"아래는 '{stock_name}({stock_code})' 종목과 관련된 최신 뉴스 헤드라인 목록입니다.\n\n"
                            f"{headlines}\n\n"
                            f"위 뉴스만을 근거로 다음 항목을 한국어로 명확하게 정리해줘.\n"
                            f"1. '{stock_name}'에 대한 현재 뉴스 흐름의 전반적인 톤(긍정/부정/중립)과 그 이유\n"
                            f"2. 주가에 긍정적 영향을 줄 수 있는 핵심 요인 (있다면 최대 3가지)\n"
                            f"3. 주가에 부정적 영향을 줄 수 있는 핵심 요인 (있다면 최대 3가지)\n"
                            f"4. 투자자가 앞으로 주의 깊게 지켜봐야 할 포인트 1~2가지\n"
                            f"※ 확정적인 투자 권유가 아니라 뉴스 기반의 참고 분석임을 명시해줘."
                        )
                        ok, text = ask_gemini(stock_prompt)
                        if ok:
                            st.success(text)
                        else:
                            st.warning(text)
    else:
        st.error(f"❌ '{stock_name}' 데이터를 가져오지 못했습니다. 신규 상장주이거나 조회 기간 설정 문제일 수 있습니다.")
