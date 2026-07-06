import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import io
import os

st.set_page_config(layout="wide", page_title="대한민국 증시 AI 통합 브리핑 분석기")
st.title("📈 대한민국 증시 AI 통합 브리핑 & 실시간 차트 분석기 (Groq Ver.)")

# =========================================================================
# 🛠️ Groq AI 세팅
# =========================================================================
try:
    from groq import Groq
    groq_key = st.secrets["GROQ_API_KEY"]
    # Groq 클라이언트 초기화
    client = Groq(api_key=groq_key)
    ai_ready = True
except Exception as e:
    ai_ready = False
    ai_error_msg = str(e)


def ask_groq(prompt, system_role="당신은 영리한 경제 수석 분석가이자 주식 전문 애널리스트입니다."):
    """Groq 호출 공통 함수. 성공 시 (True, 텍스트), 실패 시 (False, 에러메시지) 반환."""
    if not ai_ready:
        return False, "AI 엔진이 준비되지 않았습니다. Secrets에서 GROQ_API_KEY 설정을 확인하세요."
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # 속도가 매우 빠르고 성능이 검증된 그록의 대표 무료 모델
            messages=[
                {"role": "system", "content": system_role},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return True, response.choices[0].message.content
    except Exception as ai_err:
        err_msg = str(ai_err).lower()
        if "rate_limit" in err_msg or "429" in err_msg:
            return False, "⚠️ Groq AI 무료 버전의 호출 제한(레이트 리밋)에 도달했습니다. 잠시 후 다시 시도해주세요."
        return False, f"⚠️ Groq AI 분석 중 오류가 발생했습니다: {ai_err}"


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
                'リンク': link
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

            df['MA5'] = df['Close'].rolling(window=5).mean()
            df['MA20'] = df['Close'].rolling(window=20).mean()
            df['MA60'] = df['Close'].rolling(window=60).mean()
            return df
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def render_news_cards(news_df):
    for _, row in news_df.iterrows():
        with st.container(border=True):
            st.markdown(f"**{row['AI 감성판단']}** ·  {row['Date_str']}  ·  {row['언론사']}")
            st.markdown(f"🔗 [{row['제목']}]({row['링크']})")


def _to_naive(series_or_ts):
    if isinstance(series_or_ts, pd.Series):
        if hasattr(series_or_ts.dt, 'tz') and series_or_ts.dt.tz is not None:
            return series_or_ts.dt.tz_localize(None)
        return series_or_ts
    ts = pd.Timestamp(series_or_ts)
    return ts.tz_localize(None) if ts.tzinfo is not None else ts


def match_news_to_chart(stock_df, news_df, interval):
    if news_df.empty or stock_df.empty:
        return pd.DataFrame()

    tolerance = pd.Timedelta(hours=3) if interval != "1d" else pd.Timedelta(days=1)
    chart_dates_naive = _to_naive(stock_df['Date'])

    matched = []
    for _, news_row in news_df.iterrows():
        try:
            news_dt = _to_naive(news_row['Datetime'])
            diffs = (chart_dates_naive - news_dt).abs()
            closest_idx = diffs.idxmin()
            if diffs[closest_idx] <= tolerance:
                match_row = stock_df.iloc[closest_idx].copy()
                match_row['뉴스제목'] = news_row['제목']
                match_row['뉴스시간'] = news_row['Date_str']
                matched.append(match_row)
        except Exception:
            continue
    return pd.DataFrame(matched).reset_index(drop=True)


# =========================================================================
# 🔮 3. 예상가 예측 + 자기보정(학습) 로직 (Google Sheets 연동)
# =========================================================================
LOG_PATH = os.path.join(os.path.dirname(__file__), "predictions_log.csv") if "__file__" in globals() else "predictions_log.csv"
LOG_COLS = ['stock_code', 'stock_name', 'made_at', 'target_time', 'predicted', 'upper', 'lower', 'last_price', 'actual_price', 'resolved']
MARKET_CLOSE_HOUR, MARKET_CLOSE_MIN = 15, 30
GSHEET_WORKSHEET_NAME = "predictions_log"


@st.cache_resource(show_spinner=False)
def get_gsheet_worksheet():
    try:
        import gspread
        from google.oauth2.service_account import Credentials

        if "gcp_service_account" not in st.secrets or "GSHEET_ID" not in st.secrets:
            return None

        scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        creds = Credentials.from_service_account_info(dict(st.secrets["gcp_service_account"]), scopes=scopes)
        client = gspread.authorize(creds)
        sh = client.open_by_key(st.secrets["GSHEET_ID"])

        try:
            worksheet = sh.worksheet(GSHEET_WORKSHEET_NAME)
        except gspread.exceptions.WorksheetNotFound:
            worksheet = sh.add_worksheet(title=GSHEET_WORKSHEET_NAME, rows=2000, cols=len(LOG_COLS))
            worksheet.append_row(LOG_COLS)
        return worksheet
    except Exception:
        return None


def is_gsheet_connected():
    return get_gsheet_worksheet() is not None


def load_log():
    ws = get_gsheet_worksheet()
    if ws is not None:
        try:
            records = ws.get_all_records()
            df = pd.DataFrame(records) if records else pd.DataFrame(columns=LOG_COLS)
            for c in LOG_COLS:
                if c not in df.columns:
                    df[c] = None
            df['made_at'] = pd.to_datetime(df['made_at'], errors='coerce')
            df['target_time'] = pd.to_datetime(df['target_time'], errors='coerce')
            df['resolved'] = df['resolved'].astype(str).str.lower().isin(['true', '1'])
            for c in ['predicted', 'upper', 'lower', 'last_price', 'actual_price']:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            return df
        except Exception:
            pass

    if os.path.exists(LOG_PATH):
        try:
            df = pd.read_csv(LOG_PATH, parse_dates=['made_at', 'target_time'])
            for c in LOG_COLS:
                if c not in df.columns:
                    df[c] = None
            return df
        except Exception:
            return pd.DataFrame(columns=LOG_COLS)
    return pd.DataFrame(columns=LOG_COLS)


def save_log(df):
    ws = get_gsheet_worksheet()
    if ws is not None:
        try:
            df_out = df.copy()
            df_out['made_at'] = df_out['made_at'].astype(str)
            df_out['target_time'] = df_out['target_time'].astype(str)
            df_out = df_out[LOG_COLS].fillna('')
            ws.clear()
            ws.update([LOG_COLS] + df_out.astype(str).values.tolist())
            return
        except Exception:
            pass

    try:
        df.to_csv(LOG_PATH, index=False)
    except Exception:
        pass


def generate_prediction(stock_df, view_type):
    df = stock_df.dropna(subset=['Close']).reset_index(drop=True)
    if len(df) < 5:
        return None

    lookback = min(30, len(df))
    recent = df.tail(lookback).reset_index(drop=True)
    x = np.arange(len(recent))
    y = recent['Close'].values
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    resid_std = float(np.std(y - fitted)) if len(y) > 2 else 0.0

    last_time = df['Date'].iloc[-1]
    last_idx = len(recent) - 1
    last_price = float(df['Close'].iloc[-1])

    if view_type == "일 단위 (Daily)":
        target_time = last_time + timedelta(days=1)
        steps_ahead = 1
        label = "다음 거래일 예상 종가 (근사치)"
    else:
        now = datetime.now()
        market_close = last_time.replace(hour=MARKET_CLOSE_HOUR, minute=MARKET_CLOSE_MIN, second=0, microsecond=0)
        if last_time.date() != now.date() or last_time >= market_close:
            return None
        if len(recent) >= 2:
            interval_minutes = max((recent['Date'].iloc[-1] - recent['Date'].iloc[-2]).total_seconds() / 60, 1)
        else:
            interval_minutes = 1
        remaining_minutes = (market_close - last_time).total_seconds() / 60
        steps_ahead = max(1, int(remaining_minutes / interval_minutes))
        target_time = market_close
        label = "오늘 장마감 예상가"

    predicted = float(slope * (last_idx + steps_ahead) + intercept)
    upper = predicted + 1.96 * resid_std
    lower = predicted - 1.96 * resid_std

    return {
        'predicted': predicted, 'upper': upper, 'lower': lower,
        'target_time': target_time, 'label': label,
        'last_time': last_time, 'last_price': last_price, 'slope': slope
    }


def get_calibration_bias(stock_code):
    log_df = load_log()
    if log_df.empty:
        return 0.0, 0
    resolved = log_df[(log_df['stock_code'] == stock_code) & (log_df['resolved'] == True)]
    if resolved.empty:
        return 0.0, 0
    resolved = resolved.tail(10)
    try:
        errors = resolved['actual_price'].astype(float) - resolved['predicted'].astype(float)
        return float(errors.mean()), len(resolved)
    except Exception:
        return 0.0, 0


def log_prediction(stock_code, stock_name, pred_info):
    log_df = load_log()
    today_str = datetime.now().strftime('%Y-%m-%d')
    if not log_df.empty:
        made_at_str = log_df['made_at'].astype(str).str[:10]
        mask = (log_df['stock_code'] == stock_code) & (made_at_str == today_str) & \
               (log_df['target_time'].astype(str).str[:10] == str(pred_info['target_time'])[:10])
        if mask.any():
            return
    new_row = {
        'stock_code': stock_code, 'stock_name': stock_name,
        'made_at': datetime.now(), 'target_time': pred_info['target_time'],
        'predicted': pred_info['predicted'], 'upper': pred_info['upper'], 'lower': pred_info['lower'],
        'last_price': pred_info['last_price'], 'actual_price': None, 'resolved': False
    }
    log_df = pd.concat([log_df, pd.DataFrame([new_row])], ignore_index=True)
    save_log(log_df)


def resolve_pending_predictions(stock_code):
    log_df = load_log()
    if log_df.empty:
        return log_df
    now = pd.Timestamp.now()
    pending_mask = (log_df['stock_code'] == stock_code) & (log_df['resolved'] != True)
    pending = log_df[pending_mask]
    if pending.empty:
        return log_df

    actual_hist = None
    for idx, row in pending.iterrows():
        try:
            target_time = pd.to_datetime(row['target_time'])
        except Exception:
            continue
        if target_time > now:
            continue
        if actual_hist is None:
            try:
                hist = yf.download(f"{stock_code}.KS", period="7d", interval="15m", progress=False)
                if hist is None or hist.empty:
                    hist = yf.download(f"{stock_code}.KQ", period="7d", interval="15m", progress=False)
                if hist is not None and not hist.empty:
                    hist = hist.reset_index()
                    if isinstance(hist.columns, pd.MultiIndex):
                        hist.columns = [c[0] for c in hist.columns]
                    time_col = 'Datetime' if 'Datetime' in hist.columns else 'Date'
                    hist = hist.rename(columns={time_col: 'Date'})
                    hist['Date'] = _to_naive(pd.to_datetime(hist['Date']))
                    actual_hist = hist
                else:
                    actual_hist = pd.DataFrame()
            except Exception:
                actual_hist = pd.DataFrame()

        if actual_hist is not None and not actual_hist.empty:
            target_time_naive = _to_naive(target_time)
            diffs = (actual_hist['Date'] - target_time_naive).abs()
            nearest_idx = diffs.idxmin()
            if diffs[nearest_idx] <= pd.Timedelta(hours=6):
                actual_price = float(actual_hist.loc[nearest_idx, 'Close'])
                log_df.loc[idx, 'actual_price'] = actual_price
                log_df.loc[idx, 'resolved'] = True

    save_log(log_df)
    return log_df


def add_prediction_to_chart(fig, pred_info):
    fig.add_trace(go.Scatter(
        x=[pred_info['last_time'], pred_info['target_time']],
        y=[pred_info['last_price'], pred_info['predicted']],
        mode='lines+markers', line=dict(dash='dash', color='#e67e22', width=2),
        marker=dict(size=9, symbol='star'), name='🔮 AI 예상 추세선'
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[pred_info['target_time'], pred_info['target_time']],
        y=[pred_info['lower'], pred_info['upper']],
        mode='lines', line=dict(color='rgba(230,126,34,0.5)', width=7),
        name='예상 범위(신뢰구간)'
    ), row=1, col=1)
    return fig


def build_price_chart(stock_df, stock_name, stock_code, interval, matched_news_df=None, pred_info=None):
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

    # 🛠️ Plotly 인덱스 버그 완전 수정을 위해 순수한 파이썬 리스트 구조 적용
    if matched_news_df is not None and not matched_news_df.empty:
        fig.add_trace(go.Scatter(
            x=matched_news_df['Date'].tolist(), 
            y=(matched_news_df['High'] * 1.01).tolist(),
            mode='markers', 
            marker=dict(symbol='triangle-down', size=12, color='gold', line=dict(width=1, color='#333')),
            hovertemplate="<b>📰 %{customdata[0]}</b><br>%{customdata[1]}<extra></extra>",
            customdata=matched_news_df[['뉴스제목', '뉴스시간']].values.tolist(),
            name='관련 뉴스'
        ), row=1, col=1)

    if pred_info is not None:
        add_prediction_to_chart(fig, pred_info)

    if 'Volume' in stock_df.columns:
        vol_colors = ['#d64550' if c >= o else '#3b82f6' for o, c in zip(stock_df['Open'], stock_df['Close'])]
        fig.add_trace(go.Bar(
            x=stock_df['Date'], y=stock_df['Volume'], name='거래량',
            marker_color=vol_colors, showlegend=False
        ), row=2, col=1)

    fig.update_layout(
        template='plotly_white', height=680,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(t=60, b=20, l=10, r=10),
        hovermode='x unified'
    )
    fig.update_xaxes(rangeslider_visible=False, row=2, col=1)
    return fig


# 📌 탭 설정
tab1, tab2 = st.tabs(["📰 오늘의 한국 경제 시황 (기본 화면)", "📊 개별 종목 상세 분석 (예상가 · 뉴스 매핑 · AI 분석)"])

# =========================================================================
# [기본 화면] TAB 1
# =========================================================================
with tab1:
    st.subheader("👑 오늘의 대한민국 경제 및 증시 종합 브리핑")

    with st.spinner("Groq AI가 실시간 경제 뉴스를 분석 중입니다..."):
        economy_news_df = fetch_google_news("한국 경제 시황 OR 국내 증시 OR 코스피 코스닥", max_results=40)

        if not economy_news_df.empty:
            prompt = (
                "다음은 대한민국 경제 및 주식 시장 관련 최신 뉴스 헤드라인입니다.\n"
                + "\n".join(f"- {t}" for t in economy_news_df['제목'].head(20).tolist())
                + "\n\n위 뉴스들을 종합적으로 분석하여 다음 내용을 한국어로 친절하게 작성해줘."
                  "\n1. 현재 시장의 전반적인 분위기와 투자 심리 상태"
                  "\n2. 오늘 투자자가 꼭 알아야 할 핵심 경제 이슈 3가지."
            )
            # Groq 전용 분석 호출
            ok, text = ask_groq(prompt, system_role="당신은 영리한 경제 수석 분석가입니다.")
            st.markdown("### 🤖 전문 AI(Groq)의 시장 종합 요약 리포트")
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
# [선택 화면] TAB 2
# =========================================================================
with tab2:
    st.subheader("🔍 개별 종목 주가 차트 · 오늘 예상가 · 종목 전용 AI 뉴스 분석")
    st.sidebar.header("⚙️ 종목 및 차트 조건 설정")

    if is_gsheet_connected():
        st.sidebar.success("🟢 학습 데이터: Google Sheets 연동됨")
    else:
        st.sidebar.warning("🟡 학습 데이터: 로컬 CSV 사용 중 (Sheets 미연동)")

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
        last_close = stock_df['Close'].iloc[-1]
        prev_close = stock_df['Close'].iloc[-2] if len(stock_df) > 1 else last_close
        change = last_close - prev_close
        change_pct = (change / prev_close * 100) if prev_close else 0
        c1, c2, c3 = st.columns(3)
        c1.metric(f"{stock_name} 현재가", f"{last_close:,.0f}원", f"{change:,.0f}원 ({change_pct:.2f}%)")
        c2.metric("최고가 (조회기간)", f"{stock_df['High'].max():,.0f}원")
        c3.metric("최저가 (조회기간)", f"{stock_df['Low'].min():,.0f}원")

        resolve_pending_predictions(stock_code)
        pred_info = generate_prediction(stock_df, view_type)
        bias, n_samples = get_calibration_bias(stock_code)

        if pred_info is not None:
            calibrated = n_samples >= 3
            if calibrated:
                pred_info['predicted'] += bias
                pred_info['upper'] += bias
                pred_info['lower'] += bias

            log_prediction(stock_code, stock_name, pred_info)

            st.markdown("#### 🔮 AI 예상가")
            p1, p2, p3 = st.columns(3)
            p1.metric(pred_info['label'], f"{pred_info['predicted']:,.0f}원",
                      f"{pred_info['predicted'] - pred_info['last_price']:,.0f}원")
            p2.metric("예상 상단", f"{pred_info['upper']:,.0f}원")
            p3.metric("예상 하단", f"{pred_info['lower']:,.0f}원")

            cal_msg = f"과거 예측 {n_samples}건의 오차(평균 {bias:,.0f}원)를 반영해 보정된 값입니다." if calibrated \
                else f"아직 학습 데이터가 부족합니다 (누적 {n_samples}건, 3건 이상부터 자동 보정 적용)."
            st.caption(f"📌 {cal_msg} 최근 가격 추세를 선형회귀로 연장한 통계적 추정치이며, **투자 조언이 아닙니다.**")
        else:
            st.info("💡 현재 조회 조건에서는 예상가를 계산할 수 없습니다.")

        st.markdown(f"#### 📊 {stock_name}({stock_code}) 실시간 차트 ({interval}) — 뉴스 마커 & 예상가 포함")
        matched_news_df = match_news_to_chart(stock_df, stock_news_df, interval)
        fig = build_price_chart(stock_df, stock_name, stock_code, interval, matched_news_df, pred_info)
        st.plotly_chart(fig, use_container_width=True)
        if not matched_news_df.empty:
            st.caption(f"🔶 차트 위의 금색 삼각형 마커 = 해당 시간대에 매칭된 뉴스 (총 {len(matched_news_df)}건). 마우스를 올리면 제목이 보여요.")

        with st.expander("📊 이 종목의 과거 예측 정확도 확인 (학습 데이터)"):
            log_df = load_log()
            stock_log = log_df[log_df['stock_code'] == stock_code].copy() if not log_df.empty else pd.DataFrame()
            if stock_log.empty:
                st.write("아직 축적된 예측 기록이 없습니다.")
            else:
                stock_log['오차(원)'] = stock_log.apply(
                    lambda r: (r['actual_price'] - r['predicted']) if pd.notna(r.get('actual_price')) else None, axis=1)
                stock_log['오차율(%)'] = stock_log.apply(
                    lambda r: (r['오차(원)'] / r['predicted'] * 100) if pd.notna(r.get('오차(원)')) and r['predicted'] else None, axis=1)
                display_cols = ['made_at', 'target_time', 'last_price', 'predicted', 'actual_price', '오차(원)', '오차율(%)', 'resolved']
                st.dataframe(stock_log[display_cols].sort_values('made_at', ascending=False), use_container_width=True)

                resolved_only = stock_log[stock_log['resolved'] == True]
                if not resolved_only.empty:
                    mape = resolved_only['오차율(%)'].abs().mean()
                    st.metric("평균 절대 오차율 (MAPE)", f"{mape:.2f}%")

        st.markdown("---")

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
            st.caption("이 종목과 관련된 최신 뉴스만 모아서 Groq AI가 직접 종목 요약을 해줍니다.")

            if stock_news_df.empty:
                st.info("분석할 뉴스가 없습니다.")
            else:
                if st.button(f"🔍 {stock_name} 뉴스 AI 분석 실행", key="analyze_stock_news", use_container_width=True):
                    with st.spinner(f"Groq AI가 {stock_name} 관련 뉴스 {len(stock_news_df)}건을 종목 분석 중입니다..."):
                        headlines = "\n".join(
                            f"- ({row['Date_str']}) {row['제목']}"
                            for _, row in stock_news_df.head(25).iterrows()
                        )
                        stock_prompt = (
                            f"아래는 '{stock_name}({stock_code})' 종목과 관련된 최신 뉴스 헤드라인 목록입니다.\n\n"
                            f"{headlines}\n\n"
                            f"위 뉴스만을 근거로 다음 항목을 한국어로 명확하게 정리해줘.\n"
                            f"1. '{stock_name}'에 대한 현재 뉴스 흐름의 전반적인 톤(긍정/부정/중립)과 그 이유\n"
                            f"2. 주가에 긍정적 영향을 줄 수 있는 핵심 요인 (있다면 최대 3가지)\n"
                            f"3. 주가에 부정적 영향을 줄 수 있는 핵심 요인 (있다면 최대 3가지)\n"
                            f"4. 투자자가 앞으로 주의 깊게 지켜봐야 할 포인트 1~2가지\n"
                            f"※ 확정적인 투자 권유가 아니라 뉴스 기반의 참고 분석임을 명시해줘."
                        )
                        ok, text = ask_groq(stock_prompt, system_role="당신은 대한민국 주식 시장을 전문적으로 분석하는 계량 애널리스트입니다.")
                        if ok:
                            st.success(text)
                        else:
                            st.warning(text)
    else:
        st.error(f"❌ '{stock_name}' 데이터를 가져오지 못했습니다.")
