"""
Streamlit dashboard for the Nifty 100 Swing Trading Agent.

Run with:  streamlit run app.py
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import config
import db_handler
from scheduler import is_market_open, run_pipeline
from ta_engine import (
    calculate_buy_sell_pressure,
    calculate_volume_metrics,
    describe_macd_pattern,
    get_chart_data,
    get_reference_session,
    nearest_fib_level,
)

st.set_page_config(
    page_title="Nifty 100 Swing Trading Agent",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="collapsed",
)

# Pull the title up to sit ~12px below Streamlit's top header bar (default
# header height ~3.75rem), and outline the shortlist table in black.
st.markdown(
    """
    <style>
    .block-container { padding-top: calc(3.75rem + 12px); }
    [data-testid="stDataFrame"] { border: 1px solid #000000; }
    </style>
    """,
    unsafe_allow_html=True,
)

# init_db() runs idempotent CREATE TABLE/ALTER checks - only needed once per
# session, not on every script rerun (every button click / row selection).
if "db_initialized" not in st.session_state:
    db_handler.init_db()
    st.session_state["db_initialized"] = True

st.title("📈 Nifty 100 Swing Trading Agent")

# ---------------------------------------------------------------------------
# Authentication - Google sign-in via Streamlit's built-in auth (requires
# Authlib + a [auth]/[auth.google] section in .streamlit/secrets.toml, see
# .streamlit/secrets.toml.example). Toggle with config.AUTH_ENABLED.
# ---------------------------------------------------------------------------
if config.AUTH_ENABLED:
    if not getattr(st.user, "is_logged_in", False):
        st.write("Please sign in with your Google account to continue.")
        if st.button("🔐 Log in with Google", type="primary"):
            try:
                st.login("google")
            except Exception as e:
                st.error(
                    "Google sign-in isn't configured yet. Copy "
                    ".streamlit/secrets.toml.example to .streamlit/secrets.toml "
                    f"and fill in your Google OAuth credentials. ({e})"
                )
        st.stop()

    user_email = (st.user.email or "").lower()
    is_admin = user_email in config.AUTH_ADMIN_EMAILS

    # First-come-first-served access, capped at AUTH_MAX_USERS - admins are
    # exempt and can free up slots from the Admin tab below.
    if not is_admin:
        user_status = db_handler.get_user_status(user_email)

        if user_status == "revoked":
            st.error("Your access to this app has been revoked by the administrator.")
            if st.button("OK, sign me out"):
                st.logout()
            st.stop()

        if user_status != "active":
            if db_handler.get_authorized_user_count() >= config.AUTH_MAX_USERS:
                st.error(
                    f"This app is limited to {config.AUTH_MAX_USERS} users and that limit "
                    "has already been reached. Contact the administrator for access."
                )
                if st.button("Log out"):
                    st.logout()
                st.stop()
            db_handler.register_user(user_email, st.user.name or user_email)
else:
    user_email = ""
    is_admin = False

st.caption(
    "Mathematical screening (RSI, MACD, Fibonacci retracements) "
    "+ Grok AI commentary, tuned for 2-3 week swing setups."
)

run_clicked = st.button("🔍 Run Full Scan Now", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Sidebar - controls
# ---------------------------------------------------------------------------
with st.sidebar:
    # Force readable (black) text for everything in the sidebar - the default
    # muted caption color is low-contrast against the beige sidebar background.
    st.markdown(
        "<style>[data-testid='stSidebar'] * { color: #000000 !important; }</style>",
        unsafe_allow_html=True,
    )

    st.header("Controls")

    if config.AUTH_ENABLED:
        st.caption(f"Signed in as **{st.user.name or user_email}**" + (" 👑 (admin)" if is_admin else ""))
        if st.button("Log out", key="sidebar_logout"):
            st.logout()

        st.markdown("---")

    try:
        if is_market_open():
            st.success("NSE market is currently OPEN")
        else:
            st.info("NSE market is currently CLOSED — scan will use the latest available daily candle.")
    except Exception:
        st.warning("Could not determine market hours.")

    st.markdown("---")
    st.caption(
        f"Universe size: {len(config.SCAN_UNIVERSE)} tickers "
        f"(Nifty 100 + Gold/Silver, always tracked)"
    )
    st.caption(f"Fibonacci lookback: {config.FIB_LOOKBACK_DAYS} trading days (~3-4 months)")
    st.caption(f"RSI thresholds: oversold < {config.RSI_OVERSOLD}, overbought > {config.RSI_OVERBOUGHT}")
    st.caption(f"MACD: {config.MACD_FAST}/{config.MACD_SLOW}/{config.MACD_SIGNAL}")
    st.caption(
        f"Volume surge: >= {config.VOLUME_SURGE_RATIO}x its "
        f"{config.VOLUME_AVG_WINDOW}-day average"
    )
    st.caption(
        f"Sector trend: {config.SECTOR_TREND_LOOKBACK_DAYS}-day sector average return, "
        f">= {config.SECTOR_TREND_THRESHOLD * 100:.1f}% for a score bonus (small impact)"
    )
    st.caption(
        f"Buy/Sell pressure: {config.BUY_SELL_PRESSURE_WINDOW}-day volume split by "
        "up-days vs down-days (proxy, not live order-book data)"
    )

# ---------------------------------------------------------------------------
# Run a fresh scan on demand
# ---------------------------------------------------------------------------
if run_clicked:
    progress_bar = st.progress(0.0, text="Starting scan...")

    def _progress_cb(i, total, ticker):
        progress_bar.progress(i / total, text=f"Scanning {ticker} ({i}/{total})")

    try:
        with st.spinner("Running full Nifty 100 scan and requesting AI commentary..."):
            shortlist, ai_commentary, scan_date = run_pipeline(progress_callback=_progress_cb)
        progress_bar.empty()
        st.success(f"Scan complete for {scan_date}: {len(shortlist)} qualifying setup(s) found.")
    except Exception as e:
        progress_bar.empty()
        st.error(f"Scan failed: {e}")

# ---------------------------------------------------------------------------
# Load latest (or selected) results
# ---------------------------------------------------------------------------
latest = db_handler.get_latest_scan()

if latest is None:
    st.warning(
        "No scans have been run yet. Click **Run Full Scan Now** in the sidebar "
        "to generate the first shortlist."
    )
    st.stop()

scan_date, ai_commentary, signals_df = latest

available_dates = db_handler.get_available_scan_dates()
selected_date = st.selectbox("Viewing scan from:", available_dates, index=0)

if selected_date != scan_date:
    result = db_handler.get_scan_by_date(selected_date)
    if result is not None:
        scan_date, ai_commentary, signals_df = result

# ---------------------------------------------------------------------------
# Main views - split into tabs so the page isn't one long scroll. This also
# reads much better on mobile, where each tab's content fits the viewport on
# its own instead of stacking every section vertically.
# ---------------------------------------------------------------------------
selected_rows = []

tab_names = ["📋 Shortlist", "🤖 AI Commentary", "📐 Chart Analysis"]
if is_admin:
    tab_names.append("👑 Admin")

tabs = st.tabs(tab_names)
tab_shortlist, tab_ai, tab_chart = tabs[0], tabs[1], tabs[2]
tab_admin = tabs[3] if is_admin else None

# ---------------------------------------------------------------------------
# Shortlist table styling - color cues so strong/weak signals are visible at
# a glance instead of requiring a column-by-column read.
# ---------------------------------------------------------------------------
_MAX_SCORE = (
    config.SCORE_FIB_KEY_LEVEL + config.SCORE_RSI_EXTREME
    + config.SCORE_MACD_PROXIMITY + config.SCORE_VOLUME + config.SCORE_SECTOR_TREND
)


def _style_rsi(val):
    """Green = oversold (potential bullish reversal), red = overbought (bearish)."""
    if val <= config.RSI_OVERSOLD:
        return "background-color: #d4edda; color: #155724; font-weight: 600"
    if val >= config.RSI_OVERBOUGHT:
        return "background-color: #f8d7da; color: #721c24; font-weight: 600"
    return ""


def _style_score(val):
    """Shade Score green, more intensely the closer it is to the max of 100."""
    ratio = max(0.0, min(1.0, val / _MAX_SCORE))
    if ratio >= 0.7:
        return "background-color: #c3e6cb; color: #155724; font-weight: 700"
    if ratio >= 0.45:
        return "background-color: #e6f4ea; color: #1e7e34; font-weight: 600"
    return ""


def _style_macd_diff(val):
    """Color the MACD-Signal diff by sign: green = bullish, red = bearish."""
    try:
        diff = float(val.split(" ")[0])
    except (ValueError, IndexError):
        return ""
    if diff > 0:
        return "color: #2e8b57; font-weight: 600"
    if diff < 0:
        return "color: #c0392b; font-weight: 600"
    return ""


# ---------------------------------------------------------------------------
# Tab 1: Shortlist table
# ---------------------------------------------------------------------------
with tab_shortlist:
    st.subheader(f"Shortlist — {scan_date}")

    with st.expander("📐 How the score is calculated (max 100)", expanded=False):
        st.markdown(
            f"""
| # | Component | Max points | Awarded when... |
|---|---|---|---|
| 1 | Fibonacci proximity | **{config.SCORE_FIB_KEY_LEVEL}** | Price is within {config.FIB_PROXIMITY_PCT * 100:.0f}% of a **key** level (50% / 61.8%) |
| 1b | (or) | **{config.SCORE_FIB_OTHER_LEVEL}** | ...or within {config.FIB_PROXIMITY_PCT * 100:.0f}% of any other level (0% / 23.6% / 38.2% / 100%) |
| 2 | RSI extreme | **{config.SCORE_RSI_EXTREME}** | RSI(14) <= {config.RSI_OVERSOLD} (oversold) or >= {config.RSI_OVERBOUGHT} (overbought) |
| 3 | MACD crossover proximity | **{config.SCORE_MACD_PROXIMITY}** | MACD histogram is small and shrinking - converging toward a crossover |
| 4 | Volume confirmation | **{config.SCORE_VOLUME}** | Latest volume >= {config.VOLUME_SURGE_RATIO}x its {config.VOLUME_AVG_WINDOW}-day average |
| 5 | Sector trend alignment | **{config.SCORE_SECTOR_TREND}** | Stock's bullish/bearish bias is confirmed by its sector's {config.SECTOR_TREND_LOOKBACK_DAYS}-day average return moving >= {config.SECTOR_TREND_THRESHOLD * 100:.1f}% the same direction |
| | **Total (best case)** | **{config.SCORE_FIB_KEY_LEVEL + config.SCORE_RSI_EXTREME + config.SCORE_MACD_PROXIMITY + config.SCORE_VOLUME + config.SCORE_SECTOR_TREND}** | 1 (key level) + 2 + 3 + 4 + 5 |

Items 1/1b are mutually exclusive (only the nearest Fib level counts). Item 5
is intentionally a small/secondary factor - sector trend never decides a
setup on its own, it only adds a bit of conviction when it agrees with the
stock's own signals.
            """
        )

    if signals_df.empty:
        st.info("No stocks met the scoring threshold on this date.")
    else:
        display_df = signals_df.copy()
        display_df["macd_hist_display"] = display_df.apply(
            lambda r: f"{r['macd_hist']:.2f} ({r['macd_hist_direction']})", axis=1
        )
        display_df["buy_sell_display"] = display_df.apply(
            lambda r: f"{r['buy_pct']:.0f}% / {r['sell_pct']:.0f}%", axis=1
        )
        display_df["sector_trend_display"] = (display_df["sector_trend_pct"] * 100).round(2)

        display_df = display_df[[
            "ticker", "sector", "close", "rsi", "macd_hist_display", "nearest_fib_level",
            "nearest_fib_price", "fib_distance_pct", "week52_high",
            "pct_from_52w_high", "volume_ratio", "buy_sell_display",
            "sector_trend_display", "score",
        ]].rename(columns={
            "ticker": "Ticker",
            "sector": "Sector",
            "close": "Price",
            "rsi": "RSI",
            "macd_hist_display": "MACD-Signal Diff (dir)",
            "nearest_fib_level": "Nearest Fib",
            "nearest_fib_price": "Fib Price",
            "fib_distance_pct": "Fib Dist %",
            "week52_high": "52W High",
            "pct_from_52w_high": "% From 52W High",
            "volume_ratio": "Vol vs 20D Avg",
            "buy_sell_display": "Buy % / Sell %",
            "sector_trend_display": "Sector Trend % (10D)",
            "score": "Score",
        })
        display_df["Fib Dist %"] = (display_df["Fib Dist %"] * 100).round(2)
        display_df["% From 52W High"] = (display_df["% From 52W High"] * 100).round(2)
        display_df["Vol vs 20D Avg"] = display_df["Vol vs 20D Avg"].round(2)
        display_df[["Price", "RSI", "Fib Price", "52W High"]] = (
            display_df[["Price", "RSI", "Fib Price", "52W High"]].round(2)
        )

        # Compact view by default (better on mobile / narrow screens) - the
        # remaining columns are still available via the checkbox below, and
        # always shown in the per-row detail panel when a row is clicked.
        compact_columns = [
            "Ticker", "Sector", "Price", "RSI", "MACD-Signal Diff (dir)",
            "Nearest Fib", "Fib Dist %", "Score",
        ]
        show_all_cols = st.checkbox(
            "Show all columns (52W high, volume, buy/sell pressure, sector trend)",
            value=False,
        )
        table_df = display_df if show_all_cols else display_df[compact_columns]

        # With all columns shown, let the table keep its natural (wider)
        # width and scroll horizontally instead of squeezing every column
        # into the container.
        table_width = "content" if show_all_cols else "stretch"
        styled_table = (
            table_df.style.set_properties(**{
                "background-color": "#FFFFFF",
                "color": "#000000",
                "border": "1px solid #000000",
            })
            .map(_style_rsi, subset=["RSI"])
            .map(_style_score, subset=["Score"])
            .map(_style_macd_diff, subset=["MACD-Signal Diff (dir)"])
        )
        select_event = st.dataframe(
            styled_table, width=table_width, hide_index=True,
            on_select="rerun", selection_mode="single-row", key="shortlist_table",
        )
        st.caption(
            f"🟩 **RSI** highlighted green when oversold (<= {config.RSI_OVERSOLD}, "
            f"possible bullish reversal) or red when overbought (>= {config.RSI_OVERBOUGHT}, "
            f"possible bearish reversal). **Score** shaded green for stronger setups "
            f"(darker = closer to {_MAX_SCORE}). **MACD-Signal Diff** colored green "
            "(bullish) / red (bearish) by sign."
        )

        if show_all_cols:
            st.caption(
                "**MACD-Signal Diff (dir)**: positive = MACD line above Signal line "
                "(bullish momentum), negative = MACD line below Signal line (bearish "
                "momentum); the **(up)**/**(down)** tag shows whether this histogram "
                "value rose or fell vs. the previous session - a quick read on current "
                "MACD momentum direction. **% From 52W High**: how far the current "
                "price sits below its 52-week high (0% = at the 52-week high). "
                "**Vol vs 20D Avg**: latest session's volume as a multiple of its "
                f"20-day average (>= {config.VOLUME_SURGE_RATIO}x counts as a volume "
                "surge). **Buy % / Sell %**: a volume-weighted proxy for buy/sell "
                f"pressure over the last {config.BUY_SELL_PRESSURE_WINDOW} sessions "
                "(not live order-book data). **Sector Trend % (10D)**: this stock's "
                "sector's average return over the last 10 sessions."
            )
        else:
            st.caption(
                "**MACD-Signal Diff (dir)**: positive = MACD line above Signal line "
                "(bullish momentum), negative = below (bearish); **(up)**/**(down)** "
                "shows the change vs. the previous session. **Fib Dist %**: how close "
                "price is to its nearest Fibonacci retracement level. Click a row "
                "below for the full breakdown (52-week high, volume, buy/sell "
                "pressure, sector trend, and signals)."
            )

        selected_rows = select_event["selection"]["rows"] if select_event else []
        if selected_rows:
            sel_row = signals_df.iloc[selected_rows[0]]
            st.markdown(f"**{sel_row['ticker']}** ({sel_row['sector']}) — score {sel_row['score']}/100")

            d1, d2, d3, d4, d5 = st.columns(5)
            with d1:
                st.metric(
                    "Nearest Fib",
                    f"{sel_row['nearest_fib_level']} @ {sel_row['nearest_fib_price']:.2f}",
                )
            with d2:
                st.metric(
                    "52W High", f"₹{sel_row['week52_high']:.2f}",
                    delta=f"-{sel_row['pct_from_52w_high'] * 100:.1f}%",
                )
            with d3:
                st.metric("Vol vs 20D Avg", f"{sel_row['volume_ratio']:.2f}x")
            with d4:
                st.metric("Buy % / Sell %", f"{sel_row['buy_pct']:.0f}% / {sel_row['sell_pct']:.0f}%")
            with d5:
                st.metric(
                    f"Sector Trend ({config.SECTOR_TREND_LOOKBACK_DAYS}D)",
                    f"{sel_row['sector_trend_pct'] * 100:+.2f}%",
                )

            for reason in sel_row["reasons"]:
                st.markdown(f"- {reason}")
        else:
            st.caption("👆 Click a row above to see its full signal breakdown.")

# ---------------------------------------------------------------------------
# Tab 2: AI commentary
# ---------------------------------------------------------------------------
with tab_ai:
    st.subheader("🤖 AI Analyst Commentary")
    st.markdown(ai_commentary if ai_commentary else "_No commentary available._")

# ---------------------------------------------------------------------------
# Tab 3: Fibonacci retracement analysis
# ---------------------------------------------------------------------------
with tab_chart:
    st.subheader("📐 Fibonacci Retracement Analysis")

    if not signals_df.empty:
        ticker_list = signals_df["ticker"].tolist()

        # Sync with a clicked shortlist row above, if any - clicking a different
        # row re-points this selector (and therefore the chart below) at that
        # ticker. Manual changes to the selector below still work independently.
        if selected_rows:
            sel_ticker = signals_df.iloc[selected_rows[0]]["ticker"]
            if sel_ticker in ticker_list:
                st.session_state["chart_ticker_select"] = sel_ticker

        if st.session_state.get("chart_ticker_select") not in ticker_list:
            st.session_state["chart_ticker_select"] = ticker_list[0]

        chart_ticker = st.selectbox(
            "Select a stock to analyze", ticker_list, key="chart_ticker_select"
        )
        st.caption(
            "👆 Click a row in the **Shortlist** tab to load that stock's "
            "chart here, or pick one manually."
        )

        chart_df, levels, peak, trough = get_chart_data(chart_ticker)
        if chart_df is None:
            st.error(f"Could not load price data for {chart_ticker}.")
        else:
            # The actual swing high/low bars the Fib levels were derived from.
            fib_window = chart_df.tail(config.FIB_LOOKBACK_DAYS)
            peak_date = fib_window["High"].idxmax()
            trough_date = fib_window["Low"].idxmin()

            current_price = float(chart_df["Close"].iloc[-1])
            current_ratio = (peak - current_price) / (peak - trough) if peak != trough else float("nan")
            level_name, level_price, distance_pct = nearest_fib_level(current_price, levels)
            _, avg_volume_20, volume_ratio = calculate_volume_metrics(chart_df)

            # Limit the plotted chart to the most recent CHART_DISPLAY_MONTHS -
            # indicators above (RSI/MACD/Fib levels/swing high-low) are still
            # computed from the full history / FIB_LOOKBACK_DAYS window and
            # apply across this shorter view.
            display_cutoff = chart_df.index.max() - pd.DateOffset(months=config.CHART_DISPLAY_MONTHS)
            chart_display_df = chart_df[chart_df.index >= display_cutoff]

            # --- Price + volume + MACD chart ---------------------------------------
            fig = make_subplots(
                rows=3, cols=1, shared_xaxes=True,
                row_heights=[0.5, 0.15, 0.35], vertical_spacing=0.03,
                specs=[[{}], [{"secondary_y": True}], [{}]],
            )

            fig.add_trace(go.Candlestick(
                x=chart_display_df.index,
                open=chart_display_df["Open"], high=chart_display_df["High"],
                low=chart_display_df["Low"], close=chart_display_df["Close"],
                name=chart_ticker,
            ), row=1, col=1)

            # Shade the bands between consecutive Fibonacci levels so the
            # retracement grid reads as zones, not just lines. Opacity bumped
            # up to 0.18 (from an original 0.10) for better visibility against
            # the light beige chart background.
            zone_colors = [
                "rgba(128,128,128,0.18)", "rgba(224,123,57,0.18)",
                "rgba(184,134,11,0.18)", "rgba(46,139,87,0.18)",
                "rgba(31,119,180,0.18)", "rgba(128,128,128,0.18)",
            ]
            sorted_levels = sorted(levels.items(), key=lambda kv: kv[1])
            for i in range(len(sorted_levels) - 1):
                (_, y0), (_, y1) = sorted_levels[i], sorted_levels[i + 1]
                fig.add_hrect(
                    y0=y0, y1=y1, fillcolor=zone_colors[i % len(zone_colors)],
                    line_width=0, row=1, col=1,
                )

            # Darkgoldenrod (#b8860b) replaces the original gold (#d4af37) for
            # the 38.2% level - the original was low-contrast on a light
            # background.
            level_colors = {
                "0.0%": "grey", "23.6%": "#e07b39", "38.2%": "#b8860b",
                "50.0%": "#2e8b57", "61.8%": "#1f77b4", "100.0%": "grey",
            }
            for name, price in levels.items():
                color = level_colors.get(name, "grey")
                fig.add_hline(
                    y=price,
                    line_dash="dash",
                    line_color=color,
                    line_width=1.5,
                    annotation_text=f"{name}: {price:.2f}",
                    annotation_position="right",
                    annotation_font=dict(size=11, color=color),
                    annotation_bgcolor="rgba(255,255,255,0.7)",
                    annotation_bordercolor=color,
                    annotation_borderwidth=1,
                    row=1, col=1,
                )

            fig.add_hline(
                y=current_price,
                line_dash="solid",
                line_color="#FF00FF",
                line_width=2.5,
                annotation_text=f"CMP: {current_price:.2f}",
                annotation_position="left",
                annotation_font=dict(size=12, color="#FF00FF"),
                annotation_bgcolor="rgba(255,255,255,0.7)",
                annotation_bordercolor="#FF00FF",
                annotation_borderwidth=1,
                row=1, col=1,
            )

            # Mark the exact swing-high/swing-low bars the levels were measured
            # from - but only if they fall within the displayed window, since
            # the FIB_LOOKBACK_DAYS window (used to derive the levels) can
            # extend further back than the CHART_DISPLAY_MONTHS shown here.
            swing_x, swing_y, swing_text, swing_colors, swing_symbols = [], [], [], [], []
            if peak_date >= display_cutoff:
                swing_x.append(peak_date)
                swing_y.append(peak)
                swing_text.append(f"Swing High {peak:.2f}")
                swing_colors.append("#2e8b57")
                swing_symbols.append("triangle-down")
            if trough_date >= display_cutoff:
                swing_x.append(trough_date)
                swing_y.append(trough)
                swing_text.append(f"Swing Low {trough:.2f}")
                swing_colors.append("#c0392b")
                swing_symbols.append("triangle-up")

            if swing_x:
                fig.add_trace(go.Scatter(
                    x=swing_x,
                    y=swing_y,
                    mode="markers+text",
                    marker=dict(size=12, color=swing_colors, symbol=swing_symbols),
                    text=swing_text,
                    textposition="top center",
                    name="Swing points",
                    showlegend=False,
                ), row=1, col=1)

            # Volume bars (green/red by up/down day) + RSI line on a secondary axis.
            vol_colors = [
                "#2e8b57" if c >= o else "#c0392b"
                for c, o in zip(chart_display_df["Close"], chart_display_df["Open"])
            ]
            fig.add_trace(go.Bar(
                x=chart_display_df.index, y=chart_display_df["Volume"],
                marker_color=vol_colors, name="Volume", showlegend=False,
            ), row=2, col=1)
            fig.add_trace(go.Scatter(
                x=chart_display_df.index, y=chart_display_df["RSI"], mode="lines",
                line=dict(color="#6a3d9a", width=1.5), name="RSI (14)",
            ), row=2, col=1, secondary_y=True)
            fig.add_hline(
                y=config.RSI_OVERBOUGHT, line_dash="dot", line_color="#c0392b",
                line_width=1, row=2, col=1, secondary_y=True,
            )
            fig.add_hline(
                y=config.RSI_OVERSOLD, line_dash="dot", line_color="#2e8b57",
                line_width=1, row=2, col=1, secondary_y=True,
            )

            # MACD panel: histogram (green/red by sign) + MACD/Signal lines + zero line.
            macd_hist_colors = [
                "#2e8b57" if v >= 0 else "#c0392b" for v in chart_display_df["MACD_HIST"]
            ]
            fig.add_trace(go.Bar(
                x=chart_display_df.index, y=chart_display_df["MACD_HIST"],
                marker_color=macd_hist_colors, name="MACD Histogram", showlegend=False,
            ), row=3, col=1)
            fig.add_trace(go.Scatter(
                x=chart_display_df.index, y=chart_display_df["MACD"], mode="lines",
                line=dict(color="#1f77b4", width=1.5), name="MACD",
            ), row=3, col=1)
            fig.add_trace(go.Scatter(
                x=chart_display_df.index, y=chart_display_df["MACD_SIGNAL"], mode="lines",
                line=dict(color="#e07b39", width=1.5), name="Signal",
            ), row=3, col=1)
            fig.add_hline(y=0, line_color="grey", line_width=1, row=3, col=1)

            fig.update_layout(
                title=(
                    f"{chart_ticker} — Last {config.CHART_DISPLAY_MONTHS} Months "
                    f"(Fibonacci levels from {config.FIB_LOOKBACK_DAYS}-Day range) "
                    "+ Volume/RSI + MACD"
                ),
                xaxis_rangeslider_visible=False,
                height=900,
                yaxis_title="Price (INR)",
                yaxis2_title="Volume",
                yaxis3_title="MACD",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                paper_bgcolor="#FAF6EC",
                plot_bgcolor="#FFFDF6",
                font_color="#2B2A28",
            )
            fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1, secondary_y=True)
            fig.update_xaxes(title_text="Date", tickformat="%d %b %Y", row=3, col=1)
            st.plotly_chart(fig, use_container_width=True)

            # --- Metrics row 1, with arrows showing current MACD movement -----------
            macd_now = float(chart_df["MACD"].iloc[-1])
            macd_prev = float(chart_df["MACD"].iloc[-2])
            hist_now = float(chart_df["MACD_HIST"].iloc[-1])
            hist_prev = float(chart_df["MACD_HIST"].iloc[-2])

            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("RSI (14)", f"{chart_df['RSI'].iloc[-1]:.1f}")
            with col2:
                st.metric("MACD Line", f"{macd_now:.3f}", delta=f"{macd_now - macd_prev:+.3f}")
            with col3:
                st.metric("MACD Histogram", f"{hist_now:.3f}", delta=f"{hist_now - hist_prev:+.3f}")
            with col4:
                st.metric(f"Volume vs {config.VOLUME_AVG_WINDOW}D Avg", f"{volume_ratio:.2f}x")
            with col5:
                st.metric("Last Close", f"₹{current_price:.2f}")

            st.caption(
                "Arrows on **MACD Line** / **MACD Histogram** show the change vs. the "
                "previous session (green = rising, red = falling) - a quick read on "
                "the current direction of MACD momentum."
            )

            # --- Metrics row 2: previous session, buy/sell pressure, sector ---------
            prev_date, prev_open, prev_close = get_reference_session(chart_df)
            buy_pct, sell_pct = calculate_buy_sell_pressure(chart_df)

            chart_match = signals_df[signals_df["ticker"] == chart_ticker]
            if not chart_match.empty:
                sector = chart_match.iloc[0]["sector"]
                sector_trend_pct = chart_match.iloc[0]["sector_trend_pct"]
            else:
                sector = config.SECTOR_MAP.get(chart_ticker, "Unknown")
                sector_trend_pct = float("nan")

            col6, col7, col8, col9, col10 = st.columns(5)
            with col6:
                st.metric("Prev Session Open", f"₹{prev_open:.2f}", help=f"Session: {prev_date}")
            with col7:
                st.metric(
                    "Prev Session Close", f"₹{prev_close:.2f}",
                    delta=f"{prev_close - prev_open:+.2f}", help=f"Session: {prev_date}",
                )
            with col8:
                st.metric("Buy % / Sell %", f"{buy_pct:.0f}% / {sell_pct:.0f}%")
            with col9:
                st.metric("Sector", sector)
            with col10:
                if sector_trend_pct == sector_trend_pct:  # not NaN
                    st.metric(f"Sector Trend ({config.SECTOR_TREND_LOOKBACK_DAYS}D)", f"{sector_trend_pct * 100:+.2f}%")
                else:
                    st.metric(f"Sector Trend ({config.SECTOR_TREND_LOOKBACK_DAYS}D)", "n/a")

            if is_market_open():
                st.caption(
                    f"**Prev Session** ({prev_date}) shows the last fully completed "
                    "session's open/close - today's session is still live, so "
                    "today's bar is excluded. **Buy % / Sell %** is a volume-weighted "
                    f"proxy over the last {config.BUY_SELL_PRESSURE_WINDOW} sessions "
                    "(NOT live order-book data - real bid/ask depth as shown on "
                    "Zerodha/Groww requires a separate broker API)."
                )
            else:
                st.caption(
                    f"**Prev Session** ({prev_date}) shows today's just-closed "
                    "session's open/close (market is closed). **Buy % / Sell %** is "
                    f"a volume-weighted proxy over the last {config.BUY_SELL_PRESSURE_WINDOW} "
                    "sessions (NOT live order-book data - real bid/ask depth as shown "
                    "on Zerodha/Groww requires a separate broker API)."
                )

            st.markdown("**MACD Pattern Analysis**")
            st.info(describe_macd_pattern(chart_df))

            # --- Fibonacci retracement breakdown -------------------------------------
            st.markdown("**How These Fibonacci Levels Were Chosen**")
            st.caption(
                f"Swing High: ₹{peak:.2f} on {peak_date.date()} | "
                f"Swing Low: ₹{trough:.2f} on {trough_date.date()} "
                f"(highest High / lowest Low over the last {config.FIB_LOOKBACK_DAYS} "
                "trading days, marked on the chart above). Each level = Swing High − "
                "ratio × (Swing High − Swing Low)."
            )

            fib_rows = []
            for name, price in sorted(levels.items(), key=lambda kv: kv[1]):
                ratio = float(name.strip("%")) / 100
                dist_pct = (price - current_price) / current_price * 100 if current_price else float("nan")
                fib_rows.append({
                    "Level": name,
                    "Ratio": round(ratio, 3),
                    "Price": round(price, 2),
                    "Distance from CMP %": round(dist_pct, 2),
                })
            st.dataframe(pd.DataFrame(fib_rows), use_container_width=True, hide_index=True)
            st.caption(
                "**Ratio**: the Fibonacci retracement ratio for that level (0.0 = swing "
                "high, 1.0 = swing low). **Distance from CMP %**: positive = level is "
                "above the current price (potential resistance), negative = below "
                "(potential support)."
            )

            # --- Current position + likely next move ---------------------------------
            st.markdown("**Current Position & Likely Next Move**")
            st.markdown(
                f"Price (₹{current_price:.2f}) sits at the **{current_ratio * 100:.1f}%** "
                f"retracement of the swing range, nearest to the **{level_name}** level "
                f"(₹{level_price:.2f}), {distance_pct * 100:.2f}% away."
            )

            levels_above = [(n, p) for n, p in levels.items() if p > current_price]
            levels_below = [(n, p) for n, p in levels.items() if p < current_price]
            next_resistance = min(levels_above, key=lambda kv: kv[1]) if levels_above else None
            next_support = max(levels_below, key=lambda kv: kv[1]) if levels_below else None

            if next_resistance:
                r_name, r_price = next_resistance
                st.markdown(
                    f"- **Resistance above:** {r_name} level at ₹{r_price:.2f} "
                    f"({(r_price - current_price) / current_price * 100:.2f}% above CMP)."
                )
            if next_support:
                s_name, s_price = next_support
                st.markdown(
                    f"- **Support below:** {s_name} level at ₹{s_price:.2f} "
                    f"({(current_price - s_price) / current_price * 100:.2f}% below CMP)."
                )

            if hist_now >= 0:
                st.markdown(
                    "- MACD momentum is currently **bullish** (histogram above zero), "
                    "so price is more likely to move toward resistance / the swing "
                    "high while this holds."
                )
            else:
                st.markdown(
                    "- MACD momentum is currently **bearish** (histogram below zero), "
                    "so price is more likely to move toward support / the swing low "
                    "while this holds."
                )
    else:
        st.info("Run a scan to enable the Fibonacci retracement analysis.")

# ---------------------------------------------------------------------------
# Tab 4: Admin - manage who can access this app (admins only)
# ---------------------------------------------------------------------------
if is_admin:
    with tab_admin:
        st.subheader("👑 User Access Management")

        users = db_handler.get_all_authorized_users()
        active_count = sum(1 for u in users if u["status"] == "active")
        st.metric("Registered users", f"{active_count} / {config.AUTH_MAX_USERS}")
        admin_list = ", ".join(sorted(config.AUTH_ADMIN_EMAILS)) or "none configured"
        st.caption(
            "Users below were auto-registered on first Google sign-in "
            f"(first-come, first-served, capped at {config.AUTH_MAX_USERS}). "
            f"Admin account(s) ({admin_list}) always have access, don't count "
            "toward this limit, and are managed via the AUTH_ADMIN_EMAILS "
            "setting in .env, not from this table. **Revoke** signs a user out "
            "and frees their slot; **Restore** lets them back in."
        )

        if not users:
            st.info("No users have signed in yet.")
        else:
            h1, h2, h3, h4, h5 = st.columns([3, 3, 2, 1, 1])
            h1.markdown("**Email**")
            h2.markdown("**Name**")
            h3.markdown("**First login**")
            h4.markdown("**Status**")
            h5.markdown("**Action**")
            for u in users:
                c1, c2, c3, c4, c5 = st.columns([3, 3, 2, 1, 1])
                c1.write(u["email"])
                c2.write(u["name"] or "—")
                c3.write(u["first_login"])
                if u["status"] == "active":
                    c4.write("🟢 Active")
                    if c5.button("Revoke", key=f"revoke_{u['email']}"):
                        db_handler.revoke_user(u["email"])
                        st.rerun()
                else:
                    c4.write("🔴 Revoked")
                    if c5.button("Restore", key=f"restore_{u['email']}"):
                        db_handler.restore_user(u["email"])
                        st.rerun()

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "📈 **Nifty 100 Swing Trading Agent** — mathematical screening (RSI, MACD, "
    "Fibonacci retracements) plus AI-generated commentary, for 2-3 week swing "
    "setups on NSE-listed stocks and Gold/Silver ETFs. Price data via Yahoo "
    "Finance (yfinance); AI commentary via the configured LLM API."
)
st.caption(
    "⚠️ For educational and personal use only — this is **not** investment "
    "advice. Always do your own research and consult a registered financial "
    "advisor before trading."
)
st.caption("Built by **Rishikesh Kimmatkar**")
