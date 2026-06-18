"""
Transformada de Laplace — Exemplo 4.1: Tanque de Armazenamento de Líquido
UNIMONTES | Disciplina de Equações Diferenciais Ordinárias | Prof. Fernando Félix
Alunos: Bruno Gomes, Júlio César, Leonardo, Marcus, Samuel
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import os
import base64

# ─── Configuração da página ───────────────────────────────────────────────────

st.set_page_config(
    page_title="Exemplo 4.1 — Tanque de Líquido | UNIMONTES",
    page_icon="assets/favicon.png" if os.path.exists("assets/favicon.png") else "📐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,wght@0,300;0,400;0,600;1,400&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

  .stApp { background-color: #F7F8FA; }
  .block-container { padding-top: 1.5rem; max-width: 1200px; }
  .main .block-container { padding-left: 1.5rem; padding-right: 1.5rem; }

  .header-bar {
    background: #1A3A5C; color: #FFFFFF;
    padding: 1.2rem 2rem;
    border-bottom: 3px solid #C4A32A;
    display: flex; align-items: center; gap: 1.5rem;
    margin: -1.5rem -1rem 2rem -1rem;
  }
  .header-logo {
    width: 72px; height: 72px;
    background: rgba(255,255,255,0.10);
    border: 1px solid rgba(255,255,255,0.25);
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0; overflow: hidden;
  }
  .header-title {
    font-family: 'Source Serif 4', serif;
    font-size: 1.4rem; font-weight: 600; line-height: 1.25; margin: 0;
  }
  .header-sub {
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem; font-weight: 300;
    opacity: 0.75; margin-top: 0.25rem; letter-spacing: 0.03em;
  }
  .header-credits {
    margin-left: auto; text-align: right;
    font-family: 'Inter', sans-serif;
    font-size: 0.75rem; opacity: 0.7; line-height: 1.6;
  }

  .section-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.68rem; font-weight: 600;
    letter-spacing: 0.12em; text-transform: uppercase;
    color: #1A3A5C; opacity: 0.6; margin-bottom: 0.3rem;
  }
  .section-title {
    font-family: 'Source Serif 4', serif;
    font-size: 1.45rem; font-weight: 600;
    color: #111827; margin: 0 0 1rem 0; line-height: 1.3;
  }
  .divider { border: none; border-top: 1px solid #DEE2E9; margin: 2rem 0; }
  .gold-accent {
    display: inline-block; width: 32px; height: 3px;
    background: #C4A32A; margin-bottom: 0.6rem;
  }

  .result-step {
    background: #F7F8FA; border: 1px solid #E5E7EB;
    padding: 1rem 1.25rem; margin-bottom: 0.75rem;
  }
  .result-step .step-num {
    font-size: 0.65rem; font-weight: 700;
    letter-spacing: 0.1em; text-transform: uppercase;
    color: #C4A32A; margin-bottom: 0.4rem;
  }
  .result-step-gold {
    background: #fff; border-left: 3px solid #C4A32A;
    padding: 1.25rem 1.5rem; margin-bottom: 0.75rem;
  }
  .result-step-gold .step-num {
    font-size: 0.65rem; font-weight: 700;
    letter-spacing: 0.1em; text-transform: uppercase;
    color: #1A3A5C; margin-bottom: 0.4rem;
  }

  .result-card {
    background: #FFFFFF; border: 1px solid #DEE2E9;
    padding: 1rem 1.25rem; text-align: center;
  }
  .result-card .card-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.68rem; font-weight: 600;
    letter-spacing: 0.08em; text-transform: uppercase;
    color: #6B7280; margin-bottom: 0.4rem;
  }
  .result-card .card-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.3rem; font-weight: 600; color: #1A3A5C;
  }
  .result-card .card-value-gold {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.3rem; font-weight: 600; color: #C4A32A;
  }

  .info-card {
    background: #fff; border-left: 3px solid #1A3A5C;
    padding: 1rem 1.25rem; margin-bottom: 1rem;
    font-family: 'Inter', sans-serif; font-size: 0.83rem;
    color: #374151; line-height: 1.65;
  }
  .info-card-gold {
    background: #fff; border-left: 3px solid #C4A32A;
    padding: 1rem 1.25rem; margin-bottom: 1rem;
    font-family: 'Inter', sans-serif; font-size: 0.83rem;
    color: #374151; line-height: 1.65;
  }
  .info-card .card-title { font-size: 0.7rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: #1A3A5C; margin-bottom: 0.4rem; }
  .info-card-gold .card-title { font-size: 0.7rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: #C4A32A; margin-bottom: 0.4rem; }

  div[data-testid="stButton"] > button {
    background: #1A3A5C !important; color: #FFFFFF !important;
    border: none !important; font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem !important; font-weight: 500 !important;
    letter-spacing: 0.05em !important; padding: 0.55rem 1.75rem !important;
    border-radius: 0 !important;
  }
  div[data-testid="stButton"] > button:hover { background: #243F62 !important; }

  div[data-testid="stNumberInput"] input {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.9rem !important; border-radius: 0 !important; color: #1A3A5C !important;
  }

  .warn-box {
    background: #FEF3C7; border-left: 3px solid #C4A32A;
    padding: 0.9rem 1.2rem; font-family: 'Inter', sans-serif;
    font-size: 0.85rem; color: #78350F;
  }

  table { width: 100%; border-collapse: collapse; }
  th {
    background: #1A3A5C; color: #fff; padding: 0.6rem 0.9rem;
    font-family: 'Inter', sans-serif; font-size: 0.8rem;
    font-weight: 500; text-align: left;
  }
  td {
    padding: 0.55rem 0.9rem; font-family: 'Inter', sans-serif;
    font-size: 0.82rem; border-bottom: 1px solid #E5E7EB;
  }
  tr:nth-child(even) td { background: #F7F8FA; }

  .app-item {
    border-top: 1px solid #DEE2E9; padding: 0.85rem 0;
    font-family: 'Inter', sans-serif; font-size: 0.87rem; color: #374151;
  }
  .app-item strong { color: #1A3A5C; font-weight: 600; }

  .footer {
    margin-top: 3rem; padding: 1.5rem 0;
    border-top: 1px solid #DEE2E9;
    font-family: 'Inter', sans-serif; font-size: 0.75rem;
    color: #9CA3AF; text-align: center;
  }

  div[data-testid="stRadio"] > div { flex-direction: row !important; gap: 1rem; }
</style>
""", unsafe_allow_html=True)


# ─── Funções de cálculo ───────────────────────────────────────────────────────

def calcular(A, Rv, M, h_bar, t_max, n_pts=600):
    ARv = A * Rv
    h_dev_inf = Rv * M
    h_inf = h_bar + h_dev_inf
    t = np.linspace(0, t_max, n_pts)
    h_dev = h_dev_inf * (1 - np.exp(-t / ARv))
    h = h_bar + h_dev
    return {
        "ARv": ARv,
        "h_dev_inf": h_dev_inf,
        "h_inf": h_inf,
        "t": t,
        "h_dev": h_dev,
        "h": h,
    }

def fmt(v, d=4):
    if not np.isfinite(v):
        return "—"
    return f"{v:.{d}f}".rstrip("0").rstrip(".") or "0"


# ─── Logo ─────────────────────────────────────────────────────────────────────

logo_path = "assets/logo_unimontes.png"
if os.path.exists(logo_path):
    with open(logo_path, "rb") as img_file:
        logo_b64 = base64.b64encode(img_file.read()).decode()
    logo_html = (
        f'<div class="header-logo">'
        f'<img src="data:image/png;base64,{logo_b64}" '
        f'style="width:72px;height:72px;object-fit:contain;" alt="UNIMONTES">'
        f'</div>'
    )
else:
    logo_html = """
    <div class="header-logo" style="display:flex;flex-direction:column;align-items:center;justify-content:center;">
      <span style="font-family:'Source Serif 4',serif;font-size:1.4rem;font-weight:700;color:#C4A32A;">U</span>
      <span style="font-family:'Inter',sans-serif;font-size:0.45rem;font-weight:600;color:rgba(255,255,255,0.85);letter-spacing:0.08em;">UNIMONTES</span>
      <div style="width:32px;height:2px;background:#C4A32A;margin-top:4px;"></div>
    </div>
    """

# ─── Cabeçalho ────────────────────────────────────────────────────────────────

st.markdown(f"""
<div class="header-bar">
  {logo_html}
  <div>
    <p class="header-title">Transformada de Laplace — Exemplo 4.1</p>
    <p class="header-sub">Universidade Estadual de Montes Claros — UNIMONTES &nbsp;·&nbsp; Função de transferência e resposta ao degrau</p>
  </div>
  <div class="header-credits">
    Prof. Fernando Félix<br>
    Bruno Gomes · Júlio César<br>
    Leonardo · Marcus · Samuel
  </div>
</div>
""", unsafe_allow_html=True)


# ─── Seção 1: Modelo físico + Diagrama ───────────────────────────────────────

col_intro, col_diag = st.columns([3, 2], gap="large")

with col_intro:
    st.markdown('<div class="gold-accent"></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-label">Resumo técnico</p>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Funcionamento do aplicativo</h2>', unsafe_allow_html=True)

    st.markdown("""
<p style="font-family:'Inter',sans-serif;font-size:0.88rem;color:#374151;line-height:1.8;margin:0 0 1rem 0;">
  O aplicativo foi desenvolvido em <strong>Python</strong> com <strong>Streamlit</strong>,
  ferramenta que permite criar interfaces web interativas a partir de um script.
</p>

<p style="font-family:'Inter',sans-serif;font-size:0.88rem;color:#374151;line-height:1.8;margin:0 0 1rem 0;">
  Para o design, foi utilizado <strong>CSS personalizado</strong>, responsável por definir
  cores, fontes, espaçamentos, cartões, cabeçalho, divisórias e botões. Isso deixou
  o app organizado.
</p>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="info-card">
  <div class="card-title">Lógica principal</div>
  Entrada dos parâmetros → cálculo de AR<sub>v</sub>, R<sub>v</sub>M e h(∞) →
  aplicação da fórmula final → plotagem da curva h(t).
</div>
""", unsafe_allow_html=True)

    st.latex(r"h(t) = \bar{h} + R_vM\left(1 - e^{-t/AR_v}\right)")

    st.markdown("""
<p style="font-family:'Inter',sans-serif;font-size:0.84rem;color:#6B7280;line-height:1.7;margin:0.75rem 0 0 0;">
  O <strong>NumPy</strong> foi usado nos cálculos numéricos, enquanto o
  <strong>Plotly</strong> foi utilizado para gerar o gráfico interativo da resposta
  do nível do líquido ao longo do tempo.
</p>
""", unsafe_allow_html=True)

with col_diag:
    st.markdown('<div class="gold-accent"></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-label">Diagrama esquemático</p>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Representação do sistema</h2>', unsafe_allow_html=True)

    st.markdown("""
<div style="background:#fff;border:1px solid #DEE2E9;padding:1.25rem;">
<svg viewBox="0 0 220 170" width="100%" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="arr" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto">
      <path d="M0,0 L6,3 L0,6 Z" fill="#1A3A5C"/>
    </marker>
  </defs>

  <!-- Tubo de entrada -->
  <line x1="90" y1="5" x2="90" y2="36" stroke="#1A3A5C" stroke-width="3"/>
  <polygon points="84,30 96,30 90,40" fill="#1A3A5C"/>
  <text x="100" y="20" font-family="Inter,sans-serif" font-size="11" fill="#6B7280">qᵢ(t)</text>

  <!-- Corpo do tanque -->
  <rect x="40" y="42" width="140" height="95" fill="none" stroke="#1A3A5C" stroke-width="2"/>

  <!-- Líquido -->
  <rect x="41" y="82" width="138" height="54" fill="#DBEAFE" opacity="0.65"/>
  <line x1="41" y1="82" x2="179" y2="82" stroke="#1A3A5C" stroke-width="1.5" stroke-dasharray="5 3"/>

  <!-- Label A -->
  <text x="52" y="65" font-family="Inter,sans-serif" font-size="11" fill="#9CA3AF" font-style="italic">A</text>

  <!-- Cota h -->
  <line x1="186" y1="82" x2="186" y2="137" stroke="#1A3A5C" stroke-width="1"
        marker-start="url(#arr)" marker-end="url(#arr)"/>
  <text x="192" y="113" font-family="Inter,sans-serif" font-size="11" fill="#1A3A5C" font-style="italic">h(t)</text>

  <!-- Saída -->
  <line x1="110" y1="137" x2="110" y2="160" stroke="#C4A32A" stroke-width="3"/>
  <polygon points="104,152 116,152 110,162" fill="#C4A32A"/>
  <text x="118" y="155" font-family="Inter,sans-serif" font-size="10" fill="#C4A32A">h / Rᵥ</text>
</svg>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ─── Seção 2: Desenvolvimento passo a passo ───────────────────────────────────

st.markdown('<div class="gold-accent"></div>', unsafe_allow_html=True)
st.markdown('<p class="section-label">Desenvolvimento</p>', unsafe_allow_html=True)
st.markdown('<h2 class="section-title">Resolução passo a passo</h2>', unsafe_allow_html=True)

col_passos, col_info = st.columns([3, 1], gap="large")

with col_passos:

    # Passo 1
    st.markdown('<div class="result-step">', unsafe_allow_html=True)
    st.markdown('<div class="step-num">Passo 1 — Variáveis de desvio</div>', unsafe_allow_html=True)
    st.markdown("""
<p style="font-family:'Inter',sans-serif;font-size:0.87rem;color:#374151;margin:0 0 0.5rem 0;">
  Define-se a variável de desvio subtraindo o estado estacionário:
</p>""", unsafe_allow_html=True)
    st.latex(r"h' \triangleq h - \bar{h}, \qquad q_i' \triangleq q_i - \bar{q}_i")
    st.markdown("""
<p style="font-family:'Inter',sans-serif;font-size:0.87rem;color:#374151;margin:0.5rem 0;">
  Subtraindo a versão em regime permanente da EDO original, e usando que
  <em>dh'/dt = dh/dt</em>, obtém-se a EDO em variáveis de desvio, com <em>h'(0) = 0</em>:
</p>""", unsafe_allow_html=True)
    st.latex(r"A\,\frac{dh'}{dt} = q_i' - \frac{1}{R_v}\,h'")
    st.markdown('</div>', unsafe_allow_html=True)

    # Passo 2
    st.markdown('<div class="result-step">', unsafe_allow_html=True)
    st.markdown('<div class="step-num">Passo 2 — Transformada de Laplace</div>', unsafe_allow_html=True)
    st.markdown("""
<p style="font-family:'Inter',sans-serif;font-size:0.87rem;color:#374151;margin:0 0 0.5rem 0;">
  Aplicando a transformada com <em>h'(0) = 0</em>:
</p>""", unsafe_allow_html=True)
    st.latex(r"A\bigl[sH'(s) - h'(0)\bigr] = Q_i'(s) - \frac{1}{R_v}H'(s)")
    st.latex(r"A\,s\,H'(s) = Q_i'(s) - \frac{1}{R_v}H'(s)")
    st.markdown('</div>', unsafe_allow_html=True)

    # Passo 3
    st.markdown('<div class="result-step">', unsafe_allow_html=True)
    st.markdown('<div class="step-num">Passo 3 — Função de transferência G(s)</div>', unsafe_allow_html=True)
    st.markdown("""
<p style="font-family:'Inter',sans-serif;font-size:0.87rem;color:#374151;margin:0 0 0.5rem 0;">
  Reagrupando e isolando a razão <em>H'(s) / Q'ᵢ(s)</em>:
</p>""", unsafe_allow_html=True)
    st.latex(r"G(s) = \frac{H'(s)}{Q_i'(s)} = \frac{R_v}{AR_v s + 1}")
    st.markdown('</div>', unsafe_allow_html=True)

    # Passo 4
    st.markdown('<div class="result-step">', unsafe_allow_html=True)
    st.markdown('<div class="step-num">Passo 4 — Resposta ao degrau de magnitude M</div>', unsafe_allow_html=True)
    st.markdown("""
<p style="font-family:'Inter',sans-serif;font-size:0.87rem;color:#374151;margin:0 0 0.5rem 0;">
  Para uma entrada degrau <em>qᵢ(t) = M</em>, tem-se <em>Q'ᵢ(s) = M/s</em>. Substituindo:
</p>""", unsafe_allow_html=True)
    st.latex(r"H'(s) = \frac{R_v}{AR_v s + 1}\cdot\frac{M}{s}")
    st.markdown('</div>', unsafe_allow_html=True)

    # Resultado
    st.markdown('<div class="result-step-gold">', unsafe_allow_html=True)
    st.markdown('<div class="step-num">Resultado — Transformada inversa (Eq. 4-13)</div>', unsafe_allow_html=True)
    st.markdown("""
<p style="font-family:'Inter',sans-serif;font-size:0.87rem;color:#374151;margin:0 0 0.6rem 0;">
  Pela tabela de transformadas, a transformada inversa de <em>H'(s)</em> fornece o desvio do nível.
  Voltando à variável original:
</p>""", unsafe_allow_html=True)
    st.latex(r"h'(t) = R_v M\!\left(1 - e^{-t/AR_v}\right)")
    st.latex(r"h(t) = \bar{h} + R_v M\!\left(1 - e^{-t/AR_v}\right)")
    st.markdown('</div>', unsafe_allow_html=True)

with col_info:
    st.markdown("""
<div class="info-card">
  <div class="card-title">Variáveis de desvio</div>
  Simplificam a EDO eliminando o ponto de operação, permitindo condição inicial nula: <em>h'(0) = 0</em>.
</div>
<div class="info-card">
  <div class="card-title">Função de transferência</div>
  <em>G(s) = Rᵥ / (ARᵥs + 1)</em> é um sistema de <strong>primeira ordem</strong>.
  O ganho estático é Rᵥ e a constante de tempo é ARᵥ.
</div>
<div class="info-card-gold">
  <div class="card-title">Resposta ao degrau</div>
  O nível parte de <em>h̄</em> e converge a <em>h̄ + RᵥM</em>, com curva exponencial
  característica de sistemas de primeira ordem.
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ─── Seção 3: Calculadora ─────────────────────────────────────────────────────

st.markdown('<div class="gold-accent"></div>', unsafe_allow_html=True)
st.markdown('<p class="section-label">Calculadora</p>', unsafe_allow_html=True)
st.markdown('<h2 class="section-title">Defina os parâmetros</h2>', unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    A = st.number_input("A — Área (m²)", min_value=0.01, value=2.0, step=0.1, format="%.2f")
with col2:
    Rv = st.number_input("Rv — Resistência (m·s/m³)", min_value=0.01, value=1.5, step=0.1, format="%.2f")
with col3:
    M = st.number_input("M — Magnitude do degrau (m³/s)", value=0.5, step=0.1, format="%.2f")
with col4:
    h_bar = st.number_input("h̄ — Nível inicial (m)", min_value=0.0, value=1.0, step=0.1, format="%.2f")
with col5:
    t_max = st.number_input("t máx (s)", min_value=1.0, value=20.0, step=1.0, format="%.1f")

calcular_btn = st.button("Calcular solução")


# ─── Resultados ───────────────────────────────────────────────────────────────

if calcular_btn or "res" in st.session_state:

    if calcular_btn:
        if A <= 0 or Rv <= 0 or t_max <= 0:
            st.markdown('<div class="warn-box">⚠️ Os parâmetros A, Rv e t máximo devem ser positivos.</div>',
                        unsafe_allow_html=True)
            st.stop()
        st.session_state["res"] = calcular(A, Rv, M, h_bar, t_max)
        st.session_state["params"] = dict(A=A, Rv=Rv, M=M, h_bar=h_bar, t_max=t_max)

    res = st.session_state["res"]
    p   = st.session_state["params"]

    ARv       = res["ARv"]
    h_dev_inf = res["h_dev_inf"]
    h_inf     = res["h_inf"]
    t_arr     = res["t"]
    h_dev_arr = res["h_dev"]
    h_arr     = res["h"]

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="gold-accent"></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-label">Resultados</p>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Parâmetros calculados</h2>', unsafe_allow_html=True)

    # ── Cards ────────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
<div class="result-card" style="border-top:3px solid #1A3A5C;">
  <div class="card-label">Constante de tempo</div>
  <div style="font-size:0.8rem;color:#374151;margin-bottom:0.4rem;">AR<sub>v</sub></div>
  <div class="card-value">{fmt(ARv, 4)} s</div>
</div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
<div class="result-card" style="border-top:3px solid #1A3A5C;">
  <div class="card-label">Desvio final h'(∞)</div>
  <div style="font-size:0.8rem;color:#374151;margin-bottom:0.4rem;">R<sub>v</sub> · M</div>
  <div class="card-value">{fmt(h_dev_inf, 4)} m</div>
</div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
<div class="result-card" style="border-top:3px solid #C4A32A;">
  <div class="card-label">Nível final h(∞)</div>
  <div style="font-size:0.8rem;color:#374151;margin-bottom:0.4rem;">h̄ + R<sub>v</sub> · M</div>
  <div class="card-value-gold">{fmt(h_inf, 4)} m</div>
</div>""", unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
<div class="result-card" style="border-top:3px solid #6B7280;">
  <div class="card-label">Nível inicial h̄</div>
  <div style="font-size:0.8rem;color:#374151;margin-bottom:0.4rem;">Regime permanente inicial</div>
  <div class="card-value" style="color:#6B7280;">{fmt(p["h_bar"], 4)} m</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Expressão numérica ───────────────────────────────────────────────────
    st.markdown("""
<div style="background:#fff;border-left:3px solid #1A3A5C;padding:1.25rem 1.5rem;margin-bottom:1.25rem;">
  <div style="font-family:'Inter',sans-serif;font-size:0.7rem;font-weight:600;
              letter-spacing:0.1em;text-transform:uppercase;color:#6B7280;margin-bottom:0.75rem;">
    Expressão com os valores inseridos
  </div>
""", unsafe_allow_html=True)

    st.latex(
        rf"G(s) = \frac{{{fmt(p['Rv'], 4)}}}{{{fmt(ARv, 4)}\,s + 1}}"
    )
    st.latex(
        rf"h(t) = {fmt(p['h_bar'], 4)} + {fmt(h_dev_inf, 4)}"
        rf"\!\left(1 - e^{{-t/{fmt(ARv, 4)}}}\right)"
    )

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Gráfico ──────────────────────────────────────────────────────────────
    st.markdown('<div class="gold-accent"></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-label">Visualização</p>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Resposta temporal h(t)</h2>', unsafe_allow_html=True)

    curva = st.radio(
        "Exibir:",
        ["Nível real h(t)", "Desvio h'(t)", "Ambas as curvas"],
        horizontal=True,
        label_visibility="collapsed",
    )

    fig = go.Figure()

    fig.add_hline(
        y=p["h_bar"], line_dash="dash", line_color="#9CA3AF", line_width=1.2,
        annotation_text=f"h̄ = {fmt(p['h_bar'], 2)} m",
        annotation_position="top right",
        annotation_font=dict(color="#9CA3AF", size=11),
    )
    fig.add_hline(
        y=h_inf, line_dash="dash", line_color="#C4A32A", line_width=1.5,
        annotation_text=f"h(∞) = {fmt(h_inf, 3)} m",
        annotation_position="bottom right",
        annotation_font=dict(color="#C4A32A", size=11),
    )
    fig.add_vline(
        x=ARv, line_dash="dot", line_color="#6B7280", line_width=1.2,
        annotation_text=f"ARv = {fmt(ARv, 2)} s",
        annotation_position="top right",
        annotation_font=dict(color="#6B7280", size=10),
    )

    if curva in ("Nível real h(t)", "Ambas as curvas"):
        fig.add_trace(go.Scatter(
            x=t_arr, y=h_arr, mode="lines",
            name="h(t) — nível real",
            line=dict(color="#1A3A5C", width=2.5),
        ))

    if curva in ("Desvio h'(t)", "Ambas as curvas"):
        fig.add_trace(go.Scatter(
            x=t_arr, y=h_dev_arr, mode="lines",
            name="h'(t) — desvio",
            line=dict(color="#C4A32A", width=2, dash="dot"),
        ))

    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", size=12, color="#374151"),
        margin=dict(l=50, r=50, t=30, b=60),
        xaxis=dict(title="Tempo (s)", showgrid=True, gridcolor="#E5E7EB",
                   zeroline=True, zerolinecolor="#9CA3AF", linecolor="#D1D5DB"),
        yaxis=dict(title="Nível (m)", showgrid=True, gridcolor="#E5E7EB",
                   zeroline=True, zerolinecolor="#9CA3AF", linecolor="#D1D5DB"),
        legend=dict(orientation="h", y=1.08, x=0,
                    bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter, sans-serif", size=12)),
        height=430,
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ─── Seção 4: Aplicações da Transformada de Laplace ──────────────────────────

st.markdown('<div class="gold-accent"></div>', unsafe_allow_html=True)
st.markdown('<p class="section-label">Contexto</p>', unsafe_allow_html=True)
st.markdown('<h2 class="section-title">O que essa transformação permite fazer</h2>', unsafe_allow_html=True)

col_a1, col_a2 = st.columns(2, gap="large")

with col_a1:
    st.markdown("""
<div class="app-item">
  <strong>Resolver EDOs lineares com condição inicial</strong><br>
  Converte a equação diferencial em algébrica, resolve em <em>s</em> e aplica a transformada inversa para obter a solução no tempo.
</div>
<div class="app-item">
  <strong>Obter a função de transferência de um processo</strong><br>
  A razão <em>Y(s)/U(s)</em> descreve completamente a dinâmica de um sistema linear em torno de um ponto de operação.
</div>
<div class="app-item">
  <strong>Analisar a resposta a diferentes entradas</strong><br>
  Com <em>G(s)</em> definido, basta multiplicar pela transformada da entrada — degrau, rampa, impulso — para obter a saída correspondente.
</div>
""", unsafe_allow_html=True)

with col_a2:
    st.markdown("""
<div class="app-item">
  <strong>Estudar estabilidade pelo plano complexo</strong><br>
  Os polos de <em>G(s)</em> determinam o comportamento dinâmico: polos com parte real negativa garantem estabilidade.
</div>
<div class="app-item">
  <strong>Projetar controladores (PID, etc.)</strong><br>
  A função de transferência do processo é a entrada para métodos de projeto de malhas de controle realimentadas.
</div>
<div class="app-item">
  <strong>Compor sistemas em série ou paralelo</strong><br>
  Funções de transferência podem ser multiplicadas ou somadas diretamente, facilitando a análise de processos com múltiplos estágios.
</div>
""", unsafe_allow_html=True)


# ─── Rodapé ───────────────────────────────────────────────────────────────────

st.markdown("""
<div class="footer">
  UNIMONTES — Universidade Estadual de Montes Claros &nbsp;·&nbsp;
  Disciplina de Equações Diferenciais Ordinárias &nbsp;·&nbsp; Prof. Fernando Félix<br>
  Bruno Gomes · Júlio César · Leonardo · Marcus · Samuel &nbsp;·&nbsp;
  Construído com Streamlit, NumPy e Plotly
</div>
""", unsafe_allow_html=True)
