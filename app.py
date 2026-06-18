"""
Transformada de Laplace — Tanque de Armazenamento de Líquido
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
    page_title="Transformada de Laplace — Tanque de Líquido",
    page_icon="assets/favicon.png" if os.path.exists("assets/favicon.png") else "📐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CSS personalizado ────────────────────────────────────────────────────────

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,wght@0,300;0,400;0,600;1,400&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

  .stApp { background-color: #F7F8FA; }
  .block-container { padding-top: 1.5rem; max-width: 1200px; }

  /* Remove padding extra do Streamlit */
  .main .block-container { padding-left: 1.5rem; padding-right: 1.5rem; }

  .header-bar {
    background: #1A3A5C;
    color: #FFFFFF;
    padding: 1.2rem 2rem;
    border-bottom: 3px solid #C4A32A;
    display: flex;
    align-items: center;
    gap: 1.5rem;
    margin: -1.5rem -1rem 2rem -1rem;
  }
  .header-logo {
    width: 72px; height: 72px;
    background: rgba(255,255,255,0.10);
    border: 1px solid rgba(255,255,255,0.25);
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
    overflow: hidden;
  }
  .header-title {
    font-family: 'Source Serif 4', serif;
    font-size: 1.4rem; font-weight: 600;
    line-height: 1.25; margin: 0;
  }
  .header-sub {
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem; font-weight: 300;
    opacity: 0.75; margin-top: 0.25rem;
    letter-spacing: 0.03em;
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
    color: #1A3A5C; opacity: 0.6;
    margin-bottom: 0.3rem;
  }
  .section-title {
    font-family: 'Source Serif 4', serif;
    font-size: 1.45rem; font-weight: 600;
    color: #111827; margin: 0 0 1rem 0;
    line-height: 1.3;
  }
  .divider {
    border: none; border-top: 1px solid #DEE2E9;
    margin: 2rem 0;
  }
  .gold-accent {
    display: inline-block;
    width: 32px; height: 3px;
    background: #C4A32A; margin-bottom: 0.6rem;
  }

  /* Painel de resultado */
  .result-step {
    background: #F7F8FA;
    border: 1px solid #E5E7EB;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
  }
  .result-step .step-num {
    font-size: 0.65rem; font-weight: 700;
    letter-spacing: 0.1em; text-transform: uppercase;
    color: #C4A32A; margin-bottom: 0.4rem;
  }
  .result-step-gold {
    background: #fff;
    border-left: 3px solid #C4A32A;
    padding: 1.25rem 1.5rem;
    margin-bottom: 0.75rem;
  }
  .result-step-gold .step-num {
    font-size: 0.65rem; font-weight: 700;
    letter-spacing: 0.1em; text-transform: uppercase;
    color: #1A3A5C; margin-bottom: 0.4rem;
  }

  /* Cards de resultado */
  .result-card {
    background: #FFFFFF;
    border: 1px solid #DEE2E9;
    padding: 1rem 1.25rem;
    text-align: center;
  }
  .result-card .card-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.68rem; font-weight: 600;
    letter-spacing: 0.08em; text-transform: uppercase;
    color: #6B7280; margin-bottom: 0.4rem;
  }
  .result-card .card-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.3rem; font-weight: 600;
    color: #1A3A5C;
  }
  .result-card .card-value-gold {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.3rem; font-weight: 600;
    color: #C4A32A;
  }

  /* Bloco de interpretação */
  .interp-block {
    background: #1A3A5C;
    color: #E8EDF5;
    padding: 1.5rem 1.75rem;
    margin-top: 1rem;
    font-family: 'Inter', sans-serif;
    font-size: 0.88rem;
    line-height: 1.7;
  }
  .interp-block h4 {
    font-family: 'Source Serif 4', serif;
    font-size: 1rem; color: #C4A32A;
    margin: 0 0 0.75rem 0;
  }

  /* Info card azul escuro */
  .info-card {
    background: #fff;
    border-left: 3px solid #1A3A5C;
    padding: 1rem 1.25rem;
    margin-bottom: 1rem;
    font-family: 'Inter', sans-serif;
    font-size: 0.83rem;
    color: #374151;
    line-height: 1.65;
  }
  .info-card-gold {
    background: #fff;
    border-left: 3px solid #C4A32A;
    padding: 1rem 1.25rem;
    margin-bottom: 1rem;
    font-family: 'Inter', sans-serif;
    font-size: 0.83rem;
    color: #374151;
    line-height: 1.65;
  }
  .info-card .card-title, .info-card-gold .card-title {
    font-size: 0.7rem; font-weight: 600;
    letter-spacing: 0.08em; text-transform: uppercase;
    color: #1A3A5C; margin-bottom: 0.4rem;
  }
  .info-card-gold .card-title { color: #C4A32A; }

  /* Botão */
  div[data-testid="stButton"] > button {
    background: #1A3A5C !important;
    color: #FFFFFF !important;
    border: none !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.05em !important;
    padding: 0.55rem 1.75rem !important;
    border-radius: 0 !important;
  }
  div[data-testid="stButton"] > button:hover {
    background: #243F62 !important;
  }

  /* Inputs */
  div[data-testid="stNumberInput"] input {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.9rem !important;
    border-radius: 0 !important;
    color: #1A3A5C !important;
  }

  code {
    font-family: 'JetBrains Mono', monospace;
    background: #EEF0F4;
    padding: 0.1em 0.4em;
    font-size: 0.85em;
    color: #1A3A5C;
  }

  /* App items */
  .app-item {
    border-top: 1px solid #DEE2E9;
    padding: 0.85rem 0;
    font-family: 'Inter', sans-serif;
    font-size: 0.87rem; color: #374151;
  }
  .app-item strong { color: #1A3A5C; font-weight: 600; }

  /* Warn */
  .warn-box {
    background: #FEF3C7;
    border-left: 3px solid #C4A32A;
    padding: 0.9rem 1.2rem;
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem; color: #78350F;
  }

  /* Tabela */
  table { width: 100%; border-collapse: collapse; }
  th {
    background: #1A3A5C; color: #fff;
    padding: 0.6rem 0.9rem;
    font-family: 'Inter', sans-serif; font-size: 0.8rem;
    font-weight: 500; text-align: left;
  }
  td {
    padding: 0.55rem 0.9rem;
    font-family: 'Inter', sans-serif; font-size: 0.82rem;
    border-bottom: 1px solid #E5E7EB;
  }
  tr:nth-child(even) td { background: #F7F8FA; }

  .footer {
    margin-top: 3rem; padding: 1.5rem 0;
    border-top: 1px solid #DEE2E9;
    font-family: 'Inter', sans-serif;
    font-size: 0.75rem; color: #9CA3AF;
    text-align: center;
  }

  /* Radio horizontal */
  div[data-testid="stRadio"] > div {
    flex-direction: row !important;
    gap: 1rem;
  }
</style>
""", unsafe_allow_html=True)


# ─── Funções de cálculo ───────────────────────────────────────────────────────

def calcular(A, Rv, M, h_bar, t_max, n_pts=600):
    tau = A * Rv
    h_dev_inf = Rv * M
    h_inf = h_bar + h_dev_inf
    t = np.linspace(0, t_max, n_pts)
    h_dev = h_dev_inf * (1 - np.exp(-t / tau))
    h = h_bar + h_dev
    return {
        "tau": tau,
        "h_dev_inf": h_dev_inf,
        "h_inf": h_inf,
        "t": t,
        "h_dev": h_dev,
        "h": h,
    }


def fmt(v, d=4):
    if not np.isfinite(v):
        return "—"
    return f"{v:.{d}f}".rstrip("0").rstrip(".")


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
      <span style="font-family:\'Source Serif 4\',serif;font-size:1.4rem;font-weight:700;color:#C4A32A;">U</span>
      <span style="font-family:\'Inter\',sans-serif;font-size:0.45rem;font-weight:600;color:rgba(255,255,255,0.85);letter-spacing:0.08em;">UNIMONTES</span>
      <div style="width:32px;height:2px;background:#C4A32A;margin-top:4px;"></div>
    </div>
    """

# ─── Cabeçalho ────────────────────────────────────────────────────────────────

st.markdown(f"""
<div class="header-bar">
  {logo_html}
  <div>
    <p class="header-title">Transformada de Laplace aplicada à resolução de EDOs</p>
    <p class="header-sub">Universidade Estadual de Montes Claros — UNIMONTES &nbsp;·&nbsp; Ferramenta interativa</p>
  </div>
  <div class="header-credits">
    Prof. Fernando Félix<br>
    Bruno Gomes · Júlio César<br>
    Leonardo · Marcus · Samuel
  </div>
</div>
""", unsafe_allow_html=True)


# ─── Introdução ───────────────────────────────────────────────────────────────

col_intro, col_prop = st.columns([3, 2], gap="large")

with col_intro:
    st.markdown('<div class="gold-accent"></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-label">O modelo físico</p>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Tanque de armazenamento de líquido</h2>', unsafe_allow_html=True)

    st.markdown("""
<div style="background:#fff;border:1px solid #DEE2E9;padding:1.25rem 1.5rem;margin-bottom:1rem;">
  <p style="font-family:\'Inter\',sans-serif;font-size:0.88rem;color:#374151;line-height:1.75;margin:0;">
    O sistema representa um <strong style="color:#1A3A5C;">tanque de líquido</strong>
    com entrada de vazão <em>qᵢ(t)</em> e nível <em>h(t)</em>.
    A dinâmica é governada por um balanço de massa:
  </p>
</div>
""", unsafe_allow_html=True)

    st.latex(r"A\,\frac{dh}{dt} = q_i - \frac{1}{R_v}\,h")

    st.markdown("""
<p style="font-family:\'Inter\',sans-serif;font-size:0.85rem;color:#374151;line-height:1.7;margin:0.5rem 0 1rem 0;">
  onde <strong>A</strong> é a área da seção transversal do tanque e
  <strong>Rᵥ</strong> é a resistência da válvula de saída.
</p>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="interp-block">
  <h4>📘 O que é a Transformada de Laplace?</h4>
  <p style="margin:0 0 0.6rem 0;">
    A <strong>Transformada de Laplace</strong> converte uma EDO no domínio do tempo
    em uma <em>equação algébrica</em> no domínio de <em>s</em>, muito mais simples de resolver.
  </p>
  <ul style="margin:0.5rem 0 0 1.2rem;padding:0;">
    <li><strong>qᵢ</strong> — vazão de entrada (perturbação de processo)</li>
    <li><strong>h</strong> — nível do líquido (variável de saída)</li>
    <li>Uma <strong>entrada degrau</strong> representa uma mudança súbita na vazão</li>
    <li>A resposta é típica de um <strong>sistema de primeira ordem</strong></li>
  </ul>
</div>
""", unsafe_allow_html=True)

with col_prop:
    st.markdown('<div class="gold-accent"></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-label">Propriedades fundamentais</p>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Transformadas usadas</h2>', unsafe_allow_html=True)

    st.markdown("""
| Função | Transformada |
|---|---|
| $h'(t)$ | $sH'(s) - h'(0)$ |
| $e^{-at}$ | $\\dfrac{1}{s+a}$ |
| $\\dfrac{1}{s}$ | degrau $u(t)$ |
| $\\dfrac{M}{s}$ | degrau de mag. $M$ |
| $\\dfrac{1}{\\tau s + 1}$ | $\\dfrac{1}{\\tau}e^{-t/\\tau}$ |
""")

    st.markdown('<div class="gold-accent" style="margin-top:1.5rem;"></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-label">Diagrama esquemático</p>', unsafe_allow_html=True)

    # Diagrama do tanque em SVG
    st.markdown("""
<div style="background:#fff;border:1px solid #DEE2E9;padding:1rem 1.25rem;">
<svg viewBox="0 0 220 160" width="100%" xmlns="http://www.w3.org/2000/svg">
  <!-- Entrada de vazão -->
  <line x1="90" y1="5" x2="90" y2="38" stroke="#1A3A5C" stroke-width="3"/>
  <polygon points="84,32 96,32 90,42" fill="#1A3A5C"/>
  <text x="100" y="22" font-family="Inter,sans-serif" font-size="10" fill="#6B7280">qᵢ(t)</text>

  <!-- Corpo do tanque -->
  <rect x="45" y="42" width="130" height="90" fill="none" stroke="#1A3A5C" stroke-width="2"/>

  <!-- Nível do líquido -->
  <rect x="46" y="82" width="128" height="49" fill="#DBEAFE" opacity="0.7"/>
  <line x1="46" y1="82" x2="174" y2="82" stroke="#1A3A5C" stroke-width="1.5" stroke-dasharray="5 3"/>

  <!-- Cota h(t) -->
  <line x1="180" y1="82" x2="180" y2="132" stroke="#1A3A5C" stroke-width="1" marker-start="url(#arr)" marker-end="url(#arr)"/>
  <text x="185" y="111" font-family="Inter,sans-serif" font-size="10" fill="#1A3A5C">h(t)</text>

  <!-- Área A -->
  <text x="58" y="66" font-family="Inter,sans-serif" font-size="10" fill="#9CA3AF">A</text>

  <!-- Seta bidirecional para h -->
  <defs>
    <marker id="arr" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto">
      <path d="M0,0 L6,3 L0,6 Z" fill="#1A3A5C"/>
    </marker>
  </defs>

  <!-- Saída -->
  <line x1="110" y1="132" x2="110" y2="155" stroke="#C4A32A" stroke-width="3"/>
  <polygon points="104,146 116,146 110,156" fill="#C4A32A"/>
  <text x="118" y="150" font-family="Inter,sans-serif" font-size="10" fill="#C4A32A">h/Rᵥ</text>
</svg>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ─── Desenvolvimento simbólico ────────────────────────────────────────────────

st.markdown('<div class="gold-accent"></div>', unsafe_allow_html=True)
st.markdown('<p class="section-label">Desenvolvimento simbólico</p>', unsafe_allow_html=True)
st.markdown('<h2 class="section-title">Resolução passo a passo</h2>', unsafe_allow_html=True)

col_passos, col_info = st.columns([3, 1], gap="large")

with col_passos:

    # Passo 1
    st.markdown('<div class="result-step">', unsafe_allow_html=True)
    st.markdown('<div class="step-num">Passo 1 — EDO em variáveis de desvio</div>', unsafe_allow_html=True)
    st.markdown("""
<p style="font-family:\'Inter\',sans-serif;font-size:0.88rem;color:#374151;margin:0 0 0.5rem 0;">
  Definindo as <strong>variáveis de desvio</strong> em relação ao estado estacionário:
</p>
""", unsafe_allow_html=True)
    st.latex(r"h'(t) = h(t) - \bar{h}, \quad q_i'(t) = q_i(t) - \bar{q}_i")
    st.markdown("""
<p style="font-family:\'Inter\',sans-serif;font-size:0.88rem;color:#374151;margin:0.5rem 0;">
  A EDO original se torna (com condição inicial <em>h'(0) = 0</em>):
</p>
""", unsafe_allow_html=True)
    st.latex(r"A\,\frac{dh'}{dt} = q_i' - \frac{1}{R_v}\,h'")
    st.markdown('</div>', unsafe_allow_html=True)

    # Passo 2
    st.markdown('<div class="result-step">', unsafe_allow_html=True)
    st.markdown('<div class="step-num">Passo 2 — Aplicando a Transformada de Laplace</div>', unsafe_allow_html=True)
    st.markdown("""
<p style="font-family:\'Inter\',sans-serif;font-size:0.88rem;color:#374151;margin:0 0 0.5rem 0;">
  Com <em>h'(0) = 0</em>, a transformada da derivada é simplesmente <em>sH'(s)</em>:
</p>
""", unsafe_allow_html=True)
    st.latex(r"A\,s\,H'(s) = Q_i'(s) - \frac{1}{R_v}\,H'(s)")
    st.latex(r"\left(A\,s + \frac{1}{R_v}\right) H'(s) = Q_i'(s)")
    st.markdown('</div>', unsafe_allow_html=True)

    # Passo 3
    st.markdown('<div class="result-step">', unsafe_allow_html=True)
    st.markdown('<div class="step-num">Passo 3 — Função de transferência G(s)</div>', unsafe_allow_html=True)
    st.markdown("""
<p style="font-family:\'Inter\',sans-serif;font-size:0.88rem;color:#374151;margin:0 0 0.5rem 0;">
  A <strong>função de transferência</strong> relaciona saída e entrada no domínio <em>s</em>:
</p>
""", unsafe_allow_html=True)
    st.latex(r"G(s) = \frac{H'(s)}{Q_i'(s)} = \frac{R_v}{A\,R_v\,s + 1} = \frac{R_v}{\tau s + 1}")
    st.markdown("""
<p style="font-family:\'Inter\',sans-serif;font-size:0.88rem;color:#374151;margin:0.5rem 0 0 0;">
  onde a <strong>constante de tempo</strong> é τ = A · Rᵥ.
</p>
""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Passo 4
    st.markdown('<div class="result-step">', unsafe_allow_html=True)
    st.markdown('<div class="step-num">Passo 4 — Entrada degrau e resposta no domínio s</div>', unsafe_allow_html=True)
    st.markdown("""
<p style="font-family:\'Inter\',sans-serif;font-size:0.88rem;color:#374151;margin:0 0 0.5rem 0;">
  Para uma entrada degrau de magnitude <em>M</em>:
</p>
""", unsafe_allow_html=True)
    st.latex(r"Q_i'(s) = \frac{M}{s}")
    st.latex(r"H'(s) = G(s)\cdot Q_i'(s) = \frac{R_v}{\tau s + 1}\cdot\frac{M}{s}")
    st.markdown("""
<p style="font-family:\'Inter\',sans-serif;font-size:0.88rem;color:#374151;margin:0.5rem 0;">
  Decompondo em frações parciais:
</p>
""", unsafe_allow_html=True)
    st.latex(r"H'(s) = R_v M \left(\frac{1}{s} - \frac{\tau}{\tau s + 1}\right) = R_v M \left(\frac{1}{s} - \frac{1}{s + 1/\tau}\right)")
    st.markdown('</div>', unsafe_allow_html=True)

    # Resultado final
    st.markdown('<div class="result-step-gold">', unsafe_allow_html=True)
    st.markdown('<div class="step-num">Resultado — Transformada Inversa de Laplace</div>', unsafe_allow_html=True)
    st.latex(r"h'(t) = R_v\,M\!\left(1 - e^{-t/\tau}\right)")
    st.latex(r"h(t) = \bar{h} + R_v\,M\!\left(1 - e^{-t/\tau}\right)")
    st.markdown("""
<p style="font-family:\'Inter\',sans-serif;font-size:0.85rem;color:#374151;margin:0.5rem 0 0 0;">
  Esta é a <strong>resposta de um sistema de primeira ordem</strong> a uma entrada degrau:
  a curva parte de <em>h̄</em>, cresce exponencialmente e tende ao novo regime permanente.
</p>
""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_info:
    st.markdown("""
<div class="info-card">
  <div class="card-title">Sistema de 1ª Ordem</div>
  O tanque é um sistema de <strong>primeira ordem</strong>: a resposta cresce
  exponencialmente até um novo equilíbrio, com velocidade controlada por τ.
</div>
<div class="info-card">
  <div class="card-title">Constante de tempo τ</div>
  Em <em>t = τ</em>, o sistema atingiu <strong>63,2%</strong> da variação total.
  Em <em>t = 5τ</em> considera-se em regime permanente (99,3%).
</div>
<div class="info-card-gold">
  <div class="card-title">Ganho estático Rᵥ</div>
  O <strong>ganho estático</strong> K = Rᵥ indica a variação final do nível
  por unidade de degrau na vazão: Δh(∞) = Rᵥ · M.
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ─── Calculadora ──────────────────────────────────────────────────────────────

st.markdown('<div class="gold-accent"></div>', unsafe_allow_html=True)
st.markdown('<p class="section-label">Calculadora</p>', unsafe_allow_html=True)
st.markdown('<h2 class="section-title">Defina os parâmetros do sistema</h2>', unsafe_allow_html=True)

with st.container():
    st.markdown("""
<div style="background:#fff;border:1px solid #DEE2E9;padding:1.5rem 1.75rem;margin-bottom:1rem;">
  <h3 style="font-family:\'Source Serif 4\',serif;font-size:1.1rem;color:#1A3A5C;margin:0 0 1.2rem 0;">
    Parâmetros do tanque e da perturbação
  </h3>
</div>
""", unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        A = st.number_input(
            "A — Área (m²)",
            min_value=0.01, value=2.0, step=0.1, format="%.2f",
            help="Área da seção transversal do tanque. Deve ser positiva."
        )
    with col2:
        Rv = st.number_input(
            "Rv — Resistência (m·s/m³)",
            min_value=0.01, value=1.5, step=0.1, format="%.2f",
            help="Resistência da válvula de saída. Deve ser positiva."
        )
    with col3:
        M = st.number_input(
            "M — Magnitude do degrau (m³/s)",
            value=0.5, step=0.1, format="%.2f",
            help="Magnitude da mudança súbita na vazão de entrada."
        )
    with col4:
        h_bar = st.number_input(
            "h̄ — Nível inicial (m)",
            min_value=0.0, value=1.0, step=0.1, format="%.2f",
            help="Nível de equilíbrio antes da perturbação."
        )
    with col5:
        t_max = st.number_input(
            "t máx (s)",
            min_value=1.0, value=20.0, step=1.0, format="%.1f",
            help="Tempo máximo do gráfico."
        )

calcular_btn = st.button("Calcular solução")

# ─── Cálculo e resultados ─────────────────────────────────────────────────────

if calcular_btn or "res" in st.session_state:

    if calcular_btn:
        if A <= 0 or Rv <= 0 or t_max <= 0:
            st.markdown("""
<div class="warn-box">
  ⚠️ Os parâmetros A, Rv e t máximo devem ser positivos.
</div>
""", unsafe_allow_html=True)
            st.stop()
        st.session_state["res"] = calcular(A, Rv, M, h_bar, t_max)
        st.session_state["params"] = dict(A=A, Rv=Rv, M=M, h_bar=h_bar, t_max=t_max)

    res = st.session_state["res"]
    p = st.session_state["params"]
    tau = res["tau"]
    h_dev_inf = res["h_dev_inf"]
    h_inf = res["h_inf"]
    t_arr = res["t"]
    h_dev_arr = res["h_dev"]
    h_arr = res["h"]

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="gold-accent"></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-label">Resultados</p>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Parâmetros calculados</h2>', unsafe_allow_html=True)

    # Cards de resultado
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
<div class="result-card" style="border-top:3px solid #1A3A5C;">
  <div class="card-label">Constante de tempo</div>
  <div style="font-size:0.8rem;color:#374151;margin-bottom:0.4rem;">τ = A · Rᵥ</div>
  <div class="card-value">{fmt(tau, 4)} s</div>
</div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
<div class="result-card" style="border-top:3px solid #1A3A5C;">
  <div class="card-label">Desvio final do nível</div>
  <div style="font-size:0.8rem;color:#374151;margin-bottom:0.4rem;">h'(∞) = Rᵥ · M</div>
  <div class="card-value">{fmt(h_dev_inf, 4)} m</div>
</div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
<div class="result-card" style="border-top:3px solid #C4A32A;">
  <div class="card-label">Nível final (real)</div>
  <div style="font-size:0.8rem;color:#374151;margin-bottom:0.4rem;">h(∞) = h̄ + Rᵥ · M</div>
  <div class="card-value-gold">{fmt(h_inf, 4)} m</div>
</div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
<div class="result-card" style="border-top:3px solid #6B7280;">
  <div class="card-label">Nível inicial</div>
  <div style="font-size:0.8rem;color:#374151;margin-bottom:0.4rem;">h̄ (equilíbrio)</div>
  <div class="card-value" style="color:#6B7280;">{fmt(p["h_bar"], 4)} m</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Expressão numérica
    st.markdown("""
<div style="background:#fff;border-left:3px solid #1A3A5C;padding:1.25rem 1.5rem;margin-bottom:1rem;">
  <div style="font-family:\'Inter\',sans-serif;font-size:0.7rem;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;color:#6B7280;margin-bottom:0.75rem;">
    Expressão numérica com os parâmetros inseridos
  </div>
""", unsafe_allow_html=True)
    st.latex(
        rf"h'(t) = {fmt(h_dev_inf, 4)}\left(1 - e^{{-t/{fmt(tau, 4)}}}\right)"
    )
    st.latex(
        rf"h(t) = {fmt(p['h_bar'], 4)} + {fmt(h_dev_inf, 4)}\left(1 - e^{{-t/{fmt(tau, 4)}}}\right)"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Tabela de valores-chave
    st.markdown("""
<div style="background:#fff;border:1px solid #DEE2E9;padding:1.25rem 1.5rem;margin-bottom:1rem;">
  <div style="font-family:\'Inter\',sans-serif;font-size:0.7rem;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;color:#6B7280;margin-bottom:0.75rem;">
    Valores em instantes representativos
  </div>
""", unsafe_allow_html=True)

    fatores = [1, 2, 3, 4, 5]
    labels = ["τ (63,2%)", "2τ (86,5%)", "3τ (95,0%)", "4τ (98,2%)", "5τ (99,3%)"]
    rows = []
    for fator, label in zip(fatores, labels):
        t_val = fator * tau
        hd = h_dev_inf * (1 - np.exp(-fator))
        hv = p["h_bar"] + hd
        pct = f"{(hd / h_dev_inf * 100):.1f}%" if h_dev_inf != 0 else "—"
        rows.append(f"""
<tr>
  <td>{label}</td>
  <td style="font-family:\'JetBrains Mono\',monospace;">{fmt(t_val, 3)}</td>
  <td style="font-family:\'JetBrains Mono\',monospace;">{fmt(hd, 4)}</td>
  <td style="font-family:\'JetBrains Mono\',monospace;">{fmt(hv, 4)}</td>
  <td>{pct}</td>
</tr>""")

    st.markdown(f"""
<table>
  <thead>
    <tr>
      <th>Instante</th>
      <th>t (s)</th>
      <th>h'(t) (m)</th>
      <th>h(t) (m)</th>
      <th>% do regime</th>
    </tr>
  </thead>
  <tbody>
    {"".join(rows)}
  </tbody>
</table>
</div>
""", unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ─── Gráfico ──────────────────────────────────────────────────────────────

    st.markdown('<div class="gold-accent"></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-label">Visualização</p>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Resposta temporal do nível do tanque</h2>', unsafe_allow_html=True)

    curva = st.radio(
        "Exibir:",
        ["Nível real h(t)", "Desvio h'(t)", "Ambas as curvas"],
        horizontal=True,
        label_visibility="collapsed",
    )

    fig = go.Figure()

    # Linha do nível inicial
    fig.add_hline(
        y=p["h_bar"], line_dash="dash", line_color="#9CA3AF", line_width=1.2,
        annotation_text=f"h̄ = {fmt(p['h_bar'], 2)} m",
        annotation_position="top right",
        annotation_font=dict(color="#9CA3AF", size=11),
    )
    # Linha do nível final
    fig.add_hline(
        y=h_inf, line_dash="dash", line_color="#C4A32A", line_width=1.5,
        annotation_text=f"h(∞) = {fmt(h_inf, 3)} m",
        annotation_position="bottom right",
        annotation_font=dict(color="#C4A32A", size=11),
    )
    # Linha de τ
    fig.add_vline(
        x=tau, line_dash="dot", line_color="#6B7280", line_width=1.2,
        annotation_text=f"τ = {fmt(tau, 2)} s",
        annotation_position="top right",
        annotation_font=dict(color="#6B7280", size=10),
    )

    if curva in ("Nível real h(t)", "Ambas as curvas"):
        fig.add_trace(go.Scatter(
            x=t_arr, y=h_arr,
            mode="lines",
            name="h(t) — nível real",
            line=dict(color="#1A3A5C", width=2.5),
        ))

    if curva in ("Desvio h'(t)", "Ambas as curvas"):
        fig.add_trace(go.Scatter(
            x=t_arr, y=h_dev_arr,
            mode="lines",
            name="h'(t) — desvio",
            line=dict(color="#C4A32A", width=2, dash="dot"),
        ))

    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", size=12, color="#374151"),
        margin=dict(l=50, r=50, t=30, b=60),
        xaxis=dict(
            title="Tempo (s)",
            showgrid=True, gridcolor="#E5E7EB",
            zeroline=True, zerolinecolor="#9CA3AF",
            linecolor="#D1D5DB",
        ),
        yaxis=dict(
            title="Nível (m)",
            showgrid=True, gridcolor="#E5E7EB",
            zeroline=True, zerolinecolor="#9CA3AF",
            linecolor="#D1D5DB",
        ),
        legend=dict(
            orientation="h", y=1.08, x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", size=12),
        ),
        height=430,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Interpretação
    st.markdown(f"""
<div class="interp-block">
  <h4>🔍 Interpretação física do resultado</h4>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:1.5rem;">
    <div>
      <p style="margin:0 0 0.5rem 0;">
        O sistema parte do nível de equilíbrio
        <strong style="color:#C4A32A;">h̄ = {fmt(p['h_bar'], 3)} m</strong>
        e, após a perturbação degrau de magnitude
        <strong style="color:#C4A32A;">M = {fmt(p['M'], 3)} m³/s</strong>,
        converge exponencialmente para o novo regime permanente.
      </p>
      <p style="margin:0;">
        A constante de tempo <strong style="color:#C4A32A;">τ = {fmt(tau, 4)} s</strong>
        determina a velocidade da resposta: quanto maior τ, mais lentamente
        o tanque se adapta à nova vazão.
      </p>
    </div>
    <div>
      <p style="margin:0 0 0.5rem 0;">
        O ganho estático <strong style="color:#C4A32A;">K = Rᵥ = {fmt(p['Rv'], 3)}</strong>
        amplifica a perturbação: para cada unidade de aumento na vazão,
        o nível sobe <strong style="color:#C4A32A;">{fmt(p['Rv'], 3)} m</strong> em regime permanente.
      </p>
      <p style="margin:0;">
        Após <strong style="color:#C4A32A;">5τ = {fmt(5 * tau, 3)} s</strong>,
        o sistema atingiu 99,3% da variação total,
        sendo considerado em regime permanente.
      </p>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ─── Aplicações ───────────────────────────────────────────────────────────────

st.markdown('<div class="gold-accent"></div>', unsafe_allow_html=True)
st.markdown('<p class="section-label">Contexto</p>', unsafe_allow_html=True)
st.markdown('<h2 class="section-title">Aplicações em Engenharia Química e de Processos</h2>', unsafe_allow_html=True)

col_app1, col_app2 = st.columns(2, gap="large")

with col_app1:
    st.markdown("""
<div class="app-item">
  <strong>Controle de nível em reatores</strong><br>
  O modelo de tanque é a base para o projeto de controladores PID em reatores
  químicos contínuos (CSTR), onde o nível deve ser mantido estável.
</div>
<div class="app-item">
  <strong>Sistemas de abastecimento d'água</strong><br>
  Reservatórios de distribuição de água seguem exatamente este modelo,
  com variações de demanda representadas como entradas degrau.
</div>
<div class="app-item">
  <strong>Colunas de destilação</strong><br>
  O nível nos refervededores e condensadores de colunas de destilação é
  controlado usando modelos de primeira ordem similares a este.
</div>
""", unsafe_allow_html=True)

with col_app2:
    st.markdown("""
<div class="app-item">
  <strong>Indústria farmacêutica</strong><br>
  Tanques de mistura e diluição em processos farmacêuticos exigem controle
  preciso do nível para garantir as concentrações corretas.
</div>
<div class="app-item">
  <strong>Processos de evaporação</strong><br>
  Evaporadores industriais utilizam balanços de massa equivalentes para
  modelar a variação do nível do líquido concentrado.
</div>
<div class="app-item">
  <strong>Malhas de controle industrial</strong><br>
  A função de transferência G(s) = Rᵥ/(τs+1) é o modelo de processo padrão
  em sintonizações de controladores pelo método de Ziegler-Nichols.
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
