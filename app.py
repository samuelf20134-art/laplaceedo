"""
Transformada de Laplace — Ferramenta interativa para resolução de EDOs lineares
UNIMONTES | Disciplina de Matemática Aplicada | Prof. Fernando Félix
Alunos: Bruno Gomes, Júlio César, Leonardo, Marcus
"""

import streamlit as st
import sympy as sp
import numpy as np
import plotly.graph_objects as go
from scipy.integrate import solve_ivp
import os

# ─── Configuração da página ──────────────────────────────────────────────────

st.set_page_config(
    page_title="Transformada de Laplace — EDOs",
    page_icon="assets/favicon.png" if os.path.exists("assets/favicon.png") else "📐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CSS personalizado ────────────────────────────────────────────────────────

st.markdown("""
<style>
  /* Importação de fonte acadêmica */
  @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,wght@0,300;0,400;0,600;1,400&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

  /* Resetar fundo padrão */
  .stApp { background-color: #F7F8FA; }
  .block-container { padding-top: 1.5rem; max-width: 1200px; }

  /* Barra institucional */
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
  .header-bar .logo-placeholder {
    width: 72px; height: 72px;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.25);
    display: flex; align-items: center; justify-content: center;
    font-size: 0.65rem; color: rgba(255,255,255,0.6);
    text-align: center; flex-shrink: 0;
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

  /* Seções */
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

  /* Painel de entrada */
  .input-panel {
    background: #FFFFFF;
    border: 1px solid #DEE2E9;
    padding: 1.5rem 1.75rem;
    margin-bottom: 1.5rem;
  }
  .input-panel h3 {
    font-family: 'Source Serif 4', serif;
    font-size: 1.1rem; color: #1A3A5C;
    margin: 0 0 1.2rem 0;
  }

  /* Blocos de resultado */
  .result-block {
    background: #FFFFFF;
    border-left: 3px solid #1A3A5C;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
    font-family: 'Inter', sans-serif;
    font-size: 0.9rem;
  }
  .result-block .label {
    font-size: 0.7rem; font-weight: 600;
    letter-spacing: 0.1em; text-transform: uppercase;
    color: #6B7280; margin-bottom: 0.5rem;
  }
  .result-step {
    background: #F7F8FA;
    border: 1px solid #E5E7EB;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    font-family: 'Inter', sans-serif;
  }
  .result-step .step-num {
    font-size: 0.65rem; font-weight: 700;
    letter-spacing: 0.1em; text-transform: uppercase;
    color: #C4A32A; margin-bottom: 0.4rem;
  }

  /* Interpretação */
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

  /* Botão principal */
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

  /* Botão secundário (exemplo guiado) */
  .stButton.example-btn > button {
    background: transparent !important;
    color: #1A3A5C !important;
    border: 1px solid #1A3A5C !important;
  }

  /* Monospace para equações inline */
  code {
    font-family: 'JetBrains Mono', monospace;
    background: #EEF0F4;
    padding: 0.1em 0.4em;
    font-size: 0.85em;
    color: #1A3A5C;
  }

  /* Aplicações */
  .app-item {
    border-top: 1px solid #DEE2E9;
    padding: 0.85rem 0;
    font-family: 'Inter', sans-serif;
    font-size: 0.87rem; color: #374151;
  }
  .app-item strong {
    color: #1A3A5C; font-weight: 600;
  }

  /* Alerta / aviso */
  .warn-box {
    background: #FEF3C7;
    border-left: 3px solid #C4A32A;
    padding: 0.9rem 1.2rem;
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem; color: #78350F;
  }

  /* Tabela de propriedades */
  table { width: 100%; border-collapse: collapse; }
  th { background: #1A3A5C; color: #fff; padding: 0.6rem 0.9rem;
       font-family: 'Inter', sans-serif; font-size: 0.8rem;
       font-weight: 500; text-align: left; }
  td { padding: 0.55rem 0.9rem; font-family: 'Inter', sans-serif;
       font-size: 0.82rem; border-bottom: 1px solid #E5E7EB; }
  tr:nth-child(even) td { background: #F7F8FA; }

  /* Rodapé */
  .footer {
    margin-top: 3rem; padding: 1.5rem 0;
    border-top: 1px solid #DEE2E9;
    font-family: 'Inter', sans-serif;
    font-size: 0.75rem; color: #9CA3AF;
    text-align: center;
  }
</style>
""", unsafe_allow_html=True)


# ─── Funções matemáticas ──────────────────────────────────────────────────────

def parse_f(expr_str: str):
    """Converte string da força externa f(t) em expressão SymPy."""
    t = sp.Symbol('t')
    local_dict = {
        't': t,
        'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
        'exp': sp.exp, 'log': sp.log, 'sqrt': sp.sqrt,
        'pi': sp.pi, 'e': sp.E,
        'Heaviside': sp.Heaviside,
    }
    try:
        return sp.sympify(expr_str, locals=local_dict)
    except Exception:
        return None


def montar_laplace(a, b, c, f_expr, y0, v0):
    """
    Monta a equação algébrica em Y(s) após aplicar a Transformada de Laplace
    à EDO:  a*y'' + b*y' + c*y = f(t),  y(0)=y0, y'(0)=v0.

    Retorna (Y_s, s, t, F_s) onde Y_s é a expressão de Y(s).
    """
    s, t = sp.symbols('s t')
    Y = sp.Function('Y')(s)

    # Transformadas das derivadas
    # L{y''} = s²Y - s·y(0) - y'(0)
    # L{y'}  = sY - y(0)
    Ly2 = s**2 * Y - s * y0 - v0
    Ly1 = s * Y - y0
    Ly0 = Y

    # Transformada de f(t)
    try:
        F_s = sp.laplace_transform(f_expr, t, s, noconds=True)
    except Exception:
        F_s = None

    if F_s is None:
        return None, s, t, None

    # Equação algébrica: a·L{y''} + b·L{y'} + c·L{y} = F(s)
    equacao = sp.Eq(a * Ly2 + b * Ly1 + c * Ly0, F_s)

    try:
        sol = sp.solve(equacao, Y)
        if not sol:
            return None, s, t, F_s
        Y_s = sp.simplify(sol[0])
    except Exception:
        return None, s, t, F_s

    return Y_s, s, t, F_s


def decompor_fracoes(Y_s, s):
    """Aplica decomposição em frações parciais a Y(s)."""
    try:
        Y_parcial = sp.apart(Y_s, s)
        return Y_parcial
    except Exception:
        return Y_s


def transformada_inversa(Y_s, s, t):
    """Calcula a transformada inversa de Laplace de Y(s)."""
    try:
        y_t = sp.inverse_laplace_transform(Y_s, s, t, noconds=True)
        y_t = sp.simplify(sp.expand(y_t))
        return y_t
    except Exception:
        return None


def solucao_numerica(a, b, c, f_func, y0, v0, t_span, n_pontos=600):
    """
    Resolve a EDO numericamente usando scipy.integrate.solve_ivp.
    Reescreve como sistema de primeira ordem:
      u1' = u2
      u2' = (f(t) - b·u2 - c·u1) / a
    """
    def sistema(t, u):
        u1, u2 = u
        ft = f_func(t) if callable(f_func) else float(f_func)
        du2 = (ft - b * u2 - c * u1) / a
        return [u2, du2]

    t_eval = np.linspace(t_span[0], t_span[1], n_pontos)
    sol = solve_ivp(
        sistema,
        t_span,
        [y0, v0],
        t_eval=t_eval,
        method='RK45',
        rtol=1e-8, atol=1e-10,
        dense_output=False,
    )
    if sol.success:
        return sol.t, sol.y[0]
    return None, None


def avaliar_simbolico(y_expr, t_sym, t_array):
    """Avalia a expressão simbólica y(t) em um array numérico."""
    try:
        f_num = sp.lambdify(t_sym, y_expr, modules=['numpy'])
        return f_num(t_array)
    except Exception:
        return None


def interpretar_solucao(y_expr, t):
    """Gera uma descrição técnica do comportamento da solução."""
    texto = []
    expr_str = str(y_expr)

    tem_exp_pos = any(
        atom.is_Mul and any(
            isinstance(f, sp.exp) and f.args[0].coeff(t) > 0
            for f in sp.preorder_traversal(atom)
        )
        for atom in sp.preorder_traversal(y_expr)
    )

    exps = []
    for atom in sp.preorder_traversal(y_expr):
        if isinstance(atom, sp.exp):
            arg = atom.args[0]
            coef = arg.coeff(t)
            if coef != 0:
                exps.append(float(coef.evalf()) if coef.is_number else None)

    tem_sin_cos = 'sin' in expr_str or 'cos' in expr_str
    tem_exp_neg = any(e is not None and e < 0 for e in exps)
    tem_exp_pos_flag = any(e is not None and e > 0 for e in exps)

    if tem_exp_pos_flag and not tem_exp_neg:
        texto.append("A solução cresce sem limite com o tempo — o sistema é instável. "
                      "Pelo menos um modo exponencial diverge.")
    elif tem_exp_neg and not tem_exp_pos_flag:
        if tem_sin_cos:
            texto.append("A solução apresenta oscilação amortecida: os termos oscilatórios "
                          "decaem exponencialmente e a resposta converge para zero.")
        else:
            texto.append("A solução decai exponencialmente para zero. "
                          "O sistema é assintoticamente estável.")
    elif tem_exp_pos_flag and tem_exp_neg:
        texto.append("A solução contém modos que crescem e modos que decaem. "
                      "O comportamento dominante a longo prazo é ditado pelo termo exponencial positivo.")
    elif tem_sin_cos and not exps:
        texto.append("A solução oscila harmonicamente com amplitude constante. "
                      "Não há amortecimento: o sistema é marginalmente estável.")
    else:
        texto.append("O comportamento da solução depende da combinação dos termos presentes.")

    if exps:
        vals = [e for e in exps if e is not None]
        if vals:
            rates = ", ".join([f"{e:+.3f}" for e in sorted(set(vals))])
            texto.append(f"Taxas exponenciais identificadas: {rates}. "
                          "Valores negativos correspondem a amortecimento; positivos, a crescimento.")

    return " ".join(texto)


# ─── Cabeçalho institucional ──────────────────────────────────────────────────

logo_path = "assets/logo_unimontes.png"
if os.path.exists(logo_path):
    import base64
    with open(logo_path, "rb") as img_file:
        logo_b64 = base64.b64encode(img_file.read()).decode()
    logo_html = f'<img src="data:image/png;base64,{logo_b64}" style="width:72px;height:72px;object-fit:contain;" alt="UNIMONTES">'
else:
    logo_html = '<div class="logo-placeholder">LOGO<br>UNIMONTES</div>'

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
    Leonardo · Marcus
  </div>
</div>
""", unsafe_allow_html=True)


# ─── Introdução ──────────────────────────────────────────────────────────────

col_intro, col_prop = st.columns([3, 2], gap="large")

with col_intro:
    st.markdown('<div class="gold-accent"></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-label">O método</p>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Passo a passo</h2>',
                unsafe_allow_html=True)
    st.markdown("""
Se vcs acharem bom, dá pra bolar uma explicação legal pra ficar tipo um slide, tbm tem que definir o tema e tal
    """)

with col_prop:
    st.markdown('<div class="gold-accent"></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-label">Propriedades fundamentais</p>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Transformadas das derivadas</h2>',
                unsafe_allow_html=True)
    st.markdown("""
| Função | Transformada |
|---|---|
| $y(t)$ | $Y(s)$ |
| $y'(t)$ | $sY(s) - y(0)$ |
| $y''(t)$ | $s^2 Y(s) - sy(0) - y'(0)$ |
| $e^{at}$ | $\\frac{1}{s-a}$ |
| $\\sin(at)$ | $\\frac{a}{s^2+a^2}$ |
| $\\cos(at)$ | $\\frac{s}{s^2+a^2}$ |
| $t^n$ | $\\frac{n!}{s^{n+1}}$ |
    """)

st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ─── Painel de entrada ────────────────────────────────────────────────────────

st.markdown('<div class="gold-accent"></div>', unsafe_allow_html=True)
st.markdown('<p class="section-label">Calculadora</p>', unsafe_allow_html=True)
st.markdown('<h2 class="section-title">Defina a EDO e as condições iniciais</h2>',
            unsafe_allow_html=True)

# Carregar exemplo guiado
if "exemplo_carregado" not in st.session_state:
    st.session_state.exemplo_carregado = False

col_ex, _ = st.columns([1, 4])
with col_ex:
    if st.button("Carregar exemplo do seminário"):
        st.session_state.exemplo_carregado = True
        st.session_state.a_val = 1.0
        st.session_state.b_val = -1.0
        st.session_state.c_val = -2.0
        st.session_state.f_str = "0"
        st.session_state.y0_val = 1.0
        st.session_state.v0_val = 0.0
        st.session_state.t_max = 3.0

if st.session_state.exemplo_carregado:
    st.markdown("""
<div class="warn-box">
Exemplo guiado carregado: <strong>y'' − y' − 2y = 0</strong>,&nbsp;
y(0) = 1, y'(0) = 0. A solução esperada é y(t) = (1/3)e²ᵗ + (2/3)e⁻ᵗ.
</div>
    """, unsafe_allow_html=True)

st.markdown("")

with st.container():
    st.markdown('<div class="input-panel">', unsafe_allow_html=True)
    st.markdown('<h3>EDO: &nbsp; a · y″ + b · y′ + c · y = f(t)</h3>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        a_val = st.number_input("Coeficiente a (y″)",
                                 value=st.session_state.get("a_val", 1.0),
                                 key="a_input", step=0.5)
    with col2:
        b_val = st.number_input("Coeficiente b (y′)",
                                 value=st.session_state.get("b_val", -1.0),
                                 key="b_input", step=0.5)
    with col3:
        c_val = st.number_input("Coeficiente c (y)",
                                 value=st.session_state.get("c_val", -2.0),
                                 key="c_input", step=0.5)

    col4, col5, col6, col7 = st.columns([2, 1, 1, 1])
    with col4:
        f_str = st.text_input(
            "Força externa f(t)",
            value=st.session_state.get("f_str", "0"),
            placeholder="Ex: sin(2*t), exp(t), t, 0",
            key="f_input",
        )
    with col5:
        y0_val = st.number_input("y(0)", value=st.session_state.get("y0_val", 1.0),
                                  key="y0_input", step=0.5)
    with col6:
        v0_val = st.number_input("y′(0)", value=st.session_state.get("v0_val", 0.0),
                                  key="v0_input", step=0.5)
    with col7:
        t_max = st.number_input("t máximo", value=st.session_state.get("t_max", 5.0),
                                 min_value=0.5, max_value=50.0, step=0.5, key="tmax_input")

    st.markdown('</div>', unsafe_allow_html=True)

calcular = st.button("Calcular solução")

# ─── Cálculo ─────────────────────────────────────────────────────────────────

if calcular or st.session_state.get("resultado_disponivel", False):

    if calcular:
        # Validar entrada
        if a_val == 0:
            st.error("O coeficiente a não pode ser zero (não seria uma equação de segunda ordem).")
            st.stop()

        f_expr = parse_f(f_str)
        if f_expr is None:
            st.error(f"Não foi possível interpretar a expressão f(t) = '{f_str}'. "
                      "Use notação Python: sin(t), cos(2*t), exp(-t), t**2, etc.")
            st.stop()

        with st.spinner("Calculando..."):
            Y_s, s_sym, t_sym, F_s = montar_laplace(a_val, b_val, c_val, f_expr, y0_val, v0_val)

        if Y_s is None:
            st.error("Não foi possível obter Y(s). A função f(t) inserida pode não ter "
                      "transformada de Laplace expressa em forma fechada.")
            st.stop()

        Y_parcial = decompor_fracoes(Y_s, s_sym)
        y_t = transformada_inversa(Y_s, s_sym, t_sym)

        if y_t is None:
            st.error("A transformada inversa de Y(s) não pôde ser calculada simbolicamente. "
                      "Verifique os coeficientes — raízes complexas podem exigir tratamento adicional.")
            st.stop()

        st.session_state.resultado_disponivel = True
        st.session_state.resultado = {
            "a": a_val, "b": b_val, "c": c_val,
            "f_expr": f_expr, "f_str": f_str,
            "y0": y0_val, "v0": v0_val, "t_max": t_max,
            "Y_s": Y_s, "Y_parcial": Y_parcial,
            "y_t": y_t, "s_sym": s_sym, "t_sym": t_sym,
            "F_s": F_s,
        }

    # ── Recuperar resultado armazenado ──
    res = st.session_state.resultado
    a_val = res["a"]; b_val = res["b"]; c_val = res["c"]
    f_expr = res["f_expr"]; f_str = res["f_str"]
    y0_val = res["y0"]; v0_val = res["v0"]; t_max = res["t_max"]
    Y_s = res["Y_s"]; Y_parcial = res["Y_parcial"]
    y_t = res["y_t"]; s_sym = res["s_sym"]; t_sym = res["t_sym"]
    F_s = res["F_s"]

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ─── Cálculo simbólico passo a passo ─────────────────────────────────────

    st.markdown('<div class="gold-accent"></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-label">Desenvolvimento simbólico</p>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Resolução passo a passo</h2>',
                unsafe_allow_html=True)

    col_passos, col_espaco = st.columns([3, 1])
    with col_passos:

        # Passo 1: EDO original
        st.markdown('<div class="result-step">', unsafe_allow_html=True)
        st.markdown('<div class="step-num">Passo 1 — EDO no domínio do tempo</div>', unsafe_allow_html=True)
        sinal_b = f"+ {b_val}" if b_val >= 0 else f"- {abs(b_val)}"
        sinal_c = f"+ {c_val}" if c_val >= 0 else f"- {abs(c_val)}"
        f_display = f_str if f_str != "0" else "0"
        st.latex(
            rf"{sp.latex(sp.Rational(a_val).limit_denominator(1000))} \, y'' \;"
            rf"{sinal_b} \, y' \; {sinal_c} \, y = {sp.latex(f_expr)}"
        )
        st.caption(f"Condições iniciais: y(0) = {y0_val}, y'(0) = {v0_val}")
        st.markdown('</div>', unsafe_allow_html=True)

        # Passo 2: Transformada
        st.markdown('<div class="result-step">', unsafe_allow_html=True)
        st.markdown('<div class="step-num">Passo 2 — Transformada de Laplace e substituição das condições iniciais</div>',
                    unsafe_allow_html=True)
        st.latex(
            r"\mathcal{L}\{y''\} = s^2 Y(s) - s \cdot y(0) - y'(0), \quad"
            r"\mathcal{L}\{y'\} = s Y(s) - y(0)"
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # Passo 3: Y(s) simplificado
        st.markdown('<div class="result-step">', unsafe_allow_html=True)
        st.markdown('<div class="step-num">Passo 3 — Y(s) resolvido</div>', unsafe_allow_html=True)
        st.latex(r"Y(s) = " + sp.latex(Y_s))
        st.markdown('</div>', unsafe_allow_html=True)

        # Passo 4: Frações parciais
        if Y_parcial != Y_s:
            st.markdown('<div class="result-step">', unsafe_allow_html=True)
            st.markdown('<div class="step-num">Passo 4 — Decomposição em frações parciais</div>',
                        unsafe_allow_html=True)
            st.latex(r"Y(s) = " + sp.latex(Y_parcial))
            st.markdown('</div>', unsafe_allow_html=True)

        # Passo 5: Solução
        st.markdown('<div class="result-step" style="border-left: 3px solid #C4A32A;">', unsafe_allow_html=True)
        st.markdown('<div class="step-num" style="color:#1A3A5C;">Resultado — y(t)</div>',
                    unsafe_allow_html=True)
        st.latex(r"y(t) = " + sp.latex(y_t))
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ─── Visualização ─────────────────────────────────────────────────────────

    st.markdown('<div class="gold-accent"></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-label">Visualização</p>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Gráfico de y(t)</h2>', unsafe_allow_html=True)

    t_array = np.linspace(0, t_max, 800)
    y_simb = avaliar_simbolico(y_t, t_sym, t_array)

    # Solução numérica
    f_num_func = sp.lambdify(t_sym, f_expr, modules=['numpy']) if f_expr != sp.Integer(0) else lambda t: 0.0
    try:
        t_num, y_num = solucao_numerica(a_val, b_val, c_val, f_num_func, y0_val, v0_val,
                                        (0, t_max))
    except Exception:
        t_num, y_num = None, None

    modo_grafico = st.radio(
        "Exibir:",
        ["Solução simbólica", "Solução numérica (RK45)", "Comparação"],
        horizontal=True,
        label_visibility="collapsed",
    )

    fig = go.Figure()

    if modo_grafico in ("Solução simbólica", "Comparação") and y_simb is not None:
        # Limpar valores inválidos
        y_simb_clean = np.where(np.isfinite(y_simb), y_simb, np.nan)
        fig.add_trace(go.Scatter(
            x=t_array, y=y_simb_clean,
            mode='lines',
            name='Simbólica (SymPy)',
            line=dict(color='#1A3A5C', width=2.5),
        ))

    if modo_grafico in ("Solução numérica (RK45)", "Comparação") and t_num is not None:
        fig.add_trace(go.Scatter(
            x=t_num, y=y_num,
            mode='lines',
            name='Numérica (RK45)',
            line=dict(color='#C4A32A', width=2, dash='dot'),
        ))

    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Inter, sans-serif', size=12, color='#374151'),
        margin=dict(l=50, r=30, t=30, b=50),
        xaxis=dict(
            title='t', showgrid=True, gridcolor='#E5E7EB',
            zeroline=True, zerolinecolor='#9CA3AF', linecolor='#D1D5DB',
        ),
        yaxis=dict(
            title='y(t)', showgrid=True, gridcolor='#E5E7EB',
            zeroline=True, zerolinecolor='#9CA3AF', linecolor='#D1D5DB',
        ),
        legend=dict(
            orientation='h', y=1.05, x=0,
            bgcolor='rgba(0,0,0,0)',
        ),
        height=400,
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ─── Interpretação ────────────────────────────────────────────────────────

    st.markdown('<div class="gold-accent"></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-label">Análise</p>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Interpretação da solução</h2>',
                unsafe_allow_html=True)

    interpretacao = interpretar_solucao(y_t, t_sym)
    st.markdown(f"""
<div class="interp-block">
  <h4>Comportamento qualitativo</h4>
  {interpretacao}
  <br><br>
  <strong>Expressão:</strong>&nbsp; y(t) = {sp.latex(y_t)}
</div>
    """, unsafe_allow_html=True)


# ─── Aplicações ───────────────────────────────────────────────────────────────

st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown('<div class="gold-accent"></div>', unsafe_allow_html=True)
st.markdown('<p class="section-label">Contexto</p>', unsafe_allow_html=True)
st.markdown('<h2 class="section-title">Onde a Transformada de Laplace é usada</h2>',
            unsafe_allow_html=True)

col_app1, col_app2 = st.columns(2)

with col_app1:
    st.markdown("""
<div class="app-item">
  <strong>Circuitos elétricos (RLC)</strong><br>
  A equação L·I'' + R·I' + (1/C)·I = E'(t) descreve a corrente num circuito série.
  A transformada converte isso em uma expressão algébrica para I(s), facilitando
  a análise em regime permanente e transitório.
</div>
<div class="app-item">
  <strong>Sistemas massa-mola-amortecedor</strong><br>
  m·x'' + γ·x' + k·x = F(t) é formalmente idêntica à equação do circuito RLC.
  As mesmas técnicas se aplicam, e a solução revela frequências naturais, amortecimento
  e resposta a forças externas periódicas ou impulsivas.
</div>
<div class="app-item">
  <strong>Engenharia de controle</strong><br>
  A função de transferência H(s) = Y(s)/U(s) é o objeto central da teoria de controle
  clássica. A Transformada de Laplace permite representar sistemas dinâmicos como
  frações racionais em s, facilitando análise de estabilidade e projeto de controladores.
</div>
    """, unsafe_allow_html=True)

with col_app2:
    st.markdown("""
<div class="app-item">
  <strong>Processamento de sinais</strong><br>
  O plano s permite localizar polos e zeros de um sistema, determinando sua resposta
  em frequência. Filtros passa-baixa, passa-alta e passa-banda são projetados diretamente
  a partir da posição dos polos em s.
</div>
<div class="app-item">
  <strong>EDOs com forçamentos descontínuos</strong><br>
  Funções degrau (Heaviside) e impulsos (delta de Dirac) são difíceis de tratar
  no domínio do tempo. No domínio de s, elas têm transformadas simples (e⁻ᶜˢ/s
  e e⁻ᶜˢ), e o método se aplica sem modificações.
</div>
<div class="app-item">
  <strong>Equações de 4ª ordem e superiores</strong><br>
  Em teoria de vigas (equação de Euler-Bernoulli) e mecânica dos fluidos, surgem EDOs
  de ordem 4 ou superior. A transformada reduz qualquer ordem a um problema algébrico
  do mesmo grau.
</div>
    """, unsafe_allow_html=True)

# ─── Rodapé ───────────────────────────────────────────────────────────────────

st.markdown("""
<div class="footer">
  UNIMONTES — Universidade Estadual de Montes Claros &nbsp;·&nbsp;
  Disciplina de Equações Diferenciais Ordinárias &nbsp;·&nbsp; Prof. Fernando Félix<br>
  Bruno Gomes · Júlio César · Leonardo · Marcus &nbsp;·&nbsp;
  Construído com Streamlit, SymPy, NumPy, Plotly e SciPy
</div>
""", unsafe_allow_html=True)
