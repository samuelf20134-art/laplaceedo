```python
"""
Transformada de Laplace — Ferramenta interativa para resolução de EDOs lineares
UNIMONTES | Disciplina de Equações Diferenciais Ordinárias | Prof. Fernando Félix
Alunos: Bruno Gomes, Júlio César, Leonardo, Marcus e Samuel Antunes França
"""

import os
import base64

import streamlit as st
import sympy as sp
import numpy as np
import plotly.graph_objects as go
from scipy.integrate import solve_ivp


# ─────────────────────────────────────────────────────────────────────────────
# Configuração da página
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Transformada de Laplace — EDOs",
    page_icon="assets/favicon.png" if os.path.exists("assets/favicon.png") else "📐",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ─────────────────────────────────────────────────────────────────────────────
# CSS personalizado
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,wght@0,300;0,400;0,600;1,400&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

  .stApp {
    background-color: #F7F8FA;
  }

  .block-container {
    padding-top: 1.5rem;
    max-width: 1200px;
  }

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
    width: 72px;
    height: 72px;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.25);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.65rem;
    color: rgba(255,255,255,0.6);
    text-align: center;
    flex-shrink: 0;
  }

  .header-title {
    font-family: 'Source Serif 4', serif;
    font-size: 1.4rem;
    font-weight: 600;
    line-height: 1.25;
    margin: 0;
  }

  .header-sub {
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem;
    font-weight: 300;
    opacity: 0.75;
    margin-top: 0.25rem;
    letter-spacing: 0.03em;
  }

  .header-credits {
    margin-left: auto;
    text-align: right;
    font-family: 'Inter', sans-serif;
    font-size: 0.75rem;
    opacity: 0.78;
    line-height: 1.6;
  }

  .section-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #1A3A5C;
    opacity: 0.65;
    margin-bottom: 0.3rem;
  }

  .section-title {
    font-family: 'Source Serif 4', serif;
    font-size: 1.45rem;
    font-weight: 600;
    color: #111827;
    margin: 0 0 1rem 0;
    line-height: 1.3;
  }

  .divider {
    border: none;
    border-top: 1px solid #DEE2E9;
    margin: 2rem 0;
  }

  .gold-accent {
    display: inline-block;
    width: 32px;
    height: 3px;
    background: #C4A32A;
    margin-bottom: 0.6rem;
  }

  .input-panel {
    background: #FFFFFF;
    border: 1px solid #DEE2E9;
    padding: 1.5rem 1.75rem;
    margin-bottom: 1.5rem;
  }

  .input-panel h3 {
    font-family: 'Source Serif 4', serif;
    font-size: 1.1rem;
    color: #1A3A5C;
    margin: 0 0 1.2rem 0;
  }

  .result-step {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-left: 3px solid #1A3A5C;
    padding: 1rem 1.25rem;
    margin-bottom: 0.9rem;
    font-family: 'Inter', sans-serif;
  }

  .result-step .step-num {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #C4A32A;
    margin-bottom: 0.4rem;
  }

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
    font-size: 1rem;
    color: #C4A32A;
    margin: 0 0 0.75rem 0;
  }

  div[data-testid="stButton"] > button {
    background: #1A3A5C !important;
    color: #FFFFFF !important;
    border: none !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em !important;
    padding: 0.55rem 1.75rem !important;
    border-radius: 0 !important;
  }

  div[data-testid="stButton"] > button:hover {
    background: #243F62 !important;
  }

  code {
    font-family: 'JetBrains Mono', monospace;
    background: #EEF0F4;
    padding: 0.1em 0.4em;
    font-size: 0.85em;
    color: #1A3A5C;
  }

  .app-item {
    border-top: 1px solid #DEE2E9;
    padding: 0.85rem 0;
    font-family: 'Inter', sans-serif;
    font-size: 0.87rem;
    color: #374151;
    line-height: 1.6;
  }

  .app-item strong {
    color: #1A3A5C;
    font-weight: 600;
  }

  .warn-box {
    background: #FEF3C7;
    border-left: 3px solid #C4A32A;
    padding: 0.9rem 1.2rem;
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
    color: #78350F;
  }

  table {
    width: 100%;
    border-collapse: collapse;
  }

  th {
    background: #1A3A5C;
    color: #fff;
    padding: 0.6rem 0.9rem;
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem;
    font-weight: 500;
    text-align: left;
  }

  td {
    padding: 0.55rem 0.9rem;
    font-family: 'Inter', sans-serif;
    font-size: 0.82rem;
    border-bottom: 1px solid #E5E7EB;
  }

  tr:nth-child(even) td {
    background: #F7F8FA;
  }

  .footer {
    margin-top: 3rem;
    padding: 1.5rem 0;
    border-top: 1px solid #DEE2E9;
    font-family: 'Inter', sans-serif;
    font-size: 0.75rem;
    color: #9CA3AF;
    text-align: center;
  }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Funções matemáticas
# ─────────────────────────────────────────────────────────────────────────────

def exact_number(value):
    """
    Converte valores numéricos vindos do Streamlit para frações exatas.
    Isso evita erros de simplificação causados por floats como 1.0, -2.0 etc.
    """
    try:
        return sp.Rational(str(value))
    except Exception:
        return sp.nsimplify(value)


def parse_f(expr_str: str):
    """Converte string da força externa f(t) em expressão SymPy."""
    t = sp.Symbol("t")

    local_dict = {
        "t": t,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "exp": sp.exp,
        "log": sp.log,
        "sqrt": sp.sqrt,
        "pi": sp.pi,
        "e": sp.E,
        "E": sp.E,
        "Heaviside": sp.Heaviside,
    }

    try:
        expr = sp.sympify(expr_str, locals=local_dict)
        return sp.nsimplify(expr)
    except Exception:
        return None


def montar_laplace(a, b, c, f_expr, y0, v0):
    """
    Monta a equação algébrica em Y(s) após aplicar a Transformada de Laplace
    à EDO: a*y'' + b*y' + c*y = f(t), com y(0)=y0 e y'(0)=v0.
    """
    s, t = sp.symbols("s t")
    Y = sp.Symbol("Y")

    a = exact_number(a)
    b = exact_number(b)
    c = exact_number(c)
    y0 = exact_number(y0)
    v0 = exact_number(v0)
    f_expr = sp.nsimplify(f_expr)

    Ly2 = s**2 * Y - s * y0 - v0
    Ly1 = s * Y - y0
    Ly0 = Y

    try:
        F_s = sp.laplace_transform(f_expr, t, s, noconds=True)
        F_s = sp.nsimplify(F_s)
    except Exception:
        return None, s, t, None, None

    equacao = sp.Eq(a * Ly2 + b * Ly1 + c * Ly0, F_s)

    try:
        sol = sp.solve(equacao, Y)
        if not sol:
            return None, s, t, F_s, equacao

        Y_s = sp.nsimplify(sol[0])
        Y_s = sp.factor(sp.cancel(sp.together(Y_s)))
    except Exception:
        return None, s, t, F_s, equacao

    return Y_s, s, t, F_s, equacao


def decompor_fracoes(Y_s, s):
    """Aplica decomposição em frações parciais a Y(s)."""
    try:
        Y_s = sp.nsimplify(Y_s)
        return sp.apart(Y_s, s)
    except Exception:
        return Y_s


def limpar_heaviside(expr, t):
    """
    A transformada inversa de Laplace considera t >= 0 e costuma devolver Heaviside(t).
    Para a apresentação didática, removemos Heaviside(t) da forma exibida.
    """
    try:
        expr = expr.replace(sp.Heaviside(t), 1)
        expr = expr.replace(sp.Heaviside(t, 1/2), 1)
        expr = expr.replace(sp.Heaviside(t, sp.Rational(1, 2)), 1)
        return sp.simplify(expr)
    except Exception:
        return expr


def transformada_inversa(Y_s, s, t):
    """Calcula a transformada inversa de Laplace de Y(s)."""
    try:
        Y_s = sp.nsimplify(Y_s)
        Y_s = sp.factor(sp.cancel(sp.together(Y_s)))

        y_t_raw = sp.inverse_laplace_transform(Y_s, s, t)
        y_t_raw = sp.simplify(sp.nsimplify(y_t_raw))

        y_t_display = limpar_heaviside(y_t_raw, t)
        y_t_display = sp.factor(sp.simplify(y_t_display))

        return y_t_raw, y_t_display
    except Exception:
        return None, None


def solucao_numerica(a, b, c, f_func, y0, v0, t_span, n_pontos=600):
    """
    Resolve a EDO numericamente usando scipy.integrate.solve_ivp.
    Reescreve a EDO de segunda ordem como sistema de primeira ordem.
    """
    a = float(a)
    b = float(b)
    c = float(c)
    y0 = float(y0)
    v0 = float(v0)

    def sistema(t, u):
        u1, u2 = u
        try:
            ft = f_func(t) if callable(f_func) else float(f_func)
        except Exception:
            ft = 0.0

        du1 = u2
        du2 = (ft - b * u2 - c * u1) / a
        return [du1, du2]

    t_eval = np.linspace(t_span[0], t_span[1], n_pontos)

    sol = solve_ivp(
        sistema,
        t_span,
        [y0, v0],
        t_eval=t_eval,
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
        dense_output=False,
    )

    if sol.success:
        return sol.t, sol.y[0]

    return None, None


def avaliar_simbolico(y_expr, t_sym, t_array):
    """Avalia a expressão simbólica y(t) em um array numérico."""
    try:
        f_num = sp.lambdify(t_sym, y_expr, modules=["numpy"])
        y_val = f_num(t_array)

        if np.isscalar(y_val):
            y_val = np.full_like(t_array, float(y_val), dtype=float)

        return np.array(y_val, dtype=float)
    except Exception:
        return None


def interpretar_solucao(y_expr, t):
    """Gera uma descrição técnica curta sobre o comportamento da solução."""
    if y_expr is None:
        return "Não foi possível interpretar a solução simbolicamente."

    expr_str = str(y_expr)
    exps = []

    for atom in sp.preorder_traversal(y_expr):
        if getattr(atom, "func", None) == sp.exp:
            arg = atom.args[0]
            coef = sp.simplify(arg.coeff(t))
            if coef != 0:
                try:
                    exps.append(float(coef.evalf()))
                except Exception:
                    pass

    tem_sin_cos = "sin" in expr_str or "cos" in expr_str
    tem_exp_neg = any(e < 0 for e in exps)
    tem_exp_pos = any(e > 0 for e in exps)

    partes = []

    if tem_exp_pos and not tem_exp_neg:
        partes.append(
            "A solução contém termo exponencial crescente. Para valores maiores de t, esse termo domina o comportamento."
        )
    elif tem_exp_neg and not tem_exp_pos:
        if tem_sin_cos:
            partes.append(
                "A solução possui oscilação amortecida: há termos trigonométricos combinados com decaimento exponencial."
            )
        else:
            partes.append(
                "A solução apresenta decaimento exponencial. Os termos tendem a diminuir conforme t aumenta."
            )
    elif tem_exp_pos and tem_exp_neg:
        partes.append(
            "A solução combina termos que crescem e termos que decaem. No longo prazo, o termo exponencial positivo tende a dominar."
        )
    elif tem_sin_cos:
        partes.append(
            "A solução apresenta comportamento oscilatório, com termos de seno ou cosseno."
        )
    else:
        partes.append(
            "O comportamento depende da combinação algébrica dos termos obtidos na solução."
        )

    if exps:
        taxas = ", ".join([f"{e:+.3f}" for e in sorted(set(exps))])
        partes.append(
            f"Taxas exponenciais identificadas: {taxas}. Valores positivos indicam crescimento; negativos indicam decaimento."
        )

    return " ".join(partes)


def latex_num(x):
    """Converte número de entrada para LaTeX exato."""
    return sp.latex(exact_number(x))


def montar_texto_edo(a, b, c, f_expr):
    """Monta uma expressão simbólica da EDO para exibição."""
    t = sp.Symbol("t")
    y = sp.Function("y")

    a = exact_number(a)
    b = exact_number(b)
    c = exact_number(c)

    lhs = a * sp.Derivative(y(t), (t, 2)) + b * sp.Derivative(y(t), t) + c * y(t)
    return sp.Eq(lhs, f_expr)


# ─────────────────────────────────────────────────────────────────────────────
# Cabeçalho institucional
# ─────────────────────────────────────────────────────────────────────────────

logo_path = "assets/logo_unimontes.png"

if os.path.exists(logo_path):
    with open(logo_path, "rb") as img_file:
        logo_b64 = base64.b64encode(img_file.read()).decode()

    logo_html = (
        f'<img src="data:image/png;base64,{logo_b64}" '
        f'style="width:72px;height:72px;object-fit:contain;" alt="UNIMONTES">'
    )
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
    Bruno Gomes · Júlio César · Leonardo<br>
    Marcus · Samuel Antunes França
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Introdução
# ─────────────────────────────────────────────────────────────────────────────

col_intro, col_prop = st.columns([3, 2], gap="large")

with col_intro:
    st.markdown('<div class="gold-accent"></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-label">O método</p>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Ideia do cálculo</h2>', unsafe_allow_html=True)

    st.markdown(r"""
A Transformada de Laplace permite resolver certas EDOs trocando o problema de lugar.

Em vez de trabalhar diretamente com derivadas no domínio do tempo \(t\), aplicamos a transformada e obtemos uma equação algébrica em \(s\). Depois de isolar \(Y(s)\), usamos a transformada inversa para recuperar \(y(t)\).

Para uma EDO linear de segunda ordem,

\[
ay'' + by' + cy = f(t),
\]

o método usa as condições iniciais desde o começo. Por isso, \(y(0)\) e \(y'(0)\) entram diretamente na equação transformada.
""")

with col_prop:
    st.markdown('<div class="gold-accent"></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-label">Propriedades fundamentais</p>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Transformadas usadas</h2>', unsafe_allow_html=True)

    st.markdown(r"""
| Função | Transformada |
|---|---|
| $y(t)$ | $Y(s)$ |
| $y'(t)$ | $sY(s) - y(0)$ |
| $y''(t)$ | $s^2Y(s) - sy(0) - y'(0)$ |
| $e^{at}$ | $\frac{1}{s-a}$ |
| $\sin(at)$ | $\frac{a}{s^2+a^2}$ |
| $\cos(at)$ | $\frac{s}{s^2+a^2}$ |
| $t^n$ | $\frac{n!}{s^{n+1}}$ |
""")

st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Painel de entrada
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="gold-accent"></div>', unsafe_allow_html=True)
st.markdown('<p class="section-label">Calculadora</p>', unsafe_allow_html=True)
st.markdown('<h2 class="section-title">Defina a EDO e as condições iniciais</h2>', unsafe_allow_html=True)

if "exemplo_carregado" not in st.session_state:
    st.session_state.exemplo_carregado = False

col_ex, _ = st.columns([1, 4])

with col_ex:
    if st.button("Carregar exemplo do livro"):
        st.session_state.exemplo_carregado = True
        st.session_state.a_val = 1.0
        st.session_state.b_val = -1.0
        st.session_state.c_val = -2.0
        st.session_state.f_str = "0"
        st.session_state.y0_val = 1.0
        st.session_state.v0_val = 0.0
        st.session_state.t_max = 5.0
        st.session_state.resultado_disponivel = False

if st.session_state.exemplo_carregado:
    st.markdown("""
<div class="warn-box">
Exemplo carregado: <strong>y'' − y' − 2y = 0</strong>, com
<strong>y(0) = 1</strong> e <strong>y'(0) = 0</strong>.
A solução esperada é <strong>y(t) = (1/3)e<sup>2t</sup> + (2/3)e<sup>−t</sup></strong>.
</div>
""", unsafe_allow_html=True)

st.markdown("")

with st.container():
    st.markdown('<div class="input-panel">', unsafe_allow_html=True)
    st.markdown('<h3>EDO: &nbsp; a · y″ + b · y′ + c · y = f(t)</h3>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        a_val = st.number_input(
            "Coeficiente a (y″)",
            value=st.session_state.get("a_val", 1.0),
            key="a_input",
            step=0.5,
        )

    with col2:
        b_val = st.number_input(
            "Coeficiente b (y′)",
            value=st.session_state.get("b_val", -1.0),
            key="b_input",
            step=0.5,
        )

    with col3:
        c_val = st.number_input(
            "Coeficiente c (y)",
            value=st.session_state.get("c_val", -2.0),
            key="c_input",
            step=0.5,
        )

    col4, col5, col6, col7 = st.columns([2, 1, 1, 1])

    with col4:
        f_str = st.text_input(
            "Força externa f(t)",
            value=st.session_state.get("f_str", "0"),
            placeholder="Ex: sin(2*t), exp(t), t, 0",
            key="f_input",
        )

    with col5:
        y0_val = st.number_input(
            "y(0)",
            value=st.session_state.get("y0_val", 1.0),
            key="y0_input",
            step=0.5,
        )

    with col6:
        v0_val = st.number_input(
            "y′(0)",
            value=st.session_state.get("v0_val", 0.0),
            key="v0_input",
            step=0.5,
        )

    with col7:
        t_max = st.number_input(
            "t máximo",
            value=st.session_state.get("t_max", 5.0),
            min_value=0.5,
            max_value=50.0,
            step=0.5,
            key="tmax_input",
        )

    st.markdown('</div>', unsafe_allow_html=True)

calcular = st.button("Calcular solução")


# ─────────────────────────────────────────────────────────────────────────────
# Cálculo simbólico e numérico
# ─────────────────────────────────────────────────────────────────────────────

if calcular:
    if a_val == 0:
        st.error("O coeficiente a não pode ser zero. Nesse formato, a equação precisa ser de segunda ordem.")
        st.stop()

    f_expr = parse_f(f_str)

    if f_expr is None:
        st.error(
            f"Não foi possível interpretar f(t) = '{f_str}'. "
            "Use notação Python: sin(t), cos(2*t), exp(-t), t**2, etc."
        )
        st.stop()

    with st.spinner("Calculando..."):
        Y_s, s_sym, t_sym, F_s, equacao_laplace = montar_laplace(
            a_val, b_val, c_val, f_expr, y0_val, v0_val
        )

    if Y_s is None:
        st.error(
            "Não foi possível obter Y(s). Verifique a função f(t) ou os coeficientes inseridos."
        )
        st.stop()

    Y_parcial = decompor_fracoes(Y_s, s_sym)
    y_t_raw, y_t_display = transformada_inversa(Y_s, s_sym, t_sym)

    if y_t_display is None:
        st.error(
            "A transformada inversa de Y(s) não pôde ser calculada simbolicamente."
        )
        st.stop()

    st.session_state.resultado_disponivel = True
    st.session_state.resultado = {
        "a": a_val,
        "b": b_val,
        "c": c_val,
        "f_expr": f_expr,
        "f_str": f_str,
        "y0": y0_val,
        "v0": v0_val,
        "t_max": t_max,
        "Y_s": Y_s,
        "Y_parcial": Y_parcial,
        "y_t_raw": y_t_raw,
        "y_t_display": y_t_display,
        "s_sym": s_sym,
        "t_sym": t_sym,
        "F_s": F_s,
        "equacao_laplace": equacao_laplace,
    }


if st.session_state.get("resultado_disponivel", False):
    res = st.session_state.resultado

    a_val = res["a"]
    b_val = res["b"]
    c_val = res["c"]
    f_expr = res["f_expr"]
    f_str = res["f_str"]
    y0_val = res["y0"]
    v0_val = res["v0"]
    t_max = res["t_max"]
    Y_s = res["Y_s"]
    Y_parcial = res["Y_parcial"]
    y_t_raw = res["y_t_raw"]
    y_t_display = res["y_t_display"]
    s_sym = res["s_sym"]
    t_sym = res["t_sym"]
    F_s = res["F_s"]
    equacao_laplace = res["equacao_laplace"]

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Desenvolvimento simbólico
    st.markdown('<div class="gold-accent"></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-label">Desenvolvimento simbólico</p>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Resolução passo a passo</h2>', unsafe_allow_html=True)

    col_passos, col_espaco = st.columns([3, 1])

    with col_passos:
        st.markdown('<div class="result-step">', unsafe_allow_html=True)
        st.markdown('<div class="step-num">Passo 1 — EDO no domínio do tempo</div>', unsafe_allow_html=True)

        edo_display = montar_texto_edo(a_val, b_val, c_val, f_expr)
        st.latex(sp.latex(edo_display))
        st.caption(f"Condições iniciais: y(0) = {y0_val}, y'(0) = {v0_val}")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="result-step">', unsafe_allow_html=True)
        st.markdown(
            '<div class="step-num">Passo 2 — Transformada de Laplace</div>',
            unsafe_allow_html=True,
        )

        st.latex(
            r"\mathcal{L}\{y''\}=s^2Y(s)-s\,y(0)-y'(0), \qquad "
            r"\mathcal{L}\{y'\}=sY(s)-y(0)"
        )

        st.markdown("Equação algébrica obtida:")
        if equacao_laplace is not None:
            st.latex(sp.latex(equacao_laplace))
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="result-step">', unsafe_allow_html=True)
        st.markdown('<div class="step-num">Passo 3 — Y(s) resolvido</div>', unsafe_allow_html=True)
        st.latex(r"Y(s) = " + sp.latex(Y_s))
        st.markdown('</div>', unsafe_allow_html=True)

        if sp.simplify(Y_parcial - Y_s) == 0 and Y_parcial != Y_s:
            st.markdown('<div class="result-step">', unsafe_allow_html=True)
            st.markdown(
                '<div class="step-num">Passo 4 — Decomposição em frações parciais</div>',
                unsafe_allow_html=True,
            )
            st.latex(r"Y(s) = " + sp.latex(Y_parcial))
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(
            '<div class="result-step" style="border-left: 3px solid #C4A32A;">',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="step-num" style="color:#1A3A5C;">Resultado — y(t)</div>',
            unsafe_allow_html=True,
        )
        st.latex(r"y(t) = " + sp.latex(y_t_display))
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Visualização
    st.markdown('<div class="gold-accent"></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-label">Visualização</p>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Gráfico de y(t)</h2>', unsafe_allow_html=True)

    t_array = np.linspace(0, float(t_max), 800)
    y_simb = avaliar_simbolico(y_t_display, t_sym, t_array)

    try:
        if f_expr == 0:
            f_num_func = lambda x: 0.0
        else:
            f_num_func = sp.lambdify(t_sym, f_expr, modules=["numpy"])

        t_num, y_num = solucao_numerica(
            a_val,
            b_val,
            c_val,
            f_num_func,
            y0_val,
            v0_val,
            (0, float(t_max)),
        )
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
        y_simb_clean = np.where(np.isfinite(y_simb), y_simb, np.nan)

        fig.add_trace(
            go.Scatter(
                x=t_array,
                y=y_simb_clean,
                mode="lines",
                name="Simbólica (SymPy)",
                line=dict(color="#1A3A5C", width=2.5),
            )
        )

    if modo_grafico in ("Solução numérica (RK45)", "Comparação") and t_num is not None:
        fig.add_trace(
            go.Scatter(
                x=t_num,
                y=y_num,
                mode="lines",
                name="Numérica (RK45)",
                line=dict(color="#C4A32A", width=2, dash="dot"),
            )
        )

    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", size=12, color="#374151"),
        margin=dict(l=50, r=30, t=30, b=50),
        xaxis=dict(
            title="t",
            showgrid=True,
            gridcolor="#E5E7EB",
            zeroline=True,
            zerolinecolor="#9CA3AF",
            linecolor="#D1D5DB",
        ),
        yaxis=dict(
            title="y(t)",
            showgrid=True,
            gridcolor="#E5E7EB",
            zeroline=True,
            zerolinecolor="#9CA3AF",
            linecolor="#D1D5DB",
        ),
        legend=dict(
            orientation="h",
            y=1.05,
            x=0,
            bgcolor="rgba(0,0,0,0)",
        ),
        height=420,
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Interpretação
    st.markdown('<div class="gold-accent"></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-label">Análise</p>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Interpretação da solução</h2>', unsafe_allow_html=True)

    interpretacao = interpretar_solucao(y_t_display, t_sym)

    st.markdown(f"""
<div class="interp-block">
  <h4>Comportamento qualitativo</h4>
  {interpretacao}
  <br><br>
  <strong>Expressão:</strong>&nbsp; y(t) = {sp.latex(y_t_display)}
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Aplicações
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown('<div class="gold-accent"></div>', unsafe_allow_html=True)
st.markdown('<p class="section-label">Contexto</p>', unsafe_allow_html=True)
st.markdown('<h2 class="section-title">Onde a Transformada de Laplace aparece</h2>', unsafe_allow_html=True)

col_app1, col_app2 = st.columns(2)

with col_app1:
    st.markdown("""
<div class="app-item">
  <strong>Circuitos elétricos.</strong><br>
  Em circuitos RC, RL e RLC, a transformada ajuda a representar correntes e tensões em termos algébricos, principalmente quando há fontes variáveis no tempo.
</div>
<div class="app-item">
  <strong>Sistemas massa-mola.</strong><br>
  Modelos com amortecimento e força externa podem ser descritos por EDOs de segunda ordem, semelhantes às usadas nesta calculadora.
</div>
<div class="app-item">
  <strong>Condições iniciais.</strong><br>
  O método incorpora diretamente valores como posição inicial e velocidade inicial, sem precisar resolver primeiro uma solução geral.
</div>
""", unsafe_allow_html=True)

with col_app2:
    st.markdown("""
<div class="app-item">
  <strong>Engenharia de controle.</strong><br>
  A análise no domínio de s permite estudar resposta temporal, estabilidade e comportamento de sistemas dinâmicos.
</div>
<div class="app-item">
  <strong>Sinais e sistemas.</strong><br>
  Funções descontínuas, entradas por degrau e impulsos são tratadas de maneira mais organizada no domínio transformado.
</div>
<div class="app-item">
  <strong>Modelagem computacional.</strong><br>
  O mesmo procedimento usado no cálculo manual pode ser implementado em Python, permitindo testar exemplos e visualizar soluções.
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Rodapé
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="footer">
  UNIMONTES — Universidade Estadual de Montes Claros &nbsp;·&nbsp;
  Disciplina de Equações Diferenciais Ordinárias &nbsp;·&nbsp; Prof. Fernando Félix<br>
  Bruno Gomes · Júlio César · Leonardo · Marcus · Samuel Antunes França<br>
  Construído com Streamlit, SymPy, NumPy, Plotly e SciPy
</div>
""", unsafe_allow_html=True)
```
