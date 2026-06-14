"""
Transformada de Laplace — Ferramenta interativa para resolução de EDOs lineares
UNIMONTES | Disciplina de Matemática Aplicada | Prof. Fernando Félix
Alunos: Bruno Gomes, Júlio César, Leonardo, Samuel, Marcus
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
    """
    s, t = sp.symbols('s t')

    a = sp.Rational(str(a))
    b = sp.Rational(str(b))
    c = sp.Rational(str(c))
    y0 = sp.Rational(str(y0))
    v0 = sp.Rational(str(v0))
    f_expr = sp.nsimplify(f_expr)

    Y = sp.Function('Y')(s)

    Ly2 = s**2 * Y - s * y0 - v0
    Ly1 = s * Y - y0
    Ly0 = Y

    try:
        F_s = sp.laplace_transform(f_expr, t, s, noconds=True)
        F_s = sp.nsimplify(F_s)
    except Exception:
        F_s = None

    if F_s is None:
        return None, s, t, None

    equacao = sp.Eq(a * Ly2 + b * Ly1 + c * Ly0, F_s)

    try:
        sol = sp.solve(equacao, Y)
        if not sol:
            return None, s, t, F_s

        Y_s = sp.nsimplify(sol[0])
        Y_s = sp.factor(sp.cancel(sp.together(Y_s)))

    except Exception:
        return None, s, t, F_s

    return Y_s, s, t, F_s


def decompor_fracoes(Y_s, s):
    """Aplica decomposição em frações parciais a Y(s), usando forma exata."""
    try:
        Y_s = sp.nsimplify(Y_s)
        Y_s = sp.factor(sp.cancel(sp.together(Y_s)))
        return sp.apart(Y_s, s)
    except Exception:
        return Y_s


def transformada_inversa(Y_s, s, t):
    """Calcula a transformada inversa de Laplace de Y(s) em forma expandida."""
    try:
        Y_s = sp.nsimplify(Y_s)
        Y_s = sp.factor(sp.cancel(sp.together(Y_s)))

        y_t = sp.inverse_laplace_transform(Y_s, s, t, noconds=True)

        # Remove Heaviside(t), pois a Transformada de Laplace trabalha com t >= 0.
        y_t = y_t.replace(sp.Heaviside(t), 1)
        y_t = y_t.replace(sp.Heaviside(t, sp.Rational(1, 2)), 1)

        # Força a solução a aparecer como soma de termos, igual ao livro.
        y_t = sp.expand(y_t)
        y_t = sp.powsimp(y_t, force=True)
        y_t = sp.expand(y_t)
        y_t = sp.nsimplify(y_t)

        return y_t

    except Exception:
        return None
def laplace_derivada(k, Y, s, iniciais):
    """
    Fórmula geral:
    L{y^(k)} = s^k Y(s) - s^(k-1)y(0) - s^(k-2)y'(0) - ... - y^(k-1)(0)
    """
    if k == 0:
        return Y

    expr = s**k * Y

    for j in range(k):
        expr -= s**(k - 1 - j) * iniciais[j]

    return expr


def montar_laplace_geral(coefs, f_expr, iniciais):
    """
    Resolve por Laplace uma EDO linear de ordem n:

    a_n y^(n) + a_(n-1)y^(n-1) + ... + a_1 y' + a_0 y = f(t)

    coefs = [a_n, a_(n-1), ..., a_0]
    iniciais = [y(0), y'(0), ..., y^(n-1)(0)]
    """
    s, t = sp.symbols('s t')
    Y = sp.Function('Y')(s)

    n = len(coefs) - 1

    coefs = [sp.Rational(str(v)) for v in coefs]
    iniciais = [sp.Rational(str(v)) for v in iniciais]
    f_expr = sp.nsimplify(f_expr)

    try:
        F_s = sp.laplace_transform(f_expr, t, s, noconds=True)
        F_s = sp.nsimplify(F_s)

        if F_s.has(sp.LaplaceTransform):
            return None, s, t, None, None

    except Exception:
        return None, s, t, None, None

    lhs = 0

    for k in range(n, -1, -1):
        coef = coefs[n - k]
        lhs += coef * laplace_derivada(k, Y, s, iniciais)

    equacao = sp.Eq(lhs, F_s)

    try:
        sol = sp.solve(equacao, Y)

        if not sol:
            return None, s, t, F_s, equacao

        Y_s = sp.nsimplify(sol[0])
        Y_s = sp.factor(sp.cancel(sp.together(Y_s)))

    except Exception:
        return None, s, t, F_s, equacao

    return Y_s, s, t, F_s, equacao


def solucao_numerica_geral(coefs, f_func, iniciais, t_span, n_pontos=600):
    """
    Resolve numericamente uma EDO linear de ordem n usando solve_ivp.
    """
    n = len(coefs) - 1

    coefs = [float(v) for v in coefs]
    iniciais = [float(v) for v in iniciais]

    a_n = coefs[0]

    if a_n == 0:
        return None, None

    def sistema(t, u):
        try:
            ft = f_func(t) if callable(f_func) else float(f_func)
        except Exception:
            ft = 0.0

        derivadas = list(u[1:])

        soma = 0.0

        for k in range(n):
            coef_a_k = coefs[n - k]
            soma += coef_a_k * u[k]

        ultima = (ft - soma) / a_n
        derivadas.append(ultima)

        return derivadas

    t_eval = np.linspace(t_span[0], t_span[1], n_pontos)

    sol = solve_ivp(
        sistema,
        t_span,
        iniciais,
        t_eval=t_eval,
        method='RK45',
        rtol=1e-8,
        atol=1e-10,
        dense_output=False,
    )

    if sol.success:
        return sol.t, sol.y[0]

    return None, None


def rotulo_derivada_inicial(k):
    if k == 0:
        return "y(0)"
    if k == 1:
        return "y′(0)"
    if k == 2:
        return "y″(0)"
    if k == 3:
        return "y‴(0)"
    return f"y^({k})(0)"


def rotulo_coeficiente(ordem_derivada):
    if ordem_derivada == 0:
        return "Coeficiente de y"
    if ordem_derivada == 1:
        return "Coeficiente de y′"
    if ordem_derivada == 2:
        return "Coeficiente de y″"
    if ordem_derivada == 3:
        return "Coeficiente de y‴"
    if ordem_derivada == 4:
        return "Coeficiente de y⁽⁴⁾"
    return f"Coeficiente de y^({ordem_derivada})"


def montar_edo_latex(coefs, f_expr):
    t = sp.Symbol('t')
    y = sp.Function('y')
    n = len(coefs) - 1

    lhs = 0

    for k in range(n, -1, -1):
        coef = sp.Rational(str(coefs[n - k]))

        if k == 0:
            termo = y(t)
        else:
            termo = sp.Derivative(y(t), (t, k))

        lhs += coef * termo

    return sp.Eq(lhs, f_expr)

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
    Leonardo · Marcus · Samuel
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
    st.markdown('<p class="section-label">Propriedades fundamentais </p>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Tabela do autor, tô usando aqui de exemplo pra vcs</h2>',
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


def carregar_exemplo_ordem2():
    st.session_state.ordem_input = 2
    st.session_state.coef_2_0 = 1.0
    st.session_state.coef_2_1 = -1.0
    st.session_state.coef_2_2 = -2.0
    st.session_state.ini_2_0 = 1.0
    st.session_state.ini_2_1 = 0.0
    st.session_state.f_input_geral = "0"
    st.session_state.tmax_geral = 5.0
    st.session_state.resultado_disponivel = False


def carregar_exemplo_ordem4():
    st.session_state.ordem_input = 4
    st.session_state.coef_4_0 = 1.0
    st.session_state.coef_4_1 = 0.0
    st.session_state.coef_4_2 = 0.0
    st.session_state.coef_4_3 = 0.0
    st.session_state.coef_4_4 = -1.0
    st.session_state.ini_4_0 = 0.0
    st.session_state.ini_4_1 = 1.0
    st.session_state.ini_4_2 = 0.0
    st.session_state.ini_4_3 = 0.0
    st.session_state.f_input_geral = "0"
    st.session_state.tmax_geral = 8.0
    st.session_state.resultado_disponivel = False


if "ordem_input" not in st.session_state:
    st.session_state.ordem_input = 2

col_ex1, col_ex2, _ = st.columns([1.2, 1.2, 3])

with col_ex1:
    st.button("Exemplo ordem 2", on_click=carregar_exemplo_ordem2)

with col_ex2:
    st.button("Exemplo ordem 4", on_click=carregar_exemplo_ordem4)


ordem = st.selectbox(
    "Ordem da EDO",
    [2, 3, 4],
    key="ordem_input",
    help="Escolha a maior derivada presente na equação."
)

defaults_coefs = {
    2: [1.0, -1.0, -2.0],
    3: [1.0, 0.0, 0.0, -1.0],
    4: [1.0, 0.0, 0.0, 0.0, -1.0],
}

defaults_iniciais = {
    2: [1.0, 0.0],
    3: [0.0, 1.0, 0.0],
    4: [0.0, 1.0, 0.0, 0.0],
}

with st.container():
    st.markdown('<div class="input-panel">', unsafe_allow_html=True)
    st.markdown(
        f'<h3>EDO linear de ordem {ordem}: &nbsp; aₙ · y⁽ⁿ⁾ + ... + a₁ · y′ + a₀ · y = f(t)</h3>',
        unsafe_allow_html=True
    )

    st.markdown("**Coeficientes da EDO**")

    coefs = []
    cols_coef = st.columns(min(ordem + 1, 5))

    for i in range(ordem + 1):
        ordem_derivada = ordem - i
        key = f"coef_{ordem}_{i}"

        if key not in st.session_state:
            st.session_state[key] = defaults_coefs[ordem][i]

        with cols_coef[i % len(cols_coef)]:
            coefs.append(
                st.number_input(
                    rotulo_coeficiente(ordem_derivada),
                    key=key,
                    step=0.5
                )
            )

    st.markdown("**Condições iniciais**")

    iniciais = []
    cols_ini = st.columns(min(ordem, 4))

    for j in range(ordem):
        key = f"ini_{ordem}_{j}"

        if key not in st.session_state:
            st.session_state[key] = defaults_iniciais[ordem][j]

        with cols_ini[j % len(cols_ini)]:
            iniciais.append(
                st.number_input(
                    rotulo_derivada_inicial(j),
                    key=key,
                    step=0.5
                )
            )

    col_f, col_tmax = st.columns([3, 1])

    with col_f:
        if "f_input_geral" not in st.session_state:
            st.session_state.f_input_geral = "0"

        f_str = st.text_input(
            "Força externa f(t)",
            placeholder="Ex: sin(2*t), exp(t), t, 0",
            key="f_input_geral",
        )

    with col_tmax:
        if "tmax_geral" not in st.session_state:
            st.session_state.tmax_geral = 5.0

        t_max = st.number_input(
            "t máximo",
            min_value=0.5,
            max_value=50.0,
            step=0.5,
            key="tmax_geral"
        )

    st.markdown('</div>', unsafe_allow_html=True)

calcular = st.button("Calcular solução")

# ─── Cálculo ─────────────────────────────────────────────────────────────────

if calcular:
    # Validação
    if coefs[0] == 0:
        st.error("O coeficiente do termo de maior ordem não pode ser zero.")
        st.stop()

    f_expr = parse_f(f_str)
    if f_expr is None:
        st.error("Erro na expressão f(t).")
        st.stop()

    with st.spinner("Calculando..."):
        Y_s, s_sym, t_sym, F_s, equacao_laplace = montar_laplace_geral(coefs, f_expr, iniciais)
        
        if Y_s is None:
            st.error("Não foi possível obter Y(s).")
            st.stop()

        Y_parcial = decompor_fracoes(Y_s, s_sym)
        y_t = transformada_inversa(Y_s, s_sym, t_sym)

        if y_t is None:
            st.error("Erro na transformada inversa.")
            st.stop()

        st.session_state.resultado_disponivel = True
        st.session_state.resultado = {
            "ordem": ordem, "coefs": coefs, "iniciais": iniciais,
            "f_expr": f_expr, "f_str": f_str, "t_max": t_max,
            "Y_s": Y_s, "Y_parcial": Y_parcial, "y_t": y_t,
            "s_sym": s_sym, "t_sym": t_sym, "F_s": F_s,
            "equacao_laplace": equacao_laplace
        }

# Exibição dos resultados
if st.session_state.get("resultado_disponivel", False):
    # ── Recuperar resultado armazenado ──
    res = st.session_state.resultado
    
    ordem = res["ordem"]
    coefs = res["coefs"]
    iniciais = res["iniciais"]
    f_expr = res["f_expr"]
    f_str = res["f_str"]
    t_max = res["t_max"]
    equacao_laplace = res["equacao_laplace"]
    Y_s = res["Y_s"]
    Y_parcial = res["Y_parcial"]
    y_t = res["y_t"]
    s_sym = res["s_sym"]
    t_sym = res["t_sym"]
    F_s = res["F_s"]

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ─── Cálculo simbólico passo a passo ─────────────────────────────────────
    # ... (o restante do seu código de exibição continua aqui)
    # ── Recuperar resultado armazenado ──
    res = st.session_state.resultado
   ordem = res["ordem"]
coefs = res["coefs"]
iniciais = res["iniciais"]
f_expr = res["f_expr"]
f_str = res["f_str"]
t_max = res["t_max"]
equacao_laplace = res["equacao_laplace"]
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

        edo_display = montar_edo_latex(coefs, f_expr)
        st.latex(sp.latex(edo_display))

        texto_iniciais = ", ".join(
            [f"{rotulo_derivada_inicial(i)} = {iniciais[i]}" for i in range(ordem)]
        )

        st.caption(f"Condições iniciais: {texto_iniciais}")
        st.markdown('</div>', unsafe_allow_html=True)

                # Passo 2: Transformada
        st.markdown('<div class="result-step">', unsafe_allow_html=True)
        st.markdown(
            '<div class="step-num">Passo 2 — Transformada de Laplace e substituição das condições iniciais</div>',
            unsafe_allow_html=True
        )

        st.latex(
            r"\mathcal{L}\{y^{(n)}\}="
            r"s^nY(s)-s^{n-1}y(0)-s^{n-2}y'(0)-\cdots-y^{(n-1)}(0)"
        )

        st.markdown("Equação algébrica obtida:")

        if equacao_laplace is not None:
            st.latex(sp.latex(equacao_laplace))

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
       t_num, y_num = solucao_numerica_geral(
    coefs,
    f_num_func,
    iniciais,
    (0, t_max)
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


# ─── Aplicações ───────────────────────────────────────────────────────────────

st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown('<div class="gold-accent"></div>', unsafe_allow_html=True)
st.markdown('<p class="section-label">Contexto</p>', unsafe_allow_html=True)
st.markdown('<h2 class="section-title">Aqui daria pra colocar um resumo de onde isso é usado ou algo assim </h2>',
            unsafe_allow_html=True)

col_app1, col_app2 = st.columns(2)

with col_app1:
    st.markdown("""
<div class="app-item">
  <strong>.</strong><br>
  .
</div>
<div class="app-item">
  <strong>.</strong><br>
  .
</div>
<div class="app-item">
  <strong>.</strong><br>
  .
</div>
    """, unsafe_allow_html=True)

with col_app2:
    st.markdown("""
<div class="app-item">
  <strong>.</strong><br>
  .
</div>
<div class="app-item">
  <strong>.</strong><br>
  .
</div>
<div class="app-item">
  <strong>.</strong><br>
  .
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
