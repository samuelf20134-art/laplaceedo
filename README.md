# Transformada de Laplace — Ferramenta Interativa para EDOs

**Disciplina de Equações Diferenciais Ordinárias | UNIMONTES**  
Prof. Fernando Félix  
Alunos: Bruno Gomes · Júlio César · Leonardo · Marcus

---

## O que é este projeto

Aplicativo Streamlit para uso em seminário acadêmico sobre a Transformada de Laplace
aplicada à resolução de EDOs lineares de segunda ordem com condições iniciais.

A ferramenta:
- Recebe os coeficientes `a`, `b`, `c` da EDO `a·y'' + b·y' + c·y = f(t)`;
- Recebe a força externa `f(t)` e as condições iniciais `y(0)`, `y'(0)`;
- Monta automaticamente a equação algébrica no domínio de *s*;
- Resolve `Y(s)` simbolicamente via SymPy;
- Decompõe em frações parciais quando possível;
- Calcula a transformada inversa `y(t)`;
- Plota a solução com Plotly (simbólica, numérica RK45, ou comparação);
- Interpreta o comportamento qualitativo da solução.

---

## Estrutura do projeto

```
laplace_app/
├── app.py                  # Aplicativo principal
├── requirements.txt        # Dependências Python
├── README.md               # Este arquivo
├── .gitignore
└── assets/
    └── logo_unimontes.png  # Logo institucional (adicionar manualmente)
```

---

## Como rodar localmente

### 1. Clonar o repositório

```bash
git clone https://github.com/seu-usuario/laplace-unimontes.git
cd laplace-unimontes
```

### 2. Criar e ativar ambiente virtual

```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

### 4. Adicionar a logo da UNIMONTES (opcional)

Salve o arquivo da logo em:

```
assets/logo_unimontes.png
```

O app detecta automaticamente o arquivo e exibe no cabeçalho.
Se o arquivo não existir, um espaço reservado é exibido no lugar.

A logo oficial da UNIMONTES está disponível em:
https://unimontes.br/institucional/identidade-visual/

### 5. Executar

```bash
streamlit run app.py
```

O app abrirá em `http://localhost:8501`.

---

## Publicar no GitHub

```bash
git init
git add .
git commit -m "inicial: app Transformada de Laplace — UNIMONTES"
git branch -M main
git remote add origin https://github.com/seu-usuario/laplace-unimontes.git
git push -u origin main
```

> **Atenção:** não suba arquivos `.venv/` ou `__pycache__/` — eles já estão no `.gitignore`.

---

## Deploy no Streamlit Community Cloud

1. Acesse [share.streamlit.io](https://share.streamlit.io) e faça login com GitHub.
2. Clique em **New app**.
3. Selecione o repositório e a branch `main`.
4. Em **Main file path**, insira: `app.py`
5. Clique em **Deploy**.

O Streamlit Cloud lê `requirements.txt` automaticamente e instala as dependências.
O app ficará disponível em uma URL pública do tipo:
`https://seu-usuario-laplace-unimontes-app-xxxx.streamlit.app`

> **Nota sobre a logo:** para que a logo apareça no deploy, ela precisa estar
> versionada no repositório. Se preferir não versionar, o app exibe um placeholder.

---

## Exemplos de f(t) aceitos

| Entrada | Função |
|---|---|
| `0` | Sem força externa (homogênea) |
| `sin(2*t)` | sen(2t) |
| `cos(t)` | cos(t) |
| `exp(-t)` | e⁻ᵗ |
| `t` | t (rampa) |
| `t**2` | t² |
| `exp(t)*sin(t)` | eᵗ·sen(t) |

---

## Dependências

| Biblioteca | Uso |
|---|---|
| `streamlit` | Interface web |
| `sympy` | Cálculo simbólico (transformada, frações parciais, inversa) |
| `numpy` | Avaliação numérica das expressões |
| `plotly` | Gráficos interativos |
| `scipy` | Solução numérica da EDO via RK45 |

---

## Melhorias futuras

- Suporte a EDOs de ordem superior (3ª, 4ª ordem);
- Suporte a forças descontínuas com função degrau de Heaviside na interface;
- Exibição do diagrama de polos e zeros de Y(s) no plano s;
- Modo de comparação entre múltiplas soluções (variação de parâmetros);
- Exportação do passo a passo como PDF;
- Cálculo da função de transferência H(s) e diagrama de Bode;
- Modo escuro.

---

*UNIMONTES — Universidade Estadual de Montes Claros*
