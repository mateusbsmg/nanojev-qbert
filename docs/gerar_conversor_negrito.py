"""Gera etapa2/negrito_linkedin.html: um conversor de negrito Unicode que funciona sem internet.

A página já vem com o texto do artigo (ARTIGO_LINKEDIN.md), com os títulos de seção e as palavras-chave
em negrito. Rode de novo sempre que o .md mudar.

Uso: .venv\\Scripts\\python.exe etapa2\\gerar_conversor_negrito.py
"""
import json
import re
from pathlib import Path

AQUI = Path(__file__).resolve().parent
NEGRITO = {"O desafio: o Q*bert do Odyssey", "A máquina", "O treino: sim, é fine-tuning", "O esforço que ninguém vê",
           "O resultado", "O que fica", "fine-tuning"}

md = (AQUI / "ARTIGO_LINKEDIN.md").read_text(encoding="utf-8").replace("\r\n", "\n").rstrip("\n")
linhas = []
for line in md.split("\n"):
    if line.startswith("# "):
        line = f"⟦{line[2:]}⟧"  # título em negrito
    # marca com ⟦ ⟧ o que vai para negrito; a página converte ao abrir
    linhas.append(re.sub(r"\*\*(.+?)\*\*", lambda m: f"⟦{m.group(1)}⟧" if m.group(1) in NEGRITO else m.group(1), line))
artigo = "\n".join(linhas)

pagina = r"""<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<title>Negrito para LinkedIn</title>
<style>
  :root { --bg:#f6f5f1; --panel:#fff; --ink:#1d1d1b; --muted:#6b6a64; --line:#e3e1da; --accent:#0a66c2; --ok:#1f9d55; --bad:#d14343; }
  @media (prefers-color-scheme: dark) { :root { --bg:#151514; --panel:#1f1f1d; --ink:#ecebe6; --muted:#9c9a92; --line:#33322f; --accent:#70b5f9; --ok:#3fbf76; --bad:#ef6b6b; } }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.5 system-ui,"Segoe UI",sans-serif; }
  main { max-width: 920px; margin: 0 auto; padding: 24px 16px 48px; }
  h1 { font-size: 22px; margin: 0 0 4px; } .sub { color: var(--muted); margin: 0 0 16px; }
  .bar { display:flex; flex-wrap:wrap; gap:8px; align-items:center; background:var(--panel); border:1px solid var(--line);
         border-radius:10px; padding:10px; margin-bottom:10px; }
  button { font:inherit; padding:7px 12px; border-radius:7px; border:1px solid var(--line); background:var(--panel); color:var(--ink); cursor:pointer; }
  button.primary { background:var(--accent); border-color:var(--accent); color:#fff; }
  label { font-size:13px; color:var(--muted); display:flex; gap:6px; align-items:center; }
  textarea { width:100%; min-height:520px; padding:14px; border-radius:10px; border:1px solid var(--line);
             background:var(--panel); color:var(--ink); font:15px/1.55 "Segoe UI",system-ui,sans-serif; resize:vertical; }
  .count { margin-left:auto; font-weight:600; font-variant-numeric:tabular-nums; }
  .count.ok { color:var(--ok); } .count.bad { color:var(--bad); }
  .msg { min-height:20px; font-size:13px; color:var(--ok); margin-top:6px; }
  details { margin-top:14px; color:var(--muted); font-size:13px; } li { margin:3px 0; }
</style>
</head>
<body>
<main>
  <h1>Negrito para LinkedIn</h1>
  <p class="sub">Selecione um trecho e clique em <b>Negrito</b>. Tudo funciona no seu PC, sem internet. Depois clique em <b>Copiar</b> e cole no post.</p>
  <div class="bar">
    <button class="primary" id="b">𝗕 Negrito</button>
    <button id="n">Tirar negrito</button>
    <label><input type="radio" name="ac" value="sem" checked> Letras acentuadas sem negrito (mais seguro)</label>
    <label><input type="radio" name="ac" value="com"> Acento sobre a letra em negrito</label>
    <span class="count" id="count"></span>
  </div>
  <textarea id="t" spellcheck="false"></textarea>
  <div class="bar" style="margin-top:10px">
    <button class="primary" id="copy">📋 Copiar tudo</button>
    <button id="reset">Recarregar o artigo</button>
    <button id="allplain">Tirar todo o negrito</button>
  </div>
  <div class="msg" id="msg"></div>
  <details><summary>Como funciona e cuidados</summary><ul>
    <li>O post do LinkedIn não aceita formatação. Estas letras (𝗮𝘀𝘀𝗶𝗺) são <b>caracteres Unicode</b> diferentes, por isso continuam "em negrito" ao colar.</li>
    <li>O contador mostra como o LinkedIn conta: cada letra em negrito vale <b>2</b> (acentuada com acento por cima, 3). Limite: 3.000.</li>
    <li>Use pouco: títulos e poucas palavras. Leitores de tela leem essas letras de forma estranha, e a busca do LinkedIn não as reconhece como palavras normais.</li>
    <li>"Letras acentuadas sem negrito": em "máquina", o "á" fica normal e o resto em negrito. É o que aparece certo em qualquer celular.</li>
  </ul></details>
</main>
<script>
const ARTIGO = __ARTIGO__;
const $ = id => document.getElementById(id);
const up = 0x1D5D4, low = 0x1D5EE, dig = 0x1D7EC;

function toBold(s, acentoPorCima) {
  let out = "";
  for (const ch of s.normalize("NFC")) {
    const base = ch.normalize("NFD"), letra = base[0], marcas = base.slice(1);
    const c = letra.codePointAt(0);
    let b = null;
    if (c >= 65 && c <= 90) b = String.fromCodePoint(up + c - 65);
    else if (c >= 97 && c <= 122) b = String.fromCodePoint(low + c - 97);
    else if (c >= 48 && c <= 57) b = String.fromCodePoint(dig + c - 48);
    if (b === null) out += ch;
    else if (!marcas) out += b;
    else out += acentoPorCima ? b + marcas : ch;          // "mais seguro": letra acentuada fica normal
  }
  return out;
}
function toPlain(s) {
  let out = "";
  for (const ch of s) {
    const c = ch.codePointAt(0);
    if (c >= up && c < up + 26) out += String.fromCharCode(65 + c - up);
    else if (c >= low && c < low + 26) out += String.fromCharCode(97 + c - low);
    else if (c >= dig && c < dig + 10) out += String.fromCharCode(48 + c - dig);
    else out += ch;
  }
  return out.normalize("NFC");
}
const acentoPorCima = () => document.querySelector('input[name=ac]:checked').value === "com";

function carregar() {
  $("t").value = ARTIGO.replace(/⟦(.+?)⟧/g, (_, x) => toBold(x, acentoPorCima()));
  contar();
}
function contar() {
  const n = $("t").value.length;                            // JavaScript conta como o LinkedIn (UTF-16)
  $("count").textContent = `${n} / 3000 caracteres`;
  $("count").className = "count " + (n <= 3000 ? "ok" : "bad");
}
function aplicar(fn) {
  const t = $("t"), a = t.selectionStart, b = t.selectionEnd;
  if (a === b) { $("msg").textContent = "Selecione um trecho primeiro."; return; }
  const novo = fn(t.value.slice(a, b));
  t.setRangeText(novo, a, b, "select"); t.focus(); contar(); $("msg").textContent = "";
}
$("b").onclick = () => aplicar(s => toBold(toPlain(s), acentoPorCima()));
$("n").onclick = () => aplicar(toPlain);
$("allplain").onclick = () => { $("t").value = toPlain($("t").value); contar(); };
$("reset").onclick = carregar;
$("t").oninput = contar;
document.querySelectorAll('input[name=ac]').forEach(r => r.onchange = carregar);
$("copy").onclick = async () => {
  try { await navigator.clipboard.writeText($("t").value); }
  catch (e) { $("t").select(); document.execCommand("copy"); }
  $("msg").textContent = "Copiado! Agora é só colar (Ctrl+V) no post do LinkedIn.";
};
carregar();
</script>
</body>
</html>
"""
(AQUI / "negrito_linkedin.html").write_text(pagina.replace("__ARTIGO__", json.dumps(artigo, ensure_ascii=False)),
                                            encoding="utf-8")
print("salvo:", AQUI / "negrito_linkedin.html")
