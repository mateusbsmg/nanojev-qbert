"""Gera o texto do post do LinkedIn com "negrito" em letras Unicode (o post do LinkedIn não aceita formatação).

Lê etapa2/ARTIGO_LINKEDIN.md e grava etapa2/ARTIGO_LINKEDIN_post.txt.
Só ficam em negrito os títulos de seção e algumas palavras-chave: cada letra em negrito conta como 2
no limite de 3.000 caracteres (e letras acentuadas como 3).

Uso: .venv\\Scripts\\python.exe etapa2\\gerar_post_linkedin.py
"""
import re
import sys
import unicodedata
from pathlib import Path

AQUI = Path(__file__).resolve().parent
NEGRITO = {"O desafio: o Q*bert do Odyssey", "A máquina", "O treino: sim, é fine-tuning", "O esforço que ninguém vê",
           "O resultado", "O que fica", "NanoJev", "Q*bert", "fine-tuning"}


def bold(text):
    """Letras e números em 'Mathematical Sans-Serif Bold'; acento vira letra negrito + acento combinado."""
    res = []
    for ch in unicodedata.normalize("NFD", text):
        o = ord(ch)
        if "A" <= ch <= "Z":
            res.append(chr(0x1D5D4 + o - ord("A")))
        elif "a" <= ch <= "z":
            res.append(chr(0x1D5EE + o - ord("a")))
        elif "0" <= ch <= "9":
            res.append(chr(0x1D7EC + o - ord("0")))
        else:
            res.append(ch)
    return "".join(res)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    md = (AQUI / "ARTIGO_LINKEDIN.md").read_text(encoding="utf-8").replace("\r\n", "\n").rstrip("\n")
    linhas = []
    for line in md.split("\n"):
        if line.startswith("# "):
            linhas.append(bold(line[2:]))  # título em negrito
        else:
            linhas.append(re.sub(r"\*\*(.+?)\*\*",
                                 lambda m: bold(m.group(1)) if m.group(1) in NEGRITO else m.group(1), line))
    texto = "\n".join(linhas)
    (AQUI / "ARTIGO_LINKEDIN_post.txt").write_text(texto + "\n", encoding="utf-8")
    utf16 = len(texto.encode("utf-16-le")) // 2
    print(f"salvo: {AQUI / 'ARTIGO_LINKEDIN_post.txt'} | {utf16} caracteres na contagem mais rigorosa (limite 3.000)")


if __name__ == "__main__":
    main()
