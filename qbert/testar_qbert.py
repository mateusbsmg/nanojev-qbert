"""Testes das regras do Q*bert + estatísticas do robô de código.

Uso: .venv\\Scripts\\python.exe qbert\\testar_qbert.py
"""
from collections import Counter
import random
import sys

from functools import partial

import qbert_env
from qbert_env import ACTIONS, CELLS, DEADLY, bot_scores, dest, distance, on_pyramid

QbertEnv = partial(qbert_env.QbertEnv, ruleset="arcade")  # os testes abaixo usam os números do arcade
Odyssey = partial(qbert_env.QbertEnv, ruleset="odyssey")

sys.stdout.reconfigure(encoding="utf-8")
falhas = []


def check(cond, msg):
    if not cond:
        falhas.append(msg)


# --- geometria
check(len(CELLS) == 28, "pirâmide deve ter 28 cubos")
check(dest((3, 1), "up_right") == (2, 1) and dest((3, 1), "up_left") == (2, 0), "pulos para cima")
check(dest((3, 1), "down_right") == (4, 2) and dest((3, 1), "down_left") == (4, 1), "pulos para baixo")
check(not on_pyramid((2, 3)) and not on_pyramid((7, 0)) and not on_pyramid((1, -1)), "fora da pirâmide")
check(distance((0, 0), (6, 0)) == 6 and distance((6, 0), (6, 6)) == 12, "distâncias")  # base: zigue-zague

# --- cair da pirâmide custa vida e volta ao topo
env = QbertEnv(lives=3); env.reset(1); env.enemies.clear(); env.discs.clear()
env.step("up_right")
check(env.lives == 2 and env.q == (0, 0), "cair deve custar vida e voltar ao topo")

# --- regra de cor nível 1 e pontos
env = QbertEnv(level=1); env.reset(2); env.enemies.clear(); env.next_spawn = 10**9
env.step("down_left")
check(env.colors[(1, 0)] == 2 and env.score == 25, "nível 1: um pulo pinta (+25)")
env.step("up_right"); s = env.score
check(env.colors[(0, 0)] == 2 and env.score == s, "cubo já na cor-alvo não muda nem pontua")

# --- nível 2: dois pulos; nível 3: alterna
env = QbertEnv(level=2); env.reset(3); env.enemies.clear(); env.next_spawn = 10**9
env.step("down_left"); check(env.colors[(1, 0)] == 1, "nível 2: 1º pulo = intermediária")
env.step("up_right"); env.step("down_left"); check(env.colors[(1, 0)] == 2, "nível 2: 2º pulo = alvo")
env = QbertEnv(level=3); env.reset(4); env.enemies.clear(); env.next_spawn = 10**9
env.step("down_left"); env.step("up_right"); env.step("down_left")
check(env.colors[(1, 0)] == 0, "nível 3: pular no alvo desfaz")

# --- disco: leva ao topo, some, atrai o Coily (+500) e limpa inimigos
env = QbertEnv(); env.reset(5); env.next_spawn = 10**9
left = next(d for d in env.discs if d[1] == -1)
env.q = (left[0] + 1, 0)
env.enemies = [{"id": 99, "type": "coily", "pos": (6, 3), "age": 0}, {"id": 98, "type": "red_ball", "pos": (5, 5), "age": 0}]
s = env.score
env.step("up_left")
check(env.q == (0, 0) and left not in env.discs and not env.enemies and env.score >= s + 500, "disco")

# --- contato com inimigo mortal custa vida; pegar Sam dá 300; bola verde congela
env = QbertEnv(); env.reset(6); env.next_spawn = 10**9
env.enemies = [{"id": 1, "type": "red_ball", "pos": (1, 0), "age": 0}]
env.step("down_left"); check(env.lives == 2 and not env.enemies, "bola vermelha mata")
env = QbertEnv(); env.reset(7); env.next_spawn = 10**9
env.enemies = [{"id": 1, "type": "sam", "pos": (1, 0), "age": 0}]
env.step("down_left"); check(env.score >= 300 and env.lives == 3, "pegar Sam dá 300")
env = QbertEnv(); env.reset(8); env.next_spawn = 10**9
env.enemies = [{"id": 1, "type": "green_ball", "pos": (1, 1), "age": 0}, {"id": 2, "type": "red_ball", "pos": (4, 2), "age": 0}]
env.step("down_right"); check(env.frozen > 0, "bola verde congela")
pos = env.enemies[0]["pos"]; env.step("wait"); check(env.enemies and env.enemies[0]["pos"] == pos, "congelado não anda")

# --- troca de lugar também conta como contato
env = QbertEnv(); env.reset(9); env.next_spawn = 10**9
env.q = (2, 1)
env.enemies = [{"id": 1, "type": "red_ball", "pos": (1, 1), "age": 0}]
env.rng.seed(0)
lives = env.lives
env.step("up_right")  # Q*bert vai para [1,1] enquanto a bola desce para [2,1] ou [2,2]
check(env.lives == lives - 1, "pular em cima da bola custa vida")

# --- bola roxa vira Coily na última fileira; Coily se aproxima
env = QbertEnv(level=2); env.reset(10); env.next_spawn = 10**9
env.enemies = [{"id": 1, "type": "purple_ball", "pos": (6, 3), "age": 0}]
env.step("wait"); check(env.enemies[0]["type"] == "coily", "bola roxa vira Coily")
d0 = distance(env.enemies[0]["pos"], env.q); env.step("wait")
check(not env.enemies or distance(env.enemies[0]["pos"], env.q) < d0, "Coily se aproxima")

# --- rodada vencida
env = QbertEnv(); env.reset(11); env.enemies.clear(); env.next_spawn = 10**9
for c in CELLS:
    env.colors[c] = 2
env.colors[(1, 0)] = 0
env.step("down_left")
check(env.terminated and env.success and env.outcome == "round_cleared", "rodada vencida")

# --- modo Odyssey (manual brasileiro + vídeo)
env = Odyssey(); env.reset(20); env.enemies.clear(); env.next_spawn = 10**9
check(env.lives == 7, "Odyssey: 7 Q*berts")
check(env.score == 1, "Odyssey: +1 ao pousar no topo no início")
env.step("down_left"); check(env.score == 2, "Odyssey: +1 por cubo")
env = Odyssey(stop_at_round_end=False); env.reset(21); env.enemies.clear(); env.next_spawn = 10**9
for c in CELLS:
    env.colors[c] = 2
env.colors[(1, 0)] = 0
s = env.score; env.step("down_left")
check(env.score == s + 1 + 50 + 1 and env.round == 2 and not env.terminated, "Odyssey: +50 por etapa e próxima etapa (+1 no topo)")
env = Odyssey(); env.reset(22); env.next_spawn = 10**9
env.score = 299; env.enemies.clear(); lives = env.lives
env.step("down_left"); check(env.lives == lives + 1, "Odyssey: Q*bert extra aos 300 pontos")
env = Odyssey(); env.reset(23); env.next_spawn = 10**9
left = next(d for d in env.discs if d[1] == -1); env.q = (left[0] + 1, 0)
env.enemies = [{"id": 1, "type": "coily", "pos": (6, 3), "age": 0}]
s = env.score; env.step("up_left"); check(env.score == s, "Odyssey: Serpente longe NÃO cai no disco")
check(any(e["type"] == "coily" for e in env.enemies), "Odyssey: inimigos continuam depois do disco")
env = Odyssey(); env.reset(26); env.next_spawn = 10**9
left = next(d for d in env.discs if d[1] == -1); env.q = (left[0] + 1, 0)
env.enemies = [{"id": 1, "type": "coily", "pos": (left[0] + 2, 1), "age": 0}, {"id": 2, "type": "red_ball", "pos": (2, 2), "age": 0}]
s = env.score; env.step("up_left")
check(env.score == s + 25 and not any(e["type"] == "coily" for e in env.enemies) and env.enemies,
      "Odyssey: Serpente logo atrás cai (+25); bola vermelha fica")
env = Odyssey(); env.reset(24); env.next_spawn = 10**9
env.enemies = [{"id": 1, "type": "sam", "pos": (2, 1), "age": 0}]
env.colors[(3, 1)] = env.colors[(3, 2)] = 2
env.step("wait"); check(env.enemies[0]["pos"] in env.spoiled and env.colors[env.enemies[0]["pos"]] == 0, "Odyssey: Manhoso deixa cubo branco")
env = Odyssey(level=9, stop_at_round_end=False); env.reset(25); env.enemies.clear(); env.next_spawn = 10**9
env.round = 4
for c in CELLS:
    env.colors[c] = 2
env.colors[(1, 0)] = 1
env.step("down_left")
check(env.terminated and env.success and env.outcome == "game_completed", "Odyssey: fim do jogo após nível 9 etapa 4")

# --- partidas aleatórias: invariantes (os dois modos)
for seed in range(300):
    env = (QbertEnv if seed % 3 else Odyssey)(level=1 + seed % 5, max_steps=200, stop_at_round_end=seed % 2 == 0)
    obs = env.reset(seed)
    rng = random.Random(seed)
    while not env.terminated:
        check(set(obs["candidates"]) == set(ACTIONS), "sempre 5 opções")
        obs, *_ = env.step(rng.choice(ACTIONS))
        check(on_pyramid(env.q), f"Q*bert fora da pirâmide (semente {seed})")
        check(all(on_pyramid(e["pos"]) for e in env.enemies), f"inimigo fora (semente {seed})")
        check(env.lives >= 0 and env.steps <= env.max_steps, "vidas/passos")
        check(not any(e["pos"] == env.q and e["type"] in DEADLY for e in env.enemies) or env.terminated,
              f"Q*bert vivo em cima de inimigo mortal (semente {seed})")

print("Falhas:" if falhas else "Todas as regras testadas passaram.")
for f in sorted(set(falhas)):
    print("  -", f)

# --- estatísticas: robô de código x jogador aleatório (1ª rodada de cada nível)
robo = lambda e: max(bot_scores(e).items(), key=lambda kv: kv[1])[0]
for nome, escolher, nivel in [("robô de código", robo, n) for n in (1, 2, 3, 5)] + [("aleatório", None, 1)]:
    resultados, passos, vidas = Counter(), [], []
    for seed in range(200):
        env = QbertEnv(level=nivel, max_steps=250 if nivel == 1 else 400); env.reset(1000 + seed)
        rng = random.Random(seed)
        while not env.terminated:
            env.step(escolher(env) if escolher else rng.choice(ACTIONS))
        resultados[env.outcome] += 1
        passos.append(env.steps); vidas.append(env.lives)
    print(f"{nome}, nível {nivel}: {dict(resultados)}  passos médios {sum(passos)/len(passos):.0f}  vidas restantes médias {sum(vidas)/len(vidas):.1f}")
