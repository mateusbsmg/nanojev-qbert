"""Joga partidas com o modelo do servidor (porta 8765) e conta a causa de cada vida perdida.

Uso: .venv\\Scripts\\python.exe qbert\\causas_morte.py --partidas 10 --nivel 1
"""
import argparse
from collections import Counter
import sys

from avaliar_jogando import ask_model
from qbert_env import QbertEnv, dest, on_pyramid

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--partidas", type=int, default=10)
    p.add_argument("--nivel", type=int, default=1)
    a = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    causas, escolhas, fora_sem_disco = Counter(), Counter(), 0
    for k in range(a.partidas):
        env = QbertEnv(level=a.nivel, lives=3, max_steps=300, ruleset="odyssey")
        obs = env.reset(100000 * a.nivel + 10 * k)
        while not env.terminated:
            act = ask_model(obs)
            escolhas[act] += 1
            d = dest(env.q, act)
            if act != "wait" and not on_pyramid(d) and d not in env.discs:
                fora_sem_disco += 1
            obs, _, _, _, info = env.step(act)
            for e in info["events"]:
                if e["event"] == "lost_life":
                    causas[e["reason"]] += 1
    print("causas das vidas perdidas:", dict(causas))
    print("pulos para fora sem disco:", fora_sem_disco)
    print("jogadas escolhidas:", dict(escolhas))
