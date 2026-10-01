"""Coloca um modelo (via servidor na porta 8765) para jogar etapas de Q*bert e mede o resultado.

As mesmas sementes de teste (terminadas em 0, que nunca foram usadas no treino) são jogadas por:
  - o modelo que estiver no servidor (NanoJev original ou treinado),
  - o robô de código e o professor (para comparação; não usam a placa).

Uso: .venv\\Scripts\\python.exe qbert\\avaliar_jogando.py --nome treinado --partidas 20
Resultado: acrescenta uma linha em qbert\\resultados_jogando.json
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path
import sys
import time
import urllib.request

from professor import robot_action, teacher_values
from qbert_env import QbertEnv

HERE = Path(__file__).resolve().parent
INSTRUCTIONS = ("Choose the next action that maximizes the probability of completing the stated task successfully "
                "before its deadline. Use the visible state, action descriptions, remaining time, and recorded history.")
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def ask_model(obs):
    payload = {"states": [{"id": "q", "state": obs["state"], "questions": {
        "action": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": obs["candidates"]}}}]}
    req = urllib.request.Request("http://127.0.0.1:8765/api/evaluate", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with OPENER.open(req, timeout=300) as r:
        return json.load(r)["states"][0]["answers"]["action"]["choice"]


def play(who, level, seed):
    env = QbertEnv(level=level, lives=3, max_steps=300, ruleset="odyssey")
    obs = env.reset(seed)
    waits = 0
    while not env.terminated:
        if who == "modelo":
            a = ask_model(obs)
        elif who == "robô":
            a = robot_action(env)
        else:
            v = teacher_values(env, base_seed=seed * 10007 + env.steps)
            a = max(v, key=v.get)
        waits += a == "wait"
        obs, *_ = env.step(a)
    return {"quem": who, "nivel": level, "semente": seed, "resultado": env.outcome, "vidas": env.lives,
            "passos": env.steps, "cubos_faltando": env.remaining_cubes(), "esperas": waits}


def _play(args):
    return play(*args)


def resumo(rs):
    n = len(rs)
    return {"etapas_completas": f"{sum(r['resultado'] == 'round_cleared' for r in rs)}/{n}",
            "vidas_restantes_media": round(sum(r["vidas"] for r in rs) / n, 2),
            "cubos_faltando_media": round(sum(r["cubos_faltando"] for r in rs) / n, 1),
            "passos_media": round(sum(r["passos"] for r in rs) / n, 1),
            "esperas_media": round(sum(r["esperas"] for r in rs) / n, 1)}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nome", required=True, help="rótulo do modelo no servidor (ex.: original, treinado)")
    p.add_argument("--partidas", type=int, default=20, help="por nível")
    p.add_argument("--comparar", action="store_true", help="também joga robô e professor")
    a = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    seeds = {lvl: [100000 * lvl + 10 * k for k in range(a.partidas)] for lvl in (1, 2)}  # terminadas em 0 = teste
    out = {"nome": a.nome, "quando": time.strftime("%Y-%m-%d %H:%M"), "por_nivel": {}}
    t0 = time.time()
    for lvl in (1, 2):
        rs = []
        for s in seeds[lvl]:
            rs.append(play("modelo", lvl, s))
        out["por_nivel"][f"nivel_{lvl}"] = {"modelo": resumo(rs)}
        print(f"nível {lvl} — {a.nome}: {resumo(rs)}", flush=True)
    if a.comparar:
        with Pool(14) as pool:
            for lvl in (1, 2):
                for who in ("robô", "professor"):
                    rs = pool.map(_play, [(who, lvl, s) for s in seeds[lvl]])
                    out["por_nivel"][f"nivel_{lvl}"][who] = resumo(rs)
                    print(f"nível {lvl} — {who}: {resumo(rs)}", flush=True)
    out["segundos"] = round(time.time() - t0)
    f = HERE / "resultados_jogando.json"
    todos = json.loads(f.read_text(encoding="utf-8")) if f.exists() else []
    todos.append(out)
    f.write_text(json.dumps(todos, ensure_ascii=False, indent=1), encoding="utf-8")
