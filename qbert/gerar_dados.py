"""Gera exemplos de treino do Q*bert no formato do autor (uma linha JSON por decisão).

Cada linha: {"id", "state_id", "family_id": "qbert", "split", "state", "questions": {"action": {...}},
             "teacher": {"model": "qbert-professor-v1", "native_probs": {"action": {...}}}, "metadata": {...}}

O professor (qbert/professor.py) joga as partidas; em 10% das decisões executa uma jogada sorteada
(proporcional às notas dele), para o modelo também ver situações fora do caminho ideal.
A divisão treino/validação/teste é por partida (sementes diferentes), para não haver "cola".

Uso: .venv\\Scripts\\python.exe qbert\\gerar_dados.py --partidas 160 --saida qbert\\dados
"""
import argparse
import hashlib
import json
from multiprocessing import Pool
from pathlib import Path
import random
import time

from professor import teacher_values, to_probs
from qbert_env import QbertEnv

INSTRUCTIONS = ("Choose the next action that maximizes the probability of completing the stated task successfully "
                "before its deadline. Use the visible state, action descriptions, remaining time, and recorded history.")
EPSILON = 0.10


def split_for(seed):
    r = seed % 10
    return "test" if r == 0 else "dev" if r == 1 else "train"


def play_episode(args):
    level, seed = args
    rng = random.Random(seed)
    env = QbertEnv(level=level, lives=3, max_steps=300, ruleset="odyssey")
    obs = env.reset(seed)
    rows = []
    while not env.terminated:
        values = teacher_values(env, base_seed=seed * 10007 + env.steps)
        probs = to_probs(values)
        best = max(probs, key=probs.get)
        action = rng.choices(list(probs), list(probs.values()))[0] if rng.random() < EPSILON else best
        sid = hashlib.sha256(obs["state"].encode()).hexdigest()
        rows.append({
            "id": hashlib.sha256(f"qbert:{level}:{seed}:{env.steps}".encode()).hexdigest(),
            "state_id": sid, "family_id": "qbert", "split": split_for(seed),
            "state": obs["state"],
            "questions": {"action": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": obs["candidates"]}},
            "teacher": {"model": "qbert-professor-v1", "native_probs": {"action": {a: round(p, 6) for a, p in probs.items()}}},
            "metadata": {"task": "qbert", "level": level, "seed": seed, "step": env.steps, "teacher_best": best,
                         "executed_action": action, "values": {a: round(v, 3) for a, v in values.items()}},
        })
        obs, *_ = env.step(action)
    for r in rows:
        r["metadata"]["episode_outcome"] = env.outcome
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--partidas", type=int, default=160, help="partidas por nível")
    p.add_argument("--niveis", default="1,2")
    p.add_argument("--saida", default="dados")
    a = p.parse_args()
    jobs = [(int(level), 100000 * int(level) + s) for level in a.niveis.split(",") for s in range(a.partidas)]
    t0 = time.time()
    with Pool(14) as pool:
        episodes = pool.map(play_episode, jobs, chunksize=1)
    out = Path(a.saida)
    out.mkdir(parents=True, exist_ok=True)
    files = {s: open(out / f"{s}.jsonl", "w", encoding="utf-8") for s in ("train", "dev", "test")}
    counts = {s: 0 for s in files}
    outcomes = {}
    for rows in episodes:
        for r in rows:
            files[r["split"]].write(json.dumps(r, ensure_ascii=False) + "\n")
            counts[r["split"]] += 1
        o = rows[0]["metadata"]["episode_outcome"] if rows else "vazio"
        outcomes[o] = outcomes.get(o, 0) + 1
    for f in files.values():
        f.close()
    print(f"{sum(counts.values())} decisões em {time.time() - t0:.0f} s: {counts} | partidas: {outcomes}")
