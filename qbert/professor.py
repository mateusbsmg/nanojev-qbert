"""Professor do Q*bert: dá uma nota a cada jogada simulando o futuro.

Para cada uma das 5 jogadas possíveis, copia o jogo, faz a jogada e deixa o robô de código continuar
por alguns passos, várias vezes, cada vez com um sorteio diferente dos inimigos. A nota da jogada é a
média do resultado dessas continuações:
    perdeu vida ............................................. 0
    completou a etapa ....................................... 1
    sobreviveu até o fim do horizonte ......... 0,5 + 0,5 × (fração dos cubos que faltavam e foram pintados)
As notas viram probabilidades (softmax com temperatura), no mesmo formato do "teacher" do autor.

Uso (avaliar o professor contra o robô):
    .venv\\Scripts\\python.exe qbert\\professor.py --avaliar --partidas 30
"""
import argparse
import copy
import math
from multiprocessing import Pool
import random
import time

from qbert_env import ACTIONS, QbertEnv, bot_scores

ROLLOUTS, HORIZON, TEMPERATURE = 8, 20, 0.08
RISK_HORIZON, RISK_WEIGHT, SCORE_TEMPERATURE = 3, 80.0, 8.0
RISK_ROLLOUTS = 12


def robot_action(env):
    scores = bot_scores(env)
    return max(scores, key=scores.get)


def rollout_value(env, action, seed, horizon=HORIZON):
    sim = copy.deepcopy(env)
    sim.rng = random.Random(seed)          # sorteio diferente dos inimigos em cada continuação
    sim.stop_at_round_end = True
    sim.max_steps = sim.steps + horizon + 1
    lives, before = sim.lives, color_progress(sim)
    todo = max(1e-9, 1.0 - before)
    sim.step(action)
    steps = 1
    while steps <= horizon and not sim.terminated and sim.lives == lives:
        sim.step(robot_action(sim))
        steps += 1
    if sim.lives < lives:
        return 0.4 * (steps - 1) / horizon  # morrer mais tarde é menos ruim que morrer já (0 = morre nesta jogada)
    if sim.success:
        return 1.0
    return 0.5 + 0.5 * max(0.0, (color_progress(sim) - before) / todo)


def color_progress(env):
    """Fração do caminho de cores já feito (conta a cor intermediária como meio caminho)."""
    two_step = env.rule().get(0) == 1
    per = {0: 0.0, 1: 0.5 if two_step else 0.0, 2: 1.0}
    return sum(per[c] for c in env.colors.values()) / len(env.colors)


def death_risk(env, action, rollouts, base_seed, horizon):
    """Chance de perder uma vida nos próximos `horizon` passos se fizer `action` (robô continua)."""
    deaths = 0
    for k in range(rollouts):
        sim = copy.deepcopy(env)
        sim.rng = random.Random(base_seed * 1000 + k)
        sim.stop_at_round_end = True
        sim.max_steps = sim.steps + horizon + 1
        lives = sim.lives
        sim.step(action)
        n = 1
        while n < horizon and not sim.terminated and sim.lives == lives:
            sim.step(robot_action(sim))
            n += 1
        deaths += sim.lives < lives
    return deaths / rollouts


def teacher_values(env, rollouts=RISK_ROLLOUTS, base_seed=0, horizon=RISK_HORIZON):
    """Nota = preferência do robô (vontade de pintar cubos) − penalidade grande × risco de morrer."""
    bot = bot_scores(env)
    return {a: bot[a] - RISK_WEIGHT * death_risk(env, a, rollouts, base_seed, horizon) for a in ACTIONS}


def to_probs(values, temperature=SCORE_TEMPERATURE):
    top = max(values.values())
    exp = {a: math.exp((v - top) / temperature) for a, v in values.items()}
    total = sum(exp.values())
    return {a: exp[a] / total for a in ACTIONS}


def play(args):
    """Joga uma etapa inteira com o professor ou com o robô; devolve o resultado."""
    who, level, seed, lives = args
    env = QbertEnv(level=level, lives=lives, max_steps=400, ruleset="odyssey")
    env.reset(seed)
    decisions = 0
    while not env.terminated:
        if who == "professor":
            values = teacher_values(env, base_seed=seed * 10007 + env.steps)
            action = max(values, key=values.get)
        else:
            action = robot_action(env)
        env.step(action)
        decisions += 1
    return who, level, env.outcome, env.lives, env.steps


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--avaliar", action="store_true")
    p.add_argument("--partidas", type=int, default=30)
    p.add_argument("--vidas", type=int, default=3)
    a = p.parse_args()
    if a.avaliar:
        jobs = [(who, level, 5000 + s, a.vidas) for level in (1, 2, 3) for who in ("robô", "professor")
                for s in range(a.partidas)]
        t0 = time.time()
        with Pool(14) as pool:
            results = pool.map(play, jobs, chunksize=1)
        print(f"({time.time() - t0:.0f} s, {a.vidas} vidas por etapa, regras Odyssey)")
        for level in (1, 2, 3):
            for who in ("robô", "professor"):
                rs = [r for r in results if r[0] == who and r[1] == level]
                wins = sum(r[2] == "round_cleared" for r in rs)
                lives = sum(r[3] for r in rs) / len(rs)
                fins = {}
                for r in rs:
                    fins[r[2]] = fins.get(r[2], 0) + 1
                steps = sum(r[4] for r in rs) / len(rs)
                print(f"nível {level} {who:10s}: completou {wins}/{len(rs)} etapas | vidas restantes médias {lives:.1f}"
                      f" | passos médios {steps:.0f} | {fins}")
