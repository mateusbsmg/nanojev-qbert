"""Treina o NanoJev no Q*bert, continuando do modelo atual (best.safetensors).

Reaproveita as peças do autor:
  - DecisionModel (scripts/train_toy_decisions.py): Qwen3-0.6B + cabeça de escolha;
  - prepare_examples (scripts/predict_toy_decisions.py): o mesmo texto "State/Question/Candidate/Decision";
  - a mesma perda: entropia cruzada entre as notas do professor e as do modelo.
Para caber nos 12 GB da RTX 3060: bf16 na conta, pesos em fp32, gradient checkpointing, lotes pequenos
e, por padrão, a tabela de vocabulário (embeddings) congelada.
Mistura exemplos de Q*bert com exemplos originais do autor (minhoca/labirinto/tiro) para não esquecer.

Uso (teste de memória, 10 passos):
    .venv\\Scripts\\python.exe qbert\\treinar_qbert.py --passos 10 --saida runs\\qbert_teste
"""
import argparse
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "qbert"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402
from transformers import AutoConfig, AutoModel, AutoTokenizer  # noqa: E402

from predict_toy_decisions import load_decision_model_class, prepare_examples  # noqa: E402


def read_rows(path, limit=None, task_filter=None, seed=0):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    if task_filter:
        rows = [r for r in rows if r.get("metadata", {}).get("task") in task_filter]
    rows = [r for r in rows if "teacher" in r and r["teacher"].get("native_probs")]
    random.Random(seed).shuffle(rows)
    return rows[:limit] if limit else rows


def retarget(rows, temperature):
    """Recalcula as notas do professor com outra temperatura, a partir dos valores gravados em metadata.values.
    Temperatura menor = notas mais "decididas" (a melhor jogada se destaca mais)."""
    for r in rows:
        vals = r.get("metadata", {}).get("values")
        if not vals:
            continue
        top = max(vals.values())
        exp = {k: math.exp((v - top) / temperature) for k, v in vals.items()}
        total = sum(exp.values())
        r["teacher"]["native_probs"]["action"] = {k: e / total for k, e in exp.items()}


def is_danger(row, radius=2):
    """Situação de perigo: algum inimigo mortal a até `radius` pulos do Q*bert."""
    import re
    from qbert_env import DEADLY, distance, on_pyramid
    s = row["state"]
    q = re.search(r"Q\*bert at \[(\d+),(\d+)\]", s)
    en = re.search(r"Enemies: (\[.*?\])\. Enemies frozen", s)
    if not q or not en:
        return False
    qpos = (int(q.group(1)), int(q.group(2)))
    for e in json.loads(en.group(1)):
        pos = tuple(e["position"])
        if e["type"] in DEADLY and on_pyramid(pos) and distance(pos, qpos) <= radius:
            return True
    return False


def is_disc(row):
    """Situação em que a melhor jogada do professor é fugir pelo disco (raro: ~2% dos exemplos)."""
    import re
    from qbert_env import dest, on_pyramid
    q = re.search(r"Q\*bert at \[(\d+),(\d+)\]", row["state"])
    vals = row.get("metadata", {}).get("values")
    if not q or not vals:
        return False
    best = max(vals, key=vals.get)
    return best != "wait" and not on_pyramid(dest((int(q.group(1)), int(q.group(2))), best))


def to_examples(rows, tokenizer, max_length):
    examples = []
    for r in rows:
        payload = {"states": [{"id": r["id"], "state": r["state"], "questions": r["questions"]}]}
        try:
            exs = prepare_examples(payload, tokenizer, max_length)
        except ValueError:
            continue  # texto maior que max_length: fica de fora (sem cortar)
        for ex in exs:
            probs = r["teacher"]["native_probs"].get(ex["qid"])
            if not probs or set(probs) != set(ex["candidate_ids"]):
                continue
            target = [float(probs[k]) for k in ex["candidate_ids"]]
            s = sum(target)
            if s <= 0:
                continue
            ex["teacher_probs"] = [t / s for t in target]
            ex["family"] = r.get("family_id") or r.get("metadata", {}).get("task")
            ex["tokens"] = sum(len(t) for t in ex["leaf_tokens"])
            examples.append(ex)
    return examples


def loss_for(logits, examples):
    target = torch.zeros_like(logits, dtype=torch.float32)
    for i, ex in enumerate(examples):
        target[i, :len(ex["candidate_ids"])] = torch.tensor(ex["teacher_probs"], device=logits.device)
    return -(target * logits.float().log_softmax(-1)).sum(-1)


def gpu_temp():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


def esfriar(alvo, retomar):
    """Pega leve com a placa: se passou de `alvo` °C, espera até baixar para `retomar` °C."""
    t = gpu_temp()
    if t is None or t <= alvo:
        return 0.0, t
    inicio = time.time()
    while t is not None and t > retomar and time.time() - inicio < 600:
        time.sleep(5)
        t = gpu_temp()
    return time.time() - inicio, t


def microbatches(examples, max_tokens, max_questions):
    group, used = [], 0
    for ex in examples:
        if group and (used + ex["tokens"] > max_tokens or len(group) >= max_questions):
            yield group
            group, used = [], 0
        group.append(ex)
        used += ex["tokens"]
    if group:
        yield group


@torch.no_grad()
def evaluate(model, examples, pad, max_tokens, temp_alvo=75, temp_retomar=68):
    model.eval()
    ce, agree, n = 0.0, 0, 0
    for group in microbatches(examples, max_tokens, 8):
        esfriar(temp_alvo, temp_retomar)  # a avaliação também pega leve com a placa
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(group, pad)
        for ex, z in zip(group, logits):
            p = z[:len(ex["candidate_ids"])].float().softmax(-1).cpu().tolist()
            ce += -sum(t * math.log(max(q, 1e-30)) for t, q in zip(ex["teacher_probs"], p))
            agree += max(range(len(p)), key=p.__getitem__) == max(range(len(p)), key=ex["teacher_probs"].__getitem__)
            n += 1
    model.train()
    return {"ce": ce / max(n, 1), "concorda_com_professor": agree / max(n, 1), "exemplos": n}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "NanoJev-unified"))
    p.add_argument("--dados", default=str(ROOT / "qbert" / "dados"))
    p.add_argument("--originais", default=str(ROOT / "data" / "NanoJev-unified" / "unified" / "hard"))
    p.add_argument("--saida", required=True)
    p.add_argument("--passos", type=int, default=400)
    p.add_argument("--perguntas-por-passo", type=int, default=16)
    p.add_argument("--fracao-original", type=float, default=0.25, help="parte de cada passo com dados do autor")
    p.add_argument("--max-tokens-micro", type=int, default=4700, help="~1 situação de Q*bert por vez")
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--lr-backbone", type=float, default=1e-5)
    p.add_argument("--lr-cabeca", type=float, default=1e-4)
    p.add_argument("--treinar-embeddings", action="store_true", help="por padrão ficam congelados (economiza ~2 GB)")
    p.add_argument("--congelar-camadas", type=int, default=14,
                   help="congela as N primeiras camadas do Qwen (de 28); economiza memória de gradiente/otimizador")
    p.add_argument("--teto-gpu-gb", type=float, default=10.5,
                   help="limite de memória da placa; acima disso dá erro em vez de usar a RAM/pagefile do Windows")
    p.add_argument("--avaliar-cada", type=int, default=100)
    p.add_argument("--limite-avaliacao", type=int, default=300)
    p.add_argument("--temp-professor", type=float, default=0,
                   help="recalcula as notas do professor com esta temperatura (0 = usa as gravadas, que usam 8)")
    p.add_argument("--fracao-disco", type=float, default=0,
                   help="parte das situações de Q*bert de cada passo sorteada entre as de fuga pelo disco")
    p.add_argument("--salvar-cada", type=int, default=10, help="salva ponto de retomada a cada N passos")
    p.add_argument("--pular-avaliacao-inicial", action="store_true", help="usa os valores já medidos (10%% / 78%%)")
    p.add_argument("--temp-alvo", type=int, default=75, help="acima disso (°C) pausa após o passo para esfriar")
    p.add_argument("--temp-retomar", type=int, default=68, help="volta a treinar quando baixar para isso (°C)")
    p.add_argument("--limite-treino", type=int, default=0, help="usar só N linhas de treino (0 = todas)")
    p.add_argument("--seed", type=int, default=17)
    a = p.parse_args()

    torch.manual_seed(a.seed)
    random.seed(a.seed)
    ckpt, out = Path(a.checkpoint), Path(a.saida)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    run_config = json.loads((ckpt / "config.json").read_text(encoding="utf-8"))

    tok = AutoTokenizer.from_pretrained(str(ckpt / "tokenizer"), local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    body_cfg = AutoConfig.from_pretrained(str(ckpt / "backbone_config"), local_files_only=True)
    body_cfg.use_cache = False
    body = AutoModel.from_config(body_cfg, attn_implementation="sdpa").float()
    model = load_decision_model_class()(body, run_config["set_head"])
    model.load_state_dict(load_file(str(ckpt / "best.safetensors"), device="cpu"), strict=True)
    model.to(device)
    body.gradient_checkpointing_enable()
    if hasattr(body, "enable_input_require_grads"):
        body.enable_input_require_grads()
    if not a.treinar_embeddings:
        body.get_input_embeddings().weight.requires_grad_(False)
    for layer in body.layers[:a.congelar_camadas]:
        layer.requires_grad_(False)
    # No Windows, passar da memória da placa não dá erro: o driver usa a RAM (e o pagefile no C:) como extensão.
    # O teto força um erro claro em vez disso.
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, a.teto_gpu_gb * 2**30 / total), 0)
    model.train()

    t0 = time.time()
    lim = a.limite_treino or None
    # Treino: guarda só o texto e transforma em tokens na hora de usar (16 por passo). Transformar as
    # 24 mil situações de uma vez ocupava ~14 GB de RAM e fez o Windows crescer o pagefile no C:.
    q_train = read_rows(Path(a.dados) / "train.jsonl", limit=lim, seed=a.seed)
    q_dev_rows = read_rows(Path(a.dados) / "dev.jsonl", limit=a.limite_avaliacao, seed=a.seed)
    if a.temp_professor:
        retarget(q_train, a.temp_professor)
        retarget(q_dev_rows, a.temp_professor)
    q_danger = [r for r in q_train if is_disc(r)] if a.fracao_disco else []
    q_dev = to_examples(q_dev_rows, tok, a.max_length)
    if a.fracao_disco:
        print(f"situações de fuga pelo disco no treino: {len(q_danger)} de {len(q_train)}", flush=True)
    o_train = [r for r in read_rows(Path(a.originais) / "train.jsonl", limit=lim, seed=a.seed)
               if to_examples([r], tok, a.max_length)]  # só as linhas utilizáveis
    o_dev = to_examples(read_rows(Path(a.originais) / "dev.jsonl", limit=a.limite_avaliacao, seed=a.seed), tok, a.max_length)
    tk = sorted(e["tokens"] for e in to_examples(q_train[:200], tok, a.max_length))
    print(f"dados: Q*bert treino {len(q_train)} / validação {len(q_dev)}; originais treino {len(o_train)} / validação {len(o_dev)}"
          f" | tokens por pergunta Q*bert: mediana {tk[len(tk)//2]}, máx {tk[-1]} | preparo {time.time()-t0:.0f}s", flush=True)

    head = [p_ for n, p_ in model.named_parameters() if not n.startswith("backbone.") and p_.requires_grad]
    back = [p_ for n, p_ in model.named_parameters() if n.startswith("backbone.") and p_.requires_grad]
    opt = torch.optim.AdamW([{"params": back, "lr": a.lr_backbone}, {"params": head, "lr": a.lr_cabeca}],
                            weight_decay=0.01, fused=True)
    treinaveis = sum(p_.numel() for p_ in back + head)
    print(f"parâmetros treináveis: {treinaveis/1e6:.0f} M (embeddings {'treinando' if a.treinar_embeddings else 'congelados'})", flush=True)

    log, rng = [], random.Random(a.seed)
    n_orig = round(a.perguntas_por_passo * a.fracao_original)
    start_step, pausas_total, elapsed_before = 1, 0.0, 0.0
    resume_file = out / "retomar.pt"
    if resume_file.exists():
        # Continua de onde parou: pesos treináveis, otimizador, sorteio e histórico do último ponto salvo.
        st = torch.load(resume_file, map_location=device, weights_only=False)
        named = dict(model.named_parameters())
        with torch.no_grad():
            for n, v in st["treinaveis"].items():
                named[n].copy_(v)
        opt.load_state_dict(st["otimizador"])
        rng.setstate(st["sorteio"])
        torch.set_rng_state(st["torch_rng"].cpu())
        log, pausas_total, elapsed_before = st["log"], st["pausas"], st["decorrido"]
        start_step = st["passo"] + 1
        print(f"retomando do passo {st['passo']} (ponto salvo em {resume_file})", flush=True)
    else:
        before = {"qbert": {"ce": 1.6992, "concorda_com_professor": 0.10, "exemplos": 150, "nota": "medido nas corridas anteriores"},
                  "originais": {"ce": 0.7588, "concorda_com_professor": 0.78, "exemplos": 150}} if a.pular_avaliacao_inicial else {"qbert": evaluate(model, q_dev, tok.pad_token_id, a.max_tokens_micro, a.temp_alvo, a.temp_retomar),
                  "originais": evaluate(model, o_dev, tok.pad_token_id, a.max_tokens_micro, a.temp_alvo, a.temp_retomar)}
        print("antes do treino:", json.dumps(before, ensure_ascii=False), flush=True)
        log.append({"passo": 0, **before})
    t_train = time.time() - elapsed_before
    for step in range(start_step, a.passos + 1):
        ts = time.time()
        n_q = a.perguntas_por_passo - n_orig
        n_perigo = round(n_q * a.fracao_disco) if q_danger else 0
        batch = to_examples(rng.sample(q_danger, n_perigo) + rng.sample(q_train, n_q - n_perigo)
                            + rng.sample(o_train, n_orig), tok, a.max_length)
        total_q = len(batch)
        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for group in microbatches(sorted(batch, key=lambda e: e["tokens"]), a.max_tokens_micro, 8):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(group, tok.pad_token_id)
            loss = loss_for(logits, group).sum() / total_q
            loss.backward()
            loss_sum += loss.item()
            meio, _ = esfriar(a.temp_alvo + 5, a.temp_retomar)  # também no meio do passo, se esquentar demais
            pausas_total += meio
            ts += meio  # a duração do passo não conta a pausa
        torch.nn.utils.clip_grad_norm_(back + head, 1.0)
        opt.step()
        mem = torch.cuda.max_memory_allocated() / 2**30
        dur = time.time() - ts
        pausa, temp = esfriar(a.temp_alvo, a.temp_retomar)
        pausas_total += pausa
        elapsed = time.time() - t_train
        eta = elapsed / step * (a.passos - step)
        print(f"passo {step}/{a.passos}: perda {loss_sum:.4f} | {dur:.1f}s + pausa {pausa:.0f}s (placa {temp} C)"
              f" | memória pico {mem:.2f} GB | falta ~{eta/60:.0f} min", flush=True)
        if step % a.avaliar_cada == 0 or step == a.passos:
            ev = {"qbert": evaluate(model, q_dev, tok.pad_token_id, a.max_tokens_micro, a.temp_alvo, a.temp_retomar),
                  "originais": evaluate(model, o_dev, tok.pad_token_id, a.max_tokens_micro, a.temp_alvo, a.temp_retomar)}
            print(f"avaliação passo {step}:", json.dumps(ev, ensure_ascii=False), flush=True)
            log.append({"passo": step, **ev})
            save_bundle(model, ckpt, out, run_config, a)  # salvamento intermediário (último)
            anteriores = [e["qbert"]["ce"] for e in log[:-1] if "qbert" in e]
            if not anteriores or ev["qbert"]["ce"] < min(anteriores):
                # guarda também a MELHOR versão (menor erro no Q*bert), que não é sobrescrita por passos piores
                (out / "melhor").mkdir(exist_ok=True)
                save_bundle(model, ckpt, out / "melhor", run_config, a)
                (out / "melhor" / "passo.txt").write_text(f"melhor versão: passo {step}, erro {ev['qbert']['ce']:.4f}\n",
                                                          encoding="utf-8")
                print(f"melhor versão até agora (passo {step}) salva em {out / 'melhor'}", flush=True)
        (out / "log.json").write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
        ultima = log[-1]
        (out / "progresso.txt").write_text(
            f"Treino do NanoJev no Q*bert\n"
            f"Passo {step} de {a.passos} ({100*step/a.passos:.0f}%)\n"
            f"Tempo decorrido: {elapsed/60:.0f} min | falta ~{eta/60:.0f} min (término ~{time.strftime('%H:%M', time.localtime(time.time()+eta))})\n"
            f"Perda do último passo: {loss_sum:.4f} | memória pico {mem:.2f} GB\n"
            f"Placa: {temp} C após o passo | pausas para esfriar até agora: {pausas_total/60:.0f} min "
            f"(pausa se passar de {a.temp_alvo} C, volta em {a.temp_retomar} C)\n"
            f"Última avaliação (passo {ultima['passo']}): concorda com o professor em "
            f"{100*ultima['qbert']['concorda_com_professor']:.0f}% no Q*bert | "
            f"{100*ultima['originais']['concorda_com_professor']:.0f}% nos jogos originais\n", encoding="utf-8")
        if step % a.salvar_cada == 0 and step < a.passos:
            tmp = out / "retomar.pt.tmp"
            torch.save({"treinaveis": {n: p_.detach().cpu() for n, p_ in model.named_parameters() if p_.requires_grad},
                        "otimizador": opt.state_dict(), "sorteio": rng.getstate(), "torch_rng": torch.get_rng_state(),
                        "log": log, "pausas": pausas_total, "decorrido": time.time() - t_train, "passo": step}, tmp)
            os.replace(tmp, resume_file)  # troca de uma vez: nunca fica um arquivo pela metade
            print(f"ponto de retomada salvo no passo {step}", flush=True)

    save_bundle(model, ckpt, out, run_config, a)
    resume_file.unlink(missing_ok=True)
    print(f"modelo salvo em {out}", flush=True)


def save_bundle(model, ckpt, out, run_config, a):
    """Salva no mesmo formato do autor, para o serve_decisions.py carregar."""
    save_file({k: v.detach().to("cpu", torch.float32).contiguous() for k, v in model.state_dict().items()},
              str(out / "best.safetensors"))
    for d in ("tokenizer", "backbone_config"):
        shutil.copytree(ckpt / d, out / d, dirs_exist_ok=True)
    cfg = dict(run_config)
    cfg.update(init_checkpoint=str(ckpt), qbert_training=vars(a))
    (out / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
