# Treinamento do NanoJev no Q*bert: o que está acontecendo por trás dos panos

Início: 2026-09-23 · Máquina: RTX 3060 (12 GB), Ryzen 7 5700G (16 núcleos)
Pastas: jogo e scripts em `D:\Mateus\NanoJev\qbert\`; modelo treinado em `D:\Mateus\NanoJev\runs\qbert_v1\`

---

## 1. A ideia em uma frase

O NanoJev aprende **imitando as notas de um professor**. Em milhares de situações do Q*bert, um professor
diz "esta jogada vale 80%, aquela 15%, aquela 5%". O modelo é ajustado aos pouquinhos até dar notas parecidas.
É exatamente assim que o autor treinou o NanoJev na minhoca e no labirinto. A diferença é que o professor dele
era o **Jev** (um serviço pago), e o nosso é **um programa feito aqui, de graça**.

```
  ┌────────────┐   situação em texto   ┌──────────────┐   notas das 5 jogadas   ┌──────────────┐
  │  O JOGO    │ ─────────────────────▶│  PROFESSOR   │ ───────────────────────▶│  EXEMPLO     │
  │ (Python)   │                       │ (simulação)  │                         │  de treino   │
  └────────────┘                       └──────────────┘                         └──────┬───────┘
                                                                                       │ milhares
                                                                                       ▼
                                   ┌───────────────────────────────────────────────────────────┐
                                   │ TREINO: o NanoJev dá as notas dele; mede-se a diferença   │
                                   │ para as notas do professor; os números internos do modelo │
                                   │ são corrigidos um pouquinho para diminuir essa diferença. │
                                   └───────────────────────────────────────────────────────────┘
```

---

## 2. O professor (`qbert/professor.py`)

### 2.1 Primeira tentativa: só simulação (deu errado)
Para cada uma das 5 jogadas, o professor copiava o jogo, fazia a jogada e deixava o **robô de código**
continuar por 20–30 passos, 8 vezes (cada vez com um sorteio diferente dos inimigos). A nota era:
morreu = 0, completou a etapa = 1, sobreviveu = 0,5 + parte do progresso.

**Problema 1 – empates:** quando o robô morria em todas as continuações (comum no nível 2), todas as jogadas
ficavam com nota 0. No empate o professor escolhia a primeira da lista ("cima-direita"), que às vezes é pular
para fora da pirâmide. Resultado: nível 2 completou só **1 de 30** etapas (o robô sozinho: 20 de 30).

**Problema 2 – medroso:** depois de corrigir o empate, ele passou a sobreviver muito mais (200 passos contra 69
do robô), mas andava devagar demais para pintar os cubos. No nível 2 cada cubo precisa de **dois** pulos, e a
minha medida de progresso só contava cubos que chegavam à cor final — o primeiro pulo valia zero.

### 2.2 Versão final: robô + olhar o perigo
Juntei o melhor dos dois:
- **vontade de pintar cubos** → vem da nota do robô de código (que é bom nisso);
- **enxergar o perigo** → 12 simulações curtas (3 passos) para cada jogada, medindo a **chance de perder uma vida**.

```
nota(jogada) = nota do robô  −  80 × chance de morrer nos próximos 3 passos
```

As notas viram porcentagens com uma **softmax** (a mesma conta que o NanoJev usa por dentro).

Horizonte curto (3 passos) foi a chave: com 8 passos, os **erros do próprio robô dentro da simulação**
contavam como "risco" da jogada avaliada, o que confundia o professor.

### 2.3 Resultado (40 etapas por nível, 3 vidas, regras Odyssey)
| Nível | Robô de código | Professor |
|---|---|---|
| 1 | 40/40 etapas | 40/40 (terminando com mais vidas) |
| 2 | 25/40 | **29/40** |
| 3 | 0/40 | 0/40 |

O nível 3 (pular num cubo pintado desfaz) ninguém completa ainda → por enquanto só **níveis 1 e 2** no treino.

---

## 3. Os exemplos (`qbert/gerar_dados.py` → `qbert/dados/`)

- O professor joga **320 partidas** (160 no nível 1, 160 no nível 2), em 14 processos paralelos (≈ 98 s).
- Em **cada decisão** grava uma linha JSON no **mesmo formato do autor**:
  - `state`: o texto que o modelo lê (pirâmide, cores, posição, inimigos, discos, histórico…);
  - `questions.action.criteria`: as 5 opções ("Hop down-left to [1,0]." etc.);
  - `teacher.native_probs`: as notas do professor.
- Em **10% das decisões** o professor faz uma jogada sorteada (proporcional às notas) em vez da melhor,
  para o modelo ver também situações "fora do caminho ideal" (o autor fazia o mesmo, com ε = 0,1).
- Divisão **por partida**: sementes terminadas em 0 → teste, em 1 → validação, o resto → treino. Assim o modelo
  nunca é avaliado numa partida que viu no treino.

Total: **30.336 decisões** — treino 24.079 · validação 3.243 · teste 3.014.

Cada situação vira ~**4.300 tokens** (≈ 860 por opção × 5 opções), porque o texto é longo
(regras + pirâmide + inimigos + últimas 8 jogadas). É o que mais pesa no tempo de treino.

---

## 4. O treino (`qbert/treinar_qbert.py`)

### 4.1 Por que um programa próprio
O programa de treino do autor (`scripts/train_unified_games.py`) só aceita os jogos dele (labirinto, minhoca,
tiro) e tem várias travas de validação. Em vez de desmontá-lo, escrevi um programa pequeno que **reaproveita
as peças do autor**:
- o **mesmo modelo** (`DecisionModel`: Qwen3-0.6B + "cabeça de escolha" que compara as opções);
- o **mesmo texto** de entrada (`State: … Question: … Candidate: … Decision:`), via `prepare_examples`;
- a **mesma conta de erro** (entropia cruzada entre as notas do professor e as do modelo).

### 4.2 Partida do modelo atual
Não começa do zero: carrega `checkpoints\NanoJev-unified\best.safetensors` (o que já joga minhoca/labirinto).

### 4.3 Não esquecer o que já sabe
Cada passo de treino usa **16 situações**: **12 de Q*bert + 4 dos jogos originais do autor**.
Sem isso, o modelo tende a "esquecer" a minhoca e o labirinto (fenômeno conhecido como esquecimento catastrófico).

### 4.4 Um passo de treino, por dentro
1. Sorteia 16 situações.
2. Para cada uma, o Qwen lê os textos das opções e produz uma "impressão" (1024 números) por opção.
3. A cabeça de escolha transforma isso numa nota por opção → softmax → porcentagens.
4. **Erro** = −Σ (nota do professor × log da nota do modelo). Quanto mais parecidas, menor.
5. **Retropropagação:** calcula como cada um dos ~441 milhões de números treináveis deveria mudar para diminuir o erro.
6. **AdamW** aplica essa mudança, bem pequena (taxa 0,00001 no Qwen, 0,0001 na cabeça).

### 4.5 Truques para caber nos 12 GB
| Truque | O que faz | Economia |
|---|---|---|
| bf16 nas contas | contas em meia precisão; pesos guardados em precisão cheia | ~metade da memória das contas |
| *gradient checkpointing* | não guarda tudo da leitura; recalcula quando precisa | vários GB, custa ~30% de tempo |
| embeddings congelados | a tabela de vocabulário (~155 M números) não é treinada | ~2 GB |
| lotes de até 8.192 tokens | processa as 16 situações em pedaços | mantém o pico baixo |

Teste de memória (10 passos): **pico 9,4 GB**, ~26 s por passo; concordância com o professor
**10% → 35%** em 10 passos; nos jogos originais ficou em 80% (não piorou).

### 4.6 A corrida principal
- 400 passos × 16 situações ≈ 6.400 exemplos vistos (≈ ¼ do conjunto de treino).
- Avaliação a cada 50 passos em 150 situações de validação do Q*bert e 150 dos jogos originais:
  - **concorda com o professor**: % de vezes em que a jogada preferida do modelo é a mesma do professor;
  - **ce**: o erro médio (quanto menor, melhor).
- A cada avaliação o modelo é salvo em `runs\qbert_v1\` (se algo falhar, não se perde tudo).
- Progresso ao vivo: `runs\qbert_v1\progresso.txt` (passo, % e tempo que falta).
- Tempo estimado: **~3 horas**. Durante o treino o servidor do NanoJev (porta 8765) fica desligado.

---

## 5. Depois do treino
1. Religar o servidor do modelo apontando para `runs\qbert_v1`.
2. Na página do Q*bert, "Quem joga: NanoJev" → comparar com o robô e com o NanoJev original.
3. Conferir se a minhoca e o labirinto continuam funcionando (página `ao_vivo`).

## 6. Limites honestos
- O aluno tende a ficar **no nível do professor**, não acima.
- O professor não completa o nível 3 → o NanoJev também não deve completar.
- 400 passos cobrem só ¼ dos exemplos; se os resultados forem promissores, dá para treinar mais.

---

## Diário do treino
(atualizado conforme o treino avança)

| Passo | Q*bert: concorda com o professor | Q*bert: erro (ce) | Originais: concorda | Originais: erro (ce) |
|---|---|---|---|---|
| 0 (antes) | 10% | 1,70 | 78% | 0,76 |
| 50 | 35% | 1,45 | 81% | 0,76 |
| 100 | 38% | 1,38 | 77% | 0,77 |
| 150 | 42% | 1,37 | 79% | 0,77 |
| 200 | 40% | 1,36 | 77% | 0,77 |
| 250 | 37% | 1,33 | 76% | 0,77 |
| 300 | 39% | 1,33 | 78% | 0,76 |
| 350 | 45% | 1,29 | 79% | 0,77 |
| 400 (final) | 38% | **1,27** | 77% | 0,76 |

Treino concluído em ~3h40 (com pausas). Placa entre 64 °C e 81 °C; C: estável. Modelo final: `runs\qbert_v1\best.safetensors`.

## Resultado jogando de verdade (`qbert\avaliar_jogando.py`)
20 etapas por nível, 3 vidas, regras Odyssey, sementes de **teste** (nunca vistas no treino), mesmas para todos.

| Nível | Quem joga | Etapas completas | Vidas sobrando | Cubos faltando | Passos | "Esperar" por partida |
|---|---|---|---|---|---|---|
| 1 | NanoJev original | 0/20 | 0 | 24,7 | 34 | 29 |
| 1 | **NanoJev treinado** | **0/20** | 0 | **7,5** | 63 | **6** |
| 1 | Robô de código | 20/20 | 2,35 | 0 | 44 | 2 |
| 1 | Professor | 20/20 | 2,35 | 0 | 46 | 3 |
| 2 | NanoJev original | 0/20 | 0 | 27,6 | 31 | 28 |
| 2 | **NanoJev treinado** | **0/20** | 0 | **16,0** | 71 | **6** |
| 2 | Robô de código | 10/20 | 0,7 | 3,0 | 91 | 2,5 |
| 2 | Professor | 16/20 | 1,55 | 1,0 | 133 | 6 |

**Do que o treinado morre** (`qbert\causas_morte.py`, 10 partidas no nível 1, 30 vidas perdidas):
Serpente/Coily **21** · Perigo Roxo 6 · Perigo Vermelho 3 · **queda da pirâmide 0** (nenhum pulo para fora sem disco).

**Não esqueceu os jogos originais** (página `ao_vivo`, modelo treinado): minhoca 3/3; labirinto 4/5 (antes 5/5).

### Leitura
- **Aprendeu muito do jogo:** parou de só "esperar" (29 → 6 por partida), aprendeu a geometria (nunca cai),
  pinta a maior parte dos cubos (faltam 7,5 de 28 no nível 1, contra 24,7 antes) e sobrevive o dobro do tempo.
- **Ainda não aprendeu a fugir da Serpente** — é o que mata 70% das vezes, e por isso não completa etapas.
- Por que provavelmente: (1) viu só ~⅙ dos exemplos; (2) as notas do professor são "suaves" (softmax com
  temperatura 8), então em situações de perigo a jogada segura às vezes ganha só por pouco; (3) situações com
  a Serpente por perto são uma minoria dos exemplos.

### Próximos passos sugeridos
1. **Continuar o treino** a partir de `runs\qbert_v1` (mais passos, restante dos exemplos).
2. **Reforçar as situações de perigo**: sortear com mais frequência as situações em que o professor vê risco
   de morrer (Serpente perto), e deixar as notas do professor mais "decididas" (temperatura menor).
3. Depois disso, liberar mais camadas ou partir para o "aprender jogando" (item 4 da conversa sobre melhorias).

---

# Treino v2 (1000 passos) — pedido do usuário

**Antes de mudar, medi os dados** (24.079 situações de treino):
- inimigo mortal a ≤ 1 pulo: 35% · a ≤ 2 pulos: 57% · Serpente a ≤ 2 pulos: 38%;
- situações "decisivas" (alguma jogada dentro da pirâmide perde ≥ 30 pontos de nota, ou disco): 62%;
- **professor foge pelo disco: só 453 (1,9%)**.
→ O perigo **não é raro** nos dados; raro é o **uso do disco**. Por isso não sorteei "mais perigo", e sim mais disco.

**O que muda em relação ao v1:**
| | v1 | v2 |
|---|---|---|
| Ponto de partida | modelo original | **modelo v1** (`runs\qbert_v1`) |
| Passos | 400 | **1000** |
| Notas do professor | temperatura 8 | **temperatura 4** (recalculadas de `metadata.values`, mais "decididas") |
| Situações de disco por passo | ~2% (sorteio normal) | **17% das de Q*bert** (2 de 12) |
| Camadas do Qwen treinando | 15–28 (14 camadas) | **9–28 (20 camadas)** |
| Avaliação | a cada 50 passos | a cada 100 passos (notas de validação também com temperatura 4) |

Obs.: o "erro (ce)" do v2 não é comparável ao do v1, porque as notas-alvo mudaram (temperatura 4).
Pausas de resfriamento e ponto de retomada a cada 10 passos continuam. Saída: `runs\qbert_v2`. Estimativa: ~10 h.

Comando (também serve para retomar se parar):
`.venv\Scripts\python.exe qbert\treinar_qbert.py --checkpoint runs\qbert_v1 --passos 1000 --avaliar-cada 100 --limite-avaliacao 150 --temp-professor 4 --fracao-disco 0.17 --congelar-camadas 8 --saida runs\qbert_v2`

**Pergunta do usuário: "ele sabe que o objetivo é mudar a cor de todos os blocos?"** O texto de cada passo diz
o objetivo ("turn every cube to the target color…"), mostra a cor de cada cubo e quantos faltam. Mas o modelo
não "entende" o objetivo como uma pessoa: aprende imitando as notas do professor, que prefere pintar cubos e evitar morrer.

## Diário do treino v2
| Passo | Q*bert: concorda com o professor | Q*bert: erro (ce, T=4) | Originais: concorda | Originais: erro |
|---|---|---|---|---|
| 0 (modelo v1) | 38% | 1,21 | 77% | 0,76 |
| 100 | 38% | 1,29 (piorou) | 79% | 0,77 |
| 200 | 43% | **1,20** (melhor que o início) | 79% | 0,76 |
| 300 | **47%** | **1,11** | 76% | 0,77 |
| 400 | **50%** | 1,15 | 80% | 0,76 |
| 500 | **52%** | 1,16 | 79% | 0,77 |
| 600 | 50% | **1,10** | **82%** | 0,76 |
| 700 | 50% | **1,10** | 79% | 0,76 |
| 800 | 48% | 1,19 (piorou) | 79% | 0,76 |
| 900 | 48% | **1,05** (o menor até agora) | 78% | 0,76 |
| 1000 (final) | **55%** | 1,05 | 79% | 0,76 |

Treino v2 concluído em ~11 h (91+ min de pausas para esfriar; placa até 81 °C; C: estável).
Modelos: final em `runs\qbert_v2`; cópia do passo 900 em `runs\qbert_v2_passo900`.

## Resultado do v2 jogando (mesmas 20 etapas de teste por nível)
| Nível | Quem joga | Etapas completas | Cubos faltando | Jogadas | "Esperar" |
|---|---|---|---|---|---|
| 1 | original | 0/20 | 24,7 | 34 | 29 |
| 1 | v1 | 0/20 | 7,5 | 63 | 6 |
| 1 | **v2** | **2/20** | **3,8** | 81 | 3 |
| 1 | professor | 20/20 | 0 | 46 | 3 |
| 2 | original | 0/20 | 27,6 | 31 | 28 |
| 2 | v1 | 0/20 | 16,0 | 71 | 6 |
| 2 | **v2** | 0/20 | **12,2** | 85 | 1,4 |
| 2 | professor | 16/20 | 1,0 | 133 | 6 |

**Primeiras etapas completas pelo NanoJev (2/20 no nível 1).**
Causas de morte do v2 (10 partidas, nível 1, 29 vidas): Serpente **20** · Perigo Vermelho 8 · Perigo Roxo 1 · queda 0.
→ A Serpente continua sendo o principal problema (69%, praticamente igual ao v1).
Minhoca/labirinto com o v2: 3/3 e 4/5 (iguais ao v1). Obs.: 0,3 s por jogada (v1: 0,15 s) — o PC estava mais ocupado.

### Sobre os bichinhos verdes (pergunta do usuário)
Já estão no jogo: **Manhoso** (verde que deixa cubos brancos/desfeitos; pegar elimina) e **Paralizante Verde**
(pegar congela os inimigos por 6 passos). O professor dá só um bônus pequeno (+15 no robô) por pegá-los,
então quase nunca pesa. Ideia: valorizar mais caçar o Manhoso e pegar o Paralizante (congelar a Serpente
dá tempo para pintar com segurança) — possível próximo ajuste do professor.

**Lição do passo 800:** o programa sobrescrevia `best.safetensors` a cada avaliação, então a melhor versão
(passos 600–700, erro 1,10) foi perdida. **Corrigido para os próximos treinos:** além do último, o treino agora
guarda a **melhor versão** (menor erro no Q*bert) em `runs\<nome>\melhor\` (com `passo.txt` dizendo qual passo).
Possível causa da piora: os 453 exemplos de disco já apareceram ~3× cada (começo de "decorar").

**Incidente: disco C: quase cheio (primeira corrida interrompida no passo 7).**
O treino transformava as 24 mil situações em tokens logo no início e guardava tudo na RAM em listas do Python
(~14 GB reservados). Para garantir essa memória, o Windows aumentou o arquivo de memória virtual
`C:\pagefile.sys` de 8,1 GB para 20,9 GB, e o C: ficou com **0,06 GB livres**. O treino foi parado.
**Correção:** agora só as 16 situações de cada passo viram tokens, na hora de usar (a RAM cai para poucos GB).
Um vigia (`qbert\vigiar_treino.ps1`) confere o C: a cada 30 s e **para o treino se ficar abaixo de 3 GB**.
A corrida foi reiniciada do zero.

**Incidente 2: o C: continuava caindo (segunda corrida parada pelo vigia no passo 1, com 1,64 GB livres).**
Causa real: a placa de 12 GB enchia (9,3 GB do treino + ~0,9 GB da tela + reserva do PyTorch). No Windows isso
não dá erro — o driver passa a usar a **RAM como extensão da placa**, e essa memória vem do pagefile no C:.
**Correções:**
- **Só a metade de cima do Qwen é treinada** (camadas 15–28 + cabeça de escolha); as 14 primeiras ficam congeladas.
  Menos gradientes e menos memória do otimizador. Treináveis: ~220 M de números (antes ~441 M).
- **Uma situação por vez** dentro do passo (lotes de até ~4.700 tokens).
- **Teto de 10,5 GB** para o PyTorch: se passar, dá erro claro em vez de invadir a RAM/pagefile.
- O usuário desligou a hibernação (`powercfg /h off`), liberando ~12,5 GB no C:.
Teste de 4 passos: placa no máximo 7,7 GB, pico do PyTorch 5,9 GB, C: estável (~19,7 GB livres), ~24 s/passo.
Terceira corrida iniciada.

**Temperatura da placa.** Durante o treino: **87 °C**, ventoinha 100%, 159 W de 170 W, uso 100%.
Limites informados pela placa (`nvidia-smi -q -d TEMPERATURE`): alvo 83 °C · máxima de operação 93 °C ·
redução automática 95 °C · desligamento 98 °C.
O vigia agora lê a temperatura a cada 15 s, mostra junto com o progresso, avisa ≥ 90 °C e **para o treino em 93 °C**.
Opção para esfriar (precisa de administrador): `nvidia-smi -pl 140` limita a potência a 140 W
(tipicamente alguns graus a menos, com treino um pouco mais lento); volta ao normal com `nvidia-smi -pl 170`.

**Pegando leve com a placa (pedido do usuário; pode rodar a noite toda).** O treino ganhou pausas automáticas:
depois de cada passo, se a placa passou de **75 °C**, ele espera ela baixar para **68 °C**; no meio do passo,
se passar de **80 °C**, também espera. A terceira corrida foi parada no passo 25 e reiniciada do zero com isso
(quarta corrida). Tempo previsto: mais longo (estimado 4–6 h), atualizado em `progresso.txt`.

**Pausas também nas avaliações** (faltavam; a placa foi a 84 °C na avaliação inicial) → quinta corrida,
já pulando a avaliação inicial (valores conhecidos: 10% / 78%).

**Ponto de retomada a cada 10 passos (pedido do usuário).** O treino salva `runs\qbert_v1\retomar.pt`
(pesos treináveis + estado do otimizador AdamW + estado do sorteio + histórico + tempo) a cada 10 passos,
trocando o arquivo de uma vez (nunca fica pela metade). Rodando o mesmo comando de novo, ele continua do
último ponto. Testado: interrompido no passo 2, retomou no 3 e terminou no 4. Ao terminar, o arquivo é apagado.
Com isso, uma interrupção perde no máximo ~10 passos (~5 min). **Sexta corrida (definitiva) iniciada.**
Comando para retomar, se precisar:
`.venv\Scripts\python.exe qbert\treinar_qbert.py --passos 400 --avaliar-cada 50 --limite-avaliacao 150 --pular-avaliacao-inicial --saida runs\qbert_v1`
