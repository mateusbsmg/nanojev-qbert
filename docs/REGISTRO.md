# Registro — Etapa 2

Início: 2026-09-23
Pasta do projeto: `D:\Mateus\NanoJev`
Registro anterior (instalação, jogo ao vivo): `D:\Mateus\NanoJev\registro\INSTALACAO.md`

## Ponto de partida

- NanoJev instalado e rodando na RTX 3060 (12 GB), ambiente em `D:\Mateus\NanoJev\.venv` (Python 3.14.7, torch 2.14.0+cu126).
- Jogo ao vivo em `D:\Mateus\NanoJev\ao_vivo` (porta 8766), modelo na porta 8765.
- Última questão em aberto: **dá para treinar o modelo localmente?**
  Estimativa de memória para treinar como o autor: ~12–13 GB (não cabe ou fica no limite em 12 GB).
  Formas de caber: reduzir `--max-microbatch-tokens`; congelar a parte de vocabulário (embeddings); otimizador de 8 bits.
  Proposta: teste de ~15 min (validar dados + 10 rodadas de treino) para medir memória e tempo.

## Conversa

### Pedido: reproduzir o Q*bert
O usuário mandou uma captura de tela do Q*bert (versão de videogame caseiro, fundo preto, pirâmide de cubos azuis)
e perguntou se dá para reproduzir fielmente o jogo, com regras bem definidas.

Leitura da imagem:
- Pirâmide de 7 fileiras (1+2+…+7 = 28 cubos), cubos azuis; o cubo sob o Q*bert já está bege (cor-alvo).
- Quadrado bege no canto superior esquerdo: indicador da cor-alvo.
- "5" à esquerda: provavelmente vidas. "1" à direita: rodada/nível.
- Dois discos azul-claros nas laterais (fuga do Coily).
- Bola rosa/roxa na pirâmide (inimigo; provavelmente a bola que vira o Coily).
- Barra inferior: "0000→?????? 0001" (placar; significado exato não confirmado).

Resposta: dá para fazer, com ressalvas de fidelidade (ver abaixo); perguntas enviadas sobre versão e finalidade.

**Decisões do usuário:**
- Finalidade: **para o NanoJev jogar**.
- Regras: **clássicas do arcade**.

**Plano:** pasta `D:\Mateus\NanoJev\qbert\`
- `qbert_env.py`: o jogo em Python, por turnos (1 passo = 1 pulo do Q*bert), com semente (partidas repetíveis),
  estado descrito em texto no estilo dos jogos do autor.
- `servidor_qbert.py` + `index.html`: página ao vivo (porta 8767) com 3 jogadores: NanoJev, robô de código e você (teclado).
- `testar_qbert.py`: testes automáticos das regras.

### Q*bert — primeira versão pronta (regras do arcade, por turnos)
Arquivos em `D:\Mateus\NanoJev\qbert\`: `qbert_env.py`, `servidor_qbert.py`, `index.html`, `testar_qbert.py`.
Página: `http://127.0.0.1:8767` (jogadores: NanoJev, robô de código, você no teclado).

- Testes de regras: todos passaram (geometria, quedas, cores dos níveis 1–3, disco, inimigos, congelamento,
  troca de lugar, Coily, fim de rodada) + 300 partidas aleatórias sem violar invariantes.
- Robô de código (1ª rodada): nível 1 vence 199/200; nível 2 vence 118/200; níveis 3 e 5 perde sempre
  (o robô é simples demais para essas regras).
- **NanoJev sem treino em Q*bert** (semente 42): perdeu as 3 vidas em 10 passos, escolheu "esperar" 9 vezes;
  ~0,36 s por decisão. Esperado: ele nunca viu esse jogo.

### Pesquisa: Q*bert de Odyssey (imagens e regras reais)
Pergunta do usuário: se eu tinha visto imagens reais da versão de Odyssey. Resposta: não tinha; pesquisei depois.
- A imagem enviada é **da versão de Odyssey 2** (Parker Brothers; no Brasil, Philips, 1984, código 06 AV 9485).
  Capturas reais no Internet Archive batem com ela. Cópias salvas em `qbert\referencias_odyssey\`
  (inclui a capa do manual brasileiro "Regras do Jogo").
- Manual brasileiro (resumo via experienciaodyssey.com.br): **diferente do arcade** em vários pontos:
  - começa com **7 Q*berts**; vidas mostradas à esquerda; cor-alvo indicada à esquerda da pirâmide;
  - **9 níveis × 4 etapas**; discos variam em número e posição conforme nível/etapa;
  - personagens: Perigo Vermelho, Perigo Roxo (vira a Serpente na base), Serpente, Manhoso (desfaz cores;
    pode ser pego), Pula-Pula (surge na base, anda para os lados e para cima), Paralizante Verde (congela os outros);
  - pontuação: **1 ponto por cubo**, **25 pela Serpente**, **50 por etapa**; **Q*bert extra a cada 300 pontos**;
  - placar inferior: recorde à esquerda; "??????" é o lugar do nome de quem fez o recorde;
  - ao perder: reaparece no topo (se caiu) ou no cubo onde foi pego.
  - **não encontrado** no manual: regra de cores por nível, controles, velocidades, pontos do Manhoso.
- As 23 capturas reais (todas em `qbert\referencias_odyssey\`) mostram:
  - a **figurinha rosa pequena** da imagem do usuário é o **Perigo Roxo** (o "ovo" que vira cobra na base);
  - a **Serpente/Coily** aparece como uma **cobra rosa enrolada** (capturas 07, 08, 14, 15, 18–22);
  - os "Perigos" são desenhados como **gotinhas com ponta**, não bolas redondas (ex.: gota vermelha na captura 08).
  - A página atual desenha bolinhas redondas → não bate com o visual do Odyssey.
- YouTube: existe o vídeo "Classic Game Room – Q*BERT review for Magnavox Odyssey 2"
  (youtube.com/watch?v=RmaFvqQwNQI), mas o Claude não consegue assistir vídeos.
- Proposta ao usuário: aplicar as regras do manual do Odyssey + visual igual às capturas.
### Análise do vídeo do YouTube (VGDB, 7min29s)
Vídeo: https://www.youtube.com/watch?v=xpwqv0X0pDk ("Q*Bert - Odyssey 2 - VGDB"). Baixado só o vídeo (720p, sem som)
com `yt-dlp` e extraídas 224 fotos (1 a cada 2 s) com `qbert\extrair_quadros.py`.
Arquivos: `qbert\referencias_odyssey\video\` (vídeo, `quadros\q_NNNN.png`, `quadros\mosaico_NN.png`).
A partida vai do nível 1 ao 4 e termina em fim de jogo com 1371 pontos (recorde gravado com o nome "VGDB").

**Tela (confirmado):**
- Quadrado no canto superior esquerdo = cor-alvo. Número no canto superior direito = **etapa** (1–4) do nível.
- Número à esquerda = Q*berts **de reserva** (começa em 6 = 7 no total com o que está em jogo; chega a 0 e ainda joga).
- Placar de baixo: `recorde → nome (??????) pontos atuais`. A moldura muda de cor a cada nível
  (nível 1 dourada, 2 verde, 3 roxa, 4 azul). Tela "LEVEL N" entre os níveis.
- O cubo do topo é a peça pequena no alto do centro; pirâmide = 28 cubos.

**Pontuação (confirmado):**
- **+1 ponto por cubo** que muda de cor (o placar sobe de 1 em 1).
- **Q*bert extra a cada 300 pontos** (reservas sobem em ~300, ~600, ~900 e ~1200).
- Fim de etapa: o placar salta **+30 a +40** entre fotos (o manual diz 50; pode ser contagem em andamento). Incerto.

**Regras de cor observadas:**
| Nível | Comportamento visto | Exemplos (inicial → [intermediária] → alvo) |
|---|---|---|
| 1 | 1 pulo pinta | azul→amarelo; cinza→azul; branco→ciano; amarelo→ciano |
| 2 | 2 pulos (cor intermediária) | azul→cinza→amarelo; branco→amarelo→ciano; amarelo→cinza→azul |
| 3 | alterna (pular no alvo desfaz) | azul↔amarelo; amarelo↔ciano |
| 4 | intermediária + alvo que volta | cinza→amarelo→ciano, com cubos voltando |
(Batem com a regra do arcade para os níveis 1–4.)

**Personagens vistos:**
- Perigo Vermelho: gotinha vermelha que desce.
- Perigo Roxo: gotinha rosa pequena que desce → vira **cobra rosa enrolada** (Serpente) e persegue.
- Verde pequeno (Paralizante Verde); em algumas etapas aparecem **cubos brancos** perto de um personagem verde
  → provavelmente o Manhoso mudando cores (não confirmado).
- Figura rosa fora da pirâmide, nas pontas de baixo (ex.: 1302 pontos) → provavelmente o Pula-Pula.
- Discos: mudam de cor e quantidade por nível (2 no início; 4 no nível 3).
- Várias cobras ou ovos ao mesmo tempo (até 3 figuras rosa), o que difere do arcade (1 Coily por vez).

### Confirmações com fotos próximas do vídeo (`qbert\zoom_video.py`, pastas `video\zoom_*`)
- **Bônus de etapa = +50** (placar sobe de 10 em 10: 28→78, aos ~40–43 s). Bate com o manual.
- No início de cada etapa o Q*bert pousa no topo e ganha **+1** (78→79).
- **Manhoso** = figurinha **verde** que desce deixando cubos **brancos**; o Q*bert pulando no branco retoma
  o ciclo normal (+1). (~244–250 s)
- **Duas Serpentes ao mesmo tempo** confirmado; ao perder um Q*bert todos os inimigos somem. (~170–176 s)
- Tela "LEVEL 4" mostra uma demonstração: cinza→amarelo→ciano (regra do nível 4).
- **Pula-Pula não identificado** no vídeo (as figuras rosa nas pontas eram Serpentes).

### Modo "Odyssey" implementado
- `qbert_env.py`: parâmetro `ruleset` ("odyssey" padrão, ou "arcade").
  Odyssey: 7 Q*berts; +1/cubo; +1 ao pousar no topo no início da etapa; +25 Serpente no disco; +50 por etapa;
  Q*bert extra a cada 300; até 3 Serpentes/Perigos Roxos juntos; Manhoso deixa cubo branco; 9 níveis × 4 etapas
  (termina com "game_completed"); discos: 2 (nível 1), 3 (nível 2), 4 (níveis 3+).
  Pegar Paralizante/Manhoso não dá pontos (manual não cita).
- `servidor_qbert.py`: escolha de regras; recorde com nome de até 6 letras em `qbert\recorde.json`
  (como o "0000→??????" do Odyssey); `?auto=bot|model` inicia jogando sozinho.
- `index.html`: visual copiado das capturas: fundo escuro, cor-alvo à esquerda, etapa à direita, reservas,
  pirâmide nas mesmas posições, cores de cada etapa dos níveis 1–4 tiradas do vídeo, moldura do placar
  por nível (dourada/verde/roxa/azul), discos com a cor da etapa, figuras em pixel (Q*bert, gotinhas, Serpente
  enrolada, Manhoso verde, Pula-Pula), tela "LEVEL N" entre níveis, nomes em português dos personagens.
- `iniciar_qbert.bat`: liga o modelo e o Q*bert e abre `http://127.0.0.1:8767`.

**Testes:** todas as regras passaram (arcade + Odyssey: 7 vidas, +1/cubo, +1 no topo, +50 etapa, vida extra aos 300,
+25 Serpente, Manhoso branco, fim após nível 9). 300 partidas aleatórias nos dois modos sem violar regras.
Robô de código no Odyssey (partida inteira): chegou ao nível 2–3 com 643–1236 pontos (o jogador do vídeo fez 1371 no nível 4).
NanoJev no Odyssey: perdeu em 22 passos, esperando quase sempre (nunca treinou Q*bert).
Fotos da página conferidas: tela inicial e meio de partida batem com as capturas reais.

**Ainda aproximado:** regras de cor e paletas dos níveis 5–9; etapas 2–4 do nível 4; comportamento exato do Pula-Pula;
velocidades (o jogo é por turnos); tabela de quais inimigos aparecem e quando; posição dos discos.

### Ajustes pedidos pelo usuário (página do Q*bert)
- Desenho do Q*bert voltou ao anterior (corpo redondo, focinho, olhos, pés).
- **Animação de pulo**: Q*bert e inimigos fazem um arco de um cubo ao outro (~0,26 s); o cubo muda de cor ao pousar.
  Usa temporizador comum (não `requestAnimationFrame`) para não travar com a aba em segundo plano.
- O cubo sob o Q*bert não some mais.
- "Parado até perder": era o **NanoJev**, que escolhe "esperar" quase sempre (sem treino em Q*bert).
  Agora o padrão é o **Robô de código**, o NanoJev aparece como "sem treino em Q*bert" e a tela avisa
  quando ele escolhe esperar.

### IMPORTANTE: quem jogou o Q*bert
- Na imagem que o usuário mandou (etapa completa, 128 pontos), **quem jogou foi o ROBÔ DE CÓDIGO**
  (`bot_scores` em `qbert_env.py`: regras simples em Python, sem IA), **não o NanoJev**.
  Sinais na tela: "Quem joga: Robô de código", "Notas do robô", tempo por decisão 0.00 s.
- O **NanoJev não sabe jogar Q*bert**: nos testes perdeu em 10–22 passos, escolhendo "esperar" quase sempre.
- Na minhoca e no labirinto (`ao_vivo`, porta 8766) quem jogou foi **de fato o NanoJev** (modelo na porta 8765).

### Disco (plataforma) — conferido no vídeo (~30–38 s, `video\zoom_disco`)
- Q*bert pula do cubo da ponta para o disco, voa até o topo, o disco some.
- **Os outros inimigos continuam** na pirâmide e a Serpente que estava longe não caiu (placar sem +25).
- Correção no modo Odyssey: só a Serpente a até 2 pulos do cubo de onde o Q*bert saiu segue e cai (+25);
  os demais inimigos ficam. (Arcade continua limpando todos.) Testes atualizados e passando.
- Na partida da imagem (semente 1) o disco da esquerda já tinha sido usado no passo 17
  (duas Serpentes caíram, +50), por isso não aparecia no fim.

### Próximos níveis
- A partida parava no fim da etapa porque "Continuar nas próximas etapas" vinha desmarcado.
  Agora vem **marcado** e o limite de passos padrão é **3000**.

### Mais ajustes visuais pedidos
- **Voo no disco** (como no vídeo): Q*bert pula para o disco, o disco sobe suave com ele até acima do topo,
  e ele pula no cubo do topo (~1,5 s). Conferido com sequência de fotos da página.
- **Sem pulinho no lugar**: ao "esperar", o Q*bert fica parado (no jogo original não existe pulo no mesmo bloco).
- **Tela "SELECT GAME"** colorida no início de cada partida, com as cores de cada letra da captura original
  (S verde, E bege, L azul, E rosa, C ciano, T branco, G vermelho, A bege, M dourado, E azul), depois "LEVEL N".
- **Fonte pixelada** (Press Start 2P, do Google Fonts; sem internet usa Courier) no placar e nos números;
  a seta "→" do placar é desenhada à mão. Números aumentados para ficar do tamanho do original.
  Comparação lado a lado com `screenshot_00.png` e `screenshot_03.png`: praticamente iguais.

### Quedas e fim de etapa
- Vídeo, fotos de 0,25 s em cada perda de vida (`video\zoom_perda1..8`):
  - **Queda confirmada (~19–22 s)**: Q*bert na ponta esquerda da 4ª fileira pula para fora (sem disco), cai por fora
    da pirâmide passando na frente dos cubos, some embaixo e reaparece no topo; reservas 6→5; placar não muda.
  - Perda por inimigo (~180 s): reaparece no mesmo cubo onde foi pego.
  - As outras janelas escolhidas não pegaram o momento exato da perda.
- A regra já existia no jogo; faltava a animação. Agora: pulo para fora → queda pela lateral até sumir → reaparece no topo.
  As reservas só diminuem quando ele reaparece. Conferido com fotos (`?jogadas=down_left,down_left,up_left`).
- **Último pulo da etapa** não aparecia (a tela pulava direto para a etapa seguinte). Agora: último pulo →
  pirâmide completa → bônus contando de 10 em 10 (como no vídeo) → pausa → próxima etapa. Conferido com fotos
  (53 → 63 → 83 → 103 → etapa 2).
- Parâmetro de teste na página: `?jogadas=a,b,c` joga essas jogadas como "você".

### Aviso "quem está jogando" na página
Pergunta repetida do usuário: é ou não o NanoJev jogando? Resposta: **não**, por padrão é o **robô de código**.
Adicionado um quadro grande acima do jogo que diz quem joga: ROBÔ DE CÓDIGO (não é o NanoJev nem IA),
NanoJev (IA na placa de vídeo, sem treino em Q*bert) ou VOCÊ.

- Quadro "quem está jogando" removido a pedido do usuário (já entendeu).

### Como fazer o NanoJev aprender Q*bert (plano proposto, ainda não iniciado)
Formato do treino do autor (`unified\hard\train.jsonl`): cada linha tem `state` (texto), `questions.action`
(type "choice", instructions, criteria = opções) e `teacher.native_probs` = notas do professor por opção
(no autor o professor era o Jev, serviço pago). Metadados indicam "conditioned_action" + "continuation_policy".
1. **Professor** de graça: robô + simulação (para cada opção, jogar várias continuações e medir a chance de
   completar a etapa sem morrer → notas).
2. **Gerar exemplos**: milhares de situações dos níveis 1–3, no formato do autor, com treino/validação/teste.
3. **Teste de memória/tempo** na RTX 3060 (~15 min).
4. **Treinar** a partir do `best.safetensors`, misturando Q*bert com parte dos dados originais (não esquecer).
5. **Comparar** na página: NanoJev antes × depois × robô.
Expectativa: aprende ~no nível do professor; custo zero; riscos: memória de 12 GB e ajustes dos scripts (Linux).
Proposta: começar pelas etapas 1 e 3.

### Treinamento do NanoJev no Q*bert — iniciado
Detalhes completos em **`etapa2\TREINAMENTO_QBERT.md`** (documento próprio, pedido do usuário).
- Professor (`qbert\professor.py`): nota do robô − 80 × risco de morrer em 3 passos (12 simulações).
  Nível 1: 40/40 (robô 40/40); nível 2: 29/40 (robô 25/40); nível 3: 0/40 (ambos).
- Dados (`qbert\gerar_dados.py` → `qbert\dados\`): 30.336 decisões dos níveis 1–2 (treino 24.079).
- Treino próprio (`qbert\treinar_qbert.py`), reaproveitando modelo/texto/perda do autor.
  Teste de 10 passos: pico 9,4 GB, ~26 s/passo, concordância com o professor 10% → 35%, originais sem piora.
- Corrida principal: 400 passos → `runs\qbert_v1\` (progresso em `runs\qbert_v1\progresso.txt`), ~3 h.
  Servidor do modelo (8765) desligado durante o treino.

### Treinamento concluído (resumo; detalhes em `etapa2\TREINAMENTO_QBERT.md`)
- 400 passos, ~3h40 com pausas de resfriamento; modelo em `runs\qbert_v1`.
- Jogando (20 etapas/nível, teste): treinado 0/20 nos níveis 1 e 2 (original também 0/20), mas pinta muito mais
  (faltam 7,5 cubos vs 24,7), espera muito menos (6 vs 29), nunca cai da pirâmide; morre sobretudo para a Serpente (21/30).
  Robô: 20/20 e 10/20; professor: 20/20 e 16/20.
- Minhoca/labirinto continuam funcionando (3/3 e 4/5).
- Servidor do modelo (8765) agora está rodando o **modelo treinado** (`runs\qbert_v1`).

### Treino v2 (1000 passos, ~11 h) — detalhes em `etapa2\TREINAMENTO_QBERT.md`
- A partir do v1; notas do professor mais decididas (T=4); 17% das situações de Q*bert com fuga pelo disco;
  20 de 28 camadas treinando. Concordância com o professor: 38% → 55%; originais estáveis (~79%).
- Jogando: **2/20 etapas completas no nível 1** (primeiras do NanoJev; faltam 3,8 cubos em média), 0/20 no nível 2
  (faltam 12,2). Serpente ainda causa ~69% das mortes. Minhoca 3/3, labirinto 4/5.
- Servidor do modelo (8765) agora roda o **v2** (`runs\qbert_v2`). Cópia do passo 900 em `runs\qbert_v2_passo900`.
- Correção no programa: próximos treinos guardam a melhor versão em `runs\<nome>\melhor\`.

### Página do Q*bert: rótulo e gravação de vídeo
- "NanoJev (modelo, sem treino em Q*bert)" → **"NanoJev (modelo)"** (já foi treinado).
- Botão **"⏺ Gravar 3 min"**: grava só o quadro do jogo (MediaRecorder no próprio canvas, 30 fps) e baixa
  `qbert_AAAAMMDD_HHMM.webm` em Downloads; clicar de novo para antes dos 3 min.
- `qbert\converter_video.py`: converte o .webm mais recente de Downloads para **MP4** (H.264) em `qbert\videos\`
  (`--escala 2` aumenta sem borrar). Testado com um vídeo de exemplo.
- **Ajuste pedido:** gravar a área toda (título, controles, jogo e painel), não só o jogo. Agora o botão usa a
  captura da própria aba do Chrome (o Chrome pede permissão: "Compartilhar esta aba") e recorta o vídeo só na
  área `#gravavel` (do título até o fim do jogo/painel; ficam de fora as seções de texto de baixo).
- **"Vídeo bugado":** o arquivo estava bom; o defeito era do player do Windows tocando o `.webm` do Chrome
  (VP9 sem índice). Convertido para MP4, ficou limpo (conferido quadro a quadro). A gravação do teste tinha
  só 11 s (parou em 2:50 — provavelmente clicado ⏹ ou "Parar de compartilhar").
- **Agora o MP4 sai pronto:** ao terminar, a página envia o vídeo ao servidor do Q*bert (`POST /api/video`),
  que salva e converte para MP4 em `D:\Mateus\NanoJev\qbert\videos\` e mostra o caminho na página.
  Se falhar, baixa o `.webm` como antes. Testado com o vídeo do usuário.

### Mini artigo para o LinkedIn
Pedido: 500–750 palavras, introdução/desenvolvimento/conclusão, levemente empolgante, frases curtas separadas
por linha em branco, negrito em palavras-chave, ~5 hashtags; contar o que foi feito (NanoJev, PC, esforço,
Odyssey/Q*bert, treino — é fine-tuning —, resultado).
O .md que o usuário disse ter criado não foi encontrado; texto salvo em **`etapa2\ARTIGO_LINKEDIN.md`** (615 palavras).
- Ajustes: sem linhas em branco entre frases; `---` virou linha em branco.
- Limite do LinkedIn (3.000 caracteres): texto enxugado de 3.347 para **2.912 caracteres** (518 palavras),
  cortando uma frase de detalhes, o "menos de um sexto da memória" e encurtando frases longas.
- **`etapa2\ARTIGO_LINKEDIN.docx`** gerado com negrito real (40 trechos), título em destaque, uma frase por linha.
  Conferido por dentro do arquivo (sem renderizador no PC).
- Usuário: negrito do .docx não funcionou ao colar; pediu HTML (mostrou que `<em>` funciona na página dele).
  Gerado **`etapa2\ARTIGO_LINKEDIN.html`**: uma frase por `<p>`, linhas em branco como `<p><br></p>`,
  40 trechos em `<strong>`, sem `**` sobrando. Conferido com captura do navegador.
- `<strong>` também não funcionou no post. O `<em>` "em negrito" da imagem do usuário era a página de resultados do
  **Google** (que estiliza `<em>` como negrito); o post do LinkedIn não aceita nenhuma formatação.
- Solução: **letras Unicode em negrito** (𝗮𝘀𝘀𝗶𝗺), que são caracteres e sobrevivem ao colar.
  Script `etapa2\gerar_post_linkedin.py` → **`etapa2\ARTIGO_LINKEDIN_post.txt`**.
  Cada letra em negrito conta 2 (acentuada, 3) → negrito só nos 6 títulos de seção + NanoJev, Q*bert, fine-tuning.
  Texto enxugado mais um pouco (511 palavras, 2.856 caracteres normais) → **2.973** na contagem mais rigorosa.
  HTML atualizado; o .docx não foi atualizado porque estava aberto no Word.
- A pedido do usuário, o HTML passou a usar **`<em>`** no lugar de `<strong>` (40 trechos), com CSS
  `em { font-style: normal; font-weight: bold; }` para aparecer em negrito na página.
- **Revisão do usuário aplicada** a todos os arquivos (.md, _post.txt, .html, .docx): saem as frases sobre o autor
  treinar minhoca/labirinto, "Eu quis ir além" e "para não esquecer a minhoca…"; "fugindo dos inimigos";
  "com apoio do Claude Code"; "Os inimigos ainda o pega…"; hashtag #NanoJev. Negritos mantidos nos trechos que ficaram.
  Resultado: 483 palavras, 2.714 caracteres (post com letras em negrito: 2.831 na contagem rigorosa).
- Usuário: "ainda não funcionou" (trouxe orientação de outra IA: usar gerador de negrito Unicode tipo YayText).
  Solução local: **`etapa2\negrito_linkedin.html`** (gerado por `etapa2\gerar_conversor_negrito.py`), um conversor
  offline: selecionar trecho → Negrito/Tirar negrito; opção de acentos (padrão: letra acentuada fica normal,
  mais seguro em qualquer celular); contador igual ao do LinkedIn (limite 3.000); botão Copiar.
  Já abre com o artigo, negrito só nos títulos de seção e "fine-tuning" (NanoJev/Q*bert sem negrito para a busca
  do LinkedIn reconhecer). Conferido com captura: 2.809/3.000.

- Fontes: archive.org/details/Q-bert-MagnavoxOdyssey2 · experienciaodyssey.com.br/qbert-jogar ·
  videogamecritic.com/oddpr.htm (todos os inimigos e discos presentes; Pula-Pula aparece sem aviso na base).

