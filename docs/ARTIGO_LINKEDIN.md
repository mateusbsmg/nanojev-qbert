# Ensinei uma IA a jogar Q*bert do Odyssey. No meu PC de casa.
Tudo começou com uma pergunta simples.
Será que dá para rodar e **treinar uma IA de verdade** no computador de casa, sem pagar nada?
Encontrei o **NanoJev**, um projeto de código aberto no GitHub.
Ele é uma réplica em miniatura do **Jev**, um modelo que toma decisões.
Não é um chatbot.
Ele recebe uma situação e as opções, e devolve a **chance de cada uma ser a melhor**.
Por baixo, roda o **Qwen3-0.6B**, com 600 milhões de parâmetros.

**O desafio: o Q*bert do Odyssey**
Quem teve um **Philips Odyssey** nos anos 80 deve lembrar.
O **Q*bert** era aquele bichinho laranja pulando numa pirâmide de cubos, fugindo dos inimigos.
O objetivo é simples: pular em todos os cubos até mudarem de cor.
A execução, nem tanto.
Recriei o jogo do zero em **Python**, fiel à versão brasileira, com apoio do Claude Code.
Usei as **regras do manual original** da Philips.
Analisei **capturas de tela reais** do cartucho.
E quebrei um vídeo de gameplay em **224 imagens** para estudar quadro a quadro.

**A máquina**
Nada de servidor na nuvem.
Uma **RTX 3060 com 12 GB**, um **Ryzen 7 5700G** e 32 GB de RAM.
O autor do NanoJev usou uma **A100 de 80 GB**, placa de data center.

**O treino: sim, é fine-tuning**
O NanoJev original **nunca tinha visto Q*bert**.
No primeiro teste, ele simplesmente **ficava parado**, até ser pego.
Então fiz um **fine-tuning**: parti do modelo pronto e ajustei os pesos para o Q*bert.
Mas quem ensina?
Criei um **"professor"**: um programa que, a cada jogada, simula o futuro e mede o risco de cada movimento.
O professor jogou centenas de partidas e gerou **mais de 30 mil exemplos**.
O NanoJev aprendeu **imitando as notas do professor**.

**O esforço que ninguém vê**
Nada saiu de primeira.
Foram dois treinos: um de **400 passos** e outro de **1.000 passos**.
Somando, cerca de **15 horas** de placa trabalhando, madrugada adentro.
Custo de nuvem: **zero**.

**O resultado**
Antes do treino: **0 fases completas**. Ele esperava em 29 de cada 34 jogadas.
Depois do treino:
✅ **Completou fases pela primeira vez**
✅ Pinta, em média, **24 dos 28 cubos**
✅ **Nunca pula para fora** da pirâmide
✅ Concordância com o professor subiu de **10% para 55%**
Ainda não é um campeão.
Os inimigos ainda o pega na reta final, na maioria das vezes.
Mas a evolução é clara.

**O que fica**
Três lições.
Primeira: **IA de verdade cabe em casa**. Não precisa de data center.
Segunda: **o professor importa mais que o tamanho do treino**. O aluno aprende até onde quem ensina consegue ir.
Terceira: **o trabalho invisível é o que faz dar certo**. É engenharia, não mágica.
O próximo passo: um professor mais esperto contra a Serpente.
E, quem sabe, ver o Q*bert zerar a pirâmide sozinho.
Alguém aí também jogava Odyssey?

#InteligenciaArtificial #MachineLearning #FineTuning #RetroGaming #Odyssey #NanoJev
