@echo off
rem Liga o modelo NanoJev (porta 8765) e o Q*bert (porta 8767), depois abre o navegador.
cd /d "%~dp0.."
set HF_HOME=D:\Mateus\.hf-cache
start "NanoJev - modelo" .venv\Scripts\python.exe scripts\serve_decisions.py --checkpoint-dir checkpoints\NanoJev-unified --web-root web --port 8765 --disable-native-triton
start "NanoJev - Q*bert" .venv\Scripts\python.exe qbert\servidor_qbert.py
echo Carregando o modelo...
:espera
timeout /t 2 /nobreak >nul
powershell -NoProfile -Command "try { (Invoke-RestMethod http://127.0.0.1:8765/api/health -TimeoutSec 2).ready } catch { exit 1 }" >nul 2>&1 || goto espera
start http://127.0.0.1:8767
echo Pronto. Para desligar, feche as duas janelas "NanoJev".
