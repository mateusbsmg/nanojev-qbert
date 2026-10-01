"""Extrai fotos próximas (ex.: a cada 0,25 s) de um trecho do vídeo já baixado e monta mosaicos.

Uso: .venv\\Scripts\\python.exe qbert\\zoom_video.py NOME INICIO FIM [PASSO]
Saída: qbert\\referencias_odyssey\\video\\zoom_NOME\\
"""
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg

VIDEO = Path(__file__).resolve().parent / "referencias_odyssey" / "video" / "video.mp4"

name, start, end = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
step = float(sys.argv[4]) if len(sys.argv) > 4 else 0.25
ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
out = VIDEO.parent / f"zoom_{name}"
out.mkdir(exist_ok=True)
for old in out.glob("*.png"):
    old.unlink()
subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-ss", str(start), "-to", str(end), "-i", str(VIDEO),
                "-vf", f"fps=1/{step},crop=ih*4/3:ih,scale=400:300", str(out / "z_%03d.png")], check=True)
shots = sorted(out.glob("z_*.png"))
for g in range(0, len(shots), 12):
    group = shots[g:g + 12]
    inputs, filters = [], []
    for i in range(12):
        if i < len(group):
            inputs += ["-i", str(group[i])]
        else:
            inputs += ["-f", "lavfi", "-i", "color=c=gray:s=400x300:d=1"]
        filters.append(f"[{i}:v]scale=400:300[v{i}]")
    layout = "|".join(f"{(k % 4) * 400}_{(k // 4) * 300}" for k in range(12))
    graph = ";".join(filters) + ";" + "".join(f"[v{k}]" for k in range(12)) + f"xstack=inputs=12:layout={layout}"
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", *inputs, "-filter_complex", graph, "-frames:v", "1",
                    str(out / f"mosaico_{g // 12 + 1}.png")], check=True)
print(f"{len(shots)} fotos de {start}s a {end}s (passo {step}s) em {out}")
