"""Baixa um vídeo do YouTube e extrai fotos (quadros) em intervalos regulares.

Uso:
    .venv\\Scripts\\python.exe qbert\\extrair_quadros.py URL [--cada 2] [--inicio 0] [--fim 0]
Saída: qbert\\referencias_odyssey\\video\\ (vídeo) e ...\\video\\quadros\\ (imagens + mosaicos).
"""
import argparse
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg

HERE = Path(__file__).resolve().parent
OUT = HERE / "referencias_odyssey" / "video"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("url")
    p.add_argument("--cada", type=float, default=2.0, help="segundos entre fotos")
    p.add_argument("--inicio", type=float, default=0.0)
    p.add_argument("--fim", type=float, default=0.0, help="0 = até o fim")
    a = p.parse_args()
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    OUT.mkdir(parents=True, exist_ok=True)

    video = OUT / "video.mp4"
    if not video.exists():
        # Só vídeo (sem som), até 720p; usa o Node instalado para o YouTube.
        subprocess.run([sys.executable, "-m", "yt_dlp", "--js-runtimes", "node", "-f",
                        "bv*[height<=720][ext=mp4]/bv*[height<=720]/best", "--ffmpeg-location", ffmpeg,
                        "-o", str(video), a.url], check=True)

    frames = OUT / "quadros"
    frames.mkdir(exist_ok=True)
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-ss", str(a.inicio)]
    if a.fim:
        cmd += ["-to", str(a.fim)]
    cmd += ["-i", str(video), "-vf", f"fps=1/{a.cada},scale=480:-2", str(frames / "q_%04d.png")]
    subprocess.run(cmd, check=True)

    # mosaicos de 4x3 com o tempo de cada quadro, para ver muitas fotos de uma vez
    shots = sorted(frames.glob("q_*.png"))
    for start in range(0, len(shots), 12):
        group = shots[start:start + 12]
        inputs, filters = [], []
        for i, shot in enumerate(group):
            inputs += ["-i", str(shot)]
            filters.append(f"[{i}:v]scale=320:240[v{i}]")
        while len(filters) < 12:
            i = len(filters)
            inputs += ["-f", "lavfi", "-i", "color=c=gray:s=320x240:d=1"]
            filters.append(f"[{i}:v]scale=320:240[v{i}]")
        layout = "|".join(f"{(k % 4) * 320}_{(k // 4) * 240}" for k in range(12))
        graph = ";".join(filters) + ";" + "".join(f"[v{k}]" for k in range(12)) + f"xstack=inputs=12:layout={layout}:fill=gray"
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", *inputs, "-filter_complex", graph, "-frames:v", "1",
                        str(frames / f"mosaico_{start // 12 + 1:02d}.png")], check=True)
    print(f"{len(shots)} quadros (quadro N = {a.inicio}s + (N-1)*{a.cada}s); mosaicos de 12 em {frames}")


if __name__ == "__main__":
    main()
