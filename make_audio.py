# -*- coding: latin-1 -*-
"""
Gera um WAV limpo (16 kHz, mono, sem silêncio de cauda) para cada .txt.

Motivação:
  - O gTTS faz uma requisição HTTP por pedaço de ~100 caracteres e concatena os
    MP3 sem re-encodar. Isso (a) provoca rate limiting em lote e (b) deixa o
    arquivo com cabeçalho de duração inconsistente, o que faz o whisperX
    transcrever "áudio fantasma" e alucinar palavras no fim.
  - Aqui geramos frase a frase, com retry/backoff, e concatenamos via ffmpeg
    com re-encode, produzindo um arquivo com duração correta.

Depois:
    df = model.get_events_dataframe(audio_path="audio/<nome>.wav")

Uso:
    python make_audio.py --inputs-dir inputs --audio-dir audio
"""

import argparse
import random
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

MAX_CHUNK_CHARS = 180


def strip_section_header(text: str) -> str:
    """Remove a primeira linha se for um cabeçalho de seção ('1.1 Texto ...')."""
    lines = text.lstrip().split("\n", 1)
    if len(lines) == 2 and re.match(r"^\d+\.\d+\s+Texto\b", lines[0].strip()):
        return lines[1]
    return text


def split_sentences(text: str) -> list[str]:
    """Quebra em frases; junta frases curtas até MAX_CHUNK_CHARS."""
    text = " ".join(strip_section_header(text).split())
    parts = re.split(r"(?<=[.!?…])\s+", text)

    chunks, current = [], ""
    for part in parts:
        # Frase sozinha maior que o limite: quebra em vírgulas.
        # É preciso descarregar `current` ANTES, senão os pedaços da frase longa
        # entram na lista à frente do texto que os precede.
        if len(part) > MAX_CHUNK_CHARS and current:
            chunks.append(current)
            current = ""
        while len(part) > MAX_CHUNK_CHARS:
            comma = part.rfind(",", 0, MAX_CHUNK_CHARS)
            if comma > 40:
                cut = comma + 1
            else:
                # sem vírgula utilizável: quebra no último espaço,
                # nunca no meio de uma palavra
                space = part.rfind(" ", 0, MAX_CHUNK_CHARS)
                cut = space if space > 40 else MAX_CHUNK_CHARS
            chunks.append(part[:cut].strip())
            part = part[cut:].strip()

        if len(current) + len(part) + 1 <= MAX_CHUNK_CHARS:
            current = f"{current} {part}".strip()
        else:
            if current:
                chunks.append(current)
            current = part
    if current:
        chunks.append(current)
    return [c for c in chunks if c]


def tts_chunk(text: str, out_path: Path, lang: str, tld: str,
              retries: int = 4) -> None:
    """Sintetiza um pedaço com retry e backoff exponencial + jitter."""
    from gtts import gTTS

    last_exc = None
    for attempt in range(retries):
        try:
            gTTS(text=text, lang=lang, tld=tld).save(str(out_path))
            if out_path.stat().st_size > 500:  # sanidade: MP3 vazio/HTML
                return
            raise RuntimeError(f"arquivo suspeito ({out_path.stat().st_size} bytes)")
        except Exception as exc:
            last_exc = exc
            wait = (2 ** attempt) + random.uniform(0, 1.5)
            print(f"      tentativa {attempt + 1}/{retries} falhou ({exc}); "
                  f"aguardando {wait:.1f}s", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"TTS falhou após {retries} tentativas: {last_exc}")


def concat_and_clean(mp3_paths: list[Path], out_wav: Path,
                     trim_silence: bool) -> None:
    """Concatena os MP3 re-encodando para WAV 16 kHz mono e apara o silêncio final."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                     encoding="utf-8") as listfile:
        for p in mp3_paths:
            listfile.write(f"file '{p.resolve()}'\n")
        list_path = listfile.name

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", list_path,
        "-ar", "16000", "-ac", "1",
    ]
    if trim_silence:
        # remove silêncio no fim (o gatilho da alucinação do Whisper)
        cmd += ["-af", "silenceremove=stop_periods=-1:stop_duration=0.8:stop_threshold=-45dB"]
    cmd += [str(out_wav)]

    subprocess.run(cmd, check=True)
    Path(list_path).unlink(missing_ok=True)


def duration_seconds(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-dir", default="inputs")
    parser.add_argument("--audio-dir", default="audio")
    parser.add_argument("--pattern", default="*.txt")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--tld", default="com.br",
                        help="Domínio do Google Tradutor (en: English)")
    parser.add_argument("--pause", type=float, default=1.0,
                        help="Pausa entre requisições, em segundos (evita throttling)")
    parser.add_argument("--no-trim", action="store_true",
                        help="Não aparar o silêncio final")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    inputs_dir = Path(args.inputs_dir)
    audio_dir = Path(args.audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)

    text_files = sorted(inputs_dir.glob(args.pattern))
    if not text_files:
        print(f"Nenhum arquivo em '{inputs_dir}' com padrão '{args.pattern}'.")
        return

    for text_path in text_files:
        out_wav = audio_dir / f"{text_path.stem}.wav"
        if out_wav.exists() and not args.overwrite:
            print(f"[pular] {out_wav} já existe ({duration_seconds(out_wav):.1f}s)")
            continue

        print(f"\n=== {text_path.name} ===")
        text = text_path.read_text(encoding="utf-8")
        chunks = split_sentences(text)
        print(f"  {len(chunks)} pedaço(s) de texto")

        with tempfile.TemporaryDirectory() as tmpdir:
            mp3_paths = []
            for i, chunk in enumerate(chunks):
                mp3_path = Path(tmpdir) / f"{i:04d}.mp3"
                print(f"    [{i + 1}/{len(chunks)}] {chunk[:60]}...")
                tts_chunk(chunk, mp3_path, args.lang, args.tld)
                mp3_paths.append(mp3_path)
                time.sleep(args.pause)

            concat_and_clean(mp3_paths, out_wav, trim_silence=not args.no_trim)

        print(f"  Salvo: {out_wav}  ({duration_seconds(out_wav):.1f}s)")

    print("\nConcluído.")


if __name__ == "__main__":
    main()