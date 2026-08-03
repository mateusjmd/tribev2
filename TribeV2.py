# -*- coding: utf-8 -*-
"""
Roda o pipeline TribeModel/PlotBrain para todos os áudios de uma pasta.

O TribeV2 transcreve o próprio áudio para obter palavras e tempos, então os
arquivos .txt NÃO são necessários aqui — a entrada é só o áudio.

Para cada áudio, além do que o script já fazia (vídeo cobrindo toda a
predição, mosaico de timesteps representativos e imagens individuais),
também são gerados:

  1. Três gráficos de série temporal, cada um salvo separadamente:
       - Ativação Cortical Média  (grafico_ativacao_media.png)
       - Pico de Ativação por Vértice (grafico_pico_vertice.png)
       - Variabilidade de Ativação / desvio padrão (grafico_variabilidade.png)
     Os picos são marcados com um ponto + uma linha vertical suave (stem)
     ligando o pico até o eixo X.
  2. Um painel único com os três empilhados (painel_completo_series.png).
  3. Um mosaico cerebral com os N picos de maior ativação média
     (mosaico_picos_cerebrais.png), organizado em linhas de até 5 painéis
     (2 linhas de 5, no caso padrão de 10 picos), com anotação de ranking
     (#1, #2, ...) e valor de ativação acima de cada painel.

NOTA sobre a correção de fonte: a versão anterior usava
`mathtext.fontset="cm"` (Computer Modern), cuja fonte (cmr10) não tem os
glifos de ç/ã/á do português -- por isso os títulos apareciam corrompidos
("Evolu□□o"). Trocamos para a fonte "Times New Roman", que cobre
corretamente os acentos.

Uso:
    python Tribe.py --audio-dir audio --outputs-dir outputs
"""

import argparse
import io
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image as mpimg

from tribev2 import TribeModel
from tribev2.plotting import PlotBrain

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}

# ---------------------------------------------------------------------------
# Estilo dos gráficos de série temporal
# ---------------------------------------------------------------------------
plt.rcParams.update({
    # "Times New Roman" é proprietária e pode não estar instalada. font.serif
    # é uma lista de fallback: o matplotlib tenta cada nome em ordem e usa o
    # primeiro que encontrar. "DejaVu Serif" vem embutido no matplotlib e
    # garante que o script nunca quebre por falta de fonte.
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "figure.dpi": 120,
    "savefig.dpi": 140,
    "font.size": 11,
    "axes.linewidth": 1.2,
    "axes.edgecolor": "black",
    "axes.grid": False,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.8,
    "legend.frameon": False,
    # Complementos para manter o fundo branco / texto preto (não vinham no
    # bloco original, mas são necessários para o visual "sóbrio" pedido):
    "axes.labelcolor": "black",
    "text.color": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

COR_MEDIA = "black"
COR_PICO_VERTICE = "steelblue"
COR_VARIABILIDADE = "seagreen"
COR_DESTAQUE_PICOS = "firebrick"


def get_timestep_indices(n_total: int) -> list[int]:
    """Índices representativos: início, 25%, 50%, 75%, fim (sem duplicatas)."""
    return sorted(set([0, n_total // 4, n_total // 2, (3 * n_total) // 4, n_total - 1]))


def get_peak_indices(activation: np.ndarray, n_peaks: int = 10) -> list[int]:
    """
    Índices dos `n_peaks` maiores valores de `activation`, ordenados do
    MAIOR para o MENOR (rank #1 = maior ativação).
    """
    n_peaks = min(n_peaks, len(activation))
    order = np.argsort(activation)[::-1][:n_peaks]
    return order.tolist()


def align_segments(preds, segments):
    """
    `preds` é a fonte de verdade (1 timestep ~ 1 s de entrada).

    Se o áudio tiver silêncio ou ruído de cauda, o whisperX alucina palavras e
    `segments` fica mais longo que `preds`, esticando o mp4 com conteúdo que não
    existe. Aqui truncamos para manter vídeo e plots estáticos coerentes.
    """
    n = preds.shape[0]
    if len(segments) != n:
        print(f"  AVISO: len(segments)={len(segments)} != preds.shape[0]={n} "
              f"-> truncando segments para {n}")
        segments = segments[:n]
    return segments


# ---------------------------------------------------------------------------
# Séries derivadas de `preds` (não fazem parte do tribev2 -- é pós-processamento)
# ---------------------------------------------------------------------------

def compute_activation_series(preds: np.ndarray):
    """Retorna (ativação média, pico por vértice, desvio padrão) por timestep."""
    mean_activation = preds.mean(axis=1)
    peak_activation = preds.max(axis=1)
    std_activation = preds.std(axis=1)
    return mean_activation, peak_activation, std_activation


def _style_axis(ax, title, ylabel, xlabel=None):
    ax.set_title(title, pad=10)
    ax.set_ylabel(ylabel)
    if xlabel is not None:
        ax.set_xlabel(xlabel)


def _add_peak_stems(ax, timesteps, values, peak_idx):
    """Desenha uma linha vertical suave de cada pico até a base do eixo X,
    sem alterar os limites verticais originais do gráfico."""
    ylim = ax.get_ylim()
    ax.vlines(
        timesteps[peak_idx], ylim[0], values[peak_idx],
        color=COR_DESTAQUE_PICOS, alpha=0.3, linewidth=1.1, zorder=1,
    )
    ax.set_ylim(ylim)


def plot_mean_activation(timesteps, mean_activation, peak_idx, out_path):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(timesteps, mean_activation, color=COR_MEDIA, zorder=3)
    _add_peak_stems(ax, timesteps, mean_activation, peak_idx)
    ax.scatter(timesteps[peak_idx], mean_activation[peak_idx],
               color=COR_DESTAQUE_PICOS, s=40, zorder=5, label="Picos")
    _style_axis(ax, "Evolução Temporal da Ativação Cortical Média",
                "Ativação Média (u.a.)", "Tempo (s)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_peak_vertex_activation(timesteps, peak_activation, peak_idx, out_path):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(timesteps, peak_activation, color=COR_PICO_VERTICE, zorder=3)
    _add_peak_stems(ax, timesteps, peak_activation, peak_idx)
    ax.scatter(timesteps[peak_idx], peak_activation[peak_idx],
               color=COR_DESTAQUE_PICOS, s=40, zorder=5, label="Picos")
    _style_axis(ax, "Evolução Temporal do Pico de Ativação por Vértice",
                "Ativação Máxima (u.a.)", "Tempo (s)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_variability(timesteps, std_activation, out_path):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(timesteps, std_activation, color=COR_VARIABILIDADE, zorder=3)
    _style_axis(ax, "Evolução Temporal da Variabilidade de Ativação",
                "Desvio padrão (u.a.)", "Tempo (s)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_full_panel(timesteps, mean_activation, peak_activation, std_activation,
                     peak_idx, out_path):
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    axes[0].plot(timesteps, mean_activation, color=COR_MEDIA, zorder=3)
    _add_peak_stems(axes[0], timesteps, mean_activation, peak_idx)
    axes[0].scatter(timesteps[peak_idx], mean_activation[peak_idx],
                     color=COR_DESTAQUE_PICOS, s=35, zorder=5)
    _style_axis(axes[0], "Evolução Temporal da Ativação Cortical Média",
                "Ativação Média (u.a.)")

    axes[1].plot(timesteps, peak_activation, color=COR_PICO_VERTICE, zorder=3)
    _add_peak_stems(axes[1], timesteps, peak_activation, peak_idx)
    axes[1].scatter(timesteps[peak_idx], peak_activation[peak_idx],
                     color=COR_DESTAQUE_PICOS, s=35, zorder=5)
    _style_axis(axes[1], "Evolução Temporal do Pico de Ativação por Vértice",
                "Ativação Máxima (u.a.)")

    axes[2].plot(timesteps, std_activation, color=COR_VARIABILIDADE, zorder=3)
    _style_axis(axes[2], "Evolução Temporal da Variabilidade de Ativação",
                "Desvio padrão (u.a.)", "Tempo (s)")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Mosaico cerebral dos picos, em linhas de até `n_cols` painéis
# ---------------------------------------------------------------------------

def _render_brain_row(brain_plotter, preds, segments, peak_idx_row, rank_offset,
                       mean_activation, views, norm_percentile):
    """Renderiza uma linha de painéis cerebrais via plot_timesteps, anota
    ranking/tempo/ativação em cada painel e devolve a linha como array RGBA."""
    real_times = [int(i) for i in peak_idx_row]

    fig = brain_plotter.plot_timesteps(
        preds[peak_idx_row],
        segments=[segments[i] for i in peak_idx_row],
        timestamps=real_times,
        norm_percentile=norm_percentile,
        views=views,
    )

    for col, idx in enumerate(peak_idx_row):
        rank = rank_offset + col + 1
        ax = fig.axes[col]
        cor = COR_DESTAQUE_PICOS if rank == 1 else "black"
        ax.set_title(
            f"#{rank}  t={idx}s\nact={mean_activation[idx]:.4f}",
            fontsize=9, color=cor, pad=8,
        )

    # IMPORTANTE: usamos savefig(..., bbox_inches="tight") para um buffer em
    # memória, e não fig.canvas.buffer_rgba(). O canvas bruto captura só a
    # área original da figura, sem recalcular a bbox -- como os títulos
    # (#rank / t=Xs / act=Y) adicionados acima ficam fora da área que
    # plot_timesteps reservou originalmente, eles ficavam cortados no topo.
    # savefig com bbox_inches="tight" recalcula a região a salvar incluindo
    # todo o conteúdo (títulos inclusive).
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img = mpimg.imread(buf)
    plt.close(fig)
    return img


def plot_peak_brain_mosaic(brain_plotter, preds, segments, peak_idx_by_rank,
                            mean_activation, audio_name, views,
                            norm_percentile, out_path, n_cols: int = 5):
    """
    Mosaico cerebral com os N picos de maior ativação média (rank #1 = maior),
    organizado em linhas de `n_cols` painéis (padrão: 2 linhas de 5, para
    n_peaks=10).
    """
    rows = [
        peak_idx_by_rank[i:i + n_cols]
        for i in range(0, len(peak_idx_by_rank), n_cols)
    ]

    row_images = [
        _render_brain_row(
            brain_plotter, preds, segments, row_idx,
            rank_offset=r * n_cols,
            mean_activation=mean_activation,
            views=views, norm_percentile=norm_percentile,
        )
        for r, row_idx in enumerate(rows)
    ]

    fig, axes = plt.subplots(
        len(row_images), 1,
        figsize=(2.6 * n_cols, 2.3 * len(row_images)),
    )
    if len(row_images) == 1:
        axes = [axes]
    for ax, img in zip(axes, row_images):
        ax.imshow(img)
        ax.axis("off")

    view_label = views if isinstance(views, str) else ",".join(views.values())
    fig.suptitle(
        f"Momentos de Pico de Ativação Cortical Média",
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def process_audio(audio_path: Path, out_dir: Path, cache_folder: Path,
                  views: str, norm_percentile: int, n_peaks: int = 10,
                  peaks_n_cols: int = 5) -> None:
    """Pipeline completo (predição + vídeo + mosaico + imagens + séries + picos) para um áudio."""
    print(f"\n=== Processando: {audio_path.name} ===")
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_folder.mkdir(parents=True, exist_ok=True)

    # Cache próprio por áudio: o whisperX salva o JSON da transcrição ao lado do
    # arquivo e o recarrega em chamadas seguintes, então um cache compartilhado
    # pode fazer todos reaproveitarem a transcrição do primeiro.
    model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=str(cache_folder))
    brain_plotter = PlotBrain()  # fsaverage5 por padrão

    # 1) Predição -----------------------------------------------------
    df = model.get_events_dataframe(audio_path=str(audio_path))
    preds, segments = model.predict(events=df)
    print(f"  preds.shape = {preds.shape}  |  len(segments) = {len(segments)}")

    segments = align_segments(preds, segments)

    # 2) Vídeo cobrindo toda a predição --------------------------------
    video_path = out_dir / "output_video.mp4"
    brain_plotter.plot_timesteps_mp4(
        preds,
        str(video_path),
        segments=segments,
        norm_percentile=norm_percentile,
        views=views,
    )
    print(f"  Salvo: {video_path}")

    # 3) Mosaico em timesteps representativos --------------------------
    timestep_indices = get_timestep_indices(preds.shape[0])

    fig = brain_plotter.plot_timesteps(
        preds[timestep_indices],
        segments=[segments[i] for i in timestep_indices],
        timestamps=timestep_indices,
        norm_percentile=norm_percentile,
        views=views,
    )
    mosaic_path = out_dir / "output_mosaic.png"
    fig.savefig(mosaic_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {mosaic_path}")

    # 4) Imagens individuais por timestep escolhido --------------------
    for i in timestep_indices:
        fig_single = brain_plotter.plot_timesteps(
            preds[i : i + 1],
            segments=segments[i : i + 1],
            timestamps=[i],
            norm_percentile=norm_percentile,
            views=views,
        )
        single_path = out_dir / f"output_t{i:05d}.png"
        fig_single.savefig(single_path, bbox_inches="tight")
        plt.close(fig_single)
        print(f"  Salvo: {single_path}")

    # 5) Séries temporais de ativação -----------------------------------
    mean_activation, peak_activation, std_activation = compute_activation_series(preds)
    timesteps = np.arange(preds.shape[0])
    peak_idx = get_peak_indices(mean_activation, n_peaks=n_peaks)

    plot_mean_activation(timesteps, mean_activation, peak_idx,
                          out_dir / "grafico_ativacao_media.png")
    plot_peak_vertex_activation(timesteps, peak_activation, peak_idx,
                                 out_dir / "grafico_pico_vertice.png")
    plot_variability(timesteps, std_activation,
                      out_dir / "grafico_variabilidade.png")
    plot_full_panel(timesteps, mean_activation, peak_activation, std_activation,
                     peak_idx, out_dir / "painel_completo_series.png")
    print(f"  Salvo: {out_dir / 'grafico_ativacao_media.png'}")
    print(f"  Salvo: {out_dir / 'grafico_pico_vertice.png'}")
    print(f"  Salvo: {out_dir / 'grafico_variabilidade.png'}")
    print(f"  Salvo: {out_dir / 'painel_completo_series.png'}")

    # 6) Mosaico cerebral dos picos observados (2 linhas x 5 colunas) --
    peaks_mosaic_path = out_dir / "mosaico_picos_cerebrais.png"
    plot_peak_brain_mosaic(brain_plotter, preds, segments, peak_idx,
                            mean_activation, audio_path.stem, views,
                            norm_percentile, peaks_mosaic_path,
                            n_cols=peaks_n_cols)
    print(f"  Salvo: {peaks_mosaic_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", default="audio",
                        help="Pasta com os áudios de entrada (padrão: audio)")
    parser.add_argument("--outputs-dir", default="outputs",
                        help="Pasta onde salvar os resultados (padrão: outputs)")
    parser.add_argument("--views", default="left",
                        help="Vista(s) do cérebro a plotar (padrão: left)")
    parser.add_argument("--norm-percentile", type=int, default=99,
                        help="Percentil de normalização (padrão: 99)")
    parser.add_argument("--cache-folder", default="./cache",
                        help="Cache base; cada áudio ganha uma subpasta (padrão: ./cache)")
    parser.add_argument("--n-peaks", type=int, default=10,
                        help="Número de picos de ativação a destacar/plotar (padrão: 10)")
    parser.add_argument("--peaks-n-cols", type=int, default=5,
                        help="Colunas por linha no mosaico de picos (padrão: 5)")
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    outputs_dir = Path(args.outputs_dir)
    cache_base = Path(args.cache_folder)

    audio_files = sorted(p for p in audio_dir.iterdir()
                         if p.is_file() and p.suffix.lower() in AUDIO_EXTS)

    if not audio_files:
        print(f"Nenhum áudio encontrado em '{audio_dir}'.")
        return

    print(f"Encontrados {len(audio_files)} áudio(s) em '{audio_dir}'.")

    for audio_path in audio_files:
        try:
            process_audio(audio_path,
                          outputs_dir / audio_path.stem,
                          cache_base / audio_path.stem,
                          args.views, args.norm_percentile,
                          args.n_peaks, args.peaks_n_cols)
        except Exception as exc:
            print(f"  ERRO ao processar {audio_path.name}: {exc}")

    print("\nProcessamento concluído.")


if __name__ == "__main__":
    main()