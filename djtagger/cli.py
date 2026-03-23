"""Rich CLI with Typer — the beautiful interface for DJ Tagger."""

import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import typer
from rich import box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

from . import __version__
from .config import (
    DEFAULT_MUSIC_PATH,
    ERROR_FILE,
    GENRE_KEEP_PROB,
    LOG_FILE,
    STATUS_FILE,
    TAGGER_VERSION,
)

app = typer.Typer(
    name="djtagger",
    help="🎛  DJ Tagger — Autonomous DJ music tagger with Essentia ML + Beatport + Last.fm",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True,
)
console = Console()

# ─── Status file (for external polling) ─────────────────────

_status: dict = {}


def _update_status(**kwargs: object) -> None:
    _status.update(kwargs)
    _status["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(STATUS_FILE, "w") as f:
            json.dump(_status, f, indent=2)
    except Exception:
        pass


# ─── Logging ────────────────────────────────────────────────

_log_fh = None
_err_fh = None


def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    if _log_fh:
        _log_fh.write(line + "\n")
        _log_fh.flush()


def _log_error(filepath: str, msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {filepath}: {msg}"
    if _err_fh:
        _err_fh.write(line + "\n")
        _err_fh.flush()


# ─── Rich display helpers ───────────────────────────────────

SOURCE_COLORS = {
    "beatport": "green",
    "lastfm+ml": "yellow",
    "ml": "blue",
}

SOURCE_ICONS = {
    "beatport": "🟢",
    "lastfm+ml": "🟡",
    "ml": "🔵",
}


def _make_stats_panel(
    processed: int,
    total: int,
    genre_sources: dict[str, int],
    avg_speed: float,
    current_track: str,
    current_folder: str,
) -> Panel:
    """Build the live stats panel shown during processing."""
    lines: list[str] = []

    # Current track
    if current_track:
        trunc = current_track[:60] + "…" if len(current_track) > 60 else current_track
        lines.append(f"[bold]🎵 {trunc}[/bold]")
    if current_folder:
        lines.append(f"[dim]📁 {current_folder}[/dim]")
    lines.append("")

    # Genre sources
    bp = genre_sources.get("beatport", 0)
    fm = genre_sources.get("lastfm+ml", 0)
    ml = genre_sources.get("ml", 0)
    lines.append(
        f"  🟢 Beatport  [bold green]{bp:>4}[/bold green]"
        f"   🟡 Last.fm+ML  [bold yellow]{fm:>4}[/bold yellow]"
        f"   🔵 ML-only  [bold blue]{ml:>4}[/bold blue]"
    )
    lines.append("")

    # Speed
    if avg_speed > 0:
        lines.append(f"  ⏱  [dim]{avg_speed:.1f}s per track[/dim]")

    return Panel(
        "\n".join(lines),
        title="[bold]DJ Tagger[/bold]",
        border_style="cyan",
        padding=(1, 2),
    )


def _make_summary_table(
    processed: int,
    failed: int,
    skipped: int,
    total_files: int,
    genre_sources: dict[str, int],
    elapsed_hours: float,
    energies: list[float],
    valences: list[float],
) -> Table:
    """Build the final summary table."""
    table = Table(
        title="✨ Tagging Complete",
        box=box.ROUNDED,
        title_style="bold cyan",
        border_style="dim",
        padding=(0, 1),
    )
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Tracks tagged", f"[bold green]{processed - failed}[/bold green]")
    table.add_row("Failed", f"[bold red]{failed}[/bold red]" if failed else "[dim]0[/dim]")
    table.add_row("Skipped (already tagged)", f"[dim]{skipped}[/dim]")
    table.add_row("Total files", str(total_files))
    table.add_row("", "")

    bp = genre_sources.get("beatport", 0)
    fm = genre_sources.get("lastfm+ml", 0)
    ml = genre_sources.get("ml", 0)
    table.add_row("🟢 Beatport", f"[green]{bp}[/green]")
    table.add_row("🟡 Last.fm+ML", f"[yellow]{fm}[/yellow]")
    table.add_row("🔵 ML-only", f"[blue]{ml}[/blue]")
    table.add_row("", "")

    if energies:
        import numpy as np

        table.add_row(
            "Energy range",
            f"{min(energies):.2f} – {max(energies):.2f}  (avg {np.mean(energies):.2f})",
        )
        table.add_row(
            "Valence range",
            f"{min(valences):.2f} – {max(valences):.2f}  (avg {np.mean(valences):.2f})",
        )
        table.add_row("", "")

    table.add_row("Elapsed", f"{elapsed_hours:.1f}h")

    return table


# ═════════════════════════════════════════════════════════════
#  TAG command
# ═════════════════════════════════════════════════════════════


@app.command()
def tag(
    path: str = typer.Argument(
        DEFAULT_MUSIC_PATH,
        help="Folder or file to tag (recursive for folders)",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Analyze without writing tags"),
    force: bool = typer.Option(False, "--force", help="Re-tag already tagged files"),
    no_beatport: bool = typer.Option(False, "--no-beatport", help="Skip Beatport lookups"),
    fix_comments: bool = typer.Option(
        False, "--fix-comments", help="Update comments on already-tagged files (no re-analysis)"
    ),
) -> None:
    """Tag MP3 files with genre, energy, and mood metadata."""
    global _log_fh, _err_fh

    # Set socket timeout
    socket.setdefaulttimeout(10)

    if not os.path.exists(path):
        console.print(f"[bold red]Error:[/bold red] {path} not found")
        raise typer.Exit(1)

    # Open log files
    _log_fh = open(LOG_FILE, "w")
    _err_fh = open(ERROR_FILE, "w")

    try:
        _tag_inner(path, dry_run, force, no_beatport, fix_comments)
    finally:
        _cleanup()


def _tag_inner(
    path: str,
    dry_run: bool,
    force: bool,
    no_beatport: bool,
    fix_comments: bool,
) -> None:
    """Inner implementation of tag command, wrapped by try/finally for cleanup."""
    # Lazy imports (heavy — TF models)
    from .analyzer import analyze_track, load_models
    from .genres import resolve_genres
    from .scanner import filter_untagged, find_mp3s
    from .tagger import fix_comments as do_fix_comments
    from .tagger import parse_filename, write_tags
    from rich.console import Group

    # ─── Fix-comments mode ──────────────────────────────────
    if fix_comments:
        console.print("[bold cyan]🔧 Fix-comments mode[/bold cyan] — updating Serato comments\n")
        all_mp3s = find_mp3s(path)
        console.print(f"Found [bold]{len(all_mp3s)}[/bold] MP3 files")

        fixed = 0
        errors = 0
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Fixing comments", total=len(all_mp3s))
            for mp3 in all_mp3s:
                ok = do_fix_comments(mp3)
                if ok:
                    fixed += 1
                progress.advance(task)

        console.print(
            f"\n[bold green]✅ Done![/bold green] Fixed [bold]{fixed}[/bold] comments, "
            f"skipped [dim]{len(all_mp3s) - fixed}[/dim] untagged files"
        )
        return

    # ─── Scan ───────────────────────────────────────────────
    console.print(f"\n[bold cyan]📁 Scanning[/bold cyan] {path}\n")
    all_mp3s = find_mp3s(path)
    total_files = len(all_mp3s)
    console.print(f"Found [bold]{total_files}[/bold] MP3 files")

    # Filter
    if not force:
        mp3s, skipped = filter_untagged(all_mp3s)
        if skipped:
            console.print(
                f"Skipping [dim]{skipped}[/dim] already tagged → "
                f"[bold]{len(mp3s)}[/bold] to process"
            )
    else:
        mp3s = all_mp3s
        skipped = 0
        console.print(f"[yellow]Force mode[/yellow]: processing all [bold]{len(mp3s)}[/bold] tracks")

    if not mp3s:
        console.print("\n[bold green]Nothing to do![/bold green] All files already tagged. ✨")
        _update_status(state="done", total=total_files, skipped=total_files, processed=0)
        return

    # ─── Load models ────────────────────────────────────────
    console.print()
    with console.status("[bold cyan]Loading ML models…[/bold cyan]", spinner="dots"):
        models = load_models()
    console.print("[bold green]✓[/bold green] Models loaded\n")

    # ─── Init status ────────────────────────────────────────
    mode_label = "DRY RUN" if dry_run else "TAGGING"
    _update_status(
        state="running",
        mode=mode_label,
        version=TAGGER_VERSION,
        total=total_files,
        to_process=len(mp3s),
        skipped=skipped,
        processed=0,
        failed=0,
        current="",
        genre_sources={"beatport": 0, "lastfm+ml": 0, "ml": 0},
        started=time.strftime("%Y-%m-%d %H:%M:%S"),
        avg_seconds=0,
        eta_hours=0,
    )

    # ─── Header ─────────────────────────────────────────────
    header = Table.grid(padding=(0, 2))
    header.add_row(
        f"[bold cyan]🎛  DJ Tagger {TAGGER_VERSION}[/bold cyan]",
        f"[dim]{mode_label}[/dim]",
    )
    header.add_row(
        f"[dim]📀 {len(mp3s)} tracks to process[/dim]",
        f"[dim]({total_files} total, {skipped} skipped)[/dim]",
    )
    console.print(Panel(header, border_style="cyan", padding=(1, 2)))
    console.print()

    # ─── Process ────────────────────────────────────────────
    results: list[dict] = []
    failed = 0
    start_time = time.time()
    genre_sources: dict[str, int] = {"beatport": 0, "lastfm+ml": 0, "ml": 0}

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
    )
    task_id = progress.add_task("Tagging", total=len(mp3s))

    with Live(console=console, refresh_per_second=4) as live:
        for i, mp3 in enumerate(mp3s, 1):
            t0 = time.time()
            folder = os.path.basename(os.path.dirname(mp3))
            artist, artist_clean, title = parse_filename(mp3)
            track_label = f"{artist} — {title}" if artist else title

            # Update progress
            progress.update(task_id, completed=i - 1, description=f"[bold]{track_label[:50]}[/bold]")

            # Build display
            panel = _make_stats_panel(
                processed=i - 1,
                total=len(mp3s),
                genre_sources=genre_sources,
                avg_speed=(time.time() - start_time) / max(i - 1, 1),
                current_track=track_label,
                current_folder=folder,
            )
            live.update(Group(progress, panel))

            _log(f"[{i}/{len(mp3s)}] ({folder}) 🎵 {artist} — {title}")

            # Analyze
            try:
                result = analyze_track(mp3, models)
            except Exception as ex:
                _log(f"  ⚠ Analysis failed: {ex}")
                _log_error(mp3, f"Analysis: {ex}")
                failed += 1
                progress.update(task_id, completed=i)
                continue

            # Resolve genre
            try:
                final_genres, genre_source = resolve_genres(
                    artist,
                    artist_clean,
                    title,
                    result["genres"],
                    ml_electronic_genres=result.get("electronic_genres"),
                    use_beatport=not no_beatport,
                    genre_keep_prob=GENRE_KEEP_PROB,
                )
            except Exception as ex:
                _log(f"  ⚠ Genre resolution failed: {ex}")
                final_genres, genre_source = [], "ml"

            src_icon = SOURCE_ICONS.get(genre_source, "")
            src_color = SOURCE_COLORS.get(genre_source, "white")
            genre_str = "; ".join(final_genres[:4]) if final_genres else "(none)"
            _log(
                f"  Genre: {genre_str} [{genre_source}] "
                f"| E:{result['energy']:.2f} V:{result['valence']:.2f}"
            )

            # Write tags
            if not dry_run:
                ok, genre_action = write_tags(mp3, result, genre_source, final_genres)
                if ok:
                    _log(f"  ✅ Tagged (TCON: {genre_action})")
                else:
                    _log_error(mp3, f"Tag write failed: {genre_action}")
                    failed += 1
                    progress.update(task_id, completed=i)
                    continue

            # Track result
            result["final_genres"] = final_genres
            result["genre_source"] = genre_source
            results.append(result)
            genre_sources[genre_source] = genre_sources.get(genre_source, 0) + 1

            # Timing
            elapsed = time.time() - t0
            avg = (time.time() - start_time) / i
            eta = avg * (len(mp3s) - i) / 3600
            _update_status(
                processed=i,
                failed=failed,
                current=os.path.basename(mp3),
                current_folder=folder,
                genre_sources=genre_sources,
                avg_seconds=round(avg, 1),
                eta_hours=round(eta, 2),
                last_track_seconds=round(elapsed, 1),
            )

            progress.update(task_id, completed=i)

    # ─── Summary ────────────────────────────────────────────
    elapsed_total = (time.time() - start_time) / 3600
    energies = [r["energy"] for r in results]
    valences = [r["valence"] for r in results]

    console.print()
    console.print(
        _make_summary_table(
            processed=len(mp3s),
            failed=failed,
            skipped=skipped,
            total_files=total_files,
            genre_sources=genre_sources,
            elapsed_hours=elapsed_total,
            energies=energies,
            valences=valences,
        )
    )

    _update_status(
        state="done",
        processed=len(mp3s),
        failed=failed,
        finished=time.strftime("%Y-%m-%d %H:%M:%S"),
        elapsed_hours=round(elapsed_total, 2),
    )


# ═════════════════════════════════════════════════════════════
#  INFO command
# ═════════════════════════════════════════════════════════════


@app.command()
def info(
    filepath: str = typer.Argument(..., help="Path to a single MP3 file"),
) -> None:
    """Show DJ Tagger tags for a single MP3 file."""
    from .tagger import parse_filename, read_tags

    if not os.path.isfile(filepath):
        console.print(f"[bold red]Error:[/bold red] {filepath} not found")
        raise typer.Exit(1)

    artist, _, title = parse_filename(filepath)
    tags = read_tags(filepath)

    # Header
    track_label = f"{artist} — {title}" if artist else title
    console.print(f"\n[bold cyan]🎵 {track_label}[/bold cyan]")
    console.print(f"[dim]{filepath}[/dim]\n")

    if not tags.get("tagger_version"):
        console.print("[yellow]⚠ Not tagged by DJ Tagger[/yellow]\n")

    # Build table
    table = Table(box=box.SIMPLE_HEAVY, border_style="dim", padding=(0, 2))
    table.add_column("Tag", style="bold")
    table.add_column("Value")

    # Genre with source color
    src = tags.get("genre_source", "")
    src_color = SOURCE_COLORS.get(src, "white")
    src_icon = SOURCE_ICONS.get(src, "")
    genre_display = tags.get("genre", "(none)")
    table.add_row("Genre", f"[{src_color}]{genre_display}[/{src_color}]")
    table.add_row("Genre source", f"{src_icon} [{src_color}]{src}[/{src_color}]" if src else "[dim]—[/dim]")
    if tags.get("genre_detected") and tags["genre_detected"] != tags.get("genre"):
        table.add_row("Genre (detected)", f"[dim]{tags['genre_detected']}[/dim]")
    table.add_row("", "")

    # BPM & Key
    if tags.get("bpm"):
        table.add_row("BPM", f"[bold]{tags['bpm']}[/bold]")
    if tags.get("key"):
        table.add_row("Key", f"[bold]{tags['key']}[/bold]")
    if tags.get("bpm") or tags.get("key"):
        table.add_row("", "")

    # Energy & Mood
    if tags.get("energy"):
        e = float(tags["energy"])
        e_bar = _bar(e, "red", "yellow", "green")
        table.add_row("Energy", f"{e_bar}  {e:.3f}")
    if tags.get("valence"):
        v = float(tags["valence"])
        v_bar = _bar(v, "blue", "white", "yellow")
        table.add_row("Valence", f"{v_bar}  {v:.3f}")
    table.add_row("", "")

    # v5 scores
    if tags.get("danceability"):
        d = float(tags["danceability"])
        table.add_row("Danceability", f"{_mini_bar(d)}  {d:.3f}")
    if tags.get("arousal"):
        a = float(tags["arousal"])
        table.add_row("Arousal", f"{_mini_bar(a)}  {a:.3f}")
    if tags.get("mood_party"):
        p = float(tags["mood_party"])
        table.add_row("Party", f"{_mini_bar(p)}  {p:.3f}")
    table.add_row("", "")

    # Segment energy
    if tags.get("peak_energy"):
        pe = float(tags["peak_energy"])
        table.add_row("Peak energy", f"{_mini_bar(pe)}  {pe:.3f}")
    if tags.get("intro_energy"):
        ie = float(tags["intro_energy"])
        table.add_row("Intro energy", f"{_mini_bar(ie)}  {ie:.3f}")
    table.add_row("", "")

    # Moods
    for mood_key, label in [
        ("mood_happy", "😊 Happy"),
        ("mood_sad", "😢 Sad"),
        ("mood_aggressive", "🔥 Aggressive"),
        ("mood_relaxed", "😌 Relaxed"),
    ]:
        val = tags.get(mood_key)
        if val:
            v = float(val)
            table.add_row(label, f"{_mini_bar(v)}  {v:.3f}")

    table.add_row("", "")

    # Comment & version
    if tags.get("comment"):
        table.add_row("Comment", tags["comment"])
    if tags.get("tagger_version"):
        table.add_row("Tagger version", f"[dim]{tags['tagger_version']}[/dim]")

    console.print(table)
    console.print()


def _bar(value: float, low_color: str, mid_color: str, high_color: str) -> str:
    """Render a small colored bar from 0-1."""
    filled = int(value * 20)
    color = low_color if value < 0.33 else mid_color if value < 0.66 else high_color
    return f"[{color}]{'█' * filled}[/{color}][dim]{'░' * (20 - filled)}[/dim]"


def _mini_bar(value: float) -> str:
    """Small neutral bar."""
    filled = int(value * 15)
    return f"[cyan]{'█' * filled}[/cyan][dim]{'░' * (15 - filled)}[/dim]"


# ═════════════════════════════════════════════════════════════
#  STATS command
# ═════════════════════════════════════════════════════════════


@app.command()
def stats(
    path: str = typer.Argument(
        DEFAULT_MUSIC_PATH,
        help="Folder to analyze",
    ),
) -> None:
    """Show library statistics — tagged count, genre distribution, energy histograms, etc."""
    from collections import Counter, defaultdict

    from .library import scan_library

    if not os.path.exists(path):
        console.print(f"[bold red]Error:[/bold red] {path} not found")
        raise typer.Exit(1)

    console.print(f"\n[bold cyan]📊 Library Statistics[/bold cyan]")
    console.print(f"[dim]{path}[/dim]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Reading tags", total=0)

        def on_progress(current: int, total: int) -> None:
            progress.update(task, completed=current, total=total)

        tracks = scan_library(path, on_progress=on_progress)

    if not tracks:
        console.print("[yellow]No MP3 files found.[/yellow]")
        return

    tagged_tracks = [t for t in tracks if t["tagged"]]
    untagged_count = len(tracks) - len(tagged_tracks)

    genre_counter: Counter = Counter()
    source_counter: Counter = Counter()
    energies: list[float] = []
    valences: list[float] = []
    danceabilities: list[float] = []
    versions: Counter = Counter()
    # For genre×energy breakdown
    genre_energies: defaultdict[str, list[float]] = defaultdict(list)

    for t in tagged_tracks:
        genre = t["genre"]
        if genre:
            primary_genre = genre.split(";")[0].strip()
            for g in genre.split(";"):
                g = g.strip()
                if g:
                    genre_counter[g] += 1
            if t["energy"] is not None and primary_genre:
                genre_energies[primary_genre].append(t["energy"])
        source_counter[t["genre_source"] or "unknown"] += 1
        if t["energy"] is not None:
            energies.append(t["energy"])
        if t["valence"] is not None:
            valences.append(t["valence"])
        if t["danceability"] is not None:
            danceabilities.append(t["danceability"])
        versions[t["tagger_version"] or "?"] += 1

    # ─── Overview table ─────────────────────────────────────
    overview = Table(box=box.ROUNDED, border_style="dim", title="Overview", title_style="bold")
    overview.add_column("Metric", style="bold")
    overview.add_column("Value", justify="right")
    overview.add_row("Total files", str(len(tracks)))
    overview.add_row("Tagged", f"[green]{len(tagged_tracks)}[/green]")
    overview.add_row(
        "Untagged",
        f"[yellow]{untagged_count}[/yellow]" if untagged_count else "[dim]0[/dim]",
    )
    pct = (len(tagged_tracks) / len(tracks) * 100) if tracks else 0
    overview.add_row("Coverage", f"{pct:.1f}%")
    console.print(overview)
    console.print()

    # ─── Source breakdown ───────────────────────────────────
    if source_counter:
        src_table = Table(box=box.ROUNDED, border_style="dim", title="Genre Sources", title_style="bold")
        src_table.add_column("Source", style="bold")
        src_table.add_column("Count", justify="right")
        for src, count in source_counter.most_common():
            icon = SOURCE_ICONS.get(src, "")
            color = SOURCE_COLORS.get(src, "white")
            src_table.add_row(f"{icon} [{color}]{src}[/{color}]", str(count))
        console.print(src_table)
        console.print()

    # ─── Top genres ─────────────────────────────────────────
    if genre_counter:
        genre_table = Table(
            box=box.ROUNDED, border_style="dim",
            title="Top 20 Genres", title_style="bold",
        )
        genre_table.add_column("Genre", style="bold")
        genre_table.add_column("Count", justify="right")
        genre_table.add_column("Bar")
        max_count = genre_counter.most_common(1)[0][1] if genre_counter else 1
        for genre, count in genre_counter.most_common(20):
            bar_len = int(count / max_count * 30)
            genre_table.add_row(genre, str(count), f"[cyan]{'█' * bar_len}[/cyan]")
        console.print(genre_table)
        console.print()

    # ─── Energy / Valence summary ───────────────────────────
    if energies:
        import numpy as np

        ev_table = Table(box=box.ROUNDED, border_style="dim", title="Energy & Mood", title_style="bold")
        ev_table.add_column("Metric", style="bold")
        ev_table.add_column("Min", justify="right")
        ev_table.add_column("Avg", justify="right")
        ev_table.add_column("Max", justify="right")
        ev_table.add_row(
            "Energy",
            f"{min(energies):.2f}",
            f"{np.mean(energies):.2f}",
            f"{max(energies):.2f}",
        )
        ev_table.add_row(
            "Valence",
            f"{min(valences):.2f}",
            f"{np.mean(valences):.2f}",
            f"{max(valences):.2f}",
        )
        console.print(ev_table)
        console.print()

    # ─── Energy distribution histogram ──────────────────────
    if energies:
        _print_histogram("Energy Distribution", energies, "red", "yellow", "green")
        console.print()

    # ─── Valence distribution histogram ─────────────────────
    if valences:
        _print_histogram("Valence Distribution", valences, "blue", "white", "yellow")
        console.print()

    # ─── Danceability distribution histogram ─────────────────
    if danceabilities:
        _print_histogram("Danceability Distribution", danceabilities, "dim", "cyan", "magenta")
        console.print()

    # ─── Genre × Energy breakdown ───────────────────────────
    if genre_energies:
        import numpy as np

        ge_table = Table(
            box=box.ROUNDED, border_style="dim",
            title="Genre Energy Profile (top 15)", title_style="bold",
        )
        ge_table.add_column("Genre", style="bold")
        ge_table.add_column("Tracks", justify="right")
        ge_table.add_column("Low", justify="right", style="cyan")
        ge_table.add_column("Mid", justify="right", style="yellow")
        ge_table.add_column("High", justify="right", style="red")
        ge_table.add_column("Avg", justify="right")
        ge_table.add_column("Profile")

        # Sort by track count
        sorted_genres = sorted(genre_energies.items(), key=lambda x: -len(x[1]))[:15]
        for genre, e_list in sorted_genres:
            low = sum(1 for e in e_list if e < 0.4)
            mid = sum(1 for e in e_list if 0.4 <= e < 0.7)
            high = sum(1 for e in e_list if e >= 0.7)
            avg = np.mean(e_list)
            bar = _bar(avg, "cyan", "yellow", "red")
            ge_table.add_row(
                genre[:25], str(len(e_list)),
                str(low), str(mid), str(high),
                f"{avg:.2f}", bar,
            )
        console.print(ge_table)
        console.print()

    # ─── Tagger versions ───────────────────────────────────
    if versions:
        ver_table = Table(box=box.SIMPLE, border_style="dim")
        ver_table.add_column("Tagger Version", style="dim")
        ver_table.add_column("Tracks", justify="right", style="dim")
        for v, c in versions.most_common():
            ver_table.add_row(v, str(c))
        console.print(ver_table)

    console.print()


def _print_histogram(
    title: str,
    values: list[float],
    low_color: str,
    mid_color: str,
    high_color: str,
) -> None:
    """Print a 5-bucket histogram for values in 0-1 range."""
    buckets = [0] * 5
    labels = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
    colors = [low_color, low_color, mid_color, high_color, high_color]
    for v in values:
        idx = min(int(v * 5), 4)
        buckets[idx] += 1

    max_count = max(buckets) if buckets else 1
    table = Table(
        box=box.ROUNDED, border_style="dim",
        title=title, title_style="bold",
    )
    table.add_column("Range", style="bold")
    table.add_column("Count", justify="right")
    table.add_column("Distribution")
    for i, (label, count) in enumerate(zip(labels, buckets)):
        bar_len = int(count / max_count * 30) if max_count else 0
        color = colors[i]
        table.add_row(label, str(count), f"[{color}]{'█' * bar_len}[/{color}]")
    console.print(table)


# ═════════════════════════════════════════════════════════════
#  Shared helpers for new commands
# ═════════════════════════════════════════════════════════════


def _parse_range(value: str) -> tuple[float | None, float | None]:
    """Parse a 'MIN:MAX' range string. Either side can be empty.

    Examples: '0.5:0.8' -> (0.5, 0.8), '0.5:' -> (0.5, None), ':0.8' -> (None, 0.8)
    """
    if ":" not in value:
        # Treat as exact min
        v = float(value)
        return v, v
    parts = value.split(":", 1)
    lo = float(parts[0]) if parts[0].strip() else None
    hi = float(parts[1]) if parts[1].strip() else None
    return lo, hi


def _in_range(value: float | None, lo: float | None, hi: float | None) -> bool:
    """Check if value falls within [lo, hi]. None on either side means unbounded."""
    if value is None:
        return False
    if lo is not None and value < lo:
        return False
    if hi is not None and value > hi:
        return False
    return True


def _scan_with_progress(path: str) -> list[dict]:
    """Scan a library with a Rich progress bar. Shared by find/export/suggest/health."""
    from .library import scan_library

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Scanning library", total=0)

        def on_progress(current: int, total: int) -> None:
            progress.update(task, completed=current, total=total)

        return scan_library(path, on_progress=on_progress)


# ═════════════════════════════════════════════════════════════
#  FIND command
# ═════════════════════════════════════════════════════════════


@app.command()
def find(
    path: str = typer.Argument(
        DEFAULT_MUSIC_PATH,
        help="Folder to search (recursive)",
    ),
    genre: Optional[str] = typer.Option(None, "--genre", "-g", help="Filter by genre (case-insensitive substring)"),
    energy: Optional[str] = typer.Option(None, "--energy", "-e", help="Energy range, e.g. 0.7:1.0 or 0.5:"),
    valence: Optional[str] = typer.Option(None, "--valence", help="Valence range, e.g. 0.3:0.6"),
    mood_happy: Optional[str] = typer.Option(None, "--mood-happy", help="Happy score range"),
    mood_sad: Optional[str] = typer.Option(None, "--mood-sad", help="Sad score range"),
    mood_aggressive: Optional[str] = typer.Option(None, "--mood-aggressive", help="Aggressive score range"),
    mood_relaxed: Optional[str] = typer.Option(None, "--mood-relaxed", help="Relaxed score range"),
    danceability: Optional[str] = typer.Option(None, "--danceability", "-d", help="Danceability range, e.g. 0.7:"),
    peak_energy: Optional[str] = typer.Option(None, "--peak-energy", help="Peak energy range"),
    source: Optional[str] = typer.Option(None, "--source", "-s", help="Filter by genre source (beatport, lastfm+ml, ml)"),
    untagged: bool = typer.Option(False, "--untagged", help="Show only untagged tracks"),
    sort: str = typer.Option("energy", "--sort", help="Sort by: energy, valence, genre, artist, title, path"),
    reverse: bool = typer.Option(False, "--reverse", "-r", help="Reverse sort order"),
    limit: int = typer.Option(50, "--limit", "-n", help="Max results to show (0 = all)"),
) -> None:
    """Search and filter your tagged library."""
    if not os.path.exists(path):
        console.print(f"[bold red]Error:[/bold red] {path} not found")
        raise typer.Exit(1)

    tracks = _scan_with_progress(path)
    if not tracks:
        console.print("[yellow]No MP3 files found.[/yellow]")
        return

    # Parse range filters
    ranges: dict[str, tuple[float | None, float | None]] = {}
    for opt_name, opt_val, field in [
        ("energy", energy, "energy"),
        ("valence", valence, "valence"),
        ("mood-happy", mood_happy, "mood_happy"),
        ("mood-sad", mood_sad, "mood_sad"),
        ("mood-aggressive", mood_aggressive, "mood_aggressive"),
        ("mood-relaxed", mood_relaxed, "mood_relaxed"),
        ("danceability", danceability, "danceability"),
        ("peak-energy", peak_energy, "peak_energy"),
    ]:
        if opt_val is not None:
            try:
                ranges[field] = _parse_range(opt_val)
            except ValueError:
                console.print(f"[bold red]Error:[/bold red] Invalid range for --{opt_name}: {opt_val}")
                raise typer.Exit(1)

    # Filter
    results: list[dict] = []
    for t in tracks:
        if untagged:
            if t["tagged"]:
                continue
        else:
            if not t["tagged"]:
                continue

        if genre and genre.lower() not in (t["genre"] or "").lower():
            continue
        if source and t["genre_source"] != source:
            continue
        # Range filters
        skip = False
        for field, (lo, hi) in ranges.items():
            if not _in_range(t[field], lo, hi):
                skip = True
                break
        if skip:
            continue

        results.append(t)

    if not results:
        console.print("[yellow]No tracks match your filters.[/yellow]")
        return

    # Sort
    sort_key_map = {
        "energy": lambda t: t["energy"] if t["energy"] is not None else -1,
        "valence": lambda t: t["valence"] if t["valence"] is not None else -1,
        "danceability": lambda t: t["danceability"] if t["danceability"] is not None else -1,
        "genre": lambda t: (t["genre"] or "").lower(),
        "artist": lambda t: (t["artist"] or "").lower(),
        "title": lambda t: (t["title"] or "").lower(),
        "path": lambda t: t["path"].lower(),
    }
    key_fn = sort_key_map.get(sort, sort_key_map["energy"])
    # Default: energy descending, others ascending
    default_reverse = sort in ("energy", "valence", "danceability")
    actual_reverse = default_reverse if not reverse else not default_reverse
    results.sort(key=key_fn, reverse=actual_reverse)

    if limit > 0:
        results = results[:limit]

    # Display
    total_matched = len(results)
    console.print(f"\n[bold cyan]🔍 Found {total_matched} tracks[/bold cyan]\n")

    table = Table(box=box.SIMPLE_HEAVY, border_style="dim", padding=(0, 1))
    table.add_column("#", style="dim", justify="right")
    table.add_column("Artist", style="bold", max_width=25, no_wrap=True)
    table.add_column("Title", max_width=35, no_wrap=True)
    table.add_column("Genre", max_width=20, no_wrap=True)
    table.add_column("Src", justify="center")
    table.add_column("Energy", justify="right")
    table.add_column("Valence", justify="right")

    for i, t in enumerate(results, 1):
        src_icon = SOURCE_ICONS.get(t["genre_source"], "") if t["tagged"] else ""
        e_str = f"{t['energy']:.2f}" if t["energy"] is not None else "[dim]—[/dim]"
        v_str = f"{t['valence']:.2f}" if t["valence"] is not None else "[dim]—[/dim]"
        primary_genre = t["genre"].split(";")[0].strip() if t["genre"] else ""
        table.add_row(
            str(i),
            t["artist"] or "[dim]—[/dim]",
            t["title"],
            primary_genre or "[dim]—[/dim]",
            src_icon,
            e_str,
            v_str,
        )

    console.print(table)
    console.print()


# ═════════════════════════════════════════════════════════════
#  EXPORT command
# ═════════════════════════════════════════════════════════════


@app.command()
def export(
    path: str = typer.Argument(
        DEFAULT_MUSIC_PATH,
        help="Folder to export (recursive)",
    ),
    fmt: str = typer.Option("csv", "--format", "-f", help="Output format: csv or json"),
) -> None:
    """Export library data to CSV or JSON (writes to stdout)."""
    if not os.path.exists(path):
        console.print(f"[bold red]Error:[/bold red] {path} not found", err=True)
        raise typer.Exit(1)

    # Scan with progress on stderr so stdout stays clean for data
    err_console = Console(stderr=True)
    from .library import scan_library

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=err_console,
        transient=True,
    ) as progress:
        task = progress.add_task("Scanning library", total=0)

        def on_progress(current: int, total: int) -> None:
            progress.update(task, completed=current, total=total)

        tracks = scan_library(path, on_progress=on_progress)

    if not tracks:
        err_console.print("[yellow]No MP3 files found.[/yellow]")
        return

    fields = [
        "path", "artist", "title", "folder", "genre", "genre_source",
        "genre_detected", "energy", "valence", "danceability", "arousal",
        "mood_happy", "mood_sad", "mood_aggressive", "mood_relaxed",
        "mood_party", "peak_energy", "intro_energy", "energy_variance",
        "tagger_version", "comment",
    ]

    if fmt == "json":
        export_data = []
        for t in tracks:
            row = {f: t.get(f) for f in fields}
            row["tagged"] = t["tagged"]
            export_data.append(row)
        sys.stdout.write(json.dumps(export_data, indent=2, default=str) + "\n")
    else:
        import csv
        import io

        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=fields + ["tagged"], extrasaction="ignore")
        writer.writeheader()
        for t in tracks:
            row = {f: t.get(f, "") for f in fields}
            row["tagged"] = t["tagged"]
            # Convert None to empty string
            for k, v in row.items():
                if v is None:
                    row[k] = ""
            writer.writerow(row)
        sys.stdout.write(output.getvalue())

    err_console.print(f"[bold green]Exported {len(tracks)} tracks as {fmt.upper()}[/bold green]")


# ═════════════════════════════════════════════════════════════
#  SUGGEST command
# ═════════════════════════════════════════════════════════════


@app.command()
def suggest(
    path: str = typer.Argument(
        DEFAULT_MUSIC_PATH,
        help="Folder to search for suggestions",
    ),
    like: Optional[str] = typer.Option(None, "--like", "-l", help="Path to a reference track"),
    genre: Optional[str] = typer.Option(None, "--genre", "-g", help="Filter by genre"),
    energy_curve: Optional[str] = typer.Option(
        None, "--energy-curve", "-c",
        help="Energy progression: rising, falling, or steady",
    ),
    count: int = typer.Option(15, "--count", "-n", help="Number of suggestions"),
) -> None:
    """Suggest tracks for set building — find similar tracks or build energy curves."""
    if not os.path.exists(path):
        console.print(f"[bold red]Error:[/bold red] {path} not found")
        raise typer.Exit(1)

    if like and not os.path.isfile(like):
        console.print(f"[bold red]Error:[/bold red] {like} not found")
        raise typer.Exit(1)

    tracks = _scan_with_progress(path)
    tagged = [t for t in tracks if t["tagged"] and t["energy"] is not None]

    if not tagged:
        console.print("[yellow]No tagged tracks found.[/yellow]")
        return

    # Filter by genre first if specified
    if genre:
        tagged = [t for t in tagged if genre.lower() in (t["genre"] or "").lower()]
        if not tagged:
            console.print(f"[yellow]No tracks match genre '{genre}'.[/yellow]")
            return

    if like:
        _suggest_like(like, tagged, count)
    elif energy_curve:
        _suggest_curve(tagged, energy_curve, count)
    else:
        # Default: show a diverse mix of tracks across energy levels
        _suggest_diverse(tagged, count)


def _suggest_like(ref_path: str, tracks: list[dict], count: int) -> None:
    """Find tracks similar to a reference track."""
    from .tagger import parse_filename, read_tags

    ref_tags = read_tags(ref_path)
    if not ref_tags.get("tagger_version"):
        console.print("[yellow]Reference track is not tagged by DJ Tagger. Tag it first.[/yellow]")
        raise typer.Exit(1)

    ref_energy = float(ref_tags["energy"]) if ref_tags.get("energy") else 0.5
    ref_valence = float(ref_tags["valence"]) if ref_tags.get("valence") else 0.5
    ref_danceability = float(ref_tags["danceability"]) if ref_tags.get("danceability") else 0.5
    ref_moods = {
        "happy": float(ref_tags.get("mood_happy") or 0.5),
        "sad": float(ref_tags.get("mood_sad") or 0.5),
        "aggressive": float(ref_tags.get("mood_aggressive") or 0.5),
        "relaxed": float(ref_tags.get("mood_relaxed") or 0.5),
    }
    ref_genre = (ref_tags.get("genre") or "").lower()
    artist, _, title = parse_filename(ref_path)
    ref_label = f"{artist} — {title}" if artist else title

    def similarity(t: dict) -> float:
        # Don't suggest the same file
        if os.path.abspath(t["path"]) == os.path.abspath(ref_path):
            return -999.0

        score = 0.0
        # Energy similarity (weight: 3) — close energy = good for mixing
        e_diff = abs((t["energy"] or 0.5) - ref_energy)
        score -= e_diff * 3.0

        # Valence similarity (weight: 2)
        v_diff = abs((t["valence"] or 0.5) - ref_valence)
        score -= v_diff * 2.0

        # Danceability similarity (weight: 1.5)
        d_diff = abs((t["danceability"] or 0.5) - ref_danceability)
        score -= d_diff * 1.5

        # Mood similarity (weight: 1 each)
        for mood in ("happy", "sad", "aggressive", "relaxed"):
            m_diff = abs((t[f"mood_{mood}"] or 0.5) - ref_moods[mood])
            score -= m_diff

        # Genre bonus (weight: 2)
        t_genre = (t["genre"] or "").lower()
        if ref_genre and t_genre:
            ref_parts = set(g.strip() for g in ref_genre.split(";"))
            t_parts = set(g.strip() for g in t_genre.split(";"))
            if ref_parts & t_parts:
                score += 2.0

        return score

    ranked = sorted(tracks, key=similarity, reverse=True)[:count]

    console.print(f"\n[bold cyan]🎵 Tracks similar to:[/bold cyan] [bold]{ref_label}[/bold]")
    console.print(
        f"[dim]   Energy: {ref_energy:.2f}  Valence: {ref_valence:.2f}  "
        f"Genre: {ref_tags.get('genre', '—')}[/dim]\n"
    )
    _print_suggestion_table(ranked)


def _suggest_curve(tracks: list[dict], curve: str, count: int) -> None:
    """Build a set with an energy curve."""
    if curve not in ("rising", "falling", "steady"):
        console.print(f"[bold red]Error:[/bold red] --energy-curve must be rising, falling, or steady")
        raise typer.Exit(1)

    import numpy as np

    # Build target energy progression
    if curve == "rising":
        targets = np.linspace(0.3, 0.95, count)
        label = "🔺 Rising energy"
    elif curve == "falling":
        targets = np.linspace(0.95, 0.3, count)
        label = "🔻 Falling energy"
    else:
        avg_e = np.mean([t["energy"] for t in tracks if t["energy"] is not None])
        targets = np.full(count, avg_e)
        label = f"➡️  Steady energy (~{avg_e:.2f})"

    selected: list[dict] = []
    used_paths: set[str] = set()

    for target_e in targets:
        best: dict | None = None
        best_score = float("inf")
        for t in tracks:
            if t["path"] in used_paths:
                continue
            if t["energy"] is None:
                continue
            diff = abs(t["energy"] - target_e)
            if diff < best_score:
                best_score = diff
                best = t
        if best:
            selected.append(best)
            used_paths.add(best["path"])

    console.print(f"\n[bold cyan]{label}[/bold cyan] — {len(selected)} tracks\n")
    _print_suggestion_table(selected, show_index=True, energy_bar=True)


def _suggest_diverse(tracks: list[dict], count: int) -> None:
    """Suggest a diverse mix of tracks across energy levels."""
    import numpy as np

    # Pick tracks spread evenly across the energy range
    sorted_by_energy = sorted(tracks, key=lambda t: t["energy"] or 0)
    step = max(1, len(sorted_by_energy) // count)
    selected = sorted_by_energy[::step][:count]
    # Sort selected by energy for nice display
    selected.sort(key=lambda t: t["energy"] or 0)

    console.print(f"\n[bold cyan]🎲 Diverse selection[/bold cyan] — {len(selected)} tracks across energy range\n")
    _print_suggestion_table(selected, show_index=True, energy_bar=True)


def _print_suggestion_table(
    tracks: list[dict],
    show_index: bool = True,
    energy_bar: bool = False,
) -> None:
    """Print a table of suggested tracks."""
    table = Table(box=box.SIMPLE_HEAVY, border_style="dim", padding=(0, 1))
    if show_index:
        table.add_column("#", style="dim", justify="right")
    table.add_column("Artist", style="bold", max_width=25, no_wrap=True)
    table.add_column("Title", max_width=35, no_wrap=True)
    table.add_column("Genre", max_width=20, no_wrap=True)
    table.add_column("Energy", justify="right")
    table.add_column("Valence", justify="right")
    if energy_bar:
        table.add_column("Level")

    for i, t in enumerate(tracks, 1):
        e_str = f"{t['energy']:.2f}" if t["energy"] is not None else "—"
        v_str = f"{t['valence']:.2f}" if t["valence"] is not None else "—"
        primary_genre = t["genre"].split(";")[0].strip() if t["genre"] else ""
        row: list[str] = []
        if show_index:
            row.append(str(i))
        row.extend([
            t["artist"] or "[dim]—[/dim]",
            t["title"],
            primary_genre or "[dim]—[/dim]",
            e_str,
            v_str,
        ])
        if energy_bar:
            row.append(_mini_bar(t["energy"] or 0))
        table.add_row(*row)

    console.print(table)
    console.print()


# ═════════════════════════════════════════════════════════════
#  HEALTH command
# ═════════════════════════════════════════════════════════════


@app.command()
def health(
    path: str = typer.Argument(
        DEFAULT_MUSIC_PATH,
        help="Folder to check",
    ),
) -> None:
    """Check collection health — find weak spots, missing data, and quality issues."""
    from collections import Counter

    if not os.path.exists(path):
        console.print(f"[bold red]Error:[/bold red] {path} not found")
        raise typer.Exit(1)

    tracks = _scan_with_progress(path)
    if not tracks:
        console.print("[yellow]No MP3 files found.[/yellow]")
        return

    tagged = [t for t in tracks if t["tagged"]]
    untagged = [t for t in tracks if not t["tagged"]]
    ml_only = [t for t in tagged if t["genre_source"] == "ml"]
    no_artist = [t for t in tracks if not t["artist"]]
    no_genre = [t for t in tagged if not t["genre"]]
    no_energy = [t for t in tagged if t["energy"] is None]
    v4_tracks = [t for t in tagged if t["tagger_version"] and t["tagger_version"] != TAGGER_VERSION]

    # Genre frequency — find rare genres
    genre_counter: Counter = Counter()
    for t in tagged:
        primary = (t["genre"] or "").split(";")[0].strip()
        if primary:
            genre_counter[primary] += 1
    rare_genres = [(g, c) for g, c in genre_counter.items() if c <= 2]
    rare_genres.sort(key=lambda x: x[1])

    # Build report
    console.print(f"\n[bold cyan]🏥 Collection Health Report[/bold cyan]")
    console.print(f"[dim]{path} — {len(tracks)} files[/dim]\n")

    # Summary panel
    issues: list[str] = []
    good: list[str] = []

    if untagged:
        issues.append(f"[yellow]⚠[/yellow]  [bold]{len(untagged)}[/bold] tracks untagged")
    else:
        good.append("[green]✓[/green]  All tracks tagged")

    if ml_only:
        pct = len(ml_only) / len(tagged) * 100 if tagged else 0
        issues.append(
            f"[yellow]⚠[/yellow]  [bold]{len(ml_only)}[/bold] tracks with ML-only genres "
            f"({pct:.0f}% — weakest confidence)"
        )

    bp_count = sum(1 for t in tagged if t["genre_source"] == "beatport")
    fm_count = sum(1 for t in tagged if t["genre_source"] == "lastfm+ml")
    if bp_count or fm_count:
        good.append(
            f"[green]✓[/green]  [bold]{bp_count + fm_count}[/bold] tracks with "
            f"Beatport/Last.fm genres"
        )

    if no_artist:
        issues.append(
            f"[yellow]⚠[/yellow]  [bold]{len(no_artist)}[/bold] tracks with no artist in filename"
        )
    else:
        good.append("[green]✓[/green]  All tracks have artist info")

    if no_genre:
        issues.append(f"[yellow]⚠[/yellow]  [bold]{len(no_genre)}[/bold] tagged tracks with no genre")

    if no_energy:
        issues.append(f"[yellow]⚠[/yellow]  [bold]{len(no_energy)}[/bold] tagged tracks missing energy data")
    else:
        if tagged:
            good.append("[green]✓[/green]  All tagged tracks have energy/valence scores")

    if rare_genres:
        issues.append(
            f"[yellow]⚠[/yellow]  [bold]{len(rare_genres)}[/bold] genres with only 1-2 tracks"
        )

    if v4_tracks:
        issues.append(
            f"[yellow]⚠[/yellow]  [bold]{len(v4_tracks)}[/bold] tracks tagged with older version "
            f"(re-tag with --force for improved v5 accuracy)"
        )

    # Print issues first, then good
    for line in issues:
        console.print(f"  {line}")
    for line in good:
        console.print(f"  {line}")

    console.print()

    # Overall score
    if tagged:
        score_parts = []
        # Coverage: 0-30 points
        coverage = len(tagged) / len(tracks) if tracks else 0
        score_parts.append(coverage * 30)
        # High-quality sources: 0-40 points
        hq_pct = (bp_count + fm_count) / len(tagged) if tagged else 0
        score_parts.append(hq_pct * 40)
        # Artist info: 0-15 points
        artist_pct = (len(tracks) - len(no_artist)) / len(tracks) if tracks else 0
        score_parts.append(artist_pct * 15)
        # Genre completeness: 0-15 points
        genre_pct = (len(tagged) - len(no_genre)) / len(tagged) if tagged else 0
        score_parts.append(genre_pct * 15)

        total_score = sum(score_parts)
        if total_score >= 85:
            grade, grade_color = "Excellent", "green"
        elif total_score >= 70:
            grade, grade_color = "Good", "cyan"
        elif total_score >= 50:
            grade, grade_color = "Fair", "yellow"
        else:
            grade, grade_color = "Needs Work", "red"

        console.print(
            f"  [bold]Health Score:[/bold] "
            f"[bold {grade_color}]{total_score:.0f}/100 — {grade}[/bold {grade_color}]"
        )
        console.print()

    # ─── Detail sections ────────────────────────────────────

    # Untagged files
    if untagged and len(untagged) <= 20:
        ut_table = Table(
            box=box.SIMPLE, border_style="dim",
            title="Untagged Files", title_style="bold yellow",
        )
        ut_table.add_column("File", style="dim")
        for t in untagged[:20]:
            label = f"{t['artist']} — {t['title']}" if t["artist"] else t["title"]
            ut_table.add_row(label)
        console.print(ut_table)
        console.print()

    # ML-only tracks (show up to 15)
    if ml_only:
        ml_table = Table(
            box=box.SIMPLE, border_style="dim",
            title=f"ML-Only Genres ({len(ml_only)} tracks — consider re-checking)",
            title_style="bold blue",
        )
        ml_table.add_column("Artist", style="bold", max_width=25, no_wrap=True)
        ml_table.add_column("Title", max_width=35, no_wrap=True)
        ml_table.add_column("Genre (ML)", style="dim")
        for t in ml_only[:15]:
            ml_table.add_row(
                t["artist"] or "—",
                t["title"],
                t["genre"] or "—",
            )
        if len(ml_only) > 15:
            ml_table.add_row(f"[dim]… and {len(ml_only) - 15} more[/dim]", "", "")
        console.print(ml_table)
        console.print()

    # No-artist files (show up to 15)
    if no_artist:
        na_table = Table(
            box=box.SIMPLE, border_style="dim",
            title=f"Missing Artist ({len(no_artist)} files)",
            title_style="bold yellow",
        )
        na_table.add_column("Filename", style="dim")
        for t in no_artist[:15]:
            na_table.add_row(os.path.basename(t["path"]))
        if len(no_artist) > 15:
            na_table.add_row(f"[dim]… and {len(no_artist) - 15} more[/dim]")
        console.print(na_table)
        console.print()

    # Rare genres
    if rare_genres:
        rg_table = Table(
            box=box.SIMPLE, border_style="dim",
            title="Rare Genres (1-2 tracks)", title_style="bold",
        )
        rg_table.add_column("Genre", style="bold")
        rg_table.add_column("Tracks", justify="right", style="dim")
        for g, c in rare_genres:
            rg_table.add_row(g, str(c))
        console.print(rg_table)
        console.print()


# ═════════════════════════════════════════════════════════════
#  CLEAN-GENRES command
# ═════════════════════════════════════════════════════════════


@app.command("clean-genres")
def clean_genres(
    path: str = typer.Argument(
        DEFAULT_MUSIC_PATH,
        help="Folder to clean (recursive)",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be cleaned without writing"),
) -> None:
    """Remove junk genre tags (URLs, spam, foreign text) and replace with detected genres."""
    from .scanner import find_mp3s
    from .tagger import clean_junk_genre, read_tags
    from .config import is_junk_genre

    if not os.path.exists(path):
        console.print(f"[bold red]Error:[/bold red] {path} not found")
        raise typer.Exit(1)

    console.print(f"\n[bold cyan]🧹 Clean Genres[/bold cyan] {'(dry run)' if dry_run else ''}")
    console.print(f"[dim]{path}[/dim]\n")

    all_mp3s = find_mp3s(path)
    cleaned = 0
    junk_found = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
        transient=not dry_run,
    ) as progress:
        task = progress.add_task("Scanning for junk genres", total=len(all_mp3s))
        for mp3 in all_mp3s:
            tags = read_tags(mp3)
            genre = tags.get("genre", "")
            if genre and is_junk_genre(genre):
                junk_found += 1
                artist = tags.get("genre", "")
                fname = os.path.basename(mp3)

                if dry_run:
                    detected = tags.get("genre_detected", "")
                    replacement = detected if detected and not is_junk_genre(detected) else "[clear]"
                    console.print(
                        f"  [yellow]✗[/yellow] [dim]{fname[:60]}[/dim]\n"
                        f"    [red]\"{genre}\"[/red] → [green]\"{replacement}\"[/green]"
                    )
                else:
                    changed, desc = clean_junk_genre(mp3)
                    if changed:
                        cleaned += 1
                        console.print(f"  [green]✓[/green] {desc}")

            progress.advance(task)

    console.print()
    if dry_run:
        console.print(
            f"[bold yellow]Dry run:[/bold yellow] found [bold]{junk_found}[/bold] "
            f"junk genres across {len(all_mp3s)} files. Run without --dry-run to fix."
        )
    else:
        console.print(
            f"[bold green]Done![/bold green] Cleaned [bold]{cleaned}[/bold] junk genres "
            f"out of {len(all_mp3s)} files."
        )
    console.print()


# ═════════════════════════════════════════════════════════════
#  FIX-AUDIT command
# ═════════════════════════════════════════════════════════════


@app.command("fix-audit")
def fix_audit(
    report: str = typer.Argument("/tmp/audit-report.json", help="Path to audit report JSON"),
    fix_filenames: bool = typer.Option(
        False, "--fix-filenames",
        help="Also rename files with no artist (requires Serato relinking after)",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be fixed without writing"),
) -> None:
    """Fix issues from a serato-reader audit report (BPM, key, genre gaps)."""
    import socket

    if not os.path.isfile(report):
        console.print(f"[bold red]Error:[/bold red] {report} not found")
        raise typer.Exit(1)

    with open(report) as f:
        data = json.load(f)

    tracks = data.get("tracks", [])
    if not tracks:
        console.print("[yellow]No tracks with issues in report.[/yellow]")
        return

    summary = data.get("summary", {})
    console.print(f"\n[bold cyan]🔧 Fix Audit Issues[/bold cyan] {'(dry run)' if dry_run else ''}")
    console.print(f"[dim]{report}[/dim]")
    console.print(f"[dim]{summary.get('unique_tracks_with_issues', len(tracks))} tracks with issues[/dim]\n")

    # Supported formats for tagging
    TAGGABLE = {".mp3", ".m4a", ".mp4", ".aac"}

    # Categorize what we can fix
    fixable: list[dict] = []
    skipped_format: list[dict] = []
    skipped_filename: list[dict] = []

    for t in tracks:
        path = t["path"]
        issues = set(t.get("issues", []))
        ext = os.path.splitext(path)[1].lower()

        if not os.path.isfile(path):
            console.print(f"  [red]✗[/red] [dim]File not found: {os.path.basename(path)}[/dim]")
            continue

        if "no artist" in issues and not fix_filenames:
            skipped_filename.append(t)
            issues.discard("no artist")

        fixable_issues = issues & {"missing BPM", "no key", "no genre"}
        if not fixable_issues:
            continue

        if ext not in TAGGABLE:
            skipped_format.append(t)
            continue

        fixable.append({"track": t, "issues": fixable_issues, "ext": ext})

    if not fixable:
        console.print("[yellow]Nothing to fix (all issues are in unsupported formats or filenames).[/yellow]")
        if skipped_format:
            console.print(f"[dim]  {len(skipped_format)} tracks in unsupported formats (WAV)[/dim]")
        if skipped_filename:
            console.print(f"[dim]  {len(skipped_filename)} tracks need filename fix (use --fix-filenames)[/dim]")
        console.print()
        return

    # Load models for BPM/key/genre detection
    socket.setdefaulttimeout(10)
    needs_ml = any("no genre" in f["issues"] for f in fixable)

    if needs_ml or any("missing BPM" in f["issues"] or "no key" in f["issues"] for f in fixable):
        console.print("[dim]Loading analysis models...[/dim]")
        from .analyzer import analyze_track, load_models
        from .genres import resolve_genres
        from .tagger import write_tags, parse_filename
        from .config import GENRE_KEEP_PROB

        models = load_models()
        console.print("[bold green]✓[/bold green] Models loaded\n")
    else:
        models = None

    fixed_count = 0
    for item in fixable:
        t = item["track"]
        issues = item["issues"]
        path = t["path"]
        ext = item["ext"]
        fname = os.path.basename(path)
        crates = ", ".join(t.get("crates", []))

        console.print(f"  [bold]{fname[:65]}[/bold]")
        console.print(f"  [dim]Crates: {crates[:70]}[/dim]")
        console.print(f"  Issues: [yellow]{', '.join(sorted(issues))}[/yellow]")

        if dry_run:
            console.print(f"  [dim]→ Would analyze and fix[/dim]\n")
            continue

        if ext == ".mp3" and models is not None:
            try:
                artist, artist_clean, title = parse_filename(path)
                result = analyze_track(path, models)

                # Resolve genre if needed
                if "no genre" in issues:
                    final_genres, genre_source = resolve_genres(
                        artist, artist_clean, title,
                        result["genres"],
                        ml_electronic_genres=result.get("electronic_genres"),
                        use_beatport=True,
                        genre_keep_prob=GENRE_KEEP_PROB,
                    )
                else:
                    final_genres, genre_source = [], "ml"

                ok, action = write_tags(path, result, genre_source, final_genres)
                if ok:
                    fixes = []
                    if "missing BPM" in issues:
                        fixes.append(f"BPM: {result.get('bpm', '?')}")
                    if "no key" in issues:
                        fixes.append(f"Key: {result.get('key', '?')}")
                    if "no genre" in issues:
                        genre_str = "; ".join(final_genres[:3]) if final_genres else "(none)"
                        fixes.append(f"Genre: {genre_str} [{genre_source}]")
                    console.print(f"  [green]✓[/green] {', '.join(fixes)}\n")
                    fixed_count += 1
                else:
                    console.print(f"  [red]✗[/red] Write failed: {action}\n")
            except Exception as ex:
                console.print(f"  [red]✗[/red] Error: {ex}\n")

        elif ext in (".m4a", ".mp4", ".aac"):
            # M4A: can detect BPM/key and write with mutagen MP4
            try:
                from mutagen.mp4 import MP4

                result = analyze_track(path, models) if models else None
                if result is None:
                    console.print(f"  [red]✗[/red] Models not loaded\n")
                    continue

                audio = MP4(path)
                fixes = []

                if "missing BPM" in issues and result.get("bpm"):
                    audio.tags["tmpo"] = [int(round(result["bpm"]))]
                    fixes.append(f"BPM: {result['bpm']:.2f}")

                # M4A doesn't have a standard key tag, skip key for now
                if "no key" in issues:
                    fixes.append("Key: [dim]not supported for m4a[/dim]")

                if "no genre" in issues:
                    artist, artist_clean, title = parse_filename(path)
                    final_genres, genre_source = resolve_genres(
                        artist, artist_clean, title,
                        result["genres"],
                        ml_electronic_genres=result.get("electronic_genres"),
                        use_beatport=True,
                        genre_keep_prob=GENRE_KEEP_PROB,
                    )
                    if final_genres:
                        audio.tags["\xa9gen"] = ["; ".join(final_genres[:3])]
                        fixes.append(f"Genre: {'; '.join(final_genres[:3])} [{genre_source}]")

                if fixes:
                    audio.save()
                    console.print(f"  [green]✓[/green] {', '.join(fixes)}\n")
                    fixed_count += 1
            except Exception as ex:
                console.print(f"  [red]✗[/red] Error: {ex}\n")

    console.print(f"[bold green]Done![/bold green] Fixed {fixed_count} tracks.")
    if skipped_format:
        console.print(f"[dim]Skipped {len(skipped_format)} WAV files (can't write tags to WAV)[/dim]")
    if skipped_filename:
        console.print(
            f"[dim]Skipped {len(skipped_filename)} filename issues "
            f"(use --fix-filenames to rename)[/dim]"
        )
    console.print()


# ═════════════════════════════════════════════════════════════
#  CONVERT-KEYS command
# ═════════════════════════════════════════════════════════════


@app.command("convert-keys")
def convert_keys(
    path: str = typer.Argument(
        DEFAULT_MUSIC_PATH,
        help="Folder to convert (recursive)",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be converted without writing"),
) -> None:
    """Convert all standard-notation keys (Am, Fm, etc.) to Camelot notation (8A, 4A, etc.)."""
    import re

    from mutagen.id3 import ID3, TKEY

    from .config import CAMELOT_MAP
    from .scanner import find_mp3s

    if not os.path.exists(path):
        console.print(f"[bold red]Error:[/bold red] {path} not found")
        raise typer.Exit(1)

    console.print(f"\n[bold cyan]🔑 Convert Keys to Camelot[/bold cyan] {'(dry run)' if dry_run else ''}")
    console.print(f"[dim]{path}[/dim]\n")

    all_mp3s = find_mp3s(path)
    camelot_pattern = re.compile(r"^\d{1,2}[AB]$", re.IGNORECASE)

    converted = 0
    already_camelot = 0
    no_key = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
        transient=not dry_run,
    ) as progress:
        task = progress.add_task("Scanning keys", total=len(all_mp3s))
        for mp3 in all_mp3s:
            try:
                tags = ID3(mp3)
                tkey = tags.getall("TKEY")
                if not tkey or not tkey[0].text or not tkey[0].text[0].strip():
                    no_key += 1
                    progress.advance(task)
                    continue

                key = tkey[0].text[0].strip()

                if camelot_pattern.match(key):
                    # Normalize lowercase camelot (4a → 4A)
                    normalized = key.upper()
                    if normalized != key:
                        if not dry_run:
                            tags.delall("TKEY")
                            tags.add(TKEY(encoding=3, text=[normalized]))
                            tags.save(mp3)
                        converted += 1
                    else:
                        already_camelot += 1
                    progress.advance(task)
                    continue

                # Try to convert
                camelot = CAMELOT_MAP.get(key)
                if camelot:
                    if dry_run:
                        console.print(
                            f"  {key:>4s} → [green]{camelot:>3s}[/green]  "
                            f"[dim]{os.path.basename(mp3)[:60]}[/dim]"
                        )
                    else:
                        tags.delall("TKEY")
                        tags.add(TKEY(encoding=3, text=[camelot]))
                        tags.save(mp3)
                    converted += 1
                else:
                    console.print(
                        f"  [yellow]?[/yellow] Unknown key [bold]{key}[/bold]  "
                        f"[dim]{os.path.basename(mp3)[:55]}[/dim]"
                    )
            except Exception:
                pass
            progress.advance(task)

    console.print()
    if dry_run:
        console.print(
            f"[bold yellow]Dry run:[/bold yellow] {converted} keys to convert, "
            f"{already_camelot} already Camelot, {no_key} no key set"
        )
    else:
        console.print(
            f"[bold green]Done![/bold green] Converted [bold]{converted}[/bold] keys to Camelot. "
            f"{already_camelot} already Camelot, {no_key} no key set."
        )
    console.print()


# ═════════════════════════════════════════════════════════════
#  BENCH-BPM command (diagnostic)
# ═════════════════════════════════════════════════════════════


@app.command("bench-bpm")
def bench_bpm(
    path: str = typer.Argument(
        DEFAULT_MUSIC_PATH,
        help="Folder to benchmark (recursive)",
    ),
    count: int = typer.Option(500, "--count", "-n", help="Number of tracks to test"),
) -> None:
    """Benchmark BPM + Key detection: compare Serato vs DSP vs TempoCNN on random tracks."""
    import random

    import numpy as np
    from mutagen.id3 import ID3

    from .scanner import find_mp3s
    from .tagger import parse_filename
    from .config import CAMELOT_MAP

    if not os.path.exists(path):
        console.print(f"[bold red]Error:[/bold red] {path} not found")
        raise typer.Exit(1)

    console.print(f"\n[bold cyan]🎯 BPM + Key Benchmark[/bold cyan]")
    console.print(f"[dim]{path}[/dim]\n")

    all_mp3s = find_mp3s(path)
    # Only tracks that have existing BPM (from Serato)
    with_bpm: list[tuple[str, float, str]] = []
    for mp3 in all_mp3s:
        try:
            tags = ID3(mp3)
            tbpm = tags.getall("TBPM")
            tkey = tags.getall("TKEY")
            bpm_val = 0.0
            key_val = ""
            if tbpm and tbpm[0].text and tbpm[0].text[0].strip():
                bpm_val = float(tbpm[0].text[0].strip())
            if tkey and tkey[0].text and tkey[0].text[0].strip():
                key_val = tkey[0].text[0].strip()
            if bpm_val > 0:
                with_bpm.append((mp3, bpm_val, key_val))
        except Exception:
            pass

    console.print(f"Found [bold]{len(with_bpm)}[/bold] tracks with existing BPM")

    random.seed(42)
    sample = random.sample(with_bpm, min(count, len(with_bpm)))
    console.print(f"Sampling [bold]{len(sample)}[/bold] tracks\n")

    # Lazy import essentia
    console.print("[dim]Loading models...[/dim]")

    import warnings
    import logging
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    warnings.filterwarnings("ignore")
    logging.getLogger("essentia").setLevel(logging.ERROR)
    logging.getLogger("tensorflow").setLevel(logging.ERROR)

    import essentia
    essentia.log.warningActive = False
    essentia.log.infoActive = False
    import essentia.standard as es

    from .config import MODEL_DIR

    # Load TempoCNN
    tempo_cnn_path = f"{MODEL_DIR}/deepsquare-k16-3.pb"
    tempo_cnn = None
    if os.path.isfile(tempo_cnn_path):
        tempo_cnn = es.TensorflowPredictTempoCNN(graphFilename=tempo_cnn_path)
        console.print("[bold green]✓[/bold green] TempoCNN loaded")
    else:
        console.print("[yellow]⚠ TempoCNN model not found — skipping[/yellow]")

    console.print()

    # Run benchmark
    results: list[dict] = []
    errors = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Benchmarking", total=len(sample))
        for mp3, serato_bpm, serato_key in sample:
            artist, _, title = parse_filename(mp3)
            label = f"{artist} - {title}" if artist else os.path.basename(mp3)

            row: dict = {
                "label": label[:55],
                "serato_bpm": serato_bpm,
                "serato_key": serato_key,
                "dsp_bpm": None,
                "cnn_bpm": None,
                "detected_key": None,
            }

            try:
                audio_44k = es.MonoLoader(filename=mp3, sampleRate=44100)()

                # DSP BPM
                rhythm = es.RhythmExtractor2013(method="multifeature")
                dsp_bpm, _, _, _, _ = rhythm(audio_44k)
                row["dsp_bpm"] = round(float(dsp_bpm), 2)

                # TempoCNN BPM
                if tempo_cnn is not None:
                    audio_11k = es.MonoLoader(filename=mp3, sampleRate=11025)()
                    preds = tempo_cnn(audio_11k)
                    avg = np.mean(preds, axis=0)
                    peak = int(np.argmax(avg))
                    w = 5
                    lo, hi = max(0, peak - w), min(len(avg), peak + w + 1)
                    cnn_bpm = float(np.average(np.arange(lo, hi) + 30, weights=avg[lo:hi]))
                    row["cnn_bpm"] = round(cnn_bpm, 2)

                # Key detection
                key_ext = es.KeyExtractor()
                key_name, scale, _ = key_ext(audio_44k)
                standard_key = f"{key_name}{'m' if scale == 'minor' else ''}"
                row["detected_key"] = CAMELOT_MAP.get(standard_key, standard_key)

                results.append(row)
            except Exception:
                errors += 1

            progress.advance(task)

    # ─── BPM Analysis ────────────────────────────────────
    console.print()

    def _is_bpm_match(a: float, b: float, tol: float = 1.0) -> bool:
        """Match within tolerance, accounting for half/double time."""
        diff = abs(a - b)
        half = abs(a - b * 2)
        double = abs(a * 2 - b)
        return diff <= tol or half <= tol or double <= tol

    n = len(results)
    n_cnn = sum(1 for r in results if r["cnn_bpm"] is not None)

    dsp_exact = sum(1 for r in results if r["dsp_bpm"] and _is_bpm_match(r["serato_bpm"], r["dsp_bpm"], 0.5))
    dsp_close = sum(1 for r in results if r["dsp_bpm"] and _is_bpm_match(r["serato_bpm"], r["dsp_bpm"], 1.0))
    dsp_loose = sum(1 for r in results if r["dsp_bpm"] and _is_bpm_match(r["serato_bpm"], r["dsp_bpm"], 2.0))

    cnn_exact = sum(1 for r in results if r["cnn_bpm"] and _is_bpm_match(r["serato_bpm"], r["cnn_bpm"], 0.5))
    cnn_close = sum(1 for r in results if r["cnn_bpm"] and _is_bpm_match(r["serato_bpm"], r["cnn_bpm"], 1.0))
    cnn_loose = sum(1 for r in results if r["cnn_bpm"] and _is_bpm_match(r["serato_bpm"], r["cnn_bpm"], 2.0))

    bpm_table = Table(
        box=box.ROUNDED, border_style="dim",
        title="BPM Accuracy vs Serato", title_style="bold",
    )
    bpm_table.add_column("Tolerance", style="bold")
    bpm_table.add_column("DSP (RhythmExtractor)", justify="right")
    bpm_table.add_column("TempoCNN", justify="right")
    bpm_table.add_row(
        "±0.5 BPM (exact)",
        f"[green]{dsp_exact}[/green]/{n} ({dsp_exact/n*100:.1f}%)" if n else "—",
        f"[green]{cnn_exact}[/green]/{n_cnn} ({cnn_exact/n_cnn*100:.1f}%)" if n_cnn else "—",
    )
    bpm_table.add_row(
        "±1.0 BPM (close)",
        f"[green]{dsp_close}[/green]/{n} ({dsp_close/n*100:.1f}%)" if n else "—",
        f"[green]{cnn_close}[/green]/{n_cnn} ({cnn_close/n_cnn*100:.1f}%)" if n_cnn else "—",
    )
    bpm_table.add_row(
        "±2.0 BPM (loose)",
        f"[green]{dsp_loose}[/green]/{n} ({dsp_loose/n*100:.1f}%)" if n else "—",
        f"[green]{cnn_loose}[/green]/{n_cnn} ({cnn_loose/n_cnn*100:.1f}%)" if n_cnn else "—",
    )
    console.print(bpm_table)
    console.print()

    # ─── Key Analysis ────────────────────────────────────
    key_results = [r for r in results if r["serato_key"] and r["detected_key"]]
    key_match = sum(1 for r in key_results if r["serato_key"] == r["detected_key"])
    key_total = len(key_results)

    key_table = Table(
        box=box.ROUNDED, border_style="dim",
        title="Key Accuracy vs Serato (Camelot)", title_style="bold",
    )
    key_table.add_column("Metric", style="bold")
    key_table.add_column("Value", justify="right")
    key_table.add_row(
        "Exact Camelot match",
        f"[green]{key_match}[/green]/{key_total} ({key_match/key_total*100:.1f}%)" if key_total else "—",
    )
    key_table.add_row(
        "Mismatch",
        f"[yellow]{key_total - key_match}[/yellow]/{key_total} ({(key_total-key_match)/key_total*100:.1f}%)" if key_total else "—",
    )
    console.print(key_table)
    console.print()

    # ─── BPM Mismatches ─────────────────────────────────
    bpm_mismatches: list[dict] = []
    for r in results:
        dsp_off = abs(r["serato_bpm"] - r["dsp_bpm"]) if r["dsp_bpm"] else 999
        cnn_off = abs(r["serato_bpm"] - r["cnn_bpm"]) if r["cnn_bpm"] else 999
        if r["dsp_bpm"] and not _is_bpm_match(r["serato_bpm"], r["dsp_bpm"], 2.0):
            dsp_off = min(dsp_off, abs(r["serato_bpm"] - r["dsp_bpm"] * 2), abs(r["serato_bpm"] * 2 - r["dsp_bpm"]))
        if r["cnn_bpm"] and not _is_bpm_match(r["serato_bpm"], r["cnn_bpm"], 2.0):
            cnn_off = min(cnn_off, abs(r["serato_bpm"] - r["cnn_bpm"] * 2), abs(r["serato_bpm"] * 2 - r["cnn_bpm"]))

        if dsp_off > 2.0 or cnn_off > 2.0:
            r["dsp_off"] = dsp_off
            r["cnn_off"] = cnn_off
            bpm_mismatches.append(r)

    if bpm_mismatches:
        mm_table = Table(
            box=box.SIMPLE, border_style="dim",
            title=f"BPM Mismatches (>2 off) — {len(bpm_mismatches)} tracks",
            title_style="bold yellow",
        )
        mm_table.add_column("Track", max_width=45, no_wrap=True)
        mm_table.add_column("Serato", justify="right")
        mm_table.add_column("DSP", justify="right")
        mm_table.add_column("CNN", justify="right")
        for r in sorted(bpm_mismatches, key=lambda x: max(x.get("dsp_off", 0), x.get("cnn_off", 0)), reverse=True)[:25]:
            dsp_str = f"{r['dsp_bpm']:.2f}" if r["dsp_bpm"] else "—"
            cnn_str = f"{r['cnn_bpm']:.2f}" if r["cnn_bpm"] else "—"
            dsp_color = "red" if r.get("dsp_off", 0) > 2 else "green"
            cnn_color = "red" if r.get("cnn_off", 0) > 2 else "green"
            mm_table.add_row(
                r["label"],
                f"{r['serato_bpm']}",
                f"[{dsp_color}]{dsp_str}[/{dsp_color}]",
                f"[{cnn_color}]{cnn_str}[/{cnn_color}]",
            )
        console.print(mm_table)
        console.print()

    # ─── Key Mismatches (sample) ─────────────────────────
    key_mismatches = [r for r in key_results if r["serato_key"] != r["detected_key"]]
    if key_mismatches:
        km_table = Table(
            box=box.SIMPLE, border_style="dim",
            title=f"Key Mismatches — {len(key_mismatches)} tracks (showing 25)",
            title_style="bold yellow",
        )
        km_table.add_column("Track", max_width=45, no_wrap=True)
        km_table.add_column("Serato", justify="center")
        km_table.add_column("Detected", justify="center")
        for r in key_mismatches[:25]:
            km_table.add_row(
                r["label"],
                f"[green]{r['serato_key']}[/green]",
                f"[yellow]{r['detected_key']}[/yellow]",
            )
        console.print(km_table)

    if errors:
        console.print(f"\n[dim]{errors} tracks failed to analyze[/dim]")
    console.print()


# ─── Cleanup ────────────────────────────────────────────────


def _cleanup() -> None:
    global _log_fh, _err_fh
    if _log_fh:
        _log_fh.close()
        _log_fh = None
    if _err_fh:
        _err_fh.close()
        _err_fh = None


# ─── Version callback ──────────────────────────────────────


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"[bold cyan]djtagger[/bold cyan] {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        help="Show version and exit",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """🎛  DJ Tagger — Autonomous DJ music tagger."""
    pass
