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

    # Lazy imports (heavy — TF models)
    from .analyzer import analyze_track, load_models
    from .genres import resolve_genres
    from .scanner import filter_untagged, find_mp3s
    from .tagger import fix_comments as do_fix_comments
    from .tagger import parse_filename, write_tags

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
        _cleanup()
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
        _cleanup()
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
            from rich.console import Group
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

    _cleanup()


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
    """Show library statistics — tagged count, genre distribution, etc."""
    from collections import Counter

    from .scanner import find_mp3s
    from .tagger import read_tags

    if not os.path.exists(path):
        console.print(f"[bold red]Error:[/bold red] {path} not found")
        raise typer.Exit(1)

    console.print(f"\n[bold cyan]📊 Library Statistics[/bold cyan]")
    console.print(f"[dim]{path}[/dim]\n")

    with console.status("[bold cyan]Scanning…[/bold cyan]", spinner="dots"):
        all_mp3s = find_mp3s(path)

    if not all_mp3s:
        console.print("[yellow]No MP3 files found.[/yellow]")
        return

    tagged = 0
    untagged = 0
    genre_counter: Counter = Counter()
    source_counter: Counter = Counter()
    energies: list[float] = []
    valences: list[float] = []
    versions: Counter = Counter()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Reading tags", total=len(all_mp3s))
        for mp3 in all_mp3s:
            tags = read_tags(mp3)
            if tags.get("tagger_version"):
                tagged += 1
                # Genre
                genre = tags.get("genre", "")
                if genre:
                    for g in genre.split(";"):
                        g = g.strip()
                        if g:
                            genre_counter[g] += 1
                # Source
                src = tags.get("genre_source", "unknown")
                source_counter[src] += 1
                # Energy/valence
                if tags.get("energy"):
                    energies.append(float(tags["energy"]))
                if tags.get("valence"):
                    valences.append(float(tags["valence"]))
                # Version
                versions[tags.get("tagger_version", "?")] += 1
            else:
                untagged += 1
            progress.advance(task)

    # ─── Overview table ─────────────────────────────────────
    overview = Table(box=box.ROUNDED, border_style="dim", title="Overview", title_style="bold")
    overview.add_column("Metric", style="bold")
    overview.add_column("Value", justify="right")
    overview.add_row("Total files", str(len(all_mp3s)))
    overview.add_row("Tagged", f"[green]{tagged}[/green]")
    overview.add_row("Untagged", f"[yellow]{untagged}[/yellow]" if untagged else "[dim]0[/dim]")
    pct = (tagged / len(all_mp3s) * 100) if all_mp3s else 0
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

    # ─── Energy / Valence ───────────────────────────────────
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

    # ─── Tagger versions ───────────────────────────────────
    if versions:
        ver_table = Table(box=box.SIMPLE, border_style="dim")
        ver_table.add_column("Tagger Version", style="dim")
        ver_table.add_column("Tracks", justify="right", style="dim")
        for v, c in versions.most_common():
            ver_table.add_row(v, str(c))
        console.print(ver_table)

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
