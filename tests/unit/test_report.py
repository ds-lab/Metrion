from datetime import datetime, timedelta

import polars as pl
import pytest
from rich.console import Console
from typer.testing import CliRunner

from metrion.cli import app
from metrion.energy_attribution import report
from metrion.energy_attribution.wide_result import ENERGY_COLUMNS, IDENTITY_COLUMNS

T0 = datetime(2026, 9, 29, 12, 0, 0)

# pid, tid, creator, name, cmdline, cgroup
THREADS = [
    (100, 100, 1, "bash", "bash run.sh", "/system.slice/run.scope"),  # waits, never gets energy
    (200, 200, 100, "python3", "python3 train.py", "/system.slice/run.scope"),
    (200, 201, 200, "worker", "python3 train.py", "/system.slice/run.scope"),
    (300, 300, 200, "python3", "python3 -c spin", "/system.slice/run.scope"),  # grandchild of bash
    (400, 400, 1, "nginx", "nginx -g daemon off;", "/system.slice/nginx.service"),
]


def _threads() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "logical_entity_id": f"t{tid}",
                "native_pid": pid,
                "native_thread_id": tid,
                "native_parent_pid": creator,
                "executable_name": name,
                "cmdline": cmdline,
                "cgroup_name": cgroup,
                "user_id": 0,
                "group_id": 0,
            }
            for pid, tid, creator, name, cmdline, cgroup in THREADS
        ]
    )


def _rows(energy: dict[int, float], seconds: int = 2) -> pl.DataFrame:
    """seconds windows per thread, each with energy[tid] J of CPU, a quarter of it idle."""
    threads = _threads().filter(pl.col("native_thread_id").is_in(list(energy)))
    frames = []
    for second in range(seconds):
        frames.append(
            threads.with_columns(
                window_start=pl.lit(T0 + timedelta(seconds=second)),
                window_end=pl.lit(T0 + timedelta(seconds=second + 1)),
                cpu_total_j=pl.col("native_thread_id").replace_strict(energy, return_dtype=pl.Float64),
            ).with_columns(
                cpu_idle_j=pl.col("cpu_total_j") / 4,
                cpu_dynamic_j=pl.col("cpu_total_j") * 3 / 4,
                **{column: pl.lit(0.0) for column in ENERGY_COLUMNS if not column.startswith("cpu")},
            )
        )
    return pl.concat(frames).select(
        "window_start", "window_end", "logical_entity_id", *IDENTITY_COLUMNS, *ENERGY_COLUMNS
    )


ENERGY = {200: 4.0, 201: 2.0, 300: 8.0, 400: 1.0}


def test_a_process_takes_parent_and_name_from_its_main_thread():
    procs = report.processes(_threads())

    python = procs.filter(pl.col("pid") == 200).row(0, named=True)
    assert (python["ppid"], python["name"]) == (100, "python3")


def test_pids_bring_all_descendants_even_through_a_parent_without_energy():
    procs = report.processes(_threads())

    assert report.with_descendants(procs, {100}) == {100, 200, 300}


def test_a_cmdline_selects_the_matching_processes_and_their_children():
    rows, procs = _rows(ENERGY), report.processes(_threads())

    selected = report.select(rows, procs, "train.py", None, None)

    assert set(selected["native_pid"]) == {200, 300}


def test_a_cmdline_without_match_selects_nothing():
    rows, procs = _rows(ENERGY), report.processes(_threads())

    assert report.select(rows, procs, "no such program", None, None).is_empty()


def test_a_cgroup_matches_part_of_the_path():
    rows, procs = _rows(ENERGY), report.processes(_threads())

    selected = report.select(rows, procs, None, None, "nginx")

    assert set(selected["native_pid"]) == {400}


def test_the_table_has_one_row_per_process_with_its_threads_summed():
    rows, procs = _rows(ENERGY), report.processes(_threads())

    table = report.summarize(rows, procs, False)

    python = table.filter(pl.col("pid") == 200).row(0, named=True)
    assert python["threads"] == 2
    assert python["cpu_idle_j"] + python["cpu_dynamic_j"] == pytest.approx(12.0)
    assert python["total_j"] == pytest.approx(12.0)
    assert table["pid"].to_list() == [300, 200, 400], "the largest first"


def test_the_table_has_one_row_per_cgroup():
    rows, procs = _rows(ENERGY), report.processes(_threads())

    table = report.summarize(rows, procs, True)

    assert table.select("cgroup", "processes", "threads", "total_j").rows() == [
        ("/system.slice/run.scope", 2, 3, pytest.approx(28.0)),
        ("/system.slice/nginx.service", 1, 1, pytest.approx(2.0)),
    ]


def test_a_live_result_directory_is_read_as_it_is(tmp_path):
    _rows(ENERGY, seconds=4).write_parquet(tmp_path / "part-00000.parquet")

    rows, threads = report.load(tmp_path, T0 + timedelta(seconds=1), T0 + timedelta(seconds=3))

    assert rows["window_start"].unique().sort().to_list() == [T0 + timedelta(seconds=1), T0 + timedelta(seconds=2)]
    assert threads.height == len(ENERGY)


def _printed(rows, selected, table, label, width) -> str:
    console = Console(width=width, record=True)
    report.print_report(rows, selected, table, label, console=console)
    return console.export_text()


def test_the_report_fits_a_narrow_terminal_and_sums_up_what_it_leaves_out(monkeypatch):
    monkeypatch.setattr(report, "TOP", 2)
    rows, procs = _rows(ENERGY), report.processes(_threads())

    text = _printed(rows, rows, report.summarize(rows, procs, False), "all processes", 80)

    assert all(len(line) <= 80 for line in text.splitlines())
    assert "… 1 more" in text
    assert "30.0" in text, "the sum covers the rows left out"


def test_an_empty_selection_says_so_and_fails(tmp_path):
    _rows(ENERGY).write_parquet(tmp_path / "part-00000.parquet")

    result = CliRunner().invoke(app, ["analyze", str(tmp_path), "--cmdline", "no such program"])

    assert result.exit_code == 1
    assert "Nothing matched" in result.output


def test_csv_holds_every_row_and_the_full_text(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "TOP", 1)
    _rows(ENERGY).write_parquet(tmp_path / "part-00000.parquet")

    result = CliRunner().invoke(app, ["analyze", str(tmp_path), "--csv", str(tmp_path / "out"), "--unit", "Wh"])

    assert result.exit_code == 0, result.output
    table = pl.read_csv(tmp_path / "out" / "table.csv")
    assert table["pid"].to_list() == [300, 200, 400]
    assert table["total_j"].sum() == pytest.approx(30.0), "joules whatever the terminal shows"
    assert "nginx -g daemon off;" in table["cmdline"].to_list()
    rows = pl.read_csv(tmp_path / "out" / "rows.csv")
    assert rows.height == 2 * len(ENERGY)


def test_only_one_filter_at_a_time(tmp_path):
    result = CliRunner().invoke(app, ["analyze", str(tmp_path), "--cmdline", "a", "--cgroup", "b"])

    assert result.exit_code != 0
    assert "at most one" in result.output


def test_energy_too_small_to_show_is_not_shown_as_zero():
    assert report._amount(0.036, "J") == "<0.1"
    assert report._amount(0.0, "J") == "0.0"
    assert report._amount(7200.0, "Wh", True) == "2.00000 Wh"
