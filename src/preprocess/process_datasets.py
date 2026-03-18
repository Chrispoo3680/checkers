import tempfile

import polars as pl

from ..common import tools


def process_lichess_db_evals(path):

    lazy_df = pl.scan_ndjson(path)

    filtered_lazy_df = lazy_df.select(
        pl.col("fen"),
        pl.col("evals")
        .list.get(0)
        .struct.field("pvs")
        .list.agg(pl.element().struct.field("line").str.split(" ").list.get(0))
        .list.join(",")
        .alias("moves"),
        # cp based move weights (mate-aware)
        pl.when(pl.col("fen").str.split(" ").list.get(1) == "b")
        .then(
            pl.col("evals")
            .list.get(0)
            .struct.field("pvs")
            .list.agg(tools.effective_cp_element() * -1)
        )
        .otherwise(
            pl.col("evals")
            .list.get(0)
            .struct.field("pvs")
            .list.agg(tools.effective_cp_element())
        )
        .map_elements(tools.cp_weighted_moves, return_dtype=pl.List(pl.Float64))
        .alias("move_weights"),
        # Evaluation (mate-aware)
        (
            tools.effective_cp_single(
                pl.col("evals").list.get(0).struct.field("pvs").list.get(0)
            )
            * pl.when(pl.col("fen").str.split(" ").list.get(1) == "b")
            .then(-1)
            .otherwise(1)
        )
        .clip(-1000, 1000)
        .alias("evaluation"),
    ).slice(0, 15_000_000)

    return filtered_lazy_df


def process_lichess_puzzles(path, batch_size=10_000):

    lazy_df = pl.scan_parquet(path).select(["FEN", "Moves"])
    total = lazy_df.select(pl.len()).collect().item()

    tmp_dir = tempfile.mkdtemp(
        prefix="lichess_puzzles_", dir=str(tools.configure_temp_storage())
    )

    for i, offset in enumerate(range(0, total, batch_size)):
        batch_pl = lazy_df.slice(offset, batch_size).collect()
        expanded = tools.expand_game_positions(batch_pl)

        if not expanded:
            continue

        pl.DataFrame(
            {
                "fen": [d["fen"] for d in expanded],
                "moves": [d["moves"] for d in expanded],
                "move_weights": pl.Series([None] * len(expanded), dtype=pl.Null),
                "evaluation": pl.Series([None] * len(expanded), dtype=pl.Null),
            }
        ).write_parquet(f"{tmp_dir}/batch_{i:06d}.parquet")

    return pl.scan_parquet(f"{tmp_dir}/*.parquet")


def process_stockfish_evaluations_csv(path):

    lazy_df = pl.scan_csv(path)

    filtered_lazy_df = lazy_df.select(
        pl.col("fen"),
        (
            pl.col("move_1").fill_null("").cast(pl.Utf8)
            + ","
            + pl.col("move_2").fill_null("").cast(pl.Utf8)
            + ","
            + pl.col("move_3").fill_null("").cast(pl.Utf8)
            + ","
            + pl.col("move_4").fill_null("").cast(pl.Utf8)
            + ","
            + pl.col("move_5").fill_null("").cast(pl.Utf8)
        )
        .str.replace(r",+", ",")
        .str.strip_chars(",")
        .alias("moves"),
        pl.lit(None).alias("move_weights"),
        pl.col("evaluation"),
    )

    return filtered_lazy_df


def process_magnus_carlsen_games_csv(path):
    # Keep result_raw as white-absolute perspective (+1 = white wins,
    # -1 = black wins). expand_game_positions_san alternates the sign
    # at each half-move using (-1)^i where i=0 is always white's first
    # move, so this correctly produces current-player-perspective labels
    # for every position. Flipping the sign based on which side Magnus
    # played inverts ALL labels in black-sided games (bug).
    games_df = (
        pl.scan_csv(path)
        .drop_nulls()
        .with_columns(
            pl.when(pl.col("result_raw") == "1-0")
            .then(1000)
            .when(pl.col("result_raw") == "0-1")
            .then(-1000)
            .when(pl.col("result_raw") == "0.5-0.5")
            .then(0)
            .otherwise(None)
            .cast(pl.Int16)
            .alias("result_raw")
        )
        .collect()
    )
    expanded = tools.expand_game_positions_san(
        games_df, moves_col="moves", eval_col="result_raw"
    )

    return pl.LazyFrame(
        {
            "fen": expanded["fen"],
            "moves": expanded["moves"],
            "move_weights": pl.Series([None] * len(expanded["fen"]), dtype=pl.Null),
            "evaluation": expanded["evaluation"],
        }
    )
