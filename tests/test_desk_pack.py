"""Unit tests for the trading-desk pack adapter. No network."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from dexscreener_cli.desk_pack import (
    INVALIDATION_BELOW_ENTRY,
    build_desk_run,
    classify_hint,
    day_trade_levels,
    dumps_pack,
    reclaim_fraction,
    write_desk_run,
)
from dexscreener_cli.models import HotTokenCandidate, PairSnapshot


def _pair(**overrides: object) -> PairSnapshot:
    base: dict[str, object] = {
        "chain_id": "solana",
        "dex_id": "raydium",
        "pair_address": "PAIR111",
        "pair_url": "https://dexscreener.com/solana/PAIR111",
        "base_address": "TOKEN111",
        "base_symbol": "WIF",
        "base_name": "dogwifhat",
        "quote_symbol": "SOL",
        "price_usd": 1.0,
        "volume_h24": 1_200_000.0,
        "volume_h6": 300_000.0,
        "volume_h1": 40_000.0,
        "volume_m5": 4_000.0,
        "buys_h1": 80,
        "sells_h1": 60,
        "buys_h24": 800,
        "sells_h24": 700,
        "price_change_h1": 4.0,
        "price_change_h24": 12.0,
        "liquidity_usd": 340_000.0,
        "market_cap": 50_000_000.0,
        "fdv": 50_000_000.0,
        "holders_count": None,
        "holders_source": None,
        "pair_created_at_ms": None,
    }
    base.update(overrides)
    return PairSnapshot(**base)  # type: ignore[arg-type]


def _candidate(score: float = 72.0, **pair_kw: object) -> HotTokenCandidate:
    return HotTokenCandidate(
        pair=_pair(**pair_kw),
        score=score,
        boost_total=0.0,
        boost_count=0,
        has_profile=False,
        discovery="seed",
    )


class DeskLevelTests(unittest.TestCase):
    def test_reclaim_stays_inside_half_to_one_percent(self) -> None:
        for h1 in (-8.0, 0.0, 0.1, 4.0, 11.9, 12.0, 40.0):
            frac = reclaim_fraction(h1)
            self.assertGreaterEqual(frac, 0.005)
            self.assertLessEqual(frac, 0.010)

    def test_reclaim_bands(self) -> None:
        self.assertEqual(reclaim_fraction(12.0), 0.010)
        self.assertEqual(reclaim_fraction(4.0), 0.0075)
        self.assertEqual(reclaim_fraction(0.0), 0.005)
        self.assertEqual(reclaim_fraction(-3.0), 0.005)

    def test_invalidation_is_three_and_a_half_percent_below_entry(self) -> None:
        mark = 2.5
        for h1 in (0.0, 4.0, 20.0):
            frac = reclaim_fraction(h1)
            raw_entry = mark * (1.0 - frac)
            raw_inv = raw_entry * (1.0 - INVALIDATION_BELOW_ENTRY)
            got_mark, got_entry, got_inv = day_trade_levels(mark, h1)
            self.assertEqual(got_mark, float(f"{mark:.8g}"))
            self.assertEqual(got_entry, float(f"{raw_entry:.8g}"))
            self.assertEqual(got_inv, float(f"{raw_inv:.8g}"))
            gap = (raw_entry - raw_inv) / raw_entry
            self.assertAlmostEqual(gap, 0.035)


class DeskHintTests(unittest.TestCase):
    def test_clean_liquid_and_fillable(self) -> None:
        self.assertEqual(classify_hint(_candidate()), "CLEAN")
        quiet = _candidate(
            score=40,
            liquidity_usd=60_000.0,
            volume_h24=90_000.0,
            volume_h1=1_000.0,
            buys_h1=10,
            sells_h1=10,
            market_cap=2_000_000.0,
            fdv=2_000_000.0,
        )
        self.assertEqual(classify_hint(quiet), "LIQUID")
        active = _candidate(
            score=50,
            liquidity_usd=100_000.0,
            volume_h24=200_000.0,
            volume_h1=8_000.0,
            buys_h1=20,
            sells_h1=15,
            market_cap=4_000_000.0,
            fdv=4_000_000.0,
        )
        self.assertEqual(classify_hint(active), "FILLABLE-NOW")

    def test_not_fillable_cases_are_dropped(self) -> None:
        one_way = _candidate(buys_h1=40, sells_h1=0)
        self.assertEqual(classify_hint(one_way), "NOT_FILLABLE")
        thin = _candidate(volume_h24=20_000_000.0, liquidity_usd=50_000.0)
        self.assertEqual(classify_hint(thin), "NOT_FILLABLE")
        dust = _candidate(price_usd=0.0)
        self.assertEqual(classify_hint(dust), "NOT_FILLABLE")


class DeskPackTests(unittest.TestCase):
    def test_pack_shape_sleeve_and_limit(self) -> None:
        rows = [
            _candidate(base_symbol="WIF", base_address="SOL1", chain_id="solana"),
            _candidate(
                base_symbol="PEPE",
                base_address="ETH1",
                chain_id="ethereum",
                quote_symbol="WETH",
                pair_address="PAIR-ETH",
                score=50,
                buys_h1=20,
                sells_h1=15,
                liquidity_usd=100_000.0,
                volume_h24=200_000.0,
                volume_h1=8_000.0,
            ),
            _candidate(base_symbol="USDC", base_address="STABLE", chain_id="ethereum"),
            _candidate(base_symbol="DEGEN", base_address="BASE1", chain_id="base", quote_symbol="WETH"),
            _candidate(buys_h1=30, sells_h1=0, base_symbol="BAD", base_address="BAD1"),
        ]
        extra = [
            _candidate(base_symbol=f"N{i}", base_address=f"SOLX{i}", chain_id="solana")
            for i in range(8)
        ]
        run = build_desk_run(rows + extra, chains=("solana", "ethereum", "base"), limit=8, generated_at="2026-09-24T18:00:00+00:00")
        pack = run.combined
        self.assertEqual(pack["source"], "dexscreener-cli")
        self.assertEqual(pack["generated_at"], "2026-09-24T18:00:00+00:00")
        self.assertEqual(pack["sleeve_hint"], "BOTH")
        self.assertGreaterEqual(len(pack["candidates"]), 2)
        self.assertLessEqual(len(pack["candidates"]), 8)
        tickers = [row["ticker"] for row in pack["candidates"]]
        self.assertNotIn("USDC", tickers)
        self.assertNotIn("BAD", tickers)
        self.assertEqual(tickers[0], "WIF")
        first = pack["candidates"][0]
        self.assertEqual(
            list(first.keys()),
            [
                "chain",
                "ticker",
                "token_address",
                "pair_address",
                "vol_signal",
                "mark",
                "entry",
                "invalidation",
                "hint",
                "score",
            ],
        )
        self.assertEqual(first["vol_signal"], "vol24=$1.2M liq=$340K")
        self.assertEqual(first["hint"], "CLEAN")
        self.assertIsInstance(first["score"], int)
        self.assertEqual(first["entry"], day_trade_levels(1.0, 4.0)[1])
        assert run.sol is not None and run.eth is not None
        self.assertEqual(run.sol["sleeve_hint"], "SOL")
        self.assertEqual(run.eth["sleeve_hint"], "ETH")
        eth_chains = {row["chain"] for row in run.eth["candidates"]}
        self.assertTrue(eth_chains <= {"ethereum", "base"})
        self.assertIn("PEPE", [row["ticker"] for row in run.eth["candidates"]])
        self.assertIn("DEGEN", [row["ticker"] for row in run.eth["candidates"]])

    def test_sol_only_skips_eth_file(self) -> None:
        run = build_desk_run(
            [_candidate(), _candidate(base_symbol="BONK", base_address="SOL2")],
            chains=("solana",),
            limit=8,
            generated_at="2026-09-24T18:00:00+00:00",
        )
        self.assertEqual(run.combined["sleeve_hint"], "SOL")
        self.assertIsNotNone(run.sol)
        self.assertIsNone(run.eth)

    def test_write_split_files(self) -> None:
        run = build_desk_run(
            [
                _candidate(),
                _candidate(base_symbol="PEPE", base_address="ETH1", chain_id="ethereum"),
            ],
            chains=("solana", "ethereum", "base"),
            limit=8,
            generated_at="2026-09-24T18:00:00+00:00",
        )
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_desk_run(run, Path(tmp) / "latest.json")
            combined = json.loads(paths.combined.read_text(encoding="utf-8"))
            sol = json.loads(Path(tmp, "sol-latest.json").read_text(encoding="utf-8"))
            eth = json.loads(Path(tmp, "eth-latest.json").read_text(encoding="utf-8"))
            self.assertEqual(combined["sleeve_hint"], "BOTH")
            self.assertEqual(sol["candidates"][0]["ticker"], "WIF")
            self.assertEqual(eth["candidates"][0]["chain"], "ethereum")
            self.assertEqual(combined["generated_at"], sol["generated_at"])
            self.assertEqual(eth["sleeve_hint"], "ETH")


class DeskJsonTests(unittest.TestCase):
    def test_small_prices_stay_decimal(self) -> None:
        addr = "0x4ed4E862860beD51a9570b96d89aF5E1B0Efefed"
        text = dumps_pack({"token_address": addr, "mark": 0.00002145, "score": 81})
        self.assertIn(addr, text)
        self.assertIn("0.00002145", text)
        self.assertNotIn("e-", text.lower())
        loaded = json.loads(text)
        self.assertEqual(loaded["score"], 81)
        self.assertEqual(loaded["token_address"], addr)


class DeskExampleTests(unittest.TestCase):
    def test_checked_in_example_matches_schema(self) -> None:
        path = Path(__file__).resolve().parents[1] / "desk-packs" / "example.json"
        pack = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(set(pack.keys()), {"generated_at", "source", "sleeve_hint", "candidates"})
        self.assertEqual(pack["source"], "dexscreener-cli")
        self.assertIn(pack["sleeve_hint"], {"SOL", "ETH", "BOTH"})
        self.assertGreaterEqual(len(pack["candidates"]), 2)
        self.assertLessEqual(len(pack["candidates"]), 10)
        for row in pack["candidates"]:
            self.assertIn(row["chain"], {"solana", "ethereum", "base"})
            self.assertIn(row["hint"], {"CLEAN", "LIQUID", "FILLABLE-NOW", "NOT_FILLABLE"})
            self.assertLess(row["entry"], row["mark"])
            self.assertLess(row["invalidation"], row["entry"])
            self.assertIsInstance(row["score"], int)


if __name__ == "__main__":
    unittest.main()
