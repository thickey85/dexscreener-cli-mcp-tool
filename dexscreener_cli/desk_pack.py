"""Thin trading-desk adapter over the hot scanner.

One scan writes a ranked JSON shortlist for cron or a scheduled agent.
Levels are a day-trader frame around the last price:

- mark = last USD price
- entry = mark * (1 - reclaim)
  reclaim is 1.00% when the 1h change is >= +12% (extended tape),
  0.50% when the 1h change is <= 0% (already soft),
  and 0.75% otherwise
- invalidation = entry * (1 - 0.035)  # 3.5% below entry
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

from .client import DexScreenerClient
from .config import ScanFilters
from .models import HotTokenCandidate, PairSnapshot
from .scanner import HotScanner
from .state import utc_now_iso

SOURCE = "dexscreener-cli"
DEFAULT_OUT = "desk-packs/latest.json"
SOL_PACK_NAME = "sol-latest.json"
ETH_PACK_NAME = "eth-latest.json"

SOL_CHAINS = frozenset({"solana"})
ETH_SLEEVE_CHAINS = frozenset({"ethereum", "base"})

# Day-trader book. Names below these floors are not fillable size.
DESK_MIN_LIQUIDITY_USD = 50_000.0
DESK_MIN_VOLUME_H24_USD = 80_000.0
DESK_MIN_TXNS_H1 = 20
DESK_MIN_PRICE_CHANGE_H1 = -8.0
DESK_MAX_VOL_LIQ = 40.0

RECLAIM_EXTENDED = 0.010
RECLAIM_DEFAULT = 0.0075
RECLAIM_SOFT = 0.005
EXTENDED_H1_PCT = 12.0
INVALIDATION_BELOW_ENTRY = 0.035

CLEAN_MIN_SCORE = 60.0
CLEAN_MIN_LIQUIDITY_USD = 80_000.0
CLEAN_MIN_TXNS_H1 = 40
CLEAN_MAX_H1_PCT = 35.0
CLEAN_VOL_LIQ_LOW = 0.4
CLEAN_VOL_LIQ_HIGH = 12.0

FILLABLE_MIN_LIQUIDITY_USD = 75_000.0
FILLABLE_MIN_TXNS_H1 = 30
FILLABLE_MIN_VOLUME_H1_USD = 5_000.0
FILLABLE_MAX_VOL_LIQ = 25.0

# Tape defects that block a day-trader fill. concentration-risk is a
# structure flag (liq vs FDV) and does not by itself make a name unfillable.
_HARD_FLAGS = frozenset(
    {
        "thin-exit",
        "one-way-flow",
        "low-participant-flow",
        "blowoff-risk",
        "low-liquidity",
        "high-turnover",
    }
)

# Gas and stable bases are not desk names.
_SKIP_SYMBOLS = frozenset(
    {
        "USDC",
        "USDT",
        "USD1",
        "DAI",
        "USDE",
        "FDUSD",
        "SOL",
        "WSOL",
        "ETH",
        "WETH",
        "WBTC",
        "BTC",
        "CBBTC",
        "BNB",
        "WBNB",
        "BETH",
    }
)


def reclaim_fraction(price_change_h1: float) -> float:
    """Reclaim distance below mark, as a fraction. Always in [0.005, 0.010]."""
    if price_change_h1 >= EXTENDED_H1_PCT:
        return RECLAIM_EXTENDED
    if price_change_h1 <= 0.0:
        return RECLAIM_SOFT
    return RECLAIM_DEFAULT


def _round_px(value: float) -> float:
    if value <= 0.0:
        return 0.0
    return float(f"{value:.8g}")


def day_trade_levels(mark: float, price_change_h1: float) -> tuple[float, float, float]:
    """Return rounded (mark, entry, invalidation) for the day-trader frame."""
    if mark <= 0.0:
        return 0.0, 0.0, 0.0
    entry = mark * (1.0 - reclaim_fraction(price_change_h1))
    invalidation = entry * (1.0 - INVALIDATION_BELOW_ENTRY)
    return _round_px(mark), _round_px(entry), _round_px(invalidation)


def _compact_usd(value: float) -> str:
    n = abs(value)
    if n >= 1_000_000_000:
        return f"${n / 1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"${n / 1_000_000:.1f}M"
    if n >= 10_000:
        return f"${n / 1_000:.0f}K"
    if n >= 1_000:
        return f"${n / 1_000:.1f}K"
    return f"${n:.0f}"


def format_vol_signal(pair: PairSnapshot) -> str:
    return f"vol24={_compact_usd(pair.volume_h24)} liq={_compact_usd(pair.liquidity_usd)}"


def _risk_flags(pair: PairSnapshot) -> set[str]:
    _, _, flags = HotScanner._risk_profile(pair)
    return set(flags)


def _vol_liq(pair: PairSnapshot) -> float:
    return pair.volume_h24 / max(pair.liquidity_usd, 1.0)


def classify_hint(candidate: HotTokenCandidate) -> str:
    """CLEAN, LIQUID, FILLABLE-NOW, or NOT_FILLABLE."""
    pair = candidate.pair
    flags = _risk_flags(pair)
    ratio = _vol_liq(pair)
    two_sided = pair.buys_h1 > 0 and pair.sells_h1 > 0
    if (
        pair.price_usd <= 0.0
        or pair.liquidity_usd < DESK_MIN_LIQUIDITY_USD
        or pair.volume_h24 < DESK_MIN_VOLUME_H24_USD
        or pair.txns_h1 < DESK_MIN_TXNS_H1
        or not two_sided
        or ratio > DESK_MAX_VOL_LIQ
        or flags & _HARD_FLAGS
    ):
        return "NOT_FILLABLE"

    clean = (
        not (flags & _HARD_FLAGS)
        and CLEAN_VOL_LIQ_LOW <= ratio <= CLEAN_VOL_LIQ_HIGH
        and pair.txns_h1 >= CLEAN_MIN_TXNS_H1
        and pair.liquidity_usd >= CLEAN_MIN_LIQUIDITY_USD
        and pair.price_change_h1 < CLEAN_MAX_H1_PCT
        and candidate.score >= CLEAN_MIN_SCORE
    )
    if clean:
        return "CLEAN"

    if (
        pair.txns_h1 >= FILLABLE_MIN_TXNS_H1
        and pair.liquidity_usd >= FILLABLE_MIN_LIQUIDITY_USD
        and pair.volume_h1 >= FILLABLE_MIN_VOLUME_H1_USD
        and ratio <= FILLABLE_MAX_VOL_LIQ
    ):
        return "FILLABLE-NOW"
    return "LIQUID"


def _skip_candidate(candidate: HotTokenCandidate) -> bool:
    pair = candidate.pair
    symbol = pair.base_symbol.strip().upper()
    if not symbol or symbol in _SKIP_SYMBOLS:
        return True
    if not pair.base_address.strip():
        return True
    chain = pair.chain_id.strip().lower()
    if not chain:
        return True
    return False


def select_desk_candidates(candidates: list[HotTokenCandidate], limit: int) -> list[HotTokenCandidate]:
    """Keep scanner rank. Drop stables and names that are not fillable."""
    shortlist = max(2, min(10, int(limit)))
    picked: list[HotTokenCandidate] = []
    seen: set[tuple[str, str]] = set()
    for candidate in candidates:
        if len(picked) >= shortlist:
            break
        if _skip_candidate(candidate):
            continue
        if classify_hint(candidate) == "NOT_FILLABLE":
            continue
        key = (candidate.pair.chain_id.strip().lower(), candidate.pair.base_address.strip().lower())
        if key in seen:
            continue
        seen.add(key)
        picked.append(candidate)
    return picked


def candidate_record(candidate: HotTokenCandidate) -> dict[str, Any]:
    pair = candidate.pair
    mark, entry, invalidation = day_trade_levels(pair.price_usd, pair.price_change_h1)
    record: dict[str, Any] = {
        "chain": pair.chain_id.strip().lower(),
        "ticker": pair.base_symbol.strip().upper(),
        "token_address": pair.base_address,
    }
    if pair.pair_address:
        record["pair_address"] = pair.pair_address
    record["vol_signal"] = format_vol_signal(pair)
    record["mark"] = mark
    record["entry"] = entry
    record["invalidation"] = invalidation
    record["hint"] = classify_hint(candidate)
    record["score"] = int(round(max(0.0, min(100.0, candidate.score))))
    return record


def sleeve_hint_for(chains_present: set[str], requested: tuple[str, ...]) -> str:
    """SOL, ETH, or BOTH. Empty packs follow the chains that were requested."""
    if chains_present:
        scope = chains_present
    else:
        scope = {chain.strip().lower() for chain in requested}
    has_sol = bool(scope & SOL_CHAINS)
    has_eth = bool(scope & ETH_SLEEVE_CHAINS)
    if has_sol and has_eth:
        return "BOTH"
    if has_eth:
        return "ETH"
    if has_sol:
        return "SOL"
    return "BOTH"


def _pack(records: list[dict[str, Any]], *, sleeve: str, generated_at: str) -> dict[str, Any]:
    return {
        "generated_at": generated_at,
        "source": SOURCE,
        "sleeve_hint": sleeve,
        "candidates": records,
    }


def _partition(records: list[dict[str, Any]], chains: frozenset[str]) -> list[dict[str, Any]]:
    return [row for row in records if row.get("chain") in chains]


@dataclass(slots=True)
class DeskPackRun:
    combined: dict[str, Any]
    sol: dict[str, Any] | None
    eth: dict[str, Any] | None


def build_desk_run(
    candidates: list[HotTokenCandidate],
    *,
    chains: tuple[str, ...],
    limit: int,
    generated_at: str | None = None,
) -> DeskPackRun:
    stamp = generated_at or utc_now_iso()
    requested = tuple(chain.strip().lower() for chain in chains if chain.strip())
    picked = select_desk_candidates(candidates, limit)
    records = [candidate_record(row) for row in picked]
    present = {str(row["chain"]) for row in records}
    combined = _pack(records, sleeve=sleeve_hint_for(present, requested), generated_at=stamp)
    requested_set = set(requested)
    sol = None
    eth = None
    if requested_set & SOL_CHAINS:
        sol_rows = _partition(records, SOL_CHAINS)
        sol = _pack(sol_rows, sleeve="SOL", generated_at=stamp)
    if requested_set & ETH_SLEEVE_CHAINS:
        eth_rows = _partition(records, ETH_SLEEVE_CHAINS)
        eth = _pack(eth_rows, sleeve="ETH", generated_at=stamp)
    return DeskPackRun(combined=combined, sol=sol, eth=eth)


def desk_scan_filters(chains: tuple[str, ...], limit: int) -> ScanFilters:
    """Hot-scan filters wide enough to refill a 2-10 name shortlist."""
    shortlist = max(2, min(10, int(limit)))
    return ScanFilters(
        chains=chains,
        limit=min(max(shortlist * 5, 24), 48),
        min_liquidity_usd=DESK_MIN_LIQUIDITY_USD,
        min_volume_h24_usd=DESK_MIN_VOLUME_H24_USD,
        min_txns_h1=DESK_MIN_TXNS_H1,
        min_price_change_h1=DESK_MIN_PRICE_CHANGE_H1,
    )


async def scan_desk_candidates(chains: tuple[str, ...], limit: int) -> list[HotTokenCandidate]:
    """One hot scan. Holder lookups are skipped; the pack does not use them."""
    filters = desk_scan_filters(chains, limit)
    async with DexScreenerClient() as client:
        scanner = HotScanner(client)
        return await scanner.scan(filters, include_holders=False)


@dataclass(slots=True)
class DeskPackPaths:
    combined: Path
    sol: Path | None
    eth: Path | None


# Only JSON number tokens. Token addresses can contain fragments like "4E862".
_SCI = re.compile(r"(?<= )-?\d+(?:\.\d+)?[eE][+-]?\d+(?=\s*[,}\]])")


def _plain_number(text: str) -> str:
    """Expand a .8g / JSON scientific token into a plain decimal."""
    raw = text.lower()
    negative = raw.startswith("-")
    if negative:
        raw = raw[1:]
    if "e" not in raw:
        if "." not in raw:
            raw = raw + ".0"
        return ("-" if negative else "") + raw
    mant, exp_s = raw.split("e", 1)
    exp = int(exp_s)
    if "." in mant:
        whole, frac = mant.split(".", 1)
        digits = whole + frac
        point = len(whole) + exp
    else:
        digits = mant
        point = len(mant) + exp
    if point <= 0:
        body = "0." + ("0" * (-point)) + digits
    elif point >= len(digits):
        body = digits + ("0" * (point - len(digits))) + ".0"
    else:
        body = digits[:point] + "." + digits[point:]
    return ("-" if negative else "") + body


def dumps_pack(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, indent=2, ensure_ascii=True)
    text = _SCI.sub(lambda match: _plain_number(match.group(0)), text)
    return text + "\n"


def _dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps_pack(payload), encoding="utf-8")


def write_desk_run(run: DeskPackRun, out: str | Path) -> DeskPackPaths:
    """Write the combined pack and, when that sleeve was scanned, the split files."""
    combined_path = Path(out).expanduser()
    _dump(combined_path, run.combined)
    sol_path = None
    eth_path = None
    if run.sol is not None:
        sol_path = combined_path.parent / SOL_PACK_NAME
        _dump(sol_path, run.sol)
    if run.eth is not None:
        eth_path = combined_path.parent / ETH_PACK_NAME
        _dump(eth_path, run.eth)
    return DeskPackPaths(combined=combined_path, sol=sol_path, eth=eth_path)
