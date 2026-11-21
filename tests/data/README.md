# Test fixtures

Committed fixtures used by the test suite. They are test inputs, not package
data: the wheel ships only `src/gtfs2emis_py`, so nothing here reaches a user
install.

## Real sample feeds, fleets and DEM

Copied from the R reference package
[`ipeaGIT/gtfs2emis`](https://github.com/ipeaGIT/gtfs2emis) `inst/extdata` at
commit `0dbca7d779857145888ab695f36dc9e71ff57d98` (v0.1.2), retrieved
**2026-11-02**.

| File | Bytes | SHA-256 |
|---|---:|---|
| `bra_cur_gtfs.zip` | 314 526 | `0f89716a8d9c3cb73109f058ff67b66e9331058b0a0b4b284a33f98303eb61ae` |
| `irl_dub_gtfs.zip` | 282 283 | `80f74a141b6b0bbf1e745e38f85cc1f290e2bf344daab321598d50095087ccc7` |
| `usa_det_gtfs.zip` | 61 117 | `498e218454b304d87054768ea2e2ca9687c4a0a8dd5da9ba099f82dfedbb136c` |
| `bra_cur_fleet.txt` | 5 272 | `13152e6e7f905b01d722be0386fa2a07efcfaae9e395e2bae7df02b581831945` |
| `irl_dub_fleet.txt` | 247 | `3d5591e2d5bfb0021a46acd7faa40e0a9af7c3d94470d64a653645e56400c6ff` |
| `usa_det_fleet.txt` | 287 | `fc9365f2be2cb3a60df7fca916a6d8dae118a8bbc623726d8088a0637d2fdf5f` |
| `bra_cur-srtm.tif` | 13 989 | `7ff3dcb99408aabd857e753b9b01cc3b588ad1c03ac5b0dd45168356a45fdc1a` |

**Licence.** The R package is `MIT + file LICENSE` (`DESCRIPTION:20`), and these
files are redistributed under that licence. The package does not state the
original providers of the sample GTFS feeds or the SRTM tile; the file names
identify the cities (Curitiba, Dublin, Detroit). If the port publishes its own
attribution file, that is where a per-provider credit belongs
(publication-time).

`usa_det_gtfs.zip` was already in `data/raw/`; it moved here so every committed
fixture sits in one place. `data/raw/` remains the landing directory for
upstream inputs that are downloaded rather than committed.

## Synthetic edge-case feed

`synthetic_edge_cases.zip` is generated, never hand-edited:

```bash
cd gtfs2emis-py
uv run python scripts/make_synthetic_feed.py
```

It covers the contract edge cases the real feeds do not exercise: a
one-point shape, duplicate consecutive points, a loop shape, times past
24:00:00, no `shape_dist_traveled` column, and a `frequencies.txt` trip. The
generator writes a fixed zip timestamp and fixed row order, so regeneration
reproduces the file byte for byte (`tests/test_fixtures.py` checks the hash).

| File | Bytes | SHA-256 |
|---|---:|---|
| `synthetic_edge_cases.zip` | 1 559 | `2e9af3123e8c95b9e5f90ff7dfda8ba7c53afd2349bd67661cd0c5750eb93203` |

## Size budget

Total committed fixtures must stay **under 1 MB** (currently ~677 KB, dominated
by the Curitiba and Dublin feeds). `tests/test_fixtures.py::test_fixture_size_budget`
fails the build when the budget is exceeded, so adding a large feed is a
reviewed change rather than a slow creep.
