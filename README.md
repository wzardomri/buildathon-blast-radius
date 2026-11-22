# Blast Radius selector

For every patch to sqlglot (pinned at `eb11aec4`), pick the tests that can fail and
split them across CI workers. A missed failure costs 300 s, so the selector is built
to never guess. It either proves a test unaffected, runs the test, or hands it to CI.

## How it decides

1. **Bytecode diff.** Each changed `.py` file is compiled before and after the patch,
   and the code-object trees are compared with line numbers ignored. Comment-only and
   whitespace-only edits produce no difference. Every function, lambda or generator
   whose own bytecode changed is mapped to the tests that ran it.
2. **Fingerprint diff.** When a module or class body changes, or a function that runs
   at import time changes, the patched package is imported. Every module global and
   class attribute is then hashed and compared with hashes of the original package,
   which are computed during the image build. Changed attributes map to the tests that
   read them. Changed globals map to the tests that ran code naming them. Added or
   removed attributes map to every test that touched the class or one of its
   subclasses.
3. **Data files.** Changed fixtures and other files map to the tests that opened them.
   Added or deleted files map to the tests that listed their directory. Real pytest
   configuration changes (`conftest.py`, `pytest.ini`, or a pytest section in
   `pyproject.toml`, `setup.cfg` or `tox.ini`) select everything.
4. **Run or include.** The candidates are split by expected cost. Cheap candidates run
   here, in-process, on a symlink overlay of the patched tree; only the ones that fail
   are kept. Expensive ones (the TPC-DS executor test, for example) go to CI
   unverified, packed onto 1 to 8 workers by longest-first. After a local failure, the
   remaining candidates move to CI if that is cheaper. A test that errors, times out,
   cannot be loaded, or does not finish counts as failing.

## Replaying the TPC-H / TPC-DS executor tests

Most of an executor test's time goes to sqlglot's Python executor and to DuckDB
computing the expected results. Neither depends on the patch when only the front
end (parse, optimize, transpile) can change. The tracer records that front end
separately for these tests. When a patch reaches only the front end, the test runs
here in replay mode (`brreplay.py`). The front end runs for real; executor and DuckDB
results come from recordings made at image build time, keyed by a structural digest of
the optimized query and by the exact DuckDB SQL. Any input the pinned commit never
produced counts as a failure, so the test goes to CI. The TPC-DS test drops from about
28 s of CPU to about 3 s.

Changes inside the executor itself are handled per query. The recordings also hold
the lines each query ran in the planner and executor. Only the queries reaching
changed lines are executed for real, and each must reproduce its recorded result
exactly; the first one that doesn't stops the test, which then goes to CI.

The selector also times a short benchmark at startup and scales its local cost
estimates when the machine is clearly slower or faster than the one they were
measured on.

## The maps (`maps/maps.pkl`)

These are recorded once at the pinned commit by `build/trace_plugin.py`, a pytest
plugin that runs the full suite:

- the code objects each test runs (`sys.monitoring` PY_START, re-armed at every test
  boundary, with the generator's dispatch cache cleared per test);
- the class attributes of sqlglot classes each test reads, using recording
  descriptors that also note the subclass each attribute was read through;
- the files each test opens and the directories it lists, via an audit hook.

The suite is traced in both forward and reverse order. Code whose coverage depends on
the order (lazy dialect loading) is widened to its whole enclosing scope. Process pools
in the optimizer and executor tests run in-process (`brpool.py`), with the same pickle
round-trip of arguments and results that a process boundary would apply.

To rebuild the maps, see `build/make_maps.sh`.

The bundled `maps/values.tar.xz` holds comparison traces from running the
unmodified pinned SQLGlot suite with `build/value_record.py`. No patches
or failure labels are used to build it. Tests without a completed trace
continue through ordinary selection. The bundled `maps/pinned-runtime-maps.tar.xz`
holds pristine fingerprints and executor replay data produced by `brfp.py` and
`build/record_replay.py` with the pinned interpreter and dependencies.

## Validation

- Sample set, 30 commits: about 132 s simulated plus about 9 s CPU, with 0 misses. The
  starter scores 1,614 s; an oracle that selects exactly the failing tests scores 94 s.
- Mutation harness (`tools/gen.py` and `tools/label.py`), labeled by running the full
  suite under pytest with the same failure rules as the event: operator swaps, constant
  and string edits, comments, deleted statements, changed defaults, dict entries
  removed, renamed module-level functions (import-time `NameError`), syntax errors,
  edits to test files and `tests/helpers.py`, new method overrides, fixture edits,
  reordered statements, decorator removal, and package `__init__` edits.
  On 70 such commits: 0 misses, 470 s simulated against an oracle of about 300 s.

Any change the maps cannot reason about selects everything. That costs about 37 s,
roughly a ninth of a single miss.

## License and dependencies

The selector is MIT-licensed. It vendors the pinned [SQLGlot source](https://github.com/tobymao/sqlglot/tree/eb11aec4be8cb2122afe713a4816662c3f0fb2fa), which retains its own MIT license. The build uses the organizer's pinned test environment from [requirements.lock](requirements.lock).

## Layout

| path | what |
| --- | --- |
| `selector.py` | entry point |
| `brlib.py` | patch parsing and applying, code-object diff |
| `brfp.py` | fingerprints of module globals and class attributes |
| `brpool.py` | in-process `ProcessPoolExecutor` stand-in |
| `brreplay.py` | replay of the executor tests' expensive parts |
| `maps/maps.pkl` | per-test maps recorded at the pinned commit |
| `vendor/sqlglot-eb11aec.tar.xz` | `git archive` of sqlglot at the pinned commit |
| `build/` | scripts that produce the maps |
| `tools/` | mutation generator and labeller used for validation |
