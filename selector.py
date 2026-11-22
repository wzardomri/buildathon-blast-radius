"""Blast Radius selector.

For each patch:
  1. Static analysis against the pinned sqlglot checkout: which code objects
     changed (bytecode diff), which data files changed.
  2. If module/class bodies (or functions run at import time) changed, import
     the patched package and diff fingerprints of every module global and
     class attribute against the pristine ones.
  3. Map changes to candidate tests with per-test maps recorded at the pinned
     commit (code objects run, class attributes read, files opened).
  4. Run the cheap candidates here on the patched tree; include the ones that
     fail, plus the expensive candidates that are cheaper to just hand to CI.
"""

import json
import os
import pickle
import select
import signal
import sys
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import brlib  # noqa: E402

DATA = os.environ.get("BR_DATA", "/data")
SRC = os.environ.get("BR_SRC", os.path.join(HERE, "src"))
MAPS = os.environ.get("BR_MAPS", os.path.join(HERE, "maps"))
WORK = os.environ.get("BR_WORK", "/tmp/w")
PARALLEL = int(os.environ.get("BR_PARALLEL", "2"))
PER_COMMIT_WALL = float(os.environ.get("BR_PER_COMMIT", "5.0"))  # seconds of wall budget per commit
REPLAY_BUDGET = float(os.environ.get("BR_REPLAY_BUDGET", "0"))  # share of a test's CI seconds worth spending to verify it
STATES = {}
REPLAY_SKIP = {}
LINKED_REPLAY = os.environ.get("BR_LINKED_REPLAY", "0") == "1"  # replay when tpc-ds.sql/tpc-h.sql changed
RESERVE_PER_COMMIT = float(os.environ.get("BR_RESERVE", "2.0"))  # wall seconds kept for each waiting commit
LOCAL_RATIO = float(os.environ.get("BR_LOCAL_RATIO", "1.15"))  # local cpu / recorded seconds
HEAVY_PLANS = os.environ.get("BR_HEAVY_PLANS", "0") == "1"  # also try verifying only heavy cheap tests
SUFFIX_PLANS = os.environ.get("BR_SUFFIX_PLANS", "0") == "1"  # also try verifying only the costly tests
ALL_LOCAL = os.environ.get("BR_ALL_LOCAL", "0") == "1"  # verify every candidate here (measurement)
LOCAL_FIXED = float(os.environ.get("BR_LOCAL_FIXED", "0.25"))  # cpu to start running tests at all
P_FAIL = float(os.environ.get("BR_P_FAIL", "0.4"))
WORKER_COST = 2000  # ms
EVAL = os.environ.get("BR_EVAL")  # path to answers.json: print diagnostics to stderr

PYTEST_CONFIG = {"conftest.py", "pytest.ini"}
MAYBE_PYTEST_CONFIG = {"tox.ini", "setup.cfg", "pyproject.toml"}


def log(*a):
    print(*a, file=sys.stderr, flush=True)


# ------------------------------------------------------------------- packing


def pack(nums, ms):
    """Best split over 1..8 workers (LPT). Returns (cost_ms, workers)."""
    if not nums:
        return 0, []
    order = sorted(nums, key=lambda n: -ms[n])
    best = None
    total = sum(ms[n] for n in nums)
    for k in range(1, min(8, len(order)) + 1):
        if best is not None and max(ms[order[0]], total // k) + WORKER_COST * k >= best[0]:
            if total // k <= ms[order[0]]:
                break
            continue
        loads = [0] * k
        ws = [[] for _ in range(k)]
        for n in order:
            i = loads.index(min(loads))
            loads[i] += ms[n]
            ws[i].append(n)
        cost = max(loads) + WORKER_COST * k
        if best is None or cost < best[0]:
            best = (cost, ws)
    return best


# ------------------------------------------------------------------ overlay


# files that locate siblings through Path.resolve(): give them a real path in
# the overlay so they find patched neighbours
COPY_REAL = ("tests/test_sqlglotc.py",)


def build_overlay(root, files):
    """Make root a view of SRC where `files` (rel path -> text or None for
    deleted) replace the originals. Directories on the way are real; the rest
    are symlinks."""
    import shutil

    files = dict(files)
    real_copies = [c for c in COPY_REAL if c not in files]
    dirs = set()
    for rel in list(files) + real_copies:
        parts = rel.split("/")
        for i in range(1, len(parts)):
            dirs.add("/".join(parts[:i]))
    os.makedirs(root, exist_ok=True)

    def fill(reldir):
        src_dir = os.path.join(SRC, reldir) if reldir else SRC
        dst_dir = os.path.join(root, reldir) if reldir else root
        os.makedirs(dst_dir, exist_ok=True)
        entries = os.listdir(src_dir) if os.path.isdir(src_dir) else []
        for e in entries:
            rel = f"{reldir}/{e}" if reldir else e
            if rel in dirs or rel in files:
                continue
            if rel in real_copies:
                shutil.copy2(os.path.join(src_dir, e), os.path.join(dst_dir, e))
                continue
            if e == "__pycache__" and reldir and any(f.startswith(reldir + "/") and f.count("/") == reldir.count("/") + 1 for f in files):
                # keep stale pycs out of directories with replaced modules
                continue
            os.symlink(os.path.join(src_dir, e), os.path.join(dst_dir, e))
        for d in sorted(dirs):
            if (d.rsplit("/", 1)[0] if "/" in d else "") == reldir:
                fill(d)

    fill("")
    for rel, text in files.items():
        dst = os.path.join(root, rel)
        if text is None:
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "w", encoding="utf-8", newline="") as f:
            f.write(text)


# ----------------------------------------------------------------- analysis


REF_BENCH = float(os.environ.get("BR_REF_BENCH", "0.0112"))  # bench() cpu on the machine the estimates come from


def bench():
    t0 = time.process_time()
    d = {}
    acc = 0
    for i in range(60000):
        k = "k" + str(i % 997)
        d[k] = d.get(k, 0) + i
        acc += len(k) * (i & 7)
    sorted(d.items())
    return time.process_time() - t0


def calibrate():
    """How much slower this machine runs Python than the one the local cost
    estimates were measured on."""
    if os.environ.get("BR_SPEED"):
        return float(os.environ["BR_SPEED"])
    # many short runs, fastest kept: interference only ever slows a run down.
    # The collector stays off: with the maps loaded, a collection walks millions
    # of objects and would be timed as slowness of the machine.
    import gc

    was = gc.isenabled()
    gc.disable()
    try:
        runs = sorted(bench() for _ in range(6))
    finally:
        if was:
            gc.enable()
    t = runs[0]
    f = t / REF_BENCH
    if 0.85 <= f <= 1.2:
        f = 1.0  # same class of machine as the estimates: do not chase noise
    # a slower reading than this has been noise (a busy neighbour) more often than a slow machine
    return max(0.5, min(1.3, f))


class Ctx:
    def __init__(self, maps, tests, speed=1.0):
        self.speed = speed
        self.replay_sqls = {}
        for tid, path in (("tests/test_executor.py::TestExecutor::test_execute_tpcds",
                           "tests/fixtures/optimizer/tpc-ds/tpc-ds.sql"),
                          ("tests/test_executor.py::TestExecutor::test_execute_tpch",
                           "tests/fixtures/optimizer/tpc-h/tpc-h.sql")):
            try:
                with open(os.path.join(SRC, path), encoding="utf-8") as f:
                    pairs = _fixture_pairs(f.read())
                if pairs:
                    self.replay_sqls[tid] = [q for _m, q, _e in pairs]
            except OSError:
                pass
        self.maps = maps
        self.tests = tests
        self.n = len(tests)
        self.ms = [round(t["seconds"] * 1000) for t in tests]
        self.all_bits = (1 << self.n) - 1
        # (file, qualname) -> bits, for functions whose first line moved
        fq = {}
        for (f, q, _l), b in maps["code_tests"].items():
            fq[(f, q)] = fq.get((f, q), 0) | b
        self.fq_tests = fq
        mod_codes = {}
        for (f, q, _l), b in maps["code_tests"].items():
            mod_codes[f] = mod_codes.get(f, 0) | b
        self.file_code_tests = mod_codes
        self.import_codes = maps["import_codes"]
        self.order_sensitive = maps.get("order_sensitive", set())
        self._fq_classes = {(f, q.rsplit(".", 1)[0]) for (f, q) in self.fq_tests if "." in q}
        lc = maps.get("local_cpu") or [None] * self.n
        self.local_ms = []
        for i, t in enumerate(tests):
            v = lc[i] if i < len(lc) else None
            est = t["seconds"] * LOCAL_RATIO if v is None else v
            if t["id"].endswith("::test_lazy_load"):
                est = max(est, t["seconds"])  # runs a subprocess
            self.local_ms.append(est * 1000 * speed)
        self.class_setup_ms = {k: v * 1000 * speed for k, v in (maps.get("class_setup") or {}).items()}
        self.mod_import_ms = {k: v * 1000 * speed for k, v in (maps.get("module_import_cpu") or {}).items()}
        self.cls_of = []
        self.mod_of = []
        for t in tests:
            f, _, rest = t["id"].partition("::")
            mod = f[:-3].replace("/", ".")
            self.mod_of.append(mod)
            self.cls_of.append(mod + "." + rest.split("::")[0])
        self.lazy_load = [i for i, t in enumerate(tests) if t["id"].endswith("test_dialect.py::TestDialect::test_lazy_load")]

    def fq_tests_has_class(self, mod, qual):
        rel = mod.replace(".", "/")
        return (rel + ".py", qual) in self._fq_classes or (rel + "/__init__.py", qual) in self._fq_classes

    def local_cost(self, nums):
        """Estimated CPU ms to run nums here."""
        if not nums:
            return 0.0
        total = LOCAL_FIXED * 1000 * self.speed
        classes = set()
        mods = set()
        for n in nums:
            total += self.local_ms[n]
            classes.add(self.cls_of[n])
            mods.add(self.mod_of[n])
        for c in classes:
            total += self.class_setup_ms.get(c, 0.0)
        for m in mods:
            total += self.mod_import_ms.get(m, 0.0)
        return total


def modname_of(rel):
    if not rel.endswith(".py"):
        return None
    m = rel[:-3].replace("/", ".")
    if m.endswith(".__init__"):
        m = m[: -len(".__init__")]
    return m


def bits_list(bits):
    out = []
    i = 0
    while bits:
        if bits & 1:
            out.append(i)
        bits >>= 1
        i += 1
    return out


def popcount(b):
    return bin(b).count("1")


class Analysis:
    def __init__(self, ctx):
        self.ctx = ctx
        self.cands = 0  # tests to decide on (run locally or include)
        self.must = 0  # tests to include no matter what
        self.need_fp = False
        self.notes = []
        self.files = {}  # rel -> new text or None
        self.sqlglot_py_changed = False
        self.fp_modules = set()  # sqlglot modules whose import-time state may differ
        self.data_cands = 0  # candidates only because they read a changed data file
        self.lazy_load = False
        self.check_test_imports = False
        self.cands_fn = 0  # candidates from changed function bodies (lines known)
        self.changed_lines = set()  # (path, old line) deciding those changes
        self.hunks_by_file = {}
        self.skip_reason = {}  # replayable test -> why no replay was possible
        self.parse_checks = []  # (test, sqls): drop the test if all still parse
        self.infected = 0  # tests that certainly see a changed comparison outcome
        self.fn_clean = 0  # files whose only changes are function bodies
        self.n_fps = 0
        self.all_pkg_py = False
        self.warm_code_changes = {}
        self.replay_skip = {}  # replayed test -> queries the change cannot reach

    def note(self, s):
        self.notes.append(s)


def analyse_static(ctx, patch_text):
    maps = ctx.maps
    a = Analysis(ctx)
    fps = brlib.parse_patch(patch_text)
    if not fps:
        a.note("empty patch")
        return a
    a.n_fps = len(fps)
    a.all_pkg_py = all((fp.path or "").startswith("sqlglot/") and (fp.path or "").endswith(".py")
                       and fp.old_path == fp.new_path for fp in fps)
    a.py_in_patch = any((p or "").endswith(".py") for fp in fps for p in (fp.old_path, fp.new_path))
    for fp in fps:
        for p in {fp.old_path, fp.new_path} - {None}:
            bits = maps["open_tests"].get(p, 0)
            if p in _FIXTURE_READERS and fp.old_path == fp.new_path and not (fp.is_new or fp.is_deleted or fp.binary):
                bits &= ~fixture_unaffected_bits(ctx, a, p, fp)
            a.cands |= bits
            if not p.endswith(".py"):
                a.data_cands |= bits
            base = p.rsplit("/", 1)[-1]
            if base in PYTEST_CONFIG or base.startswith("conftest"):
                a.must |= ctx.all_bits
                a.note(f"config file {p}")
            elif base in MAYBE_PYTEST_CONFIG and "/" not in p:
                touched = "\n".join(l for h in fp.hunks for l in h[4])
                old_cfg = ""
                try:
                    with open(os.path.join(SRC, p), encoding="utf-8", errors="replace") as f:
                        old_cfg = f.read()
                except OSError:
                    pass
                if "pytest" in touched or "[tool:pytest]" in old_cfg or "[tool.pytest" in old_cfg or "[pytest]" in old_cfg:
                    a.must |= ctx.all_bits
                    a.note(f"pytest config in {p}")
        if fp.is_new or fp.is_deleted or fp.rename:
            for p in {fp.old_path, fp.new_path} - {None}:
                d = p.rsplit("/", 1)[0] if "/" in p else ""
                a.cands |= maps["list_tests"].get(d, 0)
                a.cands |= maps["list_tests"].get(d.rstrip("/") or ".", 0)
        if fp.binary:
            a.note(f"binary {fp.path}")
            if fp.path and fp.path.endswith(".py"):
                a.must |= ctx.all_bits
            continue

        old_text = None
        if fp.old_path is not None:
            try:
                with open(os.path.join(SRC, fp.old_path), encoding="utf-8", newline="") as f:
                    old_text = f.read()
            except (OSError, UnicodeDecodeError):
                old_text = None
        if fp.is_deleted:
            new_text = None
        else:
            try:
                new_text = brlib.apply_hunks(old_text or "", fp.hunks) if fp.hunks else (old_text or "")
            except brlib.PatchError as e:
                a.note(f"patch failed {fp.path}: {e}")
                a.must |= ctx.all_bits
                continue
        if fp.rename and fp.old_path:
            a.files[fp.old_path] = None
        a.files[fp.path] = new_text
        a.warm_code_changes.pop(fp.path, None)
        a.hunks_by_file[fp.path] = fp.hunks

        path = fp.path or fp.old_path
        if not path.endswith(".py") and not (fp.old_path or "").endswith(".py"):
            continue
        in_pkg = path.startswith("sqlglot/") or (fp.old_path or "").startswith("sqlglot/")
        in_tests = path.startswith("tests/") or (fp.old_path or "").startswith("tests/")
        mod = modname_of(path)
        if not in_pkg and not in_tests:
            # scripts outside the package and the tests (setup.py, docs,
            # benchmarks) are never imported by tests; readers were added above
            continue
        if fp.is_deleted or fp.rename or old_text is None:
            # module removed / added
            if fp.old_path:
                om = modname_of(fp.old_path)
                a.must |= maps["module_import_tests"].get(om, 0)
                a.cands |= ctx.file_code_tests.get(fp.old_path, 0)
                if fp.old_path.startswith("sqlglot/"):
                    a.need_fp = True
                    a.sqlglot_py_changed = True
                    a.fp_modules = None  # module gone: look everywhere
            if in_pkg:
                a.need_fp = True
                a.sqlglot_py_changed = True
                a.fp_modules = None
            if in_tests:
                a.must |= maps["file_tests"].get(path, 0)
            continue
        source_path = os.path.join(SRC, path)
        try:
            old_co = compile(old_text, source_path, "exec", dont_inherit=True)
        except SyntaxError:
            a.must |= ctx.all_bits
            continue
        try:
            new_co = compile(new_text, source_path, "exec", dont_inherit=True)
        except (SyntaxError, ValueError) as e:
            a.note(f"syntax error in {path}: {e}")
            # importing this module fails. Which test files then fail to
            # import is checked on the patched tree (fp_diff); tests that
            # import it lazily ran its code.
            a.cands |= ctx.file_code_tests.get(path, 0)
            a.check_test_imports = True
            if in_tests:
                a.must |= maps["file_tests"].get(path, 0)
            if in_pkg:
                a.need_fp = True
                a.sqlglot_py_changed = True
                a.fp_modules = None
            elif not in_tests:
                a.must |= maps["module_import_tests"].get(mod, 0)
            continue
        changed, added, moved = brlib.code_diff(old_co, new_co)
        if (in_pkg and not added and all(brlib.is_function_code(co) and kind == "self" for co, kind in changed)):
            a.fn_clean += 1
            if a.all_pkg_py and changed:
                a.warm_code_changes[path] = (old_co, new_co, changed)
        hunk_lines = brlib.changed_old_lines(fp.hunks)
        stmt_index = None
        file_lines = maps.get("line_tests", {}).get(path)
        if not changed and not added:
            a.note(f"{path}: no bytecode change")
            continue
        if in_pkg:
            a.sqlglot_py_changed = True
        for co, kind in changed:
            key = (path, co.co_qualname, co.co_firstlineno)
            if brlib.is_function_code(co):
                bits = ctx.maps["code_tests"].get(key, 0)
                fine = None
                if kind == "self" and file_lines is not None and bits and key not in ctx.order_sensitive:
                    if stmt_index is None:
                        try:
                            stmt_index = brlib.StmtIndex(old_text)
                        except SyntaxError:
                            stmt_index = False
                    if stmt_index:
                        fine = narrow_by_lines(stmt_index, co, hunk_lines, file_lines)
                if fine is not None:
                    a.note(f"{path}:{co.co_qualname} {kind} -> {popcount(bits)} (lines: {popcount(fine & bits)})")
                    bits &= fine
                    a.changed_lines |= {(path, l) for l in fine_lines(stmt_index, co, hunk_lines)}
                else:
                    a.note(f"{path}:{co.co_qualname} {kind} -> {popcount(bits)}")
                    a.changed_lines |= {(path, l) for l in brlib.code_line_span(co)}
                    a.changed_lines.add((path, co.co_firstlineno))
                if (VALUE_RULE and kind == "self" and in_pkg and len(changed) == 1 and not added
                        and key not in ctx.import_codes and key not in ctx.order_sensitive):
                    edit = comparison_edit(old_text, new_text)
                    if edit is not None:
                        drop, infected = value_unaffected(ctx, path, edit, bits, file_lines)
                        a.infected |= infected & bits & ~drop
                        if drop:
                            bits &= ~drop
                            a.note(f"  comparison {edit[1]} {edit[2]}: {popcount(drop)} tests see the same outcome")
                a.cands_fn |= bits
                if key in ctx.import_codes:
                    a.need_fp = True
                    _add_fp_module(a, mod, in_pkg)
                    a.note(f"  runs at import time")
                if not in_pkg and not in_tests:
                    a.must |= ctx.all_bits
            else:
                a.note(f"{path}:{co.co_qualname} body changed")
                if in_pkg:
                    a.need_fp = True
                    if co.co_qualname == "<module>":
                        # module code can reach into other modules: diff everything
                        a.fp_modules = None
                    else:
                        _add_fp_module(a, mod, in_pkg)
                elif in_tests:
                    # test module or test class body: every test in the
                    # file, and in files importing it
                    a.cands |= maps["file_tests"].get(path, 0)
                    a.cands |= maps["module_import_tests"].get(mod, 0)
                    if mod == "tests.helpers" or path.endswith("__init__.py"):
                        a.cands |= maps["module_import_tests"].get(mod, 0)
                else:
                    a.must |= ctx.all_bits
    return a


def fine_lines(si, co, hunk_lines):
    """The deciding lines narrow_by_lines used (it succeeded for co)."""
    span = brlib.code_line_span(co)
    lo, hi = min(min(span), co.co_firstlineno), max(span)
    for start, end, n, _p in si.nodes:
        if getattr(n, "lineno", None) == co.co_firstlineno or start == co.co_firstlineno:
            hi = max(hi, end)
            break
    out = set()
    for l in hunk_lines:
        if lo <= l <= hi:
            out |= si.deciding_lines(l, lo, hi) or set(range(lo, hi + 1))
    return out


# ---------------------------------------------------------- comparison values
# build/value_record.py records, per test, how every comparison in sqlglot came
# out (under each comparison operator) when the test ran alone. A patch whose
# only change is one comparison's operator or one constant operand cannot
# affect a test whose every recorded evaluation of it comes out the same.

VALUES_DIR = os.path.join(MAPS, "values")
_VALUES = {}
_CMP_OPS = ("Lt", "LtE", "Gt", "GtE", "Eq", "NotEq", "Is", "IsNot", "In", "NotIn")


def _values_index():
    if "index" not in _VALUES:
        try:
            with open(os.path.join(VALUES_DIR, "index.pkl"), "rb") as f:
                d = pickle.load(f)
            _VALUES["index"] = (frozenset(d["recorded"]), d["sites"])
        except Exception:  # noqa: BLE001
            _VALUES["index"] = None
    return _VALUES["index"]


def _values_site(path, site):
    import hashlib

    name = os.path.join(hashlib.sha1(path.encode()).hexdigest()[:16], "%d_%d_%d_%d.pkl" % site)
    try:
        with open(os.path.join(VALUES_DIR, name), "rb") as f:
            return pickle.load(f)
    except Exception:  # noqa: BLE001
        return {}


def _single_diff(a, b, out):
    """Collect differing (old, new) node pairs; False once the shapes differ
    or more than one place differs."""
    import ast as _ast

    if type(a) is not type(b):
        if isinstance(a, _ast.cmpop) and isinstance(b, _ast.cmpop):
            return False  # handled at the comparison
        out.append((a, b))
        return len(out) <= 1
    if isinstance(a, _ast.Compare) and [type(o) for o in a.ops] != [type(o) for o in b.ops]:
        # an operator change: the rest of the comparison must be identical
        if len(a.ops) != len(b.ops) or len(a.comparators) != len(b.comparators):
            return False
        sub = []
        for p, q in zip([a.left] + a.comparators, [b.left] + b.comparators):
            if not _single_diff(p, q, sub) or sub:
                return False
        out.append(("ops", a, b))
        return len(out) <= 1
    if isinstance(a, _ast.AST):
        for f in a._fields:
            x, y = getattr(a, f, None), getattr(b, f, None)
            if isinstance(x, list) or isinstance(y, list):
                if not (isinstance(x, list) and isinstance(y, list)) or len(x) != len(y):
                    return False
                for p, q in zip(x, y):
                    if not _single_diff(p, q, out):
                        return False
            elif isinstance(x, _ast.AST) or isinstance(y, _ast.AST):
                if not _single_diff(x, y, out):
                    return False
            elif x != y or type(x) is not type(y):
                if isinstance(a, _ast.Constant) and f == "value":
                    out.append((a, b))
                    return len(out) <= 1
                if f in ("kind",) and x is None and y is None:
                    continue
                return False
        return True
    if a != b or type(a) is not type(b):
        return False
    return True


def comparison_edit(old_text, new_text):
    """If the only change between two module sources is one single-operator
    comparison (its operator, or a constant operand), describe it."""
    import ast as _ast

    try:
        ta, tb = _ast.parse(old_text), _ast.parse(new_text)
    except SyntaxError:
        return None
    out = []
    if not _single_diff(ta, tb, out) or len(out) != 1:
        return None
    if out[0][0] == "ops":
        _t, cx, cy = out[0]
        if len(cx.ops) != 1:
            return None
        o, n = type(cx.ops[0]).__name__, type(cy.ops[0]).__name__
        if o not in _CMP_OPS or n not in _CMP_OPS:
            return None
        return ((cx.lineno, cx.col_offset, cx.end_lineno, cx.end_col_offset), "op", (o, n))
    x, y = out[0]
    # otherwise: a constant directly under a comparison
    def parent_compare(tree, node):
        for n in _ast.walk(tree):
            if isinstance(n, _ast.Compare) and (n.left is node or node in n.comparators):
                return n
        return None

    if isinstance(x, _ast.Constant) and isinstance(y, _ast.Constant):
        cx, cy = parent_compare(ta, x), parent_compare(tb, y)
        if cx is None or cy is None or len(cx.ops) != 1 or len(cy.ops) != 1:
            return None
        op = type(cx.ops[0]).__name__
        if op in ("Is", "IsNot") or op not in _CMP_OPS:
            return None
        side = 0 if cx.left is x else 1
        return ((cx.lineno, cx.col_offset, cx.end_lineno, cx.end_col_offset), "const", (op, side, y.value))
    return None


def _same_after(records, kind, info):
    """records: (outcome signatures, literal operand pairs or None)."""
    import operator as _op

    sigs, pairs = records
    if not sigs:
        return False
    if kind == "op":
        o, n = _CMP_OPS.index(info[0]), _CMP_OPS.index(info[1])
        for sig in sigs:
            if sig[n] not in (True, False) or sig[o] != sig[n]:
                return False
        return True
    op, side, newc = info
    if pairs is None or not pairs:
        return False
    fns = {"Lt": _op.lt, "LtE": _op.le, "Gt": _op.gt, "GtE": _op.ge, "Eq": _op.eq, "NotEq": _op.ne,
           "In": lambda a, b: a in b, "NotIn": lambda a, b: a not in b}
    f = fns[op]
    for a, b in pairs:
        try:
            old = f(a, b)
            new = f(newc, b) if side == 0 else f(a, newc)
        except Exception:  # noqa: BLE001
            return False
        if type(old) is not bool or type(new) is not bool or old != new:
            return False
    return True


def _known_differs(records, kind, info):
    """Some recorded evaluation certainly comes out differently after the edit."""
    sigs, pairs = records
    if kind == "op":
        o, n = _CMP_OPS.index(info[0]), _CMP_OPS.index(info[1])
        return any(sig[o] in (True, False) and sig[n] in (True, False) and sig[o] != sig[n] for sig in sigs)
    return False


def value_unaffected(ctx, path, edit, bits, file_lines):
    """Tests among bits whose recorded evaluations of the edited comparison
    all come out the same after the edit."""
    idx = _values_index()
    if idx is None:
        return 0, 0
    recorded, sites = idx
    site, kind, info = edit
    if (path,) + site not in sites:
        return 0, 0
    shard = _values_site(path, site)
    line = site[0]
    drop = 0
    infected = 0
    for n in bits_list(bits):
        if n >= ctx.n:
            continue  # pseudo tests (regions of a replayed test)
        tid = ctx.tests[n]["id"]
        if tid not in recorded:
            continue
        recs = shard.get(tid)
        if recs is None:
            # never evaluated when run alone; kept if a full-suite run reached
            # the line (a path only a warm cache takes)
            if file_lines is not None and (file_lines.get(line, 0) >> n) & 1:
                continue
            drop |= 1 << n
            continue
        if recs == "U" or not isinstance(recs, tuple):
            continue
        if _same_after(recs, kind, info):
            drop |= 1 << n
        elif _known_differs(recs, kind, info):
            infected |= 1 << n
    return drop, infected


def narrow_by_lines(si, co, hunk_lines, file_lines):
    """Tests that ran a statement deciding whether the changed lines of `co`
    are reached, or None when that cannot be told."""
    span = brlib.code_line_span(co)
    if not span:
        return None
    lo, hi = min(min(span), co.co_firstlineno), max(span)
    # the function's own source extent (its last statement may end later)
    for start, end, n, _p in si.nodes:
        if getattr(n, "lineno", None) == co.co_firstlineno or start == co.co_firstlineno:
            hi = max(hi, end)
            break
    mine = [l for l in hunk_lines if lo <= l <= hi]
    if not mine:
        return None
    lines = set()
    for l in mine:
        d = si.deciding_lines(l, lo, hi)
        if d is None:
            return None
        lines |= d
    bits = 0
    for l in lines:
        bits |= file_lines.get(l, 0)
    return bits


def _add_fp_module(a, mod, in_pkg):
    if a.fp_modules is not None and in_pkg:
        a.fp_modules.add(mod)


def fp_diff(ctx, a, pristine):
    """Runs in the child, after sys.path points at the patched tree."""
    import sys as _s

    import brfp

    maps = ctx.maps
    t0 = time.process_time()
    rdeps = maps.get("rdeps")
    if rdeps is None or a.fp_modules is None:
        errors = brfp.import_all()
        scope = None
    else:
        import importlib

        scope = set()
        for m in a.fp_modules:
            scope.add(m)
            scope.update(rdeps.get(m, ()))
        errors = {}
        import sqlglot  # noqa: F401  (package side effects first)

        scope.discard("sqlglot.__main__")
        for m in sorted(scope):
            try:
                importlib.import_module(m)
            except BaseException as e:  # noqa: BLE001
                errors[m] = repr(e)
        a.note(f"fp scope {len(scope)} modules")
    if errors:
        touch_by_mod = {}
        for ck, b in maps["touch"].items():
            m = ck.split(":", 1)[0]
            touch_by_mod[m] = touch_by_mod.get(m, 0) | b
    for mod, err in errors.items():
        if len(a.notes) < 60:
            a.note(f"import error {mod}: {err[:200]}")
        rel = mod.replace(".", "/") + ".py"
        # tests that ran its code import it one way or another
        a.cands |= ctx.file_code_tests.get(rel, 0)
        a.cands |= ctx.file_code_tests.get(mod.replace(".", "/") + "/__init__.py", 0)
        a.cands |= touch_by_mod.get(mod, 0)
        # function-level imports of it now raise
        a.cands |= maps.get("lazy_import_tests", {}).get(mod, 0)
    mods = brfp.sqlglot_modules() if scope is None else sorted(m for m in scope if m in _s.modules)
    new = brfp.fingerprint_modules(mods)
    unstable = pristine["unstable"]
    still = pristine["still"]
    if unstable:
        newu = brfp.fingerprint_modules(mods, unordered=True, only=unstable)
    old = pristine["fps"]
    if scope is not None:
        old = {k: v for k, v in old.items() if k.split("|", 2)[1] in scope}
        # modules in scope that were never imported here (e.g. failed)
    # globals present in the pristine full import but not here: lazily
    # provided package attributes need a getattr to appear
    fpx = None
    for k in old:
        if k in new or not k.startswith("G|"):
            continue
        _g, mod, name = k.split("|", 2)
        if old[k][1].startswith("M:"):
            continue
        m = _s.modules.get(mod)
        if m is None:
            continue
        try:
            v = getattr(m, name)
        except BaseException:  # noqa: BLE001
            continue
        if fpx is None:
            fpx = brfp._FP()
        new[k] = (brfp._tag(v), fpx.h(v))
    diffs = []
    for k in set(old) | set(new):
        if k in still:
            continue
        if k in unstable:
            if pristine["unordered"].get(k) != newu.get(k):
                diffs.append(k)
        elif old.get(k) != new.get(k):
            diffs.append(k)
    # modules that failed to import show up as missing keys; their tests are
    # already in must
    failed = set(errors)

    for k in diffs:
        kind, mod, rest = k.split("|", 2)
        if mod in failed:
            continue
        o, n = old.get(k), new.get(k)
        if kind == "G" and ((o and o[1].startswith("M:")) or (n and n[1].startswith("M:"))):
            continue  # submodule binding: the submodule is diffed itself
        if kind == "G" and o is None and "__getattr__" in vars(_s.modules.get(mod) or brfp):
            continue  # cached by a lazy package __getattr__
        tag = (n or o)[0]
        if kind == "G":
            name = rest
            bits = global_readers(ctx, mod, name, o, n)
            a.cands |= bits
            a.note(f"global {mod}.{name} changed -> {popcount(bits)}")
        elif kind == "A":
            qual, attr = rest.rsplit("|", 1)
            ck = f"{mod}:{qual}"
            fam = [ck] + maps["descendants"].get(ck, [])
            if o is None or n is None or o[0] != n[0]:
                bits = 0
                avt = maps.get("attr_via_tests")
                rk = f"{mod}:{qual}.{attr}"
                if avt is not None and (n or o)[0] == "D" and (o is None or o[0] == "D"):
                    # a data attribute added (overriding an inherited one) or
                    # removed: whoever read that name through the class or a
                    # subclass, plus readers of the class's own old value
                    for c in fam:
                        bits |= avt.get((c, attr), 0)
                    bits |= maps["read_tests"].get(rk, 0)
                    if not bits:
                        # nothing read this name through the family: a name new
                        # to the family; tests using the family and the name
                        fb = 0
                        for c in fam:
                            fb |= maps["touch"].get(c, 0)
                        bits = fb & maps.get("name_attr_tests", maps["name_tests"]).get(attr, ctx.all_bits)
                elif o is not None and o[0] == "F" and n is None:
                    # a method removed: only calls that ran it now reach the parent
                    rel = mod.replace(".", "/")
                    for f in (rel + ".py", rel + "/__init__.py"):
                        bits |= ctx.fq_tests.get((f, f"{qual}.{attr}"), 0)
                    bits |= maps["read_tests"].get(rk, 0)
                    if not bits and not ctx.fq_tests_has_class(mod, qual):
                        for c in fam:
                            bits |= maps["touch"].get(c, 0)
                elif o is None and n is not None and n[0] == "F" and attr.endswith("_sql") \
                        and (sql_bits := added_sql_method_tests(ctx, mod, qual, attr, fam)) is not None:
                    bits = sql_bits
                elif o is None and n is not None and n[0] == "F" and not attr.endswith("_sql"):
                    # a method added: calls by name on the family, or calls that
                    # ran an inherited version of it
                    fb = 0
                    for c in fam:
                        fb |= maps["touch"].get(c, 0)
                    names = maps.get("name_attr_tests", maps["name_tests"]).get(attr, 0)
                    inh = 0
                    cls = _resolve(mod, qual)
                    for base in (cls.__mro__[1:] if isinstance(cls, type) else ()):
                        if attr in base.__dict__:
                            for f, q in brfp.func_qualnames(base.__dict__[attr]):
                                relf = os.path.relpath(f, os.getcwd()) if os.path.isabs(f) else f
                                inh |= ctx.fq_tests.get((relf, q), 0)
                    bits = fb & (names | inh)
                else:
                    for c in fam:
                        bits |= maps["touch"].get(c, 0)
                    bits |= maps["read_tests"].get(rk, 0)
                a.note(f"attr {ck}.{attr} added/removed/retyped -> {popcount(bits)}")
            elif tag == "D":
                rk = f"{mod}:{qual}.{attr}"
                keyed_old = pristine.get("keyed", {}).get(k)
                keyed_new = None
                if keyed_old is not None and "key_tests" in maps and rk in maps.get("tracked", ()):
                    cls = _resolve(mod, qual)
                    v = cls.__dict__.get(attr) if cls is not None else None
                    keyed_new = brfp.keyed_fp(v)
                if keyed_old is not None and keyed_new is not None:
                    changed_keys = [kk for kk in set(keyed_old) | set(keyed_new)
                                    if keyed_old.get(kk) != keyed_new.get(kk)]
                    kt = maps["key_tests"]
                    bits = kt.get((rk, "*"), 0)
                    for kk in changed_keys:
                        bits |= kt.get((rk, kk), 0)
                    a.note(f"attr {rk}: {len(changed_keys)} keys changed -> {popcount(bits)} "
                           f"(of {popcount(maps['read_tests'].get(rk, 0))} readers)")
                elif rk in maps["read_tests"] or rk in maps.get("tracked", ()):
                    bits = maps["read_tests"].get(rk, 0)
                else:
                    bits = 0
                    for c in fam:
                        bits |= maps["touch"].get(c, 0)
                    a.note("  (untracked attribute, using class touch)")
                a.note(f"attr {rk} changed -> {popcount(bits)}")
            elif tag == "F":
                bits = 0
                cls = _resolve(mod, qual)
                v = cls.__dict__.get(attr) if cls is not None else None
                for f, q in brfp.func_qualnames(v):
                    rel = os.path.relpath(f, os.getcwd()) if os.path.isabs(f) else f
                    bits |= ctx.fq_tests.get((rel, q), 0)
                a.note(f"method {ck}.{attr} defaults/closure changed -> {popcount(bits)}")
            else:  # attribute bound to a different class
                rk = f"{mod}:{qual}.{attr}"
                bits = maps["read_tests"].get(rk, 0)
                for c in fam:
                    bits |= maps["touch"].get(c, 0)
                a.note(f"class-valued attr {rk} rebound -> {popcount(bits)}")
            a.cands |= bits
        elif kind == "B":
            ck = f"{mod}:{rest}"
            bits = 0
            for c in [ck] + maps["descendants"].get(ck, []):
                bits |= maps["touch"].get(c, 0)
            a.cands |= bits
            a.note(f"class {ck} bases/new -> {popcount(bits)}")
    if errors or a.check_test_imports:
        # test files that cannot be imported fail whole, as at collection
        import importlib

        bad = 0
        if "sqlglot" in errors:
            # the package itself is broken: every file importing it fails
            a.must |= maps["module_import_tests"].get("sqlglot", ctx.all_bits)
        for file in [] if "sqlglot" in errors else sorted({t["file"] for t in ctx.tests}):
            modname = file[:-3].replace("/", ".")
            try:
                importlib.import_module(modname)
            except BaseException as e:  # noqa: BLE001
                bad += 1
                a.must |= maps["file_tests"].get(file, 0)
                a.note(f"test file {file} fails to import: {e!r}"[:200])
        a.note(f"{bad} test files fail to import")
    a.note(f"fp diff: {len(diffs)} keys, {time.process_time() - t0:.2f}s cpu")


def added_sql_method_tests(ctx, mod, qual, attr, fam):
    """A `<key>_sql` method added to a generator class: the generator reaches
    it through its dispatch table for expressions of that key, whose lookups
    were recorded as key reads of the TRANSFORMS they came from. None when
    that does not hold (not a generator, binary expressions, unknown key)."""
    import sys as _s

    import brfp

    maps = ctx.maps
    exp = _s.modules.get("sqlglot.expressions")
    gen = _s.modules.get("sqlglot.generator")
    cls = _resolve(mod, qual)
    if exp is None or gen is None or not isinstance(cls, type) or not issubclass(cls, gen.Generator):
        return None
    klass = getattr(exp, "EXPR_CLASSES", {}).get(attr[:-4])
    if klass is None:
        return None
    binary = getattr(exp, "Binary", None)
    if binary is not None and isinstance(klass, type) and issubclass(klass, binary):
        return None  # nested binary nodes are generated by name, bypassing dispatch
    reg = brfp.class_registry()
    kfp = brfp._FP().h(klass)
    kt = maps.get("key_tests") or {}
    bits = 0
    found = False
    stack = [cls]
    seen = set()
    while stack:
        c = stack.pop()
        if id(c) in seen:
            continue
        seen.add(id(c))
        owner = next((b for b in c.__mro__ if "TRANSFORMS" in b.__dict__), None)
        okey = reg.get(id(owner)) if owner is not None else None
        if okey is None:
            return None
        found = True
        bits |= kt.get((f"{okey}.TRANSFORMS", kfp), 0) | kt.get((f"{okey}.TRANSFORMS", "*"), 0)
        stack.extend(type.__subclasses__(c))
    if not found:
        return None
    # the TRANSFORMS read may be inherited from a class shared with other
    # dialects: keep tests that used this family; add explicit calls by name
    fb = 0
    for c in fam:
        fb |= maps["touch"].get(c, 0)
    return fb & (bits | maps.get("name_attr_tests", maps["name_tests"]).get(attr, 0))


def global_readers(ctx, mod, name, o, n):
    """Tests that can observe module global `name` of `mod` changing (o: old
    fingerprint, n: new; None when absent)."""
    import sys as _s

    import brfp

    maps = ctx.maps
    ngt = maps.get("name_global_tests")
    if ngt is None:  # older maps
        return maps["name_tests"].get(name, 0)
    base = mod.replace(".", "/")
    files = (base + ".py", base + "/__init__.py")
    bits = 0
    for f in files:
        bits |= ngt.get((f, name), 0)  # loads of the global inside its module
    bits |= maps.get("lazy_import_tests", {}).get(f"{mod}.{name}", 0)  # from mod import name, in a function
    functions_only = True
    for side in (o, n):
        if side is not None and side[0] != "F":
            functions_only = False
    if functions_only:
        # a function: whoever reaches it from elsewhere calls it, so its callers
        pristine_q = []
        for f in files:
            if ctx.fq_tests.get((f, name)):
                pristine_q.append((f, name))
        for key in pristine_q:
            bits |= ctx.fq_tests[key]
        v = getattr(_s.modules.get(mod), name, None) if n is not None else None
        aliased = False
        for f, q in brfp.func_qualnames(v):
            rel = os.path.relpath(f, os.getcwd()) if os.path.isabs(f) else f
            bits |= ctx.fq_tests.get((rel, q), 0)
            if q != name:
                aliased = True
        if aliased or (o is not None and not pristine_q):
            bits |= maps["name_attr_tests"].get(name, 0)
    else:
        bits |= maps["name_attr_tests"].get(name, 0)
    return bits


def bits_or_all(ctx, fam):
    return ctx.all_bits


def _resolve(mod, qual):
    import sys as _s

    obj = _s.modules.get(mod)
    for part in qual.split("."):
        obj = getattr(obj, part, None) if obj is not None else None
        if obj is None:
            return None
        if hasattr(obj, "__get__") and not isinstance(obj, type):
            return None
    return obj


# ------------------------------------------------------------- local tests


class _Timeout(BaseException):
    pass


def run_local(ctx, nums, deadline, blind=(), replay=frozenset()):
    """Run tests (in tests.json order) in this process. Returns (failed set,
    not_run set)."""
    import unittest

    import brpool

    brpool.install()
    tests = ctx.tests
    failed = set()
    completed = set()
    hooks = None
    if replay:
        import brreplay

        hooks = brreplay.Hooks(ctx.maps["_replay_data"], "replay")
    why = []
    global LAST_WHY
    LAST_WHY = why
    # cheapest classes first, cheapest tests first inside a class
    by_cls = {}
    for n in nums:
        by_cls.setdefault(ctx.cls_of[n], []).append(n)
    cls_order = sorted(by_cls, key=lambda c: min(ctx.local_ms[n] for n in by_cls[c]) + ctx.class_setup_ms.get(c, 0))
    by_mod = {}
    run_order = []
    for c in cls_order:
        for n in sorted(by_cls[c], key=lambda n: ctx.local_ms[n]):
            tid = tests[n]["id"]
            file, _, rest = tid.partition("::")
            by_mod.setdefault(file, []).append((n, rest))
            run_order.append(n)
    if os.environ.get("BR_DEBUG_ORDER"):
        log("  run order: " + " ".join(tests[n]["id"].rsplit("::", 1)[1] for n in run_order))
    blind = list(blind)

    id_to_num = {}
    loaded = {}
    for file, items in by_mod.items():
        modname = file[:-3].replace("/", ".")
        try:
            module = __import__(modname, fromlist=["*"])
        except BaseException as e:  # noqa: BLE001
            log(f"  import of {file} failed: {e!r}"[:300])
            # the whole file fails in CI
            failed |= set(bits_list(ctx.maps["file_tests"].get(file, 0)))
            continue
        for n, rest in items:
            parts = rest.split("::")
            try:
                obj = module
                for p in parts[:-1]:
                    obj = getattr(obj, p)
                test = obj(parts[-1])
            except BaseException as e:  # noqa: BLE001
                log(f"  load {tests[n]['id']} failed: {e!r}"[:300])
                failed.add(n)
                continue
            id_to_num[test.id()] = n
            loaded[n] = test
    restore_setup = None
    if hooks is not None and REPLAY_BUDGET == 0:
        classes = {type(loaded[n]) for n in replay if n in loaded}
        for cls in classes:
            if all(n in replay and not replay[n] for n in loaded if type(loaded[n]) is cls):
                restore_setup = brreplay.light_setup(cls)
    suite = unittest.TestSuite([loaded[n] for n in run_order if n in loaded])
    started = set()

    class Result(unittest.TestResult):
        def __init__(self):
            super().__init__()
            self.cur = None

        def _mark(self, test):
            n = id_to_num.get(test.id()) if hasattr(test, "id") else None
            if n is None:
                # class / module fixture failure: description like
                # "setUpClass (tests.x.TestY)"
                desc = str(test)
                hit = False
                for tid, m in id_to_num.items():
                    clsname = tid.rsplit(".", 1)[0]
                    if clsname in desc or clsname.rsplit(".", 1)[0] in desc:
                        failed.add(m)
                        hit = True
                if not hit:
                    failed.update(id_to_num.values())
            else:
                failed.add(n)

        def startTest(self, test):
            super().startTest(test)
            n = id_to_num.get(test.id())
            if n is not None:
                started.add(n)
            if hooks is not None and n in replay:
                try:
                    hooks.budget = REPLAY_BUDGET * tests[n]["seconds"] / ctx.speed
                    rk = set(replay[n])
                    unk = [x for x in rk if isinstance(x, tuple) and x and x[0] == "__unknown__"]
                    rk -= set(unk)
                    hooks.real_keys = rk
                    hooks.abort = False
                    hooks.real_ok = 0
                    hooks.real_cap = partial_cap(len(rk))
                    hooks.skip_sqls = REPLAY_SKIP.get(n, frozenset())
                    hooks.unknown_left = 3 * unk[0][1] if unk else 0
                    hooks.duck_left = 3 * unk[0][1] if unk else 0
                    hooks.install(test)
                except BaseException as e:  # noqa: BLE001
                    log(f"  replay hooks failed: {e!r}")
                    failed.add(n)
                    hooks.uninstall()
                    # do not pay for a full run: skip the test, it is handed to CI
                    test._outcome = None
                    setattr(test, test._testMethodName, lambda: None)
            self.cur = n
            self.t0 = time.process_time()
            if n is not None:
                # CPU-time limit: wall time depends on what else the machine runs;
                # a test stuck without burning CPU is caught by the parent's deadline
                limit = max(2.0, 4 * max(tests[n]["seconds"], ctx.local_ms[n] / 1000) + 1.5)
                signal.setitimer(signal.ITIMER_PROF, limit)

        def stopTest(self, test):
            signal.setitimer(signal.ITIMER_PROF, 0)
            n = id_to_num.get(test.id())
            if hooks is not None and hooks.active:
                hooks.uninstall()
            if n is not None:
                completed.add(n)
            super().stopTest(test)
            if n in failed and not self.shouldStop:
                # a worker is needed anyway now: maybe the rest is cheaper in CI
                rest = [m for m in loaded if m not in started]
                if rest:
                    base = blind + sorted(failed)
                    inc = pack(base + rest, ctx.ms)[0] - pack(base, ctx.ms)[0]
                    if not ALL_LOCAL and inc < ctx.local_cost(rest) - LOCAL_FIXED * 1000 * ctx.speed:
                        log(f"  stopping after failure: {len(rest)} left go to CI")
                        self.shouldStop = True

        def _why(self, test, err):
            if len(why) < 6:
                try:
                    why.append(f"{test.id()}: {err[0].__name__}: {str(err[1])[:120]}")
                except Exception:  # noqa: BLE001
                    pass

        def addError(self, test, err):
            if hooks is not None and err and _is_accept(err[0]):
                return  # partial check passed
            self._why(test, err)
            self._mark(test)

        def addFailure(self, test, err):
            self._why(test, err)
            self._mark(test)

        def addSubTest(self, test, subtest, err):
            if err is not None and hooks is not None and _is_accept(err[0]):
                return
            if err is not None:
                self._why(test, err)
                self._mark(test)
                if hooks is not None:
                    hooks.abort = True  # the test has failed: stop its remaining queries

        def addUnexpectedSuccess(self, test):
            self._mark(test)

    def on_alarm(signum, frame):
        raise _Timeout()

    signal.signal(signal.SIGPROF, on_alarm)
    result = Result()
    if os.environ.get("BR_GC_THRESH"):
        import gc

        gc.set_threshold(int(os.environ["BR_GC_THRESH"]), 50, 100)
    # the first failing subtest decides the test: stop it there (our add*
    # methods never call stop(), so the suite itself carries on)
    result.failfast = True
    try:
        suite.run(result)
    except _Timeout:
        pass
    signal.setitimer(signal.ITIMER_PROF, 0)
    if restore_setup is not None:
        restore_setup()
    # whatever did not run to the end counts as failing
    failed |= set(id_to_num.values()) - completed
    for w in why:
        log("  local failure: " + w)
    return failed


# -------------------------------------------------------------- per commit


P_DATA = float(os.environ.get("BR_P_DATA", "0.85"))
P_INFECTED = float(os.environ.get("BR_P_INFECTED", "0.4"))  # a recorded comparison outcome changes


def expected_local(ctx, a, blind, local):
    """Expected cost (ms) of running `local` here with `blind` handed to CI."""
    ms = ctx.ms
    cost = ctx.local_cost(local)
    lb = 0
    for n in local:
        lb |= 1 << n
    p = P_INFECTED if (lb & a.infected) else P_DATA if (lb & a.data_cands) else P_FAIL
    lms = sum(ms[n] for n in local)
    if blind:
        cost += pack(blind, ms)[0] + p * 0.3 * lms
    else:
        cost += p * (WORKER_COST + 0.5 * lms)
    return cost


# The executor tests read their query files through load_sql_fixture_pairs:
# test_execute_* use only each pair's input SQL (and its #meta), and
# test_optimized_tpch also uses the expected SQL. An edit that leaves the
# fields a test reads unchanged cannot change that test.
_FIXTURE_READERS = {
    "tests/fixtures/optimizer/tpc-ds/tpc-ds.sql": (("TestExecutor::test_execute_tpcds", False),),
    "tests/fixtures/optimizer/tpc-h/tpc-h.sql": (("TestExecutor::test_execute_tpch", False),
                                                 ("TestExecutor::test_optimized_tpch", True)),
}


def _fixture_pairs(text):
    stmts = "\n".join(l for l in text.splitlines() if l and not l.startswith("--")).split(";")
    out = []
    for i in range(0, len(stmts), 2):
        if i + 1 < len(stmts):
            sql = stmts[i].strip()
            lines = sql.split("\n")
            j, meta = 0, []
            while j < len(lines) and lines[j].startswith("#"):
                if ":" not in lines[j]:
                    return None
                k, v = lines[j].split(":", 1)
                meta.append((k.lstrip("#").strip(), v.strip()))
                j += 1
            if j == len(lines):
                return None  # the loader itself would fail here
            out.append((tuple(meta), "\n".join(lines[j:]), stmts[i + 1].strip()))
    return out


def _parses(sql):
    try:
        import sqlglot

        sqlglot.parse_one(sql)
        return True
    except Exception as e:  # noqa: BLE001
        if os.environ.get("BR_DEBUG_PARSE"):
            log(f"parse check failed: {e!r}")
        return False


def fixture_unaffected_bits(ctx, a, path, fp):
    """Executor tests whose fields of this query file an edit leaves as they
    were (they read the file only through load_sql_fixture_pairs)."""
    try:
        with open(os.path.join(SRC, path), encoding="utf-8") as f:
            old_text = f.read()
        old = _fixture_pairs(old_text)
        new = _fixture_pairs(brlib.apply_hunks(old_text, fp.hunks))
    except Exception:  # noqa: BLE001
        return 0
    if old is None or new is None:
        return 0
    affected = set()
    parse_needed = {}
    code_changed = a.py_in_patch
    for suffix, reads_expected in _FIXTURE_READERS[path]:
        if len(old) != len(new):
            affected.add(suffix)
        elif reads_expected:
            if old != new:
                affected.add(suffix)
        elif [(m, q) for m, q, _ in old] != [(m, q) for m, q, _ in new]:
            # test_execute_* run only the pairs whose meta says "execute";
            # the others are just parsed (parse_one) and their result unused
            to_parse = []
            for (m0, q0, _), (m1, q1, _) in zip(old, new):
                if (m0, q0) == (m1, q1):
                    continue
                if m0 != m1 or dict(m1).get("execute") or code_changed:
                    affected.add(suffix)
                    break
                to_parse.append(q1)
            else:
                # decided once the patched tree is importable (process_commit)
                parse_needed.setdefault(suffix, []).extend(to_parse)
    readers = {sfx for rs in _FIXTURE_READERS.values() for sfx, _ in rs}
    out = 0
    for n, t in enumerate(ctx.tests):
        sfx = next((r for r in readers if t["id"].endswith(r)), None)
        if sfx is not None and sfx in parse_needed and sfx not in affected:
            a.parse_checks.append((n, parse_needed[sfx]))
        elif sfx is not None and sfx not in affected:
            out |= 1 << n
        elif sfx is None and t["id"].startswith("tests/test_executor.py::TestExecutor::"):
            # the other tests of the class only share its setUpClass, which
            # loads the pairs (it does not fail: the file still loads)
            out |= 1 << n
    if out:
        a.note(f"{path}: executor tests reading unchanged fields dropped ({popcount(out)}); affected {sorted(affected)}")
    return out


# Measurement tag: for a commit that sends test_execute_tpcds, a few
# milliseconds of tiny tests are added to its busiest worker. The board's
# simulated seconds (exact to the millisecond) then say why it was sent:
# ones digit = why no replay was possible, hundreds = how a local check failed.
_SKIP_CODE = {"must": 1, "inner": 2, "exec_other": 3, "test_file": 4, "linked": 5, "nodata": 6}


def tag_tpcds(ctx, tests, cid, ws, st):
    t_ds = next((t["number"] for t in tests if t["id"].endswith("TestExecutor::test_execute_tpcds")), None)
    if t_ds is None or not any(t_ds in w for w in ws):
        return ws
    plan, done = st.get("plan"), st.get("done")
    code = 0
    if plan is None:
        code = 900
    elif t_ds in plan["blind"]:
        r = (plan.get("skip") or {}).get(str(t_ds))
        if r:
            code = _SKIP_CODE.get(r, 7)
        elif any("replay test_execute_tpcds" in x for x in plan["notes"]):
            # replay possible, planner chose CI; tens digit: its local estimate in 3 s steps
            est = (plan.get("est") or {}).get(str(t_ds), 0) / 1000
            code = 800 + 10 * max(1, min(9, -(-int(est) // 3)))
        else:
            code = 8  # no replay entry at all
    elif done is None:
        code = 500  # local run incomplete / killed
    else:
        w = [x for x in done.get("why", []) if "test_execute_tpcds" in x]
        if not w:
            code = 600  # stopped before it ran, or reasons list full
        elif "budget spent" in w[0]:
            code = 100
        elif "earlier query failed" in w[0] or "executor result changed" in w[0]:
            code = 300
        elif "Diverge" in w[0]:
            code = 200
        else:
            code = 400  # a real assertion / error
    log(f"{cid}: tpcds tag {code}")
    return add_ms(ctx, ws, code)


def add_ms(ctx, ws, code):
    sent = {n for w in ws for n in w}
    ms = ctx.ms
    pool = sorted((n for n in range(ctx.n) if n not in sent and ms[n] >= 1), key=lambda n: -ms[n])
    extra, left = [], code
    for n in pool:
        if 0 < ms[n] <= left:
            extra.append(n)
            left -= ms[n]
        if left == 0:
            break
    if left:
        return ws
    ws = [list(w) for w in ws]
    loads = [sum(ms[n] for n in w) for w in ws]
    ws[loads.index(max(loads))].extend(extra)
    return ws


PARTIAL = float(os.environ.get("BR_PARTIAL", "0.25"))  # share of changed-executor queries to confirm
PARTIAL_MIN = int(os.environ.get("BR_PARTIAL_MIN", "8"))


VALUE_RULE = os.environ.get("BR_VALUE_RULE", "1") == "1"
PER_QUERY = os.environ.get("BR_PER_QUERY", "0") == "1"


def partial_cap(n_real):
    if PARTIAL >= 1 or n_real <= PARTIAL_MIN:
        return None
    return max(PARTIAL_MIN, -(-int(n_real * PARTIAL * 1000) // 1000))


def _is_accept(tp):
    try:
        import brreplay

        return isinstance(tp, type) and issubclass(tp, brreplay.Accept)
    except Exception:  # noqa: BLE001
        return False


def resolve_replay(ctx, a):
    """Fold the pseudo tests (everything but the front end; the executor alone)
    back into their tests, and pick the replayable tests a replay can verify:
    nothing outside the front end and the executor is reached, and executor
    changes are function bodies with known lines (the queries reaching them
    run for real and must give the pinned result)."""
    mode = {}
    rmap = ctx.maps.get("replay") or {}
    emap = ctx.maps.get("replay_exec") or {}
    data = ctx.maps.get("_replay_data")
    import brreplay

    test_file_changed = brreplay.TEST_FILE in a.files
    other = a.cands | a.must
    for t, p in rmap.items():
        inner = (other >> p) & 1 or (a.cands_fn >> p) & 1
        p2 = emap.get(t)
        exec_other = p2 is not None and (other >> p2) & 1
        exec_fn = p2 is not None and (a.cands_fn >> p2) & 1
        if exec_fn and os.environ.get("BR_NO_EXEC_REPLAY"):
            exec_other = True
        hit = ((other | a.cands_fn) >> t) & 1
        if not hit:
            continue
        why_not = ("must" if (a.must >> t) & 1 else "inner" if inner else "exec_other" if exec_other
                   else "test_file" if test_file_changed else "nodata" if not data else None)
        if why_not:
            a.skip_reason[t] = why_not
            continue
        tid = ctx.tests[t]["id"]
        cpu = (data.get("cpu") or {}).get(tid)
        if cpu is None:
            continue
        real = set()
        if exec_fn:
            cov = data.get("cov")
            if cov is None:
                continue
            real = {k for k, lines in cov.items() if lines & a.changed_lines}
        # queries whose recorded front-end+executor lines miss every changed
        # line: their results come straight from the recording
        fe = data.get("fe") or {}
        fe_cpu = data.get("fe_cpu") or {}
        fidx = {f: i for i, f in enumerate(data.get("fe_files") or ())}
        changed_enc = frozenset(fidx[f] * 1_000_000 + l for f, l in a.changed_lines if f in fidx)
        mine = ctx.replay_sqls.get(tid)
        if (fe and mine and a.all_pkg_py and a.fn_clean == a.n_fps and not a.need_fp and not a.must
                and a.changed_lines and PER_QUERY
                and data.get("fe_files") is not None
                and not ((data.get("fe_outer") or {}).get(tid.replace(".py::", ".").replace("/", ".").replace("::", "."), frozenset([-1])) & changed_enc)
                and (data.get("fe_outer") or {}).get(tid.replace(".py::", ".").replace("/", ".").replace("::", ".")) is not None):
            skip = {q for q in mine if q in fe and not (fe[q]["lines"] & changed_enc)}
            if skip:
                a.replay_skip[t] = frozenset(skip)
                run_q = [q for q in mine if q not in skip]
                cpu = sum(fe_cpu.get(q, 0.0) for q in run_q) + 0.05
                real = {k for k in real if any((fe.get(q) or {}).get("key") == k for q in run_q)}
                a.note(f"replay {tid.rsplit('::', 1)[1]}: {len(skip)}/{len(mine)} queries untouched by the change")
        costs = [data.get("exec_cpu", {}).get(k, 0.0) for k in real]
        cap = partial_cap(len(real))
        extra = sum(costs) if cap is None else sum(costs) * min(1.0, 1.5 * cap / len(costs))
        # its query file edited: the few queries with new SQL run for real
        edited = [f for f in a.files if f in set(ctx.maps.get("setup_linked_files") or ())]
        if edited and not LINKED_REPLAY:
            a.skip_reason[t] = "linked"
            continue  # an edited query file: hand the test to CI (no local attempt)
        if edited:
            nh = sum(len(a.hunks_by_file.get(f, [])) for f in edited) or 1
            real = set(real) | {("__unknown__", nh)}
            avg = (sum(data.get("exec_cpu", {}).values()) / max(1, len(data.get("exec_cpu", {})))) or 0.5
            extra += nh * (avg + 0.3)
        mode[t] = real
        ctx.local_ms[t] = (cpu + extra) * 1000 * ctx.speed
        if not real and REPLAY_BUDGET == 0 and data.get("light_setup_cpu") is not None:
            ctx.class_setup_ms[ctx.cls_of[t]] = data["light_setup_cpu"] * 1000 * ctx.speed
        a.note(f"replay {tid.rsplit('::', 1)[1]}: {len(real)} queries run for real, "
               f"{cpu + extra:.2f}s cpu to verify")
    a.cands |= a.cands_fn
    lim = (1 << ctx.n) - 1  # never let a pseudo number through
    a.cands &= lim
    a.must &= lim
    a.data_cands &= lim
    return mode


def decide(ctx, a, max_local_ms=float("inf")):
    """Split candidates into (blind, local)."""
    ms = ctx.ms
    must = bits_list(a.must)
    cands = [n for n in bits_list(a.cands & ~a.must)]
    if not cands:
        return must, []
    if ALL_LOCAL:
        cheap = [n for n in cands if ctx.local_ms[n] <= 8000 * ctx.speed]
        return must + [n for n in cands if n not in set(cheap)], cheap
    blind_base = list(must)
    best = (pack(blind_base + cands, ms)[0], cands, [])  # everything blind
    orders = (
        sorted(cands, key=lambda n: -ctx.local_ms[n]),  # costly to verify first
        sorted(cands, key=lambda n: -(ctx.local_ms[n] + 1) / (ms[n] + 1)),  # costly relative to CI first
    )
    for order in orders:
        # verify the first few, or all but the last few (the cheap-to-verify
        # costly tests sit at the end of the second order)
        L = len(order)
        js = range(0, min(L, 80)) if not SUFFIX_PLANS else sorted(set(range(0, min(L, 80))) | set(range(max(0, L - 80), L)))
        for j in js:
            blind = blind_base + order[:j]
            if ctx.local_cost(order[j:]) > max_local_ms:
                continue
            cost = expected_local(ctx, a, blind, order[j:])
            if cost < best[0]:
                best = (cost, order[:j], order[j:])
    # verify only the few heavy tests that are cheap to check here (a replayed
    # executor test among hundreds of small candidates), the rest to CI
    heavy = sorted((n for n in cands if ms[n] >= 2000 and ctx.local_ms[n] < ms[n] * 1.5),
                   key=lambda n: -(ms[n] - ctx.local_ms[n]))
    if os.environ.get("BR_DEBUG_DECIDE"):
        log(f"  heavy {[(ctx.tests[n]['id'].rsplit('::',1)[1], ms[n], int(ctx.local_ms[n])) for n in cands if ms[n] >= 2000]} must-heavy {[ctx.tests[n]['id'].rsplit('::',1)[1] for n in must if ms[n] >= 2000]}")
    for k in range(1, (min(3, len(heavy)) if HEAVY_PLANS else 0) + 1):
        loc = heavy[:k]
        if ctx.local_cost(loc) > max_local_ms:
            if os.environ.get("BR_DEBUG_DECIDE"):
                log(f"  heavy plan over limit: {ctx.local_cost(loc)/1000:.1f} > {max_local_ms/1000:.1f}")
            continue
        rest = [n for n in cands if n not in set(loc)]
        cost = expected_local(ctx, a, blind_base + rest, loc)
        if os.environ.get("BR_DEBUG_DECIDE"):
            log(f"  heavy plan {[ctx.tests[n]['id'].rsplit('::',1)[1] for n in loc]}: {cost/1000:.2f} vs {best[0]/1000:.2f}; local {ctx.local_cost(loc)/1000:.2f} limit {max_local_ms/1000:.1f}")
        if cost < best[0]:
            best = (cost, rest, loc)
    return blind_base + list(best[1]), list(best[2])


# ------------------------------------------------------------ warm parent
# The parent imports the pristine sqlglot and every test module once, before
# forking. A child whose patch only edits function bodies installs the new
# bytecode into the live function objects instead of importing everything
# again (a cold import and first use cost tenths of a second per test file).
# Any other patch drops the warm modules and imports the patched tree afresh.

WARM = os.environ.get("BR_WARM", "1") == "1"
WARM_LIVE = None  # (file, qualname, firstlineno) -> live function objects


def warm_parent(tests):
    global WARM_LIVE
    import gc
    import importlib
    import types as _types

    sys.path.insert(0, SRC)
    import brfp
    import brpool

    brpool.install()
    brfp.import_all()
    for mod in sorted({t["file"][:-3].replace("/", ".") for t in tests}):
        try:
            importlib.import_module(mod)
        except BaseException:  # noqa: BLE001
            pass
    prefix = os.path.join(SRC, "sqlglot") + os.sep
    live = {}
    for o in gc.get_objects():
        if type(o) is _types.FunctionType:
            c = o.__code__
            f = c.co_filename
            if f.startswith(prefix):
                live.setdefault((os.path.relpath(f, SRC), c.co_qualname, c.co_firstlineno), []).append(o)
    WARM_LIVE = live
    sys.path.remove(SRC)


def purge_warm():
    """Forget the warm pristine modules: the patched tree is imported afresh."""
    for k in list(sys.modules):
        if k == "sqlglot" or k.startswith("sqlglot.") or k == "tests" or k.startswith("tests."):
            del sys.modules[k]


def _code_paths(co, path=()):
    """(qualname, occurrence) paths of every code object under co."""
    out = {}
    counts = {}
    for c in co.co_consts:
        if isinstance(c, types.CodeType):
            k = counts.get(c.co_qualname, 0)
            counts[c.co_qualname] = k + 1
            p = path + ((c.co_qualname, k),)
            out[p] = c
            out.update(_code_paths(c, p))
    return out


def swap_in(ctx, a):
    """Install patched function bodies into the warm live functions. False when
    the patch is not a pure function-body change this can express."""
    if not (a.all_pkg_py and a.n_fps and a.fn_clean == a.n_fps and not a.need_fp and not a.must):
        return False
    plan = []
    for rel, (old_co, new_co, changed) in a.warm_code_changes.items():
        old_paths = _code_paths(old_co)
        new_paths = _code_paths(new_co)
        by_obj = {id(c): p for p, c in old_paths.items()}
        targets = set()
        for co, kind in changed:
            if kind != "self" or not brlib.is_function_code(co):
                return False
            p = by_obj.get(id(co))
            if p is None:
                return False
            # the function and every enclosing function: an enclosing body holds
            # the nested code as a constant
            for i in range(1, len(p) + 1):
                targets.add(p[:i])
        for p in targets:
            oc, nc = old_paths.get(p), new_paths.get(p)
            if oc is None or nc is None:
                return False
            if not brlib.is_function_code(oc):
                continue  # a class body: holds no code of its own at run time
            key = (rel, oc.co_qualname, oc.co_firstlineno)
            if key in ctx.import_codes:
                return False  # ran at import: its results may be cached
            if oc.co_freevars != nc.co_freevars:
                return False
            plan.append((key, nc))
    if not plan:
        return False
    # every changed function must have a live object somewhere up its chain
    n = 0
    for key, nc in plan:
        for fn in WARM_LIVE.get(key, ()):
            fn.__code__ = nc
            n += 1
    if n == 0:
        return False
    a.note(f"warm: new bytecode in {n} live functions")
    return True


def process_commit(ctx, cid, patch_text, pristine, out_fd, deadline):
    def send(obj):
        data = (json.dumps(obj) + "\n").encode()
        os.write(out_fd, data)

    a = analyse_static(ctx, patch_text)
    root = os.path.join(WORK, cid)
    try:
        build_overlay(root, a.files)
    except Exception as e:  # noqa: BLE001
        a.note(f"overlay failed {e!r}")
        a.must |= ctx.all_bits
    os.chdir(root)
    swapped = False
    if WARM_LIVE is not None:
        try:
            swapped = swap_in(ctx, a)
        except Exception as e:  # noqa: BLE001
            a.note(f"swap failed {e!r}")
            swapped = False
        if not swapped:
            purge_warm()
    sys.path.insert(0, root)
    sys.dont_write_bytecode = True
    for n, sqls in a.parse_checks:
        if not a.py_in_patch and all(_parses(q) for q in sqls):
            a.cands &= ~(1 << n)
            a.data_cands &= ~(1 << n)
            a.note(f"{ctx.tests[n]['id'].rsplit('::', 1)[1]}: edited queries are not executed and still parse: dropped")
    if a.need_fp:
        base = ctx.maps.get("base_import")
        if base is None or a.fp_modules is None or (a.fp_modules & base):
            for n in ctx.lazy_load:
                a.cands |= 1 << n
        try:
            fp_diff(ctx, a, pristine)
        except Exception as e:  # noqa: BLE001
            a.note(f"fp diff failed: {e!r}")
            a.cands |= ctx.all_bits
    if os.environ.get("BR_DUMP"):
        with open(os.path.join(os.environ["BR_DUMP"], cid + ".json"), "w") as f:
            json.dump({"cands": bits_list(a.cands & ((1 << ctx.n) - 1)), "fn": bits_list(a.cands_fn & ((1 << ctx.n) - 1)),
                       "must": bits_list(a.must & ((1 << ctx.n) - 1)), "data": bits_list(a.data_cands & ((1 << ctx.n) - 1)),
                       "need_fp": bool(a.need_fp), "files": list(a.files), "notes": a.notes}, f)
    a.replay_mode = resolve_replay(ctx, a)
    # local work must fit well inside this commit's share of the time limit
    blind, local = decide(ctx, a, max_local_ms=max(0.0, (deadline - time.monotonic()) * 1000 * 0.6))
    if local:
        a.note(f"local estimate {ctx.local_cost(local)/1000:.2f}s cpu; tests: " + ", ".join(
            ctx.tests[n]["id"].split("::", 1)[1] for n in sorted(local, key=lambda n: -ctx.local_ms[n])[:4]))
    send({"stage": "plan", "blind": blind, "local": local, "notes": a.notes,
          "skip": {str(k): v for k, v in a.skip_reason.items()},
          "est": {str(k): ctx.local_ms[k] for k in a.replay_mode}})
    failed = set()
    if local:
        REPLAY_SKIP.update(a.replay_skip)
        failed = run_local(ctx, set(local), deadline, blind,
                           {t: r for t, r in a.replay_mode.items() if t in set(local)})
    send({"stage": "done", "failed": sorted(failed), "why": list(globals().get("LAST_WHY") or [])})


def child_main(ctx, cid, patch_text, pristine, w, deadline):
    # keep stdout clean: tests print
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    sys.stdout = sys.stderr
    try:
        process_commit(ctx, cid, patch_text, pristine, w, deadline)
    except BaseException as e:  # noqa: BLE001
        try:
            os.write(w, (json.dumps({"stage": "error", "error": repr(e)}) + "\n").encode())
        except Exception:
            pass
    os._exit(0)


def main():
    t_start = time.monotonic()
    tests = json.load(open(os.path.join(DATA, "tests.json")))["tests"]
    with open(os.path.join(MAPS, "maps.pkl"), "rb") as f:
        maps = pickle.load(f)
    with open(os.path.join(MAPS, "fp.pkl"), "rb") as f:
        pristine = pickle.load(f)
    replay_data = None
    rp = os.path.join(MAPS, "replay.pkl")
    if os.path.exists(rp):
        with open(rp, "rb") as f:
            replay_data = pickle.load(f)
    if replay_data is not None and os.environ.get("BR_FIXED_REPLAY_CPU", "1") == "1":
        # build-time timings of the replay vary by a third from build to build,
        # which would make the plan depend on the build: use typical values
        fixed = {"tests/test_executor.py::TestExecutor::test_execute_tpcds": 3.4,
                 "tests/test_executor.py::TestExecutor::test_execute_tpch": 0.03,
                 "tests/test_executor.py::TestExecutor::test_optimized_tpch": 0.08}
        cpu = replay_data.get("cpu") or {}
        replay_data["cpu"] = {k: fixed.get(k, v) for k, v in cpu.items()}
    maps["_replay_data"] = replay_data
    if maps["test_ids"] != [t["id"] for t in tests]:
        log("WARNING: tests.json differs from the map; remapping by id")
        maps = remap(maps, tests)
    speed = calibrate()
    ctx = Ctx(maps, tests, speed)
    log(f"speed factor {speed:.2f}")
    preimport(maps)
    if WARM:
        try:
            warm_parent(tests)
            log(f"warm parent: {len(WARM_LIVE)} live sqlglot functions")
        except Exception as e:  # noqa: BLE001
            log(f"warm parent failed: {e!r}")
            globals()["WARM_LIVE"] = None
            purge_warm()
    if os.environ.get("BR_GC_FREEZE", "1") == "1":
        import gc

        # the maps are millions of objects: keep the children's collector off them
        gc.collect()
        gc.freeze()
    cdir = os.path.join(DATA, "commits")
    commits = sorted(f[:-6] for f in os.listdir(cdir) if f.endswith(".patch"))
    n = len(commits)
    global_deadline = t_start + PER_COMMIT_WALL * n / max(1, PARALLEL) * PARALLEL / 2 + 5
    global_deadline = t_start + max(20.0, min(PER_COMMIT_WALL * n, float(os.environ.get("BR_WALL_CAP", "150"))))
    os.makedirs(WORK, exist_ok=True)

    results = {}
    pending = list(commits)
    running = {}  # pid -> state
    answers = None
    if EVAL:
        answers = json.load(open(EVAL))["commits"]

    def finish(st, killed=False):
        STATES[st["cid"]] = dict(st, killed=killed)
        plan = st.get("plan")
        if plan is None:
            sel = list(range(ctx.n))
            why = "no plan"
        elif "done" in st:
            sel = sorted(set(plan["blind"]) | set(st["done"]["failed"]))
            why = "ok"
        else:
            sel = sorted(set(plan["blind"]) | set(plan["local"]))
            why = "local run incomplete"
        cost, workers = pack(sel, ctx.ms)
        results[st["cid"]] = workers
        msg = f"{st['cid']}: cpu={st.get('cpu', -1):.2f} {why} blind={len(plan['blind']) if plan else '-'} local={len(plan['local']) if plan else '-'} sel={len(sel)} cost={cost/1000:.2f}s"
        if answers is not None:
            fail = set(answers[st["cid"]]["failing"])
            miss = fail - set(sel)
            msg += f" failing={len(fail)} MISS={len(miss)}" if miss else f" failing={len(fail)}"
            if plan:
                for nt in plan["notes"][:int(os.environ.get("BR_NOTES", "12"))]:
                    msg += "\n     " + nt
            if miss:
                msg += "\n     missed: " + ", ".join(tests[m]["id"] for m in sorted(miss)[:5])
        log(msg)

    while pending or running:
        while pending and len(running) < PARALLEL:
            cid = pending.pop(0)
            with open(os.path.join(cdir, cid + ".patch"), encoding="utf-8", errors="surrogateescape") as f:
                text = f.read()
            remaining = global_deadline - time.monotonic()
            # most commits take well under a second: keep a reserve for the
            # ones still waiting and let this one use the rest
            reserve = RESERVE_PER_COMMIT * ctx.speed * len(pending) / PARALLEL
            per = max(1.0, min(remaining - reserve, remaining))
            deadline = time.monotonic() + per
            r, w = os.pipe()
            pid = os.fork()
            if pid == 0:
                os.close(r)
                child_main(ctx, cid, text, pristine, w, deadline - 0.3)
            os.close(w)
            running[pid] = {"cid": cid, "fd": r, "buf": b"", "deadline": deadline + 0.5, "t0": time.monotonic()}
        fds = {st["fd"]: pid for pid, st in running.items()}
        timeout = max(0.0, min(st["deadline"] for st in running.values()) - time.monotonic())
        rl, _, _ = select.select(list(fds), [], [], timeout)
        for fd in rl:
            pid = fds[fd]
            st = running[pid]
            chunk = os.read(fd, 1 << 20)
            if chunk:
                st["buf"] += chunk
                while b"\n" in st["buf"]:
                    line, st["buf"] = st["buf"].split(b"\n", 1)
                    msg = json.loads(line)
                    if msg["stage"] == "plan":
                        st["plan"] = msg
                    elif msg["stage"] == "done":
                        st["done"] = msg
                    elif msg["stage"] == "error":
                        log(f"{st['cid']}: child error {msg['error']}")
                continue
            # EOF
            os.close(fd)
            _, _, ru = os.wait4(pid, 0)
            st["cpu"] = ru.ru_utime + ru.ru_stime
            del running[pid]
            finish(st)
            cleanup(st["cid"])
        now = time.monotonic()
        for pid, st in list(running.items()):
            if now > st["deadline"]:
                log(f"{st['cid']}: over time, killing")
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                os.waitpid(pid, 0)
                os.close(st["fd"])
                del running[pid]
                finish(st, killed=True)
                cleanup(st["cid"])

    tag = os.environ.get("BR_TAG", "0") == "1"
    speed_tagged = not tag
    for cid in commits:
        ws = results.get(cid, [])
        if tag:
            try:
                ws = tag_tpcds(ctx, tests, cid, ws, STATES.get(cid) or {})
            except Exception as e:  # noqa: BLE001
                log(f"tag failed {e!r}")
        print(json.dumps({"commit": cid, "workers": ws}))
    sys.stdout.flush()
    log(f"total wall {time.monotonic() - t_start:.1f}s")


def preimport(maps):
    """Import what test files import (but not sqlglot) once, before forking,
    so children share it instead of paying for it every commit."""
    import brfp

    brfp.preimport(maps.get("thirdparty", ()))
    bad = [m for m in sys.modules if m == "sqlglot" or m.startswith("sqlglot.")]
    for m in bad:  # never let the pristine package leak into children
        del sys.modules[m]


def cleanup(cid):
    import shutil

    shutil.rmtree(os.path.join(WORK, cid), ignore_errors=True)


def remap(maps, tests):
    old_ids = maps["test_ids"]
    new_num = {t["id"]: t["number"] for t in tests}
    trans = {i: new_num[tid] for i, tid in enumerate(old_ids) if tid in new_num}
    unknown_new = [t["number"] for t in tests if t["id"] not in set(old_ids)]
    unk_bits = 0
    for u in unknown_new:
        unk_bits |= 1 << u

    def conv(b):
        out = 0
        for i in bits_list(b):
            if i in trans:
                out |= 1 << trans[i]
        # tests we know nothing about: always candidates
        return out | unk_bits if b else out

    for key in ("code_tests", "read_tests", "touch", "name_tests", "open_tests", "list_tests",
                "file_tests", "module_import_tests"):
        maps[key] = {k: conv(v) for k, v in maps[key].items()}
    file_tests = {}
    for t in tests:
        file_tests[t["file"]] = file_tests.get(t["file"], 0) | (1 << t["number"])
    maps["file_tests"] = file_tests
    maps["test_ids"] = [t["id"] for t in tests]
    return maps


if __name__ == "__main__":
    main()
