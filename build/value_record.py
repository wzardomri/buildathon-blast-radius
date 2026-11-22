"""Record, per test, what every comparison in sqlglot evaluated to.

Build-time only. Every single-operator comparison inside a sqlglot function
(``a < b``, ``x is None``, ``k in d`` ...) is rewritten in a private copy of
the pinned source into a call that evaluates it exactly as Python would and
remembers, for the running test, how that comparison would have come out
under each comparison operator, plus the operand values themselves when they
are plain literals. Each test runs alone in a freshly forked process, so a
test's record holds every comparison it needs, caches included.

At selection time a patch that only changes one comparison (its operator, or
one constant operand) cannot affect a test whose recorded comparisons all
come out the same before and after the edit.

Operands whose comparison could run user code (a custom __eq__, __lt__,
__contains__ ...) are not evaluated under other operators: the site is marked
unmeasured for that test and the test stays selected.

usage: value_record.py SQLGLOT_ROOT TESTS_JSON OUT_DIR [JOBS]
"""

import ast
import enum
import hashlib
import json
import os
import pickle
import select
import shutil
import signal
import sys
import tempfile
import time

OPS = ("Lt", "LtE", "Gt", "GtE", "Eq", "NotEq", "Is", "IsNot", "In", "NotIn")
CAP = 16  # distinct records per (test, site); more marks the site unmeasured
PLAIN = (type(None), bool, int, float, str, bytes)


# ------------------------------------------------------------- rewriting


class _Rewrite(ast.NodeTransformer):
    def __init__(self, path, sites):
        self.path = path
        self.sites = sites
        self.depth = 0

    def _func(self, node):
        self.depth += 1
        self.generic_visit(node)
        self.depth -= 1
        return node

    visit_FunctionDef = visit_AsyncFunctionDef = visit_Lambda = _func

    def visit_Compare(self, node):
        self.generic_visit(node)
        if self.depth == 0 or len(node.ops) != 1:
            return node
        op = type(node.ops[0]).__name__
        if op not in OPS:
            return node
        sid = len(self.sites)
        self.sites.append((self.path, node.lineno, node.col_offset, node.end_lineno, node.end_col_offset, op))
        call = ast.Call(
            func=ast.Name(id="__brv__", ctx=ast.Load()),
            args=[ast.Constant(sid), ast.Constant(OPS.index(op)), node.left, node.comparators[0]],
            keywords=[],
        )
        return ast.copy_location(call, node)


def rewrite_tree(root, dst):
    """Copy root/sqlglot to dst/sqlglot with comparisons instrumented."""
    sites = []
    src_pkg = os.path.join(root, "sqlglot")
    for dp, dn, fn in os.walk(src_pkg):
        dn[:] = [d for d in dn if d != "__pycache__"]
        rel = os.path.relpath(dp, root)
        os.makedirs(os.path.join(dst, rel), exist_ok=True)
        for f in fn:
            sp = os.path.join(dp, f)
            dp_ = os.path.join(dst, rel, f)
            if not f.endswith(".py"):
                shutil.copy2(sp, dp_)
                continue
            text = open(sp, encoding="utf-8").read()
            relf = os.path.join(rel, f)
            try:
                tree = ast.parse(text)
                tree = _Rewrite(relf, sites).visit(tree)
                ast.fix_missing_locations(tree)
                code = compile(tree, sp, "exec")
            except Exception:  # noqa: BLE001
                shutil.copy2(sp, dp_)
                continue
            # keep the source file (tracebacks, inspect) and ship the code as a .pyc
            shutil.copy2(sp, dp_)
            import importlib.util
            import marshal

            cache = importlib.util.cache_from_source(dp_)
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            st = os.stat(dp_)
            data = bytearray(importlib.util.MAGIC_NUMBER)
            data += (0).to_bytes(4, "little")
            data += (int(st.st_mtime) & 0xFFFFFFFF).to_bytes(4, "little")
            data += (st.st_size & 0xFFFFFFFF).to_bytes(4, "little")
            data += marshal.dumps(code)
            with open(cache, "wb") as fh:
                fh.write(data)
    return sites


# ------------------------------------------------------------- recording


SIG_CAP = 64  # distinct outcome signatures per (test, site); more marks it unmeasured
PAIR_CAP = 16  # literal operand pairs kept for constant edits; more drops them

_CMP_NAMES = ("__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__")
_SAFE_MODULES = {"builtins", "datetime", "_datetime", "decimal", "_decimal", "fractions", "uuid"}
_CONTAINERS = (tuple, list, set, frozenset, dict)
_KIND = {}


def _kind(t):
    """How == and the orderings behave for values of type t:
    'plain'     builtin scalars (str, int, float, None, bool, bytes)
    'builtin'   other stdlib values with C or side-effect-free comparisons
    'container' tuple, list, set, frozenset, dict: element comparisons
    'identity'  object's own comparisons (identity, no ordering)
    'expr'      sqlglot expression nodes: identity unless the same type
    'custom'    anything else: comparing would run user code
    """
    k = _KIND.get(t)
    if k is not None:
        return k
    if t in PLAIN:
        k = "plain"
    elif t in _CONTAINERS:
        k = "container"
    elif isinstance(t, type) and issubclass(t, enum.Enum):
        ok = all(getattr(t, n).__qualname__.split(".")[0] in ("object", "str", "int", "float", "Enum", "IntEnum",
                                                               "StrEnum", "Flag", "IntFlag", "ReprEnum")
                 for n in _CMP_NAMES)
        k = "builtin" if ok else "custom"
    elif t.__module__ in _SAFE_MODULES and type(t) is type:
        k = "builtin"
    elif all(getattr(t, n) is getattr(object, n) for n in _CMP_NAMES) and type(t) is type:
        k = "identity"
    elif (getattr(t.__eq__, "__qualname__", "") == "Expr.__eq__" and getattr(t.__ne__, "__qualname__", "") == "Expr.__ne__"
          and all(getattr(t, n) is getattr(object, n) for n in ("__lt__", "__le__", "__gt__", "__ge__"))
          and type(t) is type):
        k = "expr"
    else:
        k = "custom"
    _KIND[t] = k
    return k


def _deep_ok(v, depth=0):
    """Comparing v element by element runs no user code."""
    k = _kind(type(v))
    if k in ("plain", "builtin", "identity"):
        return True
    if k == "container" and depth < 3:
        items = v.keys() if type(v) is dict else v
        return len(v) <= 64 and all(_deep_ok(x, depth + 1) for x in items)
    return False


def _eq_safe(a, b):
    ka, kb = _kind(type(a)), _kind(type(b))
    if "custom" in (ka, kb):
        return False
    if ka == "expr" and kb == "expr":
        return type(a) is not type(b)  # same type would hash both trees
    if ka == "container" and kb == "container":
        return _deep_ok(a) and _deep_ok(b)
    return True  # mismatched kinds: the comparison returns without user code


def _order_safe(a, b):
    ka, kb = _kind(type(a)), _kind(type(b))
    if "custom" in (ka, kb):
        return False
    if ka == "container" or kb == "container":
        return _deep_ok(a) and _deep_ok(b)
    return True


def _contain_safe(a, b):
    tb = type(b)
    if tb in (str, bytes):
        return type(a) in (str, bytes) or _kind(type(a)) != "custom"
    if tb in _CONTAINERS:
        if len(b) > 64:
            return False
        if tb in (set, frozenset, dict):
            # hashing a: builtin hashes only
            if _kind(type(a)) not in ("plain", "builtin", "identity") and not (type(a) is tuple and _deep_ok(a)):
                return False
        return _deep_ok(b) and _kind(type(a)) != "custom" and (_kind(type(a)) != "container" or _deep_ok(a))
    return False


def _outcomes(a, b):
    out = []
    eq = _eq_safe(a, b)
    order = _order_safe(a, b)
    contain = _contain_safe(a, b)
    for k, f in enumerate((lambda: a < b, lambda: a <= b, lambda: a > b, lambda: a >= b, lambda: a == b,
                           lambda: a != b, lambda: a is b, lambda: a is not b, lambda: a in b, lambda: a not in b)):
        if (k < 4 and not order) or (k in (4, 5) and not eq) or (k >= 8 and not contain):
            out.append("?")  # would run user code: unknown
            continue
        try:
            r = f()
            out.append(bool(r) if type(r) is bool else "X")
        except Exception as e:  # noqa: BLE001
            out.append("E:" + type(e).__name__)
    return tuple(out)


_NOKEY = object()
_IDM = ("<obj>",)
_HASHABLE_KINDS = ("plain", "builtin")


def _part(v, k):
    if k in ("identity", "expr"):
        return _IDM
    if k in _HASHABLE_KINDS:
        try:
            hash(v)
        except TypeError:
            return _NOKEY
        return v
    t = type(v)
    if t is tuple or t is frozenset:
        if len(v) <= 32 and all((_KIND.get(type(x)) or _kind(type(x))) in _HASHABLE_KINDS for x in v):
            try:
                hash(v)
            except TypeError:
                return _NOKEY
            return v
    return _NOKEY


def _plain(v, depth=0):
    t = type(v)
    if t in PLAIN:
        return not (t is float and v != v)
    return t is tuple and depth < 3 and len(v) <= 16 and all(_plain(x, depth + 1) for x in v)


def run_record(root, tests, out_dir, jobs):
    tmp = tempfile.mkdtemp(prefix="brv_")
    sites = rewrite_tree(root, tmp)
    os.symlink(os.path.join(root, "tests"), os.path.join(tmp, "tests"))
    for extra in ("setup.cfg", "pyproject.toml", "conftest.py"):
        if os.path.exists(os.path.join(root, extra)):
            os.symlink(os.path.join(root, extra), os.path.join(tmp, extra))
    print(f"rewrote {len(sites)} comparison sites", file=sys.stderr)
    sys.path.insert(0, tmp)
    os.chdir(tmp)
    os.environ.setdefault("SKIP_INTEGRATION", "0")

    import builtins
    import operator
    import unittest

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import brpool

    brpool.install()

    fns = (operator.lt, operator.le, operator.gt, operator.ge, operator.eq, operator.ne,
           operator.is_, operator.is_not, lambda a, b: a in b, lambda a, b: a not in b)
    rec = {}  # site -> set of records, or "U"

    U = "U"
    sig_memo = {}

    def brv(sid, op, a, b):
        r = fns[op](a, b)
        cur = rec.get(sid)
        if cur is U:
            return r
        ta = type(a)
        tb = type(b)
        ka = _KIND.get(ta) or _kind(ta)
        kb = _KIND.get(tb) or _kind(tb)
        # a key that fixes the outcomes, when one is cheap to form: literal
        # values (and tuples of them) by value, other objects by identity
        pa = _part(a, ka)
        pb = _part(b, kb) if pa is not _NOKEY else _NOKEY
        if pb is not _NOKEY:
            key = (ta, tb, pa, pb, a is b)
        elif pa is not _NOKEY and ka != "container" and kb == "container" and len(b) > 64:
            # a scalar against a big container: containment is left unknown, and
            # nothing else the comparison does looks inside b
            key = (ta, tb, pa, "<big>", a is b)
        else:
            key = None
        sig = None
        if key is not None:
            try:
                sig = sig_memo.get(key)
            except TypeError:
                key = None
        if sig is None:
            sig = _outcomes(a, b)
            if key is not None:
                if len(sig_memo) > 200000:
                    sig_memo.clear()
                sig_memo[key] = sig
        if cur is None:
            cur = rec[sid] = [set(), set()]
        sigs, pairs = cur
        if sig not in sigs:
            if len(sigs) >= SIG_CAP:
                rec[sid] = U
                return r
            sigs.add(sig)
        if pairs is not None:
            if ka == "plain" and kb == "plain" and _plain(a) and _plain(b):
                pr = (a, b)
                if pr not in pairs:
                    if len(pairs) >= PAIR_CAP:
                        cur[1] = None
                    else:
                        pairs.add(pr)
            else:
                cur[1] = None  # a non-literal operand: constant edits cannot be judged
        return r

    builtins.__brv__ = brv
    import sqlglot  # noqa: F401  (instrumented package, imported once)

    # unittest names for the pytest ids
    loader = unittest.TestLoader()
    work = []
    for t in tests:
        tid = t["id"]
        file, _, rest = tid.partition("::")
        mod = file[:-3].replace("/", ".")
        work.append((t["number"], tid, mod, rest.replace("::", ".")))
    mods = sorted({w[2] for w in work})
    for m in mods:
        try:
            __import__(m)
        except Exception:  # noqa: BLE001
            pass
    rec.clear()  # import-time comparisons belong to no test

    results = {}
    running = {}
    pending = list(work)
    test_timeout = int(os.environ.get("BR_VALUE_TEST_TIMEOUT", "120"))
    t0 = time.time()
    while pending or running:
        while pending and len(running) < jobs:
            num, tid, mod, name = pending.pop(0)
            r, w = os.pipe()
            pid = os.fork()
            if pid == 0:
                os.close(r)
                ok = False
                try:
                    signal.alarm(test_timeout)
                    devnull = os.open(os.devnull, os.O_RDWR)
                    os.dup2(devnull, 0)  # a test reading stdin must not wait
                    os.dup2(devnull, 1)
                    os.dup2(devnull, 2)
                    rec.clear()
                    suite = loader.loadTestsFromName(name, sys.modules[mod])
                    res = unittest.TestResult()
                    suite.run(res)
                    ok = res.testsRun > 0 and res.wasSuccessful() and not res.skipped
                except BaseException:  # noqa: BLE001
                    ok = False
                payload = pickle.dumps({"ok": ok, "rec": {k: (v if v == "U" else (frozenset(v[0]), None if v[1] is None else frozenset(v[1])))
                                                         for k, v in rec.items()}})
                with os.fdopen(w, "wb") as fh:
                    fh.write(payload)
                os._exit(0)
            os.close(w)
            running[r] = (pid, tid, bytearray())
        # A worker may produce more than the pipe can hold. Drain available
        # bytes before waiting for it to exit, as it may be blocked in write.
        ready, _, _ = select.select(running, [], [])
        for fd in ready:
            pid, tid, data = running[fd]
            chunk = os.read(fd, 65536)
            if chunk:
                data.extend(chunk)
                continue
            os.close(fd)
            _, status = os.waitpid(pid, 0)
            del running[fd]
            results[tid] = None
            if status == 0:
                try:
                    d = pickle.loads(data)
                    if d["ok"]:
                        results[tid] = d["rec"]
                except Exception:  # noqa: BLE001
                    pass
            if len(results) % 256 == 0:
                print(f"processed {len(results)}/{len(work)} test traces", file=sys.stderr)
    print(f"recorded {sum(1 for v in results.values() if v is not None)}/{len(results)} tests "
          f"in {time.time() - t0:.0f}s", file=sys.stderr)

    # shard by source file: site key -> {test id: records or "U"}
    by_file = {}
    for tid, recs in results.items():
        if recs is None:
            continue
        for sid, v in recs.items():
            path, l0, c0, l1, c1, op = sites[sid]
            by_file.setdefault(path, {}).setdefault((l0, c0, l1, c1), {})[tid] = v
    os.makedirs(out_dir, exist_ok=True)
    for path, d in by_file.items():
        # one small file per comparison site: a run loads only the one it needs
        sub = os.path.join(out_dir, hashlib.sha1(path.encode()).hexdigest()[:16])
        os.makedirs(sub, exist_ok=True)
        for site, tv in d.items():
            with open(os.path.join(sub, "%d_%d_%d_%d.pkl" % site), "wb") as fh:
                pickle.dump(tv, fh, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(out_dir, "index.pkl"), "wb") as fh:
        pickle.dump({"recorded": sorted(t for t, v in results.items() if v is not None),
                     "sites": {(s[0], s[1], s[2], s[3], s[4]): s[5] for s in sites}}, fh)
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    root, tests_json, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    jobs = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    if tests_json.endswith(".pkl"):
        ids = pickle.load(open(tests_json, "rb"))["test_ids"]
        tests = [{"id": t, "number": i} for i, t in enumerate(ids)]
    else:
        tests = json.load(open(tests_json))["tests"]
    run_record(os.path.abspath(root), tests, out_dir, jobs)
