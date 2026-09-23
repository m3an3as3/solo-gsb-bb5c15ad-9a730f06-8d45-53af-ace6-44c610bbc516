"""固定窗口 GC 含量范围接口测试。"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "test.db"))
    with TestClient(app) as c:
        yield c


def _post(client, **payload):
    return client.post("/gc-content", json=payload)


# ---------- 确定碱基：上下界恒等 ----------


def test_definite_bases_min_equals_max(client):
    resp = _post(client, sequence="GCGCAT", window_length=3, step=3)
    assert resp.status_code == 200
    data = resp.json()
    assert data["topology"] == "linear"
    assert data["sequence_length"] == 6
    assert data["window_length"] == 3
    assert data["step"] == 3
    assert data["window_count"] == 2
    assert data["ignored_trailing_bases"] == 0
    assert data["windows"] == [
        {"index": 0, "start": 0, "window_length": 3,
         "gc_min_count": 3, "gc_max_count": 3,
         "gc_min_percent": 100.0, "gc_max_percent": 100.0,
         "wraps_origin": False},
        {"index": 1, "start": 3, "window_length": 3,
         "gc_min_count": 1, "gc_max_count": 1,
         "gc_min_percent": pytest.approx(100.0 / 3),
         "gc_max_percent": pytest.approx(100.0 / 3),
         "wraps_origin": False},
    ]


def test_all_at_gives_zero_percent(client):
    resp = _post(client, sequence="ATATAT", window_length=6)
    assert resp.status_code == 200
    window = resp.json()["windows"][0]
    assert window["gc_min_count"] == 0
    assert window["gc_max_count"] == 0
    assert window["gc_min_percent"] == 0.0
    assert window["gc_max_percent"] == 0.0


# ---------- 简并符号：分别计入上下界 ----------


def test_degenerate_bases_split_between_bounds(client):
    # G=确定GC, S=GC(必为GC), R=AG(仅上界), W=AT(恒非GC), N=任意(仅上界), A=确定非GC
    resp = _post(client, sequence="GSRWNA", window_length=6)
    assert resp.status_code == 200
    window = resp.json()["windows"][0]
    # 最小：G、S 确定为 GC => 2；最大：再加上 R、N 可能为 GC => 4
    assert window["gc_min_count"] == 2
    assert window["gc_max_count"] == 4
    assert window["gc_min_percent"] == pytest.approx(2 / 6 * 100)
    assert window["gc_max_percent"] == pytest.approx(4 / 6 * 100)


def test_three_base_degenerate_symbols(client):
    # B=CGT、V=ACG、D=AGT（含 G）、H=ACT（含 C）均可能为 G/C；S=GC 必为
    resp = _post(client, sequence="BVDHS", window_length=5)
    window = resp.json()["windows"][0]
    assert window["gc_min_count"] == 1   # 仅 S 必为 GC
    assert window["gc_max_count"] == 5   # B/V/D/H 均含 G 或 C，S 必为


def test_purine_pyrimidine_and_keto_amino(client):
    # Y=CT（含 C）、K=GT（含 G）、M=AC（含 C）仅上界；R=AG 同理
    resp = _post(client, sequence="YKMW", window_length=4)
    window = resp.json()["windows"][0]
    assert window["gc_min_count"] == 0   # 没有必为 GC 的符号（W 恒为 AT）
    assert window["gc_max_count"] == 3   # Y/K/M 各可能贡献 1


def test_s_is_counted_in_both_bounds(client):
    resp = _post(client, sequence="SSSS", window_length=4)
    window = resp.json()["windows"][0]
    assert window["gc_min_count"] == 4
    assert window["gc_max_count"] == 4


# ---------- 滑动窗口与尾随碱基 ----------


def test_sliding_step_and_trailing_bases_ignored(client):
    # L=10, W=4, S=3：起点 0,3,6；末窗 [6,10)，尾随 0 个
    resp = _post(client, sequence="GCGCATATGC", window_length=4, step=3)
    data = resp.json()
    assert [w["start"] for w in data["windows"]] == [0, 3, 6]
    assert [w["index"] for w in data["windows"]] == [0, 1, 2]
    assert data["ignored_trailing_bases"] == 0

    # L=10, W=4, S=4：起点 0,4；尾随 2 个碱基不补长、不开短窗
    resp = _post(client, sequence="GCGCATATGC", window_length=4, step=4)
    data = resp.json()
    assert [w["start"] for w in data["windows"]] == [0, 4]
    assert data["ignored_trailing_bases"] == 2
    assert all(w["window_length"] == 4 for w in data["windows"])


def test_step_defaults_to_window_length(client):
    resp = _post(client, sequence="GCGCATATGC", window_length=4)
    assert resp.status_code == 200
    data = resp.json()
    assert data["step"] == 4
    assert [w["start"] for w in data["windows"]] == [0, 4]


def test_step_smaller_than_window_overlaps(client):
    # L=8, W=4, S=2：起点 0,2,4，全部为完整窗
    resp = _post(client, sequence="GCGCATAT", window_length=4, step=2)
    data = resp.json()
    assert [w["start"] for w in data["windows"]] == [0, 2, 4]
    assert data["ignored_trailing_bases"] == 0


def test_window_equal_sequence_length_single_window(client):
    resp = _post(client, sequence="GCGCAT", window_length=6, step=2)
    data = resp.json()
    assert data["window_count"] == 1
    assert data["windows"][0]["start"] == 0
    assert data["ignored_trailing_bases"] == 0


def test_window_length_one_every_position(client):
    resp = _post(client, sequence="GARC", window_length=1, step=1)
    windows = resp.json()["windows"]
    assert [w["start"] for w in windows] == [0, 1, 2, 3]
    assert [(w["gc_min_count"], w["gc_max_count"]) for w in windows] == [
        (1, 1),  # G
        (0, 0),  # A
        (0, 1),  # R=A/G
        (1, 1),  # C
    ]


# ---------- 环形序列：跨原点、一圈封顶 ----------


def test_circular_windows_wrap_around_origin(client):
    # L=6, W=4, S=4：起点 0([0,4)) 与 4(位置 4,5,0,1)
    resp = _post(
        client, sequence="GCGCAT", topology="circular",
        window_length=4, step=4,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["ignored_trailing_bases"] == 0
    assert data["window_count"] == 2
    assert data["windows"][0]["wraps_origin"] is False
    wrapped = data["windows"][1]
    assert wrapped["start"] == 4
    assert wrapped["wraps_origin"] is True
    # 位置 4,5,0,1 = A,T,G,C => 2 个确定 GC
    assert wrapped["gc_min_count"] == 2
    assert wrapped["gc_max_count"] == 2


def test_circular_degenerate_wrap_window(client):
    # L=6, W=4, 起点 4 覆盖位置 4,5,0,1 = A,T,G,R => 最小 1（G），最大 2（R 可为 G）
    resp = _post(
        client, sequence="GRGCAT", topology="circular",
        window_length=4, step=4,
    )
    wrapped = resp.json()["windows"][1]
    assert wrapped["wraps_origin"] is True
    assert (wrapped["gc_min_count"], wrapped["gc_max_count"]) == (1, 2)


def test_circular_starts_cover_whole_turn_with_ceil_division(client):
    # L=10, W=3, S=6：ceil(10/6)=2 个起点 0,6，覆盖完整一圈
    resp = _post(
        client, sequence="A" * 10, topology="circular",
        window_length=3, step=6,
    )
    data = resp.json()
    assert [w["start"] for w in data["windows"]] == [0, 6]
    assert data["ignored_trailing_bases"] == 0


def test_circular_full_length_window_is_single(client):
    # W == L：每个起点覆盖同一碱基多重集，仅返回起点 0 的全长窗，不绕多圈
    resp = _post(
        client, sequence="GCGC", topology="circular",
        window_length=4, step=2,
    )
    data = resp.json()
    assert data["window_count"] == 1
    window = data["windows"][0]
    assert window["start"] == 0
    assert window["wraps_origin"] is False
    assert window["gc_min_count"] == 4


def test_circular_window_never_covers_more_than_one_turn(client):
    # W < L 时所有窗口长度恒为 W，没有任何碱基被重复计入
    resp = _post(
        client, sequence="G" * 7, topology="circular",
        window_length=5, step=3,
    )
    for window in resp.json()["windows"]:
        assert window["window_length"] == 5
        assert window["gc_min_count"] == 5
        assert window["gc_max_count"] == 5


# ---------- 小写与非标准空白 ----------


def test_lowercase_sequence_accepted_and_normalized(client):
    resp = _post(client, sequence="gcRn", window_length=4)
    assert resp.status_code == 200
    window = resp.json()["windows"][0]
    # G、C 确定为 GC；R 可能为 G；N 可能为任意碱基 => 最小 2，最大 4
    assert (window["gc_min_count"], window["gc_max_count"]) == (2, 4)


# ---------- 校验与状态码 ----------


def test_window_length_exceeds_sequence_returns_422(client):
    resp = _post(client, sequence="GCGC", window_length=5)
    assert resp.status_code == 422
    assert "window_length 5 exceeds sequence length 4" in resp.text


def test_non_positive_window_length_returns_422(client):
    assert _post(client, sequence="GCGC", window_length=0).status_code == 422
    assert _post(client, sequence="GCGC", window_length=-1).status_code == 422


def test_non_positive_step_returns_422(client):
    assert _post(
        client, sequence="GCGC", window_length=2, step=0
    ).status_code == 422
    assert _post(
        client, sequence="GCGC", window_length=2, step=-3
    ).status_code == 422


def test_non_iupac_sequence_returns_422(client):
    resp = _post(client, sequence="GCXG", window_length=2)
    assert resp.status_code == 422
    assert "non-IUPAC" in resp.text


def test_empty_sequence_returns_422(client):
    resp = _post(client, sequence="   ", window_length=1)
    assert resp.status_code == 422


def test_invalid_topology_returns_422(client):
    resp = _post(
        client, sequence="GCGC", topology="plasmid", window_length=2
    )
    assert resp.status_code == 422


# ---------- 只读：不依赖、不改动数据库 ----------


def test_endpoint_is_read_only_and_independent_of_saved_records(client):
    # 无需先创建任何提交即可调用
    resp = _post(client, sequence="GCSR", window_length=4)
    assert resp.status_code == 200
    # 没有任何提交被写入
    assert client.get("/digests").json() == []


def test_endpoint_works_alongside_saved_records(client):
    created = client.post(
        "/digests",
        json={
            "sequence": "AAAAGAATTCTTTT",
            "topology": "linear",
            "enzymes": [
                {"name": "EcoRI", "pattern": "GAATTC",
                 "cut_top": 1, "cut_bottom": 5}
            ],
        },
    ).json()
    resp = _post(client, sequence="GCSRNN", window_length=3, step=3)
    assert resp.status_code == 200
    # 已保存记录不受影响
    assert client.get(f"/digests/{created['id']}").json() == created
    assert len(client.get("/digests").json()) == 1
