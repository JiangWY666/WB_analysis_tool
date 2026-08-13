"""一次实验横跨好几张膜：分组是配对单位，膜只是它的一张底片。

内参和目标蛋白拍在不同的膜上（膜 A = Actin、膜 B = CITH3、膜 C = PAD4），
泳道一一对应但几何完全不同。这批用例锁住由此产生的四件事：

1. 样品名与 fold change 的基准泳道**按分组存**——填一次覆盖 N 张膜，
   而"各膜泳道号对得上"正是跨膜归一化成立的前提；
2. 同一分组出现两条内参时**必须出警告**，否则前一块胶的目标会拿后一块胶的
   内参当分母，数字照常算出来、一个提示都没有；
3. 长表逐行记住"这个比值是拿哪两张膜相除得来的"；
4. 宽表可以直接丢进 Prism，而它自己不带溯源，所以旁注文件必须说清空格。
"""

import csv

import pytest

from wb_analyzer.export import (
    SUMMARY_LONG_FIELDS,
    SUMMARY_VALUE_NORMALIZED,
    build_summary_rows,
    build_wide_table,
    export_summary_tables,
)
from wb_analyzer.normalize import (
    ORIGIN_ROI_BASELINE,
    AnalysisSession,
    QuantEntry,
    RoiCheck,
    coerce_roi_check,
    membrane_from_source,
    source_key,
)
from wb_analyzer.roi_check import RoiIssue, RoiQualityReport


def entry(
    kind: str,
    name: str,
    group: str,
    values: dict[int, float],
    *,
    source: str = "",
    membrane: str = "",
    weak: tuple[int, ...] = (),
    origin: str = ORIGIN_ROI_BASELINE,
    roi_check: object = None,
) -> QuantEntry:
    lanes = sorted(values)
    return QuantEntry(
        kind=kind,  # type: ignore[arg-type]
        name=name,
        group=group,
        intensities=[float(values[n]) for n in lanes],
        lanes=lanes,
        below_detection=[n in weak for n in lanes],
        source=source,
        membrane=membrane,
        origin=origin,
        roi_check=roi_check,  # type: ignore[arg-type]
    )


def three_membrane_session() -> AnalysisSession:
    """用户的真实流程：开 actin 膜 → 加为内参 → 关掉 → 开 CITH3 膜 → ……"""
    session = AnalysisSession()
    session.upsert(
        entry(
            "loading_control",
            "Actin",
            "0716wb",
            {1: 100.0, 2: 100.0, 3: 100.0},
            source=r"D:\0716wb\actin1.tif",
        )
    )
    session.upsert(
        entry(
            "target",
            "CITH3",
            "0716wb",
            {1: 100.0, 2: 50.0, 3: 25.0},
            source=r"D:\0716wb\cith3.tif",
        )
    )
    session.upsert(
        entry(
            "target",
            "PAD4",
            "0716wb",
            {1: 40.0, 2: 80.0, 3: 120.0},
            source=r"D:\0716wb\pad4.tif",
        )
    )
    for lane, sample in {1: "WT", 2: "KO", 3: "处理1"}.items():
        session.set_lane_label("0716wb", lane, sample)
    return session


def read_csv(path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


# ------------------------------------------------------------------ ROI 自检摘要


def test_roi_check_snapshots_a_real_report() -> None:
    """自检结论是每膜每次定量一条，跟着记录走，不是每泳道一条。"""
    passed = RoiQualityReport(issues=(), lane_count=3, band_count=3)
    check = RoiCheck.from_report(passed)
    assert check.grade == "ok"
    assert check.summary.startswith("ROI 自检通过")

    flagged = RoiQualityReport(
        issues=(
            RoiIssue(
                kind="edge_clearance_top",
                severity="warning",
                title="ROI 上边切进条带",
                detail="把 ROI 上边往上挪 5 px",
            ),
        ),
        lane_count=3,
        band_count=3,
    )
    assert RoiCheck.from_report(flagged).grade == "warning"
    assert "ROI 上边切进条带" in RoiCheck.from_report(flagged).summary


def test_roi_check_survives_a_csv_cell() -> None:
    """一列一格：档位用 ASCII 前缀，中文句子里的全角冒号切不断它。"""
    check = RoiCheck.from_report(RoiQualityReport(issues=(), lane_count=2, band_count=2))
    assert RoiCheck.parse(check.as_field()) == check
    assert RoiCheck.parse("").recorded is False
    # 手改过的单元格：认不出档位就整句留着，不假装知道
    stray = RoiCheck.parse("我自己写的一句话")
    assert stray.grade == "" and stray.summary == "我自己写的一句话"


def test_coerce_roi_check_takes_whatever_the_caller_holds() -> None:
    report = RoiQualityReport(issues=(), lane_count=1, band_count=1)
    assert coerce_roi_check(report).grade == "ok"
    assert coerce_roi_check(None) == RoiCheck()
    assert coerce_roi_check(RoiCheck("warning", "x")) == RoiCheck("warning", "x")
    assert coerce_roi_check("ok: 通过了").grade == "ok"


# ---------------------------------------------------------------------- 膜名


def test_membrane_defaults_to_the_file_stem_and_can_be_renamed() -> None:
    session = AnalysisSession()
    session.upsert(entry("target", "PAD4", "G", {1: 1.0}, source=r"D:\a\pad4_001.tif"))
    assert session.entries[0].membrane == "pad4_001"
    assert session.entries[0].label == "PAD4（膜 pad4_001）"

    session.set_membrane(0, "膜C（第二次转膜）")
    assert session.entries[0].membrane == "膜C（第二次转膜）"
    assert session.entries[0].source == r"D:\a\pad4_001.tif", "改的是名字，不是来源"
    # 清空又退回文件名，而不是留下一个没有身份的空格
    assert session.set_membrane(0, "  ").membrane == "pad4_001"


def test_membrane_is_stored_not_re_derived() -> None:
    """同名不同目录时，现推文件名会把两张膜说成一张。"""
    assert membrane_from_source(r"D:\0716wb\actin.tif") == "actin"
    assert membrane_from_source(r"D:\0722wb\actin.tif") == "actin"
    session = AnalysisSession()
    session.upsert(
        entry("loading_control", "Actin", "G", {1: 1.0}, source=r"D:\0716wb\actin.tif")
    )
    session.set_membrane(0, "0716-actin")
    session.upsert(
        entry(
            "loading_control",
            "Actin",
            "G",
            {1: 2.0},
            source=r"D:\0722wb\actin.tif",
        ),
        key_on_source=True,
    )
    assert [e.membrane for e in session.entries] == ["0716-actin", "actin"]


# ---------------------------------------------------------- 泳道名（按分组存）


def test_lane_labels_are_per_group_not_per_membrane() -> None:
    """填一次覆盖整场实验的 N 张膜——这正是跨膜归一化成立的前提。"""
    session = three_membrane_session()
    assert session.lane_label("0716wb", 2) == "KO"
    assert session.lane_labels_of("0716wb") == {1: "WT", 2: "KO", 3: "处理1"}
    # 另一个分组是另一场实验，样品名不会串过去
    assert session.lane_label("0722wb", 2) == ""

    session.set_lane_label("0716wb", 2, "")
    assert session.lane_label("0716wb", 2) == ""
    session.set_lane_labels("0716wb", {1: "A", 2: "B"})
    assert session.lane_labels_of("0716wb") == {1: "A", 2: "B"}, "整批替换就是替换"


def test_lane_labels_do_not_survive_a_clear() -> None:
    """上一批实验的样品名跟进下一批，是要写进论文图里的错。"""
    session = three_membrane_session()
    session.set_control_lane("0716wb", 2)
    session.clear()
    assert session.lane_labels == {} and session.control_lanes == {}
    assert session.lane_label("0716wb", 1) == ""


def test_group_names_are_canonical_everywhere() -> None:
    session = AnalysisSession()
    session.upsert(entry("target", "T", "  ", {1: 1.0}))
    session.set_lane_label("", 1, "WT")
    assert session.entries[0].group == "默认"
    assert session.lane_label("默认", 1) == "WT"


# ------------------------------------------------------- 基准泳道（按分组存）


def test_control_lane_is_a_property_of_the_group() -> None:
    session = three_membrane_session()
    session.upsert(
        entry("loading_control", "Actin", "0722wb", {1: 200.0, 2: 200.0}, source="a2.tif")
    )
    session.upsert(
        entry("target", "PAD4", "0722wb", {1: 200.0, 2: 100.0}, source="p2.tif")
    )
    session.set_control_lane("0722wb", 2)

    rows = {(r.group, r.target_name, r.lane): r for r in session.normalize_all()}
    assert rows[("0716wb", "PAD4", 1)].fold_change == 1.0, "这一组仍用默认的泳道 1"
    assert rows[("0722wb", "PAD4", 1)].fold_change == 2.0, "那一组自己定了泳道 2"
    assert rows[("0722wb", "PAD4", 2)].fold_change == 1.0
    assert rows[("0722wb", "PAD4", 1)].control_lane == 2, "行里记的是实际用的那一道"


def test_global_control_lane_is_only_a_default_for_unset_groups() -> None:
    """旧的全局参数降级为默认值：设过的分组不受它影响。"""
    session = three_membrane_session()
    session.set_control_lane("0716wb", 3)
    rows = {(r.target_name, r.lane): r for r in session.normalize_all(control_lane=2)}
    assert rows[("CITH3", 3)].fold_change == 1.0, "分组自己设的泳道 3 说了算"
    assert all(r.control_lane == 3 for r in rows.values())

    session.control_lanes.clear()
    rows = {(r.target_name, r.lane): r for r in session.normalize_all(control_lane=2)}
    assert rows[("CITH3", 2)].fold_change == 1.0, "没设过的分组才轮到参数"


def test_control_lane_must_be_positive() -> None:
    session = AnalysisSession()
    with pytest.raises(ValueError, match="基准泳道"):
        session.set_control_lane("G", 0)


def unpairable_reference_session() -> AnalysisSession:
    """可配对泳道是 {2,3}，而基准泳道要的是 9：没有一道能当分母。"""
    session = AnalysisSession()
    session.upsert(entry("loading_control", "Actin", "G", {2: 100.0, 3: 100.0}))
    session.upsert(entry("target", "T", "G", {2: 50.0, 3: 100.0}))
    session.set_control_lane("G", 9)
    return session


def test_unpairable_reference_lane_leaves_fold_change_empty() -> None:
    """基准泳道不可配对时不猜。

    旧实现 `shared[np_clip_index(control - 1, len(shared))]` 把泳道号当位置用，
    要泳道 9 会拿到泳道 3——"以 WT 为 1"被悄悄换成"以处理组为 1"，整组比值
    看着完全正常。宁可不给也不给错。
    """
    session = unpairable_reference_session()
    rows = session.normalize_all()

    assert [r.fold_change for r in rows] == [None, None], "一个都不给"
    assert [r.normalized for r in rows] == [0.5, 1.0], "比值本身不受影响，照常导出"
    assert {r.control_lane for r in rows} == {9}, "行里记的是你要的那一道，不是替你换的"


def test_unpairable_reference_lane_says_which_lanes_are_available() -> None:
    session = unpairable_reference_session()
    session.normalize_all()
    warning = next((w for w in session.last_warnings if "基准泳道 9" in w), "")
    assert warning, f"必须进 last_warnings：{session.last_warnings}"
    assert "可配对的是泳道 2、3" in warning, "光说不行还得说能改成哪几道"
    assert "fold change 全部留空" in warning
    assert all("基准泳道 9 没有可配对的数据" in r.note for r in session.normalize_all())


def test_a_valid_reference_lane_brings_fold_change_back() -> None:
    session = unpairable_reference_session()
    session.set_control_lane("G", 2)
    rows = session.normalize_all()
    assert [r.fold_change for r in rows] == [1.0, 2.0]
    assert not [w for w in session.last_warnings if "基准泳道" in w]


def test_the_default_reference_lane_is_refused_just_the_same() -> None:
    """默认值也是"要求"：泳道 1 的条带被删掉后，不能顺势滑到泳道 2。"""
    session = AnalysisSession()
    session.upsert(entry("loading_control", "Actin", "G", {2: 100.0, 3: 100.0}))
    session.upsert(entry("target", "T", "G", {2: 50.0, 3: 100.0}))
    rows = session.normalize_all()  # 没设过，用默认的泳道 1
    assert [r.fold_change for r in rows] == [None, None]
    assert any("基准泳道 1" in w for w in session.last_warnings)


def test_a_zero_control_on_the_reference_lane_is_reported_group_wide() -> None:
    """基准泳道在，但它的比值算不出来：后果同样覆盖整组，得说出来。"""
    session = AnalysisSession()
    session.upsert(entry("loading_control", "Actin", "G", {1: 0.0, 2: 100.0}))
    session.upsert(entry("target", "T", "G", {1: 50.0, 2: 50.0}))
    rows = session.normalize_all()
    assert [r.fold_change for r in rows] == [None, None]
    assert rows[0].note == "内参强度为 0", "这一道自己的毛病更具体，注释留给它"
    assert rows[1].note == "基准泳道 1 没有比值，fold change 无法计算"
    assert any("这一组的 fold change 全部留空" in w for w in session.last_warnings)


# --------------------------------------------- 跨膜身份：upsert 的键与安全阀


def test_same_membrane_re_quantified_updates_in_place() -> None:
    """同一张膜重测后再加入：静默更新，这正是用户要的。"""
    session = AnalysisSession()
    session.upsert(entry("loading_control", "Actin", "G", {1: 100.0}, source="a.tif"))
    session.upsert(
        entry("loading_control", "Actin", "G", {1: 111.0}, source="a.tif"), key_on_source=True
    )
    assert len(session.entries) == 1
    assert session.entries[0].intensities == [111.0]


def test_second_exposure_of_one_membrane_overwrites_by_default() -> None:
    """actin1 与 actin1(1) 是同一条 Actin，用户多半只想留一次。"""
    session = AnalysisSession()
    session.upsert(entry("loading_control", "Actin", "G", {1: 100.0}, source="actin1.tif"))
    session.upsert(entry("loading_control", "Actin", "G", {1: 120.0}, source="actin1(1).tif"))
    assert len(session.entries) == 1
    assert session.entries[0].membrane == "actin1(1)"


def test_two_real_control_membranes_can_both_be_kept() -> None:
    session = AnalysisSession()
    session.upsert(entry("loading_control", "Actin", "G", {1: 100.0}, source="gel1.tif"))
    session.upsert(
        entry("loading_control", "Actin", "G", {5: 900.0}, source="gel2.tif"), key_on_source=True
    )
    assert [e.membrane for e in session.entries] == ["gel1", "gel2"]


def test_conflicting_entries_is_the_cue_to_ask_the_user() -> None:
    """核心层分不出"两次曝光"和"两块胶"，但它能指出该问这一句。"""
    session = AnalysisSession()
    fresh = entry("loading_control", "Actin", "G", {1: 100.0}, source="gel1.tif")
    assert session.conflicting_entries(fresh) == []
    session.upsert(fresh)

    same_membrane = entry("loading_control", "Actin", "G", {1: 110.0}, source="gel1.tif")
    assert session.conflicting_entries(same_membrane) == [], "同一张膜重测，不用问"

    other_membrane = entry("loading_control", "Actin", "G", {1: 900.0}, source="gel2.tif")
    assert [e.membrane for e in session.conflicting_entries(other_membrane)] == ["gel1"]


def test_two_loading_controls_in_one_group_always_warn() -> None:
    """安全阀：不做这条，就不该开放"都留下"这个选项。"""
    session = AnalysisSession()
    session.upsert(
        entry("loading_control", "Actin", "G", {1: 100.0, 2: 100.0}, source="gel1.tif")
    )
    session.upsert(
        entry("loading_control", "Actin", "G", {1: 400.0, 2: 400.0}, source="gel2.tif"),
        key_on_source=True,
    )
    session.upsert(entry("target", "CITH3", "G", {1: 100.0, 2: 50.0}, source="cith3.tif"))

    rows = session.normalize_all()
    assert rows[0].loading_control_membrane == "gel2", "取的是最后加入的那条"
    warning = next((w for w in session.last_warnings if "2 条内参" in w), "")
    assert warning, f"多内参必须进 last_warnings：{session.last_warnings}"
    assert "Actin（膜 gel1）" in warning and "Actin（膜 gel2）" in warning
    assert "拆成两个分组" in warning
    assert "“CITH3”用的是内参“Actin（膜 gel2）”。" in session.last_warnings


def test_the_multi_control_warning_states_the_group_fact_once() -> None:
    """一组 5 个目标不该出 5 段几乎一样的长文案——诊断栏里那就是噪音。"""
    session = AnalysisSession()
    session.upsert(
        entry("loading_control", "Actin", "G", {1: 100.0, 2: 100.0}, source="gel1.tif")
    )
    session.upsert(
        entry("loading_control", "Actin", "G", {1: 400.0, 2: 400.0}, source="gel2.tif"),
        key_on_source=True,
    )
    for name in ("CITH3", "PAD4", "MPO"):
        session.upsert(entry("target", name, "G", {1: 100.0, 2: 50.0}, source=f"{name}.tif"))

    session.normalize_all()
    group_level = [w for w in session.last_warnings if "2 条内参" in w]
    assert len(group_level) == 1, f"组级事实只说一次：{session.last_warnings}"
    per_target = [w for w in session.last_warnings if w.startswith("“") and "用的是内参" in w]
    assert per_target == [
        "“CITH3”用的是内参“Actin（膜 gel2）”。",
        "“PAD4”用的是内参“Actin（膜 gel2）”。",
        "“MPO”用的是内参“Actin（膜 gel2）”。",
    ], "每个目标一句短的，点名到自己"
    assert all(len(w) < 40 for w in per_target)


def test_one_loading_control_stays_quiet() -> None:
    session = three_membrane_session()
    session.normalize_all()
    assert not [w for w in session.last_warnings if "内参" in w and "条内参" in w]


# ------------------------------------------------------------------ 分组自检


def test_group_status_answers_before_normalization_does() -> None:
    """加了目标却还没有内参，界面要能当场提示，而不是等到归一化才空着。"""
    session = AnalysisSession()
    session.upsert(entry("target", "CITH3", "0716wb", {1: 1.0, 2: 2.0}, source="c.tif"))
    status = session.group_status("0716wb")
    assert status.has_loading_control is False
    assert status.target_count == 1
    assert [m.membrane for m in status.membranes] == ["c"]
    assert any("还没有内参" in p for p in status.problems)
    assert "缺内参" in status.summary()


def test_group_status_exposes_which_lanes_a_membrane_is_missing() -> None:
    """膜 A 有 3 道、膜 B 只有 2 道——按分组存泳道名就是为了让这个暴露出来。"""
    session = three_membrane_session()
    session.upsert(entry("target", "PAD4", "0716wb", {1: 1.0, 2: 2.0}, source="pad4.tif"))
    status = session.group_status("0716wb")
    assert status.lanes == (1, 2, 3)
    missing = {m.name: m.missing_lanes for m in status.membranes}
    assert missing["PAD4"] == (3,)
    assert missing["Actin"] == ()
    assert any("PAD4" in p and "缺泳道 3" in p for p in status.problems)


def test_group_status_flags_unnamed_lanes_and_a_stray_control_lane() -> None:
    session = three_membrane_session()
    session.set_lane_label("0716wb", 3, "")
    session.set_control_lane("0716wb", 7)
    status = session.group_status("0716wb")
    assert status.unnamed_lanes == (3,)
    assert status.control_lane_missing is True
    assert any("还没有样品名" in p for p in status.problems)
    assert any("基准泳道 7" in p and "fold change 会全部留空" in p for p in status.problems)


def test_group_status_answers_who_loses_fold_change_in_one_field() -> None:
    """一个字段直接回答"有没有目标拿不到 fold change"。

    并集（`lanes`）答不了这个：基准泳道在内参膜上、在某张目标膜上没有时，
    `control_lane_missing` 是 False，而那个目标仍然会被拒绝——要界面拼两句
    才能看出来，就是会漏。
    """
    session = three_membrane_session()
    session.upsert(entry("target", "PAD4", "0716wb", {1: 1.0, 2: 2.0}, source="pad4.tif"))
    session.set_control_lane("0716wb", 3)
    status = session.group_status("0716wb")

    assert status.control_lane_missing is False, "泳道 3 在内参膜上，并集里有"
    assert status.targets_without_fold_change == ("PAD4",)
    problems = status.problems
    assert any(
        "基准泳道 3 不在“PAD4”上" in p and "全部留空" in p for p in problems
    ), problems
    assert not [p for p in problems if "其中有基准泳道" in p], "同一件事只说一遍"

    # 真被拒的就是这一个目标，另一个照常出数
    folds = {(r.target_name, r.lane): r.fold_change for r in session.normalize_all()}
    assert folds[("PAD4", 1)] is None and folds[("CITH3", 1)] == 4.0


def test_no_target_loses_fold_change_when_the_reference_lane_pairs() -> None:
    session = three_membrane_session()
    status = session.group_status("0716wb")
    assert status.targets_without_fold_change == ()
    assert not [p for p in status.problems if "fold change" in p]


def test_a_reference_lane_missing_on_the_control_membrane_blocks_everyone() -> None:
    """内参膜缺这一道 → 全组都拿不到分母，说一句就够。"""
    session = AnalysisSession()
    session.upsert(entry("loading_control", "Actin", "G", {2: 100.0}, source="a.tif"))
    session.upsert(entry("target", "A", "G", {1: 50.0, 2: 100.0}, source="a1.tif"))
    session.upsert(entry("target", "B", "G", {1: 50.0, 2: 100.0}, source="b1.tif"))
    session.set_control_lane("G", 1)
    status = session.group_status("G")
    assert status.control_lane_missing is False, "泳道 1 在目标膜上，并集里有"
    assert status.targets_without_fold_change == ("A", "B")
    blocked = [p for p in status.problems if "fold change" in p]
    assert len(blocked) == 1 and "内参“Actin（膜 a）”上没有基准泳道 1" in blocked[0]


def test_a_group_without_targets_reports_nobody_blocked() -> None:
    session = AnalysisSession()
    session.upsert(entry("loading_control", "Actin", "G", {2: 100.0}, source="a.tif"))
    session.set_control_lane("G", 9)
    status = session.group_status("G")
    assert status.targets_without_fold_change == (), "没有目标，就没有谁被拒"
    assert status.control_lane_missing is True, "但这一道确实不在本组任何膜上"


def test_groups_and_group_lanes_describe_the_experiment() -> None:
    session = three_membrane_session()
    session.upsert(entry("target", "PAD4", "0722wb", {4: 1.0, 5: 2.0}, source="p.tif"))
    assert session.groups() == ["0716wb", "0722wb"], "按加入顺序，界面据此列分组"
    assert session.group_lanes("0716wb") == [1, 2, 3]
    assert session.group_lanes("0722wb") == [4, 5]


def test_forget_group_drops_only_that_experiments_data() -> None:
    session = three_membrane_session()
    session.set_control_lane("0716wb", 2)
    session.set_lane_label("0722wb", 1, "WT")
    session.forget_group("0716wb")
    assert session.lane_labels_of("0716wb") == {}
    assert session.control_lane_of("0716wb") == 1, "退回默认值"
    assert session.lane_label("0722wb", 1) == "WT"
    assert len(session.entries) == 3, "记录本身不受影响"


def test_group_status_of_an_unknown_group_is_empty_not_an_error() -> None:
    status = AnalysisSession().group_status("从来没有过的组")
    assert status.is_empty and status.problems and "还没有任何数据" in status.problems[0]


# -------------------------------------------------------------- 内参 CSV 往返


def test_loading_control_csv_carries_sample_names_both_ways(tmp_path) -> None:
    """泳道名跟内参 CSV 一起走：显式导入回来，而不是像偏好那样不请自来。"""
    session = three_membrane_session()
    path = session.export_loading_control_csv(tmp_path / "lc.csv")
    header, rows = read_csv(path)
    assert "sample" in header and "membrane" in header
    assert [r["sample"] for r in rows] == ["WT", "KO", "处理1"]

    restored = AnalysisSession()
    assert restored.import_loading_control_csv(path) == 1
    assert restored.lane_labels_of("0716wb") == {1: "WT", 2: "KO", 3: "处理1"}
    assert restored.entries[0].membrane == "actin1"


def test_old_csv_without_the_new_columns_still_imports(tmp_path) -> None:
    """0.4 之前导出的文件不能因为少了三列就打不开。"""
    path = tmp_path / "old.csv"
    path.write_text(
        "kind,name,group,lane,integrated_density,background,mean_intensity,source\n"
        "loading_control,Actin,G,1,100,0,0,D:/old/actin.tif\n"
        "loading_control,Actin,G,2,110,0,0,D:/old/actin.tif\n",
        encoding="utf-8-sig",
    )
    session = AnalysisSession()
    assert session.import_loading_control_csv(path) == 1
    assert session.lane_labels == {}, "文件里没写，就没有样品名"
    assert session.entries[0].membrane == "actin", "膜名退回文件名，不留空"
    assert session.entries[0].roi_check == RoiCheck()


def test_two_membranes_in_one_csv_come_back_as_two_records(tmp_path) -> None:
    """一个文件里两张 Actin 膜，不能在导入时被合并成一条。"""
    path = tmp_path / "two.csv"
    path.write_text(
        "kind,name,group,sample,lane,integrated_density,source,membrane\n"
        "loading_control,Actin,G,WT,1,100,D:/gel1.tif,gel1\n"
        "loading_control,Actin,G,KO,2,110,D:/gel1.tif,gel1\n"
        "loading_control,Actin,G,WT,1,900,D:/gel2.tif,gel2\n"
        "loading_control,Actin,G,KO,2,950,D:/gel2.tif,gel2\n",
        encoding="utf-8-sig",
    )
    session = AnalysisSession()
    assert session.import_loading_control_csv(path) == 2
    assert [e.membrane for e in session.entries] == ["gel1", "gel2"]
    assert [e.intensities for e in session.entries] == [[100.0, 110.0], [900.0, 950.0]]
    assert any("内参“Actin”（分组 G）有 2 张膜的数据" in w for w in session.last_warnings)


def test_one_protein_as_control_and_target_stays_two_records(tmp_path) -> None:
    """跨膜之后用户会把一整批导进同一个文件。

    分桶键不含 kind 时，同名同组同源的两行会并进一桶、kind 取第一行的——
    一条数据凭空换了身份。
    """
    path = tmp_path / "batch.csv"
    path.write_text(
        "kind,name,group,lane,integrated_density,source\n"
        "loading_control,Actin,G,1,100,shared.tif\n"
        "loading_control,Actin,G,2,110,shared.tif\n"
        "target,Actin,G,1,300,shared.tif\n"
        "target,Actin,G,2,330,shared.tif\n",
        encoding="utf-8-sig",
    )
    session = AnalysisSession()
    assert session.import_loading_control_csv(path) == 2
    assert [(e.kind, e.intensities) for e in session.entries] == [
        ("loading_control", [100.0, 110.0]),
        ("target", [300.0, 330.0]),
    ]
    assert not [w for w in session.last_warnings if "张膜的数据" in w], (
        "同一张膜的两种身份不是两张膜，不该报"
    )


# --------------------------------------------------------- source 的规范化


def test_source_key_reads_two_spellings_of_one_path_as_one_membrane() -> None:
    assert source_key(r"D:\a\x.tif") == source_key("d:/a/x.tif")
    assert source_key("  d:/a/x.tif  ") == source_key("d:/a/x.tif"), "两边的空白不算身份"
    assert source_key("d:/a/./b/../x.tif") == source_key("d:/a/x.tif")
    assert source_key("d:/a/x.tif") != source_key("d:/a/y.tif")


def test_source_key_keeps_an_image_without_a_file_empty() -> None:
    """从数组装载的图没有文件——"没有文件"和"哪个文件"不是一个问题。"""
    assert source_key("") == ""
    assert source_key(None) == ""
    assert source_key("   ") == ""
    assert source_key("x") != "", "路径不存在照样能当身份，resolve 不会抛"


def test_keeping_both_does_not_invent_a_membrane_from_a_rewritten_path() -> None:
    """source 是 key_on_source 身份键的一部分，认错了就凭空多出一条。"""
    session = AnalysisSession()
    session.upsert(entry("loading_control", "Actin", "G", {1: 100.0}, source=r"D:\a\x.tif"))
    session.upsert(
        entry("loading_control", "Actin", "G", {1: 111.0}, source="d:/a/x.tif"),
        key_on_source=True,
    )
    assert len(session.entries) == 1, "同一张膜重新定量，静默更新"
    assert session.entries[0].intensities == [111.0]
    assert session.entries[0].source == "d:/a/x.tif", "存的还是调用方给的那串字符"


def test_conflicting_entries_agrees_with_the_normalized_source() -> None:
    session = AnalysisSession()
    session.upsert(entry("loading_control", "Actin", "G", {1: 100.0}, source=r"D:\a\x.tif"))
    same = entry("loading_control", "Actin", "G", {1: 110.0}, source="d:/a/x.tif")
    assert session.conflicting_entries(same) == [], "换个写法不该被当成另一张膜"


def test_a_rewritten_path_in_a_csv_does_not_split_one_membrane(tmp_path) -> None:
    path = tmp_path / "mixed_paths.csv"
    path.write_text(
        "kind,name,group,lane,integrated_density,source\n"
        "loading_control,Actin,G,1,100,D:\\a\\x.tif\n"
        "loading_control,Actin,G,2,110,d:/a/x.tif\n",
        encoding="utf-8-sig",
    )
    session = AnalysisSession()
    assert session.import_loading_control_csv(path) == 1
    assert session.entries[0].lanes == [1, 2]


def test_conflicting_sample_names_in_one_file_are_reported(tmp_path) -> None:
    path = tmp_path / "clash.csv"
    path.write_text(
        "kind,name,group,sample,lane,integrated_density,source\n"
        "loading_control,Actin,G,WT,1,100,a.tif\n"
        "loading_control,Actin,G,KO,1,100,a.tif\n",
        encoding="utf-8-sig",
    )
    session = AnalysisSession()
    session.import_loading_control_csv(path)
    assert session.lane_label("G", 1) == "KO"
    assert any("两个样品名" in w for w in session.last_warnings)


# ------------------------------------------------------------------ 长表 / 宽表


def test_cross_membrane_export_end_to_end(tmp_path) -> None:
    """三张膜（1 内参 + 2 目标）进同一分组，出一张能丢进 Prism 的汇总表。"""
    session = three_membrane_session()
    session.upsert(
        entry(
            "loading_control",
            "Actin",
            "0722wb",
            {1: 200.0, 2: 200.0},
            source=r"D:\0722wb\actin.tif",
            roi_check=RoiQualityReport(issues=(), lane_count=2, band_count=2),
        )
    )
    session.upsert(
        entry(
            "target", "PAD4", "0722wb", {1: 200.0, 2: 100.0}, source=r"D:\0722wb\pad4.tif"
        )
    )
    session.set_lane_labels("0722wb", {1: "WT", 2: "KO"})
    session.set_control_lane("0722wb", 2)

    export = export_summary_tables(session, tmp_path)
    assert export.long_path.name == "summary_long.csv"
    assert export.wide_path.name == "summary_wide.csv"
    assert export.readme_path is not None
    assert export.readme_path.name == "summary_wide_readme.txt"

    header, long_rows = read_csv(export.long_path)
    assert header == SUMMARY_LONG_FIELDS, "列序是约定，不能漂"
    assert len(long_rows) == 3 * 2 + 2 * 1

    cith3 = {int(r["lane"]): r for r in long_rows if r["target_name"] == "CITH3"}
    assert cith3[2]["sample"] == "KO"
    assert cith3[2]["group"] == "0716wb"
    # 分子分母各自属于哪张膜——只给一个 membrane 列会让人以为它们同膜
    assert cith3[2]["target_membrane"] == "cith3"
    assert cith3[2]["control_membrane"] == "actin1"
    assert cith3[2]["target_membrane_path"] == r"D:\0716wb\cith3.tif"
    assert cith3[2]["control_membrane_path"] == r"D:\0716wb\actin1.tif"
    assert cith3[2]["target"] == "50.0" and cith3[2]["loading_control"] == "100.0"
    assert cith3[2]["normalized"] == "0.5" and cith3[2]["fold_change"] == "0.5"
    assert cith3[2]["control_lane"] == "1"
    assert cith3[2]["target_below_detection"] == "0"
    assert cith3[2]["note"] == ""
    assert cith3[2]["app_version"] and cith3[2]["exported_at"]

    pad4_0722 = {
        int(r["lane"]): r for r in long_rows if r["group"] == "0722wb"
    }
    assert pad4_0722[1]["control_lane"] == "2", "另一场实验有自己的基准泳道"
    assert pad4_0722[1]["fold_change"] == "2.0"
    assert pad4_0722[1]["control_roi_check"].startswith("ok: ROI 自检通过")

    wide_header, wide_rows = read_csv(export.wide_path)
    assert wide_header == ["group", "lane", "sample", "CITH3", "PAD4"]
    assert [(r["group"], r["lane"], r["sample"]) for r in wide_rows] == [
        ("0716wb", "1", "WT"),
        ("0716wb", "2", "KO"),
        ("0716wb", "3", "处理1"),
        ("0722wb", "1", "WT"),
        ("0722wb", "2", "KO"),
    ]
    assert [r["CITH3"] for r in wide_rows] == ["1.0", "0.5", "0.25", "", ""]
    assert [r["PAD4"] for r in wide_rows] == ["1.0", "2.0", "3.0", "2.0", "1.0"]
    assert "Actin" not in wide_header, "内参的绝对值不可跨膜比较，宽表不列"

    readme = export.readme_path.read_text(encoding="utf-8-sig")
    assert "fold change" in readme
    assert "0716wb    泳道 1（样品 WT）" in readme
    assert "0722wb    泳道 2（样品 KO）" in readme
    assert "膜 cith3" in readme and "膜 actin1" in readme
    assert "分组“0722wb”没有“CITH3”的记录" in readme
    assert "内参的绝对积分值不能跨膜比较" in readme
    assert "summary_long.csv" in readme


def test_wide_table_can_carry_the_raw_ratio_instead(tmp_path) -> None:
    session = three_membrane_session()
    export = export_summary_tables(session, tmp_path, value=SUMMARY_VALUE_NORMALIZED)
    _, rows = read_csv(export.wide_path)
    assert [r["PAD4"] for r in rows] == ["0.4", "0.8", "1.2"]
    readme = export.readme_path.read_text(encoding="utf-8-sig")
    assert "normalized" in readme


def test_a_square_with_two_records_is_left_blank_and_explained(tmp_path) -> None:
    """聚合是用户的科学决定，不该由导出代劳。"""
    session = three_membrane_session()
    session.upsert(
        entry(
            "target",
            "CITH3",
            "0716wb",
            {1: 200.0, 2: 200.0, 3: 200.0},
            source=r"D:\0716wb\cith3_second_exposure.tif",
        ),
        key_on_source=True,
    )
    export = export_summary_tables(session, tmp_path)

    _, wide_rows = read_csv(export.wide_path)
    assert [r["CITH3"] for r in wide_rows] == ["", "", ""]
    assert [r["PAD4"] for r in wide_rows] == ["1.0", "2.0", "3.0"], "隔壁列不受影响"

    _, long_rows = read_csv(export.long_path)
    cith3 = [r for r in long_rows if r["target_name"] == "CITH3"]
    assert len(cith3) == 6, "长表两条都留着"
    assert all("宽表该格留空" in r["note"] for r in cith3)
    assert all("cith3、cith3_second_exposure" in r["note"] for r in cith3)
    assert all(r["note"] == "" for r in long_rows if r["target_name"] == "PAD4")

    readme = export.readme_path.read_text(encoding="utf-8-sig")
    assert "该格对应 2 条记录" in readme


def test_the_readme_says_why_the_reference_lane_has_no_sample_name(tmp_path) -> None:
    """基准泳道压根不在本组时，旁注不能推给"忘填样品名"。

    真正的后果是整组 fold change 都空着；说成样品名的问题，用户会去改错的东西。
    """
    session = AnalysisSession()
    session.upsert(entry("loading_control", "Actin", "G", {2: 100.0, 3: 100.0}))
    session.upsert(entry("target", "T", "G", {2: 50.0, 3: 100.0}))
    session.set_control_lane("G", 9)
    export = export_summary_tables(session, tmp_path)

    readme = export.readme_path.read_text(encoding="utf-8-sig")
    assert "G    泳道 9（这一道在本组里没有数据，拿不到分母，这一组的 fold change 全部留空）" in readme
    assert "还没有样品名" not in readme, "这一道有没有样品名不是问题所在"
    assert "退到了别的道" not in readme, "已经没有退档这回事了"
    # 和空格那一段说的是同一件事，别打架
    assert "基准泳道 9 没有可配对的数据，fold change 无法计算" in readme


def test_the_readme_names_the_targets_the_reference_lane_fails(tmp_path) -> None:
    """基准泳道在内参膜上、在某个目标膜上没有：只有那一列空，得说清是哪一列。"""
    session = AnalysisSession()
    session.upsert(entry("loading_control", "Actin", "G", {1: 100.0, 2: 100.0}))
    session.upsert(entry("target", "A", "G", {1: 50.0, 2: 100.0}))
    session.upsert(entry("target", "B", "G", {2: 100.0}))
    session.set_lane_label("G", 1, "WT")
    export = export_summary_tables(session, tmp_path)

    _, wide_rows = read_csv(export.wide_path)
    assert [r["A"] for r in wide_rows] == ["1.0", "2.0"], "A 两道都有，照常算"
    assert [r["B"] for r in wide_rows] == ["", ""], "B 拿不到基准，整列留空"
    readme = export.readme_path.read_text(encoding="utf-8-sig")
    assert "G    泳道 1（样品 WT）" in readme
    assert "这一道在“B”上算不出比值，这一列的 fold change 全部留空" in readme
    assert "“A”" not in readme.split("【本次用到的膜】")[0], "A 没问题，别把它也点名"


def test_blank_cells_keep_their_reason_from_the_row(tmp_path) -> None:
    """内参缺了这道 → 宽表空格，理由是长表那一行的 note。"""
    session = AnalysisSession()
    session.upsert(entry("loading_control", "Actin", "G", {1: 100.0, 2: 100.0}))
    session.upsert(entry("target", "T", "G", {1: 50.0, 2: 50.0, 3: 50.0}))
    export = export_summary_tables(session, tmp_path)

    _, wide_rows = read_csv(export.wide_path)
    assert [r["T"] for r in wide_rows] == ["1.0", "1.0", ""]
    readme = export.readme_path.read_text(encoding="utf-8-sig")
    assert "泳道 3" in readme and "缺少泳道 3" in readme


def test_blank_fold_change_blames_the_reference_lane(tmp_path) -> None:
    """比值算得出来、基准泳道算不出来——这条理由行自己的 note 里没有。"""
    session = AnalysisSession()
    session.upsert(entry("loading_control", "Actin", "G", {1: 0.0, 2: 100.0}))
    session.upsert(entry("target", "T", "G", {1: 50.0, 2: 50.0}))
    export = export_summary_tables(session, tmp_path)

    _, wide_rows = read_csv(export.wide_path)
    assert [r["T"] for r in wide_rows] == ["", ""]
    _, long_rows = read_csv(export.long_path)
    assert [r["normalized"] for r in long_rows] == ["", "0.5"]
    assert [r["note"] for r in long_rows] == [
        "内参强度为 0",
        "基准泳道 1 没有比值，fold change 无法计算",
    ], "长表自己就写清了两种空法"
    readme = export.readme_path.read_text(encoding="utf-8-sig")
    assert "基准泳道 1 没有比值" in readme
    assert "内参强度为 0" in readme, "泳道 1 空的理由是它自己那一行的 note"


def test_below_detection_is_its_own_column_on_both_sides(tmp_path) -> None:
    """藏在 note 的中文里机器读不了，而 note 只装得下一条理由。"""
    session = AnalysisSession()
    session.upsert(
        entry("loading_control", "Actin", "G", {1: 100.0, 2: 100.0}, weak=(2,))
    )
    session.upsert(entry("target", "T", "G", {1: 50.0, 2: 50.0}, weak=(1, 2)))
    export = export_summary_tables(session, tmp_path)
    _, rows = read_csv(export.long_path)
    flags = {
        int(r["lane"]): (r["target_below_detection"], r["loading_control_below_detection"])
        for r in rows
    }
    assert flags == {1: ("1", "0"), 2: ("1", "1")}
    assert all(r["note"] == "低于检出限" for r in rows)


def test_summary_reports_a_group_that_has_no_control_yet(tmp_path) -> None:
    session = three_membrane_session()
    session.upsert(entry("target", "CITH3", "0722wb", {1: 1.0}, source="c2.tif"))
    export = export_summary_tables(session, tmp_path)
    assert export.groups == ("0716wb",)
    assert any("还没有内参" in w for w in export.warnings)
    readme = export.readme_path.read_text(encoding="utf-8-sig")
    assert "还没有内参" in readme


def test_summary_refuses_when_there_is_nothing_to_summarize(tmp_path) -> None:
    session = AnalysisSession()
    session.upsert(entry("loading_control", "Actin", "G", {1: 1.0}))
    with pytest.raises(ValueError, match="没有可导出的汇总数据"):
        export_summary_tables(session, tmp_path)


def test_summary_rows_can_be_limited_to_one_group() -> None:
    session = three_membrane_session()
    session.upsert(entry("loading_control", "Actin", "X", {1: 1.0}, source="ax.tif"))
    session.upsert(entry("target", "T", "X", {1: 2.0}, source="tx.tif"))
    rows = build_summary_rows(session, groups=["X"])
    assert {r.group for r in rows} == {"X"}


def test_wide_rows_run_in_gel_order_within_each_group() -> None:
    """先加入的那张膜从泳道 3 开始，也不能把泳道 1 挤到表尾。"""
    session = AnalysisSession()
    session.upsert(entry("loading_control", "Actin", "G", {1: 1.0, 2: 1.0, 3: 1.0}))
    session.upsert(entry("target", "A", "G", {3: 3.0}))
    session.upsert(entry("target", "B", "G", {1: 1.0, 2: 2.0}))
    table = build_wide_table(build_summary_rows(session))
    assert [(r.group, r.lane) for r in table.rows] == [("G", 1), ("G", 2), ("G", 3)]
    assert table.targets == ("A", "B")


def test_wide_table_rejects_an_unknown_value_field() -> None:
    session = three_membrane_session()
    rows = build_summary_rows(session)
    with pytest.raises(ValueError, match="fold_change"):
        build_wide_table(rows, value="whatever")


def test_summary_can_skip_the_readme(tmp_path) -> None:
    session = three_membrane_session()
    export = export_summary_tables(session, tmp_path, write_readme=False)
    assert export.readme_path is None
    assert not (tmp_path / "summary_wide_readme.txt").exists()


# --------------------------------------------------------------- 包的公开入口


def test_the_new_api_is_reachable_from_the_package_root() -> None:
    """界面按 `from wb_analyzer import ...` 接线，别让它去 import 子模块。"""
    import wb_analyzer

    expected = {
        "GroupStatus",
        "MembraneStatus",
        "RoiCheck",
        "SummaryExport",
        "build_summary_rows",
        "build_wide_table",
        "coerce_roi_check",
        "export_summary_tables",
        "membrane_from_source",
        "source_key",
    }
    assert expected <= set(wb_analyzer.__all__)
    for name in wb_analyzer.__all__:
        assert hasattr(wb_analyzer, name), f"__all__ 里有 {name}，模块里没有"
