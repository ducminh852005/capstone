"""
Vietnamese user-facing labels of the clip analysis. This is the one place that holds these strings:
core/clip_analysis.py exports them into analysis.json so the HTML viewer never hard-codes a label
of an analysis result (only the static page chrome is written in the template itself).
"""
from typing import Dict

CALL_RESULT = {
    "IN": "Trong",
    "OUT": "Ngoài",
}

CALL_HALF = {
    "near": "nửa sân gần",
    "far": "nửa sân xa",
}

CALL_METHOD = {
    "contact": "thấy cầu chạm sàn",
    "contact_unconfirmed": "chạm sàn, chưa xác nhận",
    "lost": "mất dấu khi cầu đang rơi (ngoại suy)",
    "resting": "cầu nằm yên trên sân",
}

HIT_SOURCE = {
    "physics": "Đổi hướng đột ngột (chạm vợt hoặc nảy sàn)",
    "new": "Cầu mới xuất hiện",
    "stop": "Cầu dừng lại",
}

HITTER = {
    "near": "Người chơi gần",
    "far": "Đối thủ (suy luận)",
    "unknown": "Chưa rõ người đánh",
}

HIT_EVIDENCE = {
    "wrist": "cổ tay ở gần cầu",
    "no_near_hand": "không tay nào ở gần cầu",
    "none": "không đo được vị trí tay",
}

REJECT_REASON = {
    "new_track_no_hitter": "cầu mới xuất hiện, không có tay ở gần",
    "landing_bounce": "cầu nảy sàn sát điểm rơi",
}

END_CAUSE = {
    "landed_in": "cầu rơi trong sân",
    "landed_out": "cầu rơi ngoài sân",
    "own_half": "cầu rơi ngay nửa sân của người đánh",
    "timeout": "rally dừng, không thấy điểm rơi",
    "clip_end": "hết đoạn clip",
}

WINNER = {
    "near": "Người chơi gần",
    "far": "Đối thủ",
}

RACKET_HAND = {
    "left": "trái",
    "right": "phải",
}

ZONE = {
    "rear_left": "Cuối sân, trái",
    "rear_right": "Cuối sân, phải",
    "mid_left": "Giữa sân, trái",
    "mid_right": "Giữa sân, phải",
    "front_left": "Gần lưới, trái",
    "front_right": "Gần lưới, phải",
}

SPEED_LEVEL = {
    "0": "Đứng yên",
    "1": "Đi bộ",
    "2": "Chạy",
    "3": "Chạy nước rút",
}
"""Keyed by the index of the level in MovementStats.speed_level_share."""

BASE_SOURCE = {
    "rally": "lúc đang rally",
    "clip": "trong cả đoạn phân tích",
}

SHOT_TYPE = {
    "serve": "Giao cầu",
    "smash": "Đập cầu",
    "drop": "Bỏ nhỏ",
    "clear": "Phông cầu",
    "lift": "Lốp cầu",
    "net": "Cú lưới",
    "drive": "Đánh ngang",
    "unknown": "Chưa phân loại",
}

CONTACT_HEIGHT = {
    "overhead": "trên đầu",
    "mid": "ngang thân",
    "low": "thấp",
}

SHOT_REASON = {
    "rally_opening": "mở đầu rally sau thời gian nghỉ",
    "behind_service_line": "đứng sau vạch giao cầu ngắn",
    "contact_low": "đánh ở thấp",
    "contact_mid": "đánh ngang thân",
    "contact_overhead": "đánh trên đầu",
    "contact_low_or_mid": "đánh ở thấp hoặc ngang thân",
    "shuttle_fast": "cầu đi nhanh",
    "shuttle_slow": "cầu đi chậm",
    "shuttle_flat": "cầu bay ngang",
    "shuttle_falling": "cầu đi xuống",
    "shuttle_rising": "cầu bay lên",
    "not_rising": "cầu không bay lên",
    "hit_from_back": "đánh từ cuối hoặc giữa sân",
    "hit_at_net": "đánh gần lưới",
    "arm_extended": "tay duỗi thẳng",
    "short_flight": "bay ngắn",
    "long_flight": "bay lâu",
    "short_flight_or_landing": "bay ngắn hoặc rơi gần lưới",
    "rising_or_long_flight": "cầu bay lên hoặc bay lâu",
    "pose_missing": "thiếu khung xương nên độ tin cậy bị hạ",
    "no_rule_matches": "không luật nào khớp",
    "opponent_not_tracked": "đối thủ không được theo dõi",
    "hitter_unknown": "chưa rõ người đánh",
}

WARNING_MESSAGES = {
    "alternation_break": "{count} cú đánh của người chơi gần liền nhau trong một rally: có thể bỏ sót một cú của đối thủ.",
    "doubles_unsupported": "Trận đôi: không suy luận được cú đánh của đối thủ, chỉ nhận cú của người chơi gần.",
    "far_half_extrapolated": "Hiệu chuẩn chỉ khớp nửa sân gần; nửa sân xa là ngoại suy nên các điểm rơi ở đó kém tin cậy.",
    "fps_not_60": "Video không phải 60 fps ({fps:g} fps); số liệu theo giây dùng fps thật của clip.",
    "no_player": "Không tìm thấy người chơi nào ở nửa sân gần trong đoạn phân tích.",
    "preroll_short": "Phần khởi động chỉ có {preroll_frames} frame (cần khoảng {needed_frames}); cầu và người chơi có thể chưa được nhận diện ở đầu đoạn.",
    "frame_count_mismatch": "Clip đã cắt có {actual} frame, mong đợi {expected}; chỉ số frame có thể lệch so với video gốc.",
}


def labels_for_export() -> Dict[str, Dict[str, str]]:
    """Label tables written into analysis.json under "labels"."""
    return {
        "call_result": dict(CALL_RESULT),
        "call_half": dict(CALL_HALF),
        "call_method": dict(CALL_METHOD),
        "hit_source": dict(HIT_SOURCE),
        "hitter": dict(HITTER),
        "hit_evidence": dict(HIT_EVIDENCE),
        "reject_reason": dict(REJECT_REASON),
        "end_cause": dict(END_CAUSE),
        "winner": dict(WINNER),
        "racket_hand": dict(RACKET_HAND),
        "shot_type": dict(SHOT_TYPE),
        "contact_height": dict(CONTACT_HEIGHT),
        "shot_reason": dict(SHOT_REASON),
        "zone": dict(ZONE),
        "speed_level": dict(SPEED_LEVEL),
        "base_source": dict(BASE_SOURCE),
    }


def warning(code: str, **params) -> Dict[str, str]:
    """{"code", "message"} for a warning of analysis.json (message in Vietnamese)."""
    return {"code": code, "message": WARNING_MESSAGES[code].format(**params)}
