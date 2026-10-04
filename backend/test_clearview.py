"""Unit tests for Clearview Director, grade, and plan sanitizing.

Run from backend/:  python -m unittest test_clearview -v
"""
from __future__ import annotations

import array
import hashlib
import json
import math
import struct
import subprocess
import tempfile
import threading
import time
import unittest
import wave
from collections import OrderedDict
from pathlib import Path
from unittest.mock import MagicMock, patch

from video_create import (
    GRADE_LOOKS,
    SEQUENCE_VERSION,
    active_project_clip_names,
    attach_clip_to_active_project,
    create_project,
    ensure_project_clip_membership,
    load_projects_index,
    switch_project,
    _clips_from_scenes,
    _compact_scenes,
    _default_grade,
    _heuristic_plan,
    _image_query_from_msg,
    _image_search_reply,
    _mutate_grade,
    _normalize_grade,
    _parse_seconds,
    _plan_from_library,
    _safe_float,
    _sanitize_plan,
    _scene_span,
    _scene_index,
    _wants_grade,
    _wants_load_clips,
    apply_director_command,
    chat_edit_plan,
    grok_caption_cues,
    plan_video,
    _llm_plan,
    caption_export_texts,
    filename_from_src,
    load_sequence,
    migrate_legacy,
    persist_sequence_vo,
    persist_sequence_music,
    clear_sequence_music,
    resolve_vo_source,
    save_sequence,
    sanitize_sequence,
    SHRINK_MIN_SCENES,
    SHRINK_RATIO,
    sequence_vo_path,
    sequence_music_path,
    SEQUENCE_VO_URL,
    SEQUENCE_MUSIC_URL,
)
from video_tools import (
    _color_filter,
    _grade_filter,
    still_from_image,
    check_ffmpeg,
    ffmpeg_bin,
    compute_clip_peaks,
    load_clip_peaks,
    peaks_sidecar,
    empty_peaks,
    _pack_peaks,
    CLIPS_DIR,
    ensure_clip_poster,
    ensure_clip_proxy,
    proxy_needed,
    PROXY_MIN_BYTES,
    poster_sidecar,
    probe_duration,
    _normalize_segment,
    _export_vo_signature,
    _maybe_use_stored_vo,
    _export_vo_lead,
    _texts_to_ass,
    _ass_alpha,
)
from search_images import (
    _ok_image_url,
    parse_commons_query,
    parse_openverse,
    pack_image_search,
    query_passes,
    _merge,
    _guess_ext,
    classify_upload_name,
    import_local_image,
    import_web_image,
)
from tts_audio import (
    pick_edge_voice,
    edge_style_opts,
    sanitize_engine,
    sanitize_voice_name,
    sanitize_vo_fx,
    sanitize_vo_rate,
    edge_rate_percent,
    combined_edge_rate,
    synthesize_voiceover,
    split_tts_chunks,
    split_pausa_parts,
    strip_pausa_markers,
    _synth_script_then_concat,
    split_script_to_scenes,
    sanitize_script,
    _edge_synthesize,
    word_from_boundary,
    shift_words,
    _synth_chunks_then_concat,
)
from eleven_audio import words_to_cues, cues_from_untimed_script


def lib(*names_durs):
    clips = []
    for item in names_durs:
        if isinstance(item, str):
            name, dur = item, 5.0
        else:
            name, dur = item
        clips.append({"name": name, "title": name.rsplit(".", 1)[0], "duration": dur})
    return clips


class SafeFloatTests(unittest.TestCase):
    def test_number(self):
        self.assertEqual(_safe_float(3.2, 1, 0, 10), 3.2)

    def test_string_number(self):
        self.assertEqual(_safe_float("2.5", 1, 0, 10), 2.5)

    def test_none_and_junk(self):
        self.assertEqual(_safe_float(None, 1.5, 0, 10), 1.5)
        self.assertEqual(_safe_float("nope", 1.5, 0, 10), 1.5)
        self.assertEqual(_safe_float([], 1.5, 0, 10), 1.5)
        self.assertEqual(_safe_float({"x": 1}, 1.5, 0, 10), 1.5)

    def test_clamp_and_nan(self):
        self.assertEqual(_safe_float(99, 1, 0, 4), 4)
        self.assertEqual(_safe_float(-2, 1, 0, 4), 0)
        self.assertEqual(_safe_float(float("nan"), 2, 0, 4), 2)


class WantsLoadClipsTests(unittest.TestCase):
    def test_user_message_that_crashed(self):
        self.assertTrue(_wants_load_clips("load all the clips selected into scenes"))

    def test_spanish_and_chip(self):
        self.assertTrue(_wants_load_clips("pon todos los clips en escenas"))
        self.assertTrue(_wants_load_clips("add selected clips"))
        self.assertTrue(_wants_load_clips("cargar todos los clips a escenas"))

    def test_does_not_steal_other_commands(self):
        self.assertFalse(_wants_load_clips("genera captions estilo bold"))
        self.assertFalse(_wants_load_clips("escena 2 más cálida"))
        self.assertFalse(_wants_load_clips("usa cada clip entero"))
        self.assertFalse(_wants_load_clips("look film en todas"))
        self.assertFalse(_wants_load_clips(""))


class PlanFromLibraryTests(unittest.TestCase):
    def test_one_scene_per_clip_full_duration(self):
        clips = lib(("a.mp4", 8.5), ("b.mp4", 3), ("c.mp4", 12))
        out = _plan_from_library({"title": "T"}, clips)
        scenes = out["plan"]["scenes"]
        self.assertEqual(len(scenes), 3)
        self.assertEqual([s["clip"] for s in scenes], ["a.mp4", "b.mp4", "c.mp4"])
        self.assertEqual(scenes[0]["inPoint"], 0.0)
        self.assertEqual(scenes[0]["outPoint"], 8.5)
        self.assertEqual(scenes[1]["outPoint"], 3.0)
        self.assertTrue(out["local"])

    def test_empty_library_no_crash(self):
        out = _plan_from_library({}, [])
        self.assertIsNone(out["plan"])
        self.assertIn("No hay clips", out["reply"])

    def test_ignores_string_and_empty_entries(self):
        clips = ["oops", {}, {"name": ""}, {"name": "ok.mp4", "duration": 4}, None]
        out = _plan_from_library({}, clips)
        self.assertEqual(len(out["plan"]["scenes"]), 1)
        self.assertEqual(out["plan"]["scenes"][0]["clip"], "ok.mp4")

    def test_caps_at_40(self):
        clips = lib(*[f"c{i}.mp4" for i in range(45)])
        out = _plan_from_library({}, clips)
        self.assertEqual(len(out["plan"]["scenes"]), 40)

    def test_bad_duration_does_not_raise(self):
        out = _plan_from_library({}, [{"name": "x.mp4", "duration": "nope"}])
        self.assertEqual(len(out["plan"]["scenes"]), 1)
        self.assertGreater(out["plan"]["scenes"][0]["outPoint"], 0.3)


class ApplyDirectorLoadTests(unittest.TestCase):
    def test_load_all_with_empty_board(self):
        clips = lib("rio.mp4", "humo.mp4")
        out = apply_director_command(
            "load all the clips selected into scenes",
            {"title": "x", "scenes": []},
            clips,
        )
        self.assertIsNotNone(out)
        self.assertEqual(len(out["plan"]["scenes"]), 2)

    def test_load_all_replaces_existing_scenes(self):
        clips = lib("a.mp4", "b.mp4")
        out = apply_director_command(
            "load all the clips selected into scenes",
            {"scenes": [{"clip": "old.mp4", "inPoint": 0, "outPoint": 1}]},
            clips,
        )
        names = [s["clip"] for s in out["plan"]["scenes"]]
        self.assertEqual(names, ["a.mp4", "b.mp4"])
        self.assertNotIn("old.mp4", names)

    def test_load_all_never_raises_index_error(self):
        cases = [
            ("load all the clips selected into scenes", {}, []),
            ("load all the clips selected into scenes", {"scenes": []}, None),
            ("todos los clips a escenas", {"scenes": "broken"}, [{"name": "a.mp4"}]),
            ("add selected clips", None, lib("a.mp4")),
        ]
        for msg, plan, clips in cases:
            with self.subTest(msg=msg):
                try:
                    apply_director_command(msg, plan or {}, clips or [])
                except IndexError:
                    self.fail("IndexError on: " + msg)


class SanitizePlanTests(unittest.TestCase):
    def test_empty_library_is_value_error_not_index_error(self):
        with self.assertRaises(ValueError):
            _sanitize_plan({"scenes": [{"clip": "a.mp4"}]}, [], "youtube")

    def test_no_valid_scenes_is_value_error(self):
        with self.assertRaises(ValueError):
            _sanitize_plan({"scenes": ["nope", 3, None]}, lib("a.mp4"), "youtube")

    def test_missing_clip_name_falls_back(self):
        plan = _sanitize_plan(
            {"scenes": [{"clip": ""}, {"narration": "hi"}]},
            lib("a.mp4", "b.mp4"),
            "youtube",
        )
        self.assertEqual(len(plan["scenes"]), 2)
        self.assertTrue(all(s["clip"] in {"a.mp4", "b.mp4"} for s in plan["scenes"]))

    def test_integer_clip_index(self):
        plan = _sanitize_plan(
            {"scenes": [{"clip": 1, "inPoint": 0, "outPoint": 2}]},
            lib("a.mp4", "b.mp4"),
            "youtube",
        )
        self.assertEqual(plan["scenes"][0]["clip"], "b.mp4")

    def test_junk_in_out_points(self):
        plan = _sanitize_plan(
            {"scenes": [{"clip": "a.mp4", "inPoint": [], "outPoint": {"x": 1}}]},
            lib(("a.mp4", 6)),
            "youtube",
        )
        s = plan["scenes"][0]
        self.assertGreater(s["outPoint"], s["inPoint"])

    def test_unknown_filename_does_not_crash(self):
        plan = _sanitize_plan(
            {"scenes": [{"clip": "missing_file.mp4"}]},
            lib("real.mp4"),
            "youtube",
        )
        self.assertEqual(plan["scenes"][0]["clip"], "real.mp4")

    def test_preserves_grade(self):
        plan = _sanitize_plan(
            {"scenes": [{"clip": "a.mp4", "grade": {"temp": 0.4, "sat": 1.2}}]},
            lib("a.mp4"),
            "youtube",
        )
        g = plan["scenes"][0]["grade"]
        self.assertAlmostEqual(g["temp"], 0.4)
        self.assertAlmostEqual(g["sat"], 1.2)

    def test_preserves_crop_and_speed(self):
        plan = _sanitize_plan(
            {"scenes": [{"clip": "a.mp4", "speed": 2, "crop": {"x": 0.1, "y": 0.1, "w": 0.5, "h": 0.5}}]},
            lib(("a.mp4", 10)),
            "youtube",
        )
        s = plan["scenes"][0]
        self.assertEqual(s["speed"], 2)
        self.assertEqual(s["crop"]["w"], 0.5)
        self.assertAlmostEqual(s["duration"], 5.0)

    def test_sanitize_assigns_unique_ids_when_missing(self):
        plan = _sanitize_plan(
            {"scenes": [
                {"clip": "a.mp4", "inPoint": 0, "outPoint": 2},
                {"clip": "b.mp4", "inPoint": 0, "outPoint": 2},
            ]},
            lib("a.mp4", "b.mp4"),
            "youtube",
        )
        ids = [s["id"] for s in plan["scenes"]]
        self.assertEqual(len(ids), 2)
        self.assertTrue(all(ids))
        self.assertEqual(len(set(ids)), 2)

    def test_same_clip_two_scenes_get_different_ids(self):
        plan = _sanitize_plan(
            {"scenes": [
                {"clip": "a.mp4", "inPoint": 0, "outPoint": 2},
                {"clip": "a.mp4", "inPoint": 2, "outPoint": 4},
            ]},
            lib(("a.mp4", 8)),
            "youtube",
        )
        self.assertEqual(plan["scenes"][0]["clip"], "a.mp4")
        self.assertEqual(plan["scenes"][1]["clip"], "a.mp4")
        self.assertTrue(plan["scenes"][0]["id"])
        self.assertNotEqual(plan["scenes"][0]["id"], plan["scenes"][1]["id"])

    def test_existing_id_preserved_through_sanitize(self):
        plan = _sanitize_plan(
            {"scenes": [{"clip": "a.mp4", "id": "keep-this-id", "inPoint": 0, "outPoint": 2}]},
            lib("a.mp4"),
            "youtube",
        )
        self.assertEqual(plan["scenes"][0]["id"], "keep-this-id")

    def test_strict_keeps_unknown_clip(self):
        plan = _sanitize_plan(
            {"scenes": [{"clip": "only_mine.mp4", "inPoint": 1, "outPoint": 4}]},
            lib("other.mp4"),
            "youtube",
            remap_missing=False,
        )
        self.assertEqual(plan["scenes"][0]["clip"], "only_mine.mp4")
        self.assertEqual(plan["scenes"][0]["inPoint"], 1)

    def test_clips_as_strings_raise_cleanly(self):
        with self.assertRaises(ValueError):
            _sanitize_plan({"scenes": [{"clip": "a.mp4"}]}, ["a.mp4"], "youtube")

    def test_sanitize_keeps_cap_off_and_pos(self):
        plan = _sanitize_plan(
            {"scenes": [{
                "clip": "a.mp4",
                "capOff": True,
                "capPos": "center",
                "captionStyle": "neon",
                "junk": 123,
                "narration": "río",
                "voiceover": True,
            }]},
            lib("a.mp4"),
            "youtube",
        )
        s = plan["scenes"][0]
        self.assertTrue(s["capOff"])
        self.assertEqual(s["capPos"], "center")
        self.assertEqual(s["narration"], "río")
        self.assertTrue(s["voiceover"])
        self.assertNotIn("junk", s)
        self.assertNotIn("captionStyle", s)
        defaults = _sanitize_plan(
            {"scenes": [{"clip": "a.mp4", "capPos": "flying"}]},
            lib("a.mp4"),
            "youtube",
        )
        d = defaults["scenes"][0]
        self.assertFalse(d["capOff"])
        self.assertEqual(d["capPos"], "")

    def test_freeze_duration_keeps_capoff_window(self):
        plan = _sanitize_plan(
            {"scenes": [{
                "clip": "a.mp4",
                "inPoint": 0,
                "outPoint": 0.04,
                "freeze": True,
                "duration": 5.0,
                "capOff": True,
            }]},
            lib("a.mp4"),
            "youtube",
        )
        s = plan["scenes"][0]
        self.assertTrue(s["capOff"])
        self.assertTrue(s["freeze"])
        self.assertAlmostEqual(s["duration"], 5.0)
        start, end = _scene_span(s)
        self.assertAlmostEqual(end - start, 5.0)

    def test_fit_defaults_contain_preserves_cover(self):
        plan = _sanitize_plan(
            {"scenes": [{"clip": "a.mp4", "inPoint": 0, "outPoint": 2}]},
            lib("a.mp4"),
            "youtube",
        )
        self.assertEqual(plan["scenes"][0]["fit"], "contain")
        plan2 = _sanitize_plan(
            {"scenes": [{
                "clip": "a.mp4", "inPoint": 0, "outPoint": 2, "fit": "cover",
            }]},
            lib("a.mp4"),
            "youtube",
        )
        self.assertEqual(plan2["scenes"][0]["fit"], "cover")
        junk = _sanitize_plan(
            {"scenes": [{"clip": "a.mp4", "inPoint": 0, "outPoint": 2, "fit": "stretch"}]},
            lib("a.mp4"),
            "youtube",
        )
        self.assertEqual(junk["scenes"][0]["fit"], "contain")

    def test_still_freeze_hold_20s_play_duration(self):
        plan = _sanitize_plan(
            {"scenes": [{
                "clip": "img_hold.mp4",
                "inPoint": 0,
                "outPoint": 4,
                "freeze": True,
                "duration": 20,
            }]},
            lib(("img_hold.mp4", 4.0)),
            "youtube",
        )
        s = plan["scenes"][0]
        self.assertTrue(s["freeze"])
        self.assertAlmostEqual(s["duration"], 20.0)
        self.assertAlmostEqual(plan["duration"], 20.0)
        self.assertAlmostEqual(s["outPoint"] - s["inPoint"], 4.0)


class HeuristicPlanTests(unittest.TestCase):
    def test_one_clip(self):
        plan = _heuristic_plan("Un río sucio.", "documentary", 60, lib(("rio.mp4", 9)))
        self.assertEqual(len(plan["scenes"]), 1)
        self.assertEqual(plan["scenes"][0]["clip"], "rio.mp4")
        self.assertEqual(plan["scenes"][0]["outPoint"], 9)

    def test_string_clips_raise_not_index_error(self):
        with self.assertRaises(ValueError):
            _heuristic_plan("idea", "youtube", 30, ["a.mp4"])

    def test_empty_prompt_still_builds(self):
        plan = _heuristic_plan("", "youtube", 30, lib("a.mp4", "b.mp4", "c.mp4"))
        self.assertGreaterEqual(len(plan["scenes"]), 1)


class LlmPlanTimeoutAndJsonTests(unittest.TestCase):
    def _client(self, content='{"title":"t","summary":"s","scenes":[]}'):
        client = MagicMock()
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = content
        client.chat.completions.create.return_value = resp
        return client

    def test_llm_plan_timeout_is_45(self):
        client = self._client()
        with patch("video_create._xai_key", return_value="test-key"):
            with patch("openai.OpenAI", return_value=client) as ctor:
                _llm_plan("idea", "documentary", 60, lib("a.mp4"), "es")
        self.assertEqual(ctor.call_args.kwargs.get("timeout"), 45.0)
        self.assertEqual(
            client.chat.completions.create.call_args.kwargs.get("timeout"), 45.0
        )

    def test_plan_video_missing_key_mentions_xai(self):
        with patch("video_create.list_library_clips", return_value=lib("a.mp4")):
            with patch("video_create.active_project_clip_names", return_value=None):
                with patch(
                    "video_create._llm_plan",
                    side_effect=RuntimeError("missing_key"),
                ):
                    plan = plan_video("un río")
        self.assertIn("XAI_API_KEY", plan.get("summary") or "")
        self.assertFalse(plan.get("ai"))
        self.assertTrue(plan.get("scenes"))

    def test_plan_video_timeout_does_not_blame_key(self):
        with patch("video_create.list_library_clips", return_value=lib("a.mp4")):
            with patch("video_create.active_project_clip_names", return_value=None):
                with patch(
                    "video_create._llm_plan",
                    side_effect=TimeoutError("timed out"),
                ):
                    plan = plan_video("un río")
        summary = plan.get("summary") or ""
        self.assertNotIn("XAI_API_KEY", summary)
        self.assertIn("director no respondió", summary)
        self.assertTrue(plan.get("scenes"))
        self.assertFalse(plan.get("ai"))

    def test_llm_plan_bad_json_is_distinct_error(self):
        client = self._client("esto no es json {")
        with patch("video_create._xai_key", return_value="test-key"):
            with patch("openai.OpenAI", return_value=client):
                with self.assertRaises(RuntimeError) as ctx:
                    _llm_plan("idea", "documentary", 60, lib("a.mp4"), "es")
        self.assertEqual(str(ctx.exception), "bad_json")

    def test_all_three_creates_pass_response_format(self):
        want = {"type": "json_object"}
        plan_client = self._client()
        with patch("video_create._xai_key", return_value="test-key"):
            with patch("openai.OpenAI", return_value=plan_client):
                _llm_plan("idea", "documentary", 60, lib("a.mp4"), "es")
        self.assertEqual(
            plan_client.chat.completions.create.call_args.kwargs.get("response_format"),
            want,
        )

        chat_client = self._client('{"reply":"ok"}')
        with patch("video_create._xai_key", return_value="test-key"):
            with patch("video_create.list_library_clips", return_value=[]):
                with patch("video_create.active_project_clip_names", return_value=None):
                    with patch("video_create.apply_director_command", return_value=None):
                        with patch("openai.OpenAI", return_value=chat_client):
                            chat_edit_plan("qué opinas del ritmo?", history=[], current_plan={})
        self.assertEqual(
            chat_client.chat.completions.create.call_args.kwargs.get("response_format"),
            want,
        )

        cap_client = self._client('{"cues":[]}')
        with patch("video_create._xai_key", return_value="test-key"):
            with patch("openai.OpenAI", return_value=cap_client):
                grok_caption_cues("idea", "es", [], 10.0)
        self.assertEqual(
            cap_client.chat.completions.create.call_args.kwargs.get("response_format"),
            want,
        )


class CompactScenesTests(unittest.TestCase):
    def test_skips_non_dicts(self):
        out = _compact_scenes(["x", None, {"clip": "a.mp4", "speed": "nope"}])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["speed"], 1.0)

    def test_empty(self):
        self.assertEqual(_compact_scenes(None), [])
        self.assertEqual(_compact_scenes([]), [])


class SceneIndexAndSecondsTests(unittest.TestCase):
    def test_scene_index_bounds(self):
        self.assertIsNone(_scene_index("load all into scenes", 5))
        self.assertEqual(_scene_index("escena 2 a 3s", 5), 1)
        self.assertIsNone(_scene_index("escena 9 a 3s", 2))
        self.assertIsNone(_scene_index("escena 0", 2))

    def test_parse_seconds(self):
        self.assertEqual(_parse_seconds("escena 1 a 3s"), 3.0)
        self.assertEqual(_parse_seconds("6 segundos"), 6.0)
        self.assertIsNone(_parse_seconds("load all clips"))


class DirectorEditTests(unittest.TestCase):
    def setUp(self):
        self.clips = lib(("a.mp4", 10), ("b.mp4", 8))
        self.plan = {
            "title": "T",
            "format": "youtube",
            "scenes": [
                {"clip": "a.mp4", "inPoint": 0, "outPoint": 10, "narration": ""},
                {"clip": "b.mp4", "inPoint": 0, "outPoint": 8, "narration": ""},
            ],
        }

    def test_duration_on_existing_scene(self):
        out = apply_director_command("escena 1 a 3 segundos", self.plan, self.clips)
        self.assertAlmostEqual(out["plan"]["scenes"][0]["outPoint"], 3.0)

    def test_missing_scene_does_not_index_error(self):
        out = apply_director_command("escena 9 a 3 segundos", self.plan, self.clips)
        self.assertIsNone(out["plan"])
        self.assertIn("No hay escena 9", out["reply"])

    def test_delete_last_scene_blocked(self):
        one = {"scenes": [{"clip": "a.mp4", "inPoint": 0, "outPoint": 4}]}
        out = apply_director_command("borra la escena 1", one, self.clips)
        self.assertIsNone(out["plan"])

    def test_delete_scene(self):
        out = apply_director_command("borra la escena 2", self.plan, self.clips)
        self.assertEqual(len(out["plan"]["scenes"]), 1)
        self.assertEqual(out["plan"]["scenes"][0]["clip"], "a.mp4")

    def test_full_clips(self):
        self.plan["scenes"][0]["outPoint"] = 2
        out = apply_director_command("usa cada clip entero", self.plan, self.clips)
        self.assertEqual(out["plan"]["scenes"][0]["outPoint"], 10)

    def test_undo_is_flag_not_rewrite(self):
        out = apply_director_command("undo", self.plan, self.clips)
        self.assertTrue(out.get("undo"))
        self.assertIsNone(out["plan"])

    def test_empty_message(self):
        self.assertIsNone(apply_director_command("", self.plan, self.clips))

    def test_grade_scene_warm(self):
        out = apply_director_command("escena 2 más cálida", self.plan, self.clips)
        self.assertGreater(out["plan"]["scenes"][1]["grade"]["temp"], 0)

    def test_grade_all_film(self):
        out = apply_director_command("look film en todas", self.plan, self.clips)
        for s in out["plan"]["scenes"]:
            self.assertAlmostEqual(s["grade"]["sat"], GRADE_LOOKS["film"]["sat"])


class GradeLogicTests(unittest.TestCase):
    def test_normalize_junk(self):
        g = _normalize_grade("nope")
        self.assertEqual(g, _default_grade())
        g = _normalize_grade({"sat": "2", "temp": None, "lift": []})
        self.assertEqual(g["sat"], 2.0)
        self.assertEqual(g["lift"], 0.0)

    def test_clamp(self):
        g = _normalize_grade({"sat": 99, "lift": -9, "temp": 4})
        self.assertEqual(g["sat"], 3.0)
        self.assertEqual(g["lift"], -0.4)
        self.assertEqual(g["temp"], 1.0)

    def test_vo_calida_is_not_grade(self):
        self.assertFalse(_wants_grade("pon voiceover mas calida"))
        self.assertTrue(_wants_grade("escena 2 mas calida"))
        self.assertTrue(_wants_grade("look film en todas"))

    def test_mutate_look_film(self):
        g, note = _mutate_grade(_default_grade(), "look film")
        self.assertIn("film", note)
        self.assertAlmostEqual(g["sat"], GRADE_LOOKS["film"]["sat"])


class FfmpegGradeTests(unittest.TestCase):
    def test_identity_is_empty(self):
        self.assertEqual(_grade_filter(None), "")
        self.assertEqual(_grade_filter(_default_grade()), "")
        self.assertEqual(_grade_filter("nope"), "")

    def test_warm_emits_eq_and_colorbalance(self):
        vf = _grade_filter({"temp": 0.4, "sat": 1.2, "contrast": 1.1})
        self.assertIn("eq=", vf)
        self.assertIn("colorbalance=", vf)
        self.assertIn("saturation=1.200", vf)

    def test_color_presets(self):
        self.assertEqual(_color_filter("bw"), "hue=s=0")
        self.assertEqual(_color_filter("none"), "")
        self.assertEqual(_color_filter("nope"), "")


class ImageSearchParseTests(unittest.TestCase):
    def test_rejects_bad_urls(self):
        self.assertFalse(_ok_image_url(""))
        self.assertFalse(_ok_image_url("file:///etc/passwd"))
        self.assertFalse(_ok_image_url("https://localhost/x.jpg"))
        self.assertFalse(_ok_image_url("https://example.com/a.svg"))
        self.assertFalse(_ok_image_url("https://example.com/clip.mp4"))
        self.assertTrue(_ok_image_url("https://upload.wikimedia.org/foo.jpg"))
        self.assertFalse(_ok_image_url("http://127.0.0.2/x.jpg"))
        self.assertFalse(_ok_image_url("http://10.0.0.5/x.jpg"))
        self.assertFalse(_ok_image_url("http://192.168.1.10/x.jpg"))
        self.assertFalse(_ok_image_url("http://2130706433/x.jpg"))
        self.assertTrue(_ok_image_url("https://api.openverse.org/v1/images/photo.jpg"))

    def test_import_web_image_refuses_redirect_to_private(self):
        fetched = []

        def fake_get(url, **kwargs):
            fetched.append(url)
            self.assertFalse(kwargs.get("follow_redirects"))
            if "10.0.0.5" in url or "127.0.0." in url:
                self.fail("fetched private url: " + url)
            resp = MagicMock()
            resp.status_code = 302
            resp.headers = {"location": "http://10.0.0.5/x.jpg"}
            resp.content = b"P" * 200
            resp.url = url
            return resp

        with tempfile.TemporaryDirectory() as d:
            tmp_root = Path(d)
            with patch("search_images.httpx.get", side_effect=fake_get):
                with patch("search_images.TEMP_DIR", tmp_root):
                    with patch("search_images.still_from_image") as still:
                        ok, msg, path = import_web_image(
                            "https://upload.wikimedia.org/wikipedia/commons/foo.jpg",
                            "ssrf",
                        )
            self.assertFalse(ok)
            self.assertIsNone(path)
            self.assertTrue(msg)
            still.assert_not_called()
            self.assertEqual(len(fetched), 1)
            leftover = list(tmp_root.glob("*"))
            self.assertEqual(leftover, [])

    def test_guess_ext(self):
        self.assertEqual(_guess_ext("https://x.com/a.PNG", ""), ".png")
        self.assertEqual(_guess_ext("https://x.com/a", "image/jpeg"), ".jpg")
        self.assertEqual(_guess_ext("https://x.com/a", "text/html"), ".jpg")

    def test_commons_fixture(self):
        data = {
            "query": {
                "pages": {
                    "1": {
                        "pageid": 1,
                        "index": 1,
                        "title": "File:Dirty river.jpg",
                        "imageinfo": [{
                            "url": "https://upload.wikimedia.org/wikipedia/commons/r.jpg",
                            "thumburl": "https://upload.wikimedia.org/wikipedia/commons/thumb/r.jpg",
                            "mime": "image/jpeg",
                        }],
                    },
                    "2": {
                        "pageid": 2,
                        "index": 2,
                        "title": "File:clip.webm",
                        "imageinfo": [{"url": "https://upload.wikimedia.org/c.webm", "mime": "video/webm"}],
                    },
                }
            }
        }
        hits = parse_commons_query(data, 8)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["kind"], "image")
        self.assertIn("r.jpg", hits[0]["url"])
        self.assertEqual(hits[0]["platform"], "image")

    def test_openverse_fixture(self):
        data = {
            "results": [
                {"id": "a1", "title": "River", "url": "https://cdn.example.com/river.jpg", "thumbnail": "https://cdn.example.com/t.jpg"},
                {"id": "bad", "title": "Nope", "url": "javascript:alert(1)"},
            ]
        }
        hits = parse_openverse(data, 8)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["title"], "River")

    def test_query_passes_keeps_words(self):
        passes = query_passes("río contaminado")
        self.assertEqual(passes[0], "río contaminado")
        self.assertIn("rio contaminado", [p.lower() for p in passes])
        self.assertEqual(query_passes("factory smoke"), ["factory smoke"])

    def test_pack_commons_only(self):
        packed = pack_image_search(
            commons_hits=[{"url": "https://upload.wikimedia.org/x.jpg"}],
        )
        self.assertEqual(packed["source"], "commons")
        self.assertTrue(packed["results"])
        self.assertFalse(packed["hard_fail"])

    def test_pack_both_sources(self):
        packed = pack_image_search(
            commons_hits=[{"url": "https://upload.wikimedia.org/x.jpg"}],
            openverse_hits=[{"url": "https://cdn.example.com/y.jpg"}],
        )
        self.assertEqual(packed["source"], "commons+openverse")
        self.assertEqual(len(packed["results"]), 2)

    def test_pack_both_hard_fail(self):
        packed = pack_image_search(
            commons_error="HTTP 503",
            openverse_error="HTTP 502",
        )
        self.assertEqual(packed["results"], [])
        self.assertTrue(packed["hard_fail"])
        self.assertIn("503", packed["error"])
        self.assertNotIn("GOOGLE", packed["error"])

    def test_pack_empty_ok_is_not_hard_fail(self):
        packed = pack_image_search()
        self.assertEqual(packed["results"], [])
        self.assertFalse(packed["hard_fail"])
        self.assertTrue(packed["error"])

    def test_merge_dedupes(self):
        a = [{"url": "https://a.com/1.jpg", "title": "one"}]
        b = [{"url": "https://a.com/1.jpg?x=1", "title": "dup"}, {"url": "https://b.com/2.jpg", "title": "two"}]
        merged = _merge([a, b], 8)
        self.assertEqual(len(merged), 2)

    def test_empty_payloads(self):
        self.assertEqual(parse_commons_query(None), [])
        self.assertEqual(parse_openverse("nope"), [])


class ImageDirectorTests(unittest.TestCase):
    def test_query_extract(self):
        self.assertEqual(_image_query_from_msg("busca imágenes de ríos sucios"), "ríos sucios")
        self.assertIsNone(_image_query_from_msg("look film en todas"))
        self.assertIsNone(_image_query_from_msg("usa cada clip entero"))

    def test_director_returns_search_images(self):
        out = apply_director_command(
            "busca imágenes de humo industrial",
            {"title": "Río", "scenes": []},
            [],
        )
        self.assertIsNotNone(out)
        self.assertEqual(out["search_images"]["query"], "humo industrial")
        self.assertIsNone(out["plan"])

    def test_image_search_does_not_steal_grade(self):
        clips = lib("a.mp4")
        plan = {"scenes": [{"clip": "a.mp4", "inPoint": 0, "outPoint": 4}]}
        out = apply_director_command("escena 1 más cálida", plan, clips)
        self.assertNotIn("search_images", out or {})
        self.assertTrue(out["plan"]["scenes"][0]["grade"]["temp"] > 0)

    def test_reply_helper_empty_subject_uses_title(self):
        out = _image_search_reply("busca imágenes", {"title": "La Villa"})
        self.assertEqual(out["search_images"]["query"], "La Villa")


class SequenceTests(unittest.TestCase):
    def setUp(self):
        self.clips = lib(("rio.mp4", 12), ("humo.mp4", 8))
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_filename_from_src(self):
        self.assertEqual(filename_from_src("/clips/rio.mp4"), "rio.mp4")
        self.assertEqual(filename_from_src("/uploads/a.mp4?x=1"), "a.mp4")

    def test_migrate_editor_key(self):
        raw = migrate_legacy({
            "savedAt": 100,
            "name": "Corte",
            "clips": [{
                "src": "/clips/rio.mp4",
                "name": "rio.mp4",
                "inPoint": 2,
                "outPoint": 6,
                "speed": 1,
                "grade": {"temp": 0.3, "sat": 1, "lift": 0, "gamma": 1, "gain": 1, "contrast": 1},
                "crop": {"x": 0.1, "y": 0.1, "w": 0.6, "h": 0.6},
            }],
            "texts": [{"content": "Hola", "size": 40, "position": "center", "start": 0, "end": 2}],
            "markers": [{"t": 1.5, "label": "A"}],
        })
        self.assertEqual(len(raw["scenes"]), 1)
        self.assertEqual(raw["scenes"][0]["inPoint"], 2)
        self.assertEqual(raw["scenes"][0]["outPoint"], 6)
        seq = sanitize_sequence(raw, library=self.clips)
        self.assertEqual(seq["version"], SEQUENCE_VERSION)
        self.assertEqual(seq["scenes"][0]["clip"], "rio.mp4")
        self.assertEqual(seq["scenes"][0]["inPoint"], 2)
        self.assertAlmostEqual(seq["scenes"][0]["grade"]["temp"], 0.3)
        self.assertEqual(seq["scenes"][0]["crop"]["w"], 0.6)
        self.assertEqual(seq["markers"][0]["t"], 1.5)
        self.assertEqual(seq["texts"][0]["content"], "Hola")

    def test_migrate_assemble_key(self):
        raw = migrate_legacy({
            "savedAt": 200,
            "plan": {
                "title": "Docu",
                "format": "documentary",
                "scenes": [
                    {"clip": "rio.mp4", "inPoint": 0, "outPoint": 3, "text": "Inicio"},
                    {"clip": "humo.mp4", "inPoint": 1, "outPoint": 5, "speed": 1},
                ],
            },
            "captions": {"on": True, "style": "bold"},
        })
        seq = sanitize_sequence(raw, library=self.clips)
        self.assertEqual(len(seq["scenes"]), 2)
        self.assertEqual(seq["scenes"][0]["text"], "Inicio")
        self.assertEqual(seq["ui"]["captions"]["style"], "bold")

    def test_trim_persists_other_scene_untouched_stale_rev_keeps_disk(self):
        seq, src = save_sequence(
            {
                "title": "Corte",
                "rev": 1,
                "scenes": [
                    {"id": "s-a", "clip": "rio.mp4", "inPoint": 0, "outPoint": 8, "text": "uno"},
                    {"id": "s-b", "clip": "humo.mp4", "inPoint": 1, "outPoint": 5, "text": "dos"},
                ],
            },
            folder=self.folder,
            library=self.clips,
            force=True,
        )
        self.assertEqual(src, "sequence.json")
        self.assertEqual(seq["scenes"][0]["inPoint"], 0)
        self.assertEqual(seq["scenes"][0]["outPoint"], 8)
        trimmed = json.loads(json.dumps(seq))
        trimmed["scenes"][0]["inPoint"] = 1.5
        trimmed["scenes"][0]["outPoint"] = 4.0
        saved, src = save_sequence(trimmed, folder=self.folder, library=self.clips, force=False)
        self.assertEqual(src, "sequence.json")
        self.assertEqual(saved["scenes"][0]["id"], "s-a")
        self.assertEqual(saved["scenes"][0]["clip"], "rio.mp4")
        self.assertEqual(saved["scenes"][0]["inPoint"], 1.5)
        self.assertEqual(saved["scenes"][0]["outPoint"], 4.0)
        self.assertEqual(saved["scenes"][0]["text"], "uno")
        self.assertEqual(saved["scenes"][1]["id"], "s-b")
        self.assertEqual(saved["scenes"][1]["clip"], "humo.mp4")
        self.assertEqual(saved["scenes"][1]["inPoint"], 1)
        self.assertEqual(saved["scenes"][1]["outPoint"], 5)
        self.assertEqual(saved["scenes"][1]["text"], "dos")
        self.assertNotIn("replaceAudio", saved["scenes"][0])
        disk = (self.folder / "sequence.json").read_bytes()
        stale = json.loads(json.dumps(saved))
        stale["rev"] = seq["rev"]
        stale["scenes"][0]["inPoint"] = 0
        stale["scenes"][0]["outPoint"] = 1
        stale["scenes"][1]["clip"] = "wiped.mp4"
        stale["scenes"][1]["text"] = "no"
        kept, src = save_sequence(stale, folder=self.folder, library=self.clips, force=False)
        self.assertEqual(src, "stale")
        self.assertEqual((self.folder / "sequence.json").read_bytes(), disk)
        self.assertEqual(kept["scenes"][0]["inPoint"], 1.5)
        self.assertEqual(kept["scenes"][0]["outPoint"], 4.0)
        self.assertEqual(kept["scenes"][1]["clip"], "humo.mp4")
        self.assertEqual(kept["scenes"][1]["inPoint"], 1)
        self.assertEqual(kept["scenes"][1]["outPoint"], 5)
        self.assertEqual(kept["scenes"][1]["text"], "dos")

    def test_save_load_roundtrip(self):
        seq, src = save_sequence(
            {"title": "T", "scenes": [
                {"clip": "rio.mp4", "inPoint": 1, "outPoint": 4, "speed": 2},
            ]},
            folder=self.folder,
            library=self.clips,
            force=True,
        )
        self.assertEqual(src, "sequence.json")
        self.assertTrue((self.folder / "sequence.json").is_file())
        loaded, source = load_sequence(self.folder, library=self.clips)
        self.assertEqual(source, "sequence.json")
        self.assertEqual(loaded["scenes"][0]["inPoint"], 1)
        self.assertEqual(loaded["scenes"][0]["outPoint"], 4)
        self.assertEqual(loaded["scenes"][0]["speed"], 2)

    def test_fit_survives_save_load_roundtrip(self):
        seq, _src = save_sequence(
            {"title": "T", "scenes": [
                {"clip": "rio.mp4", "inPoint": 0, "outPoint": 3, "fit": "cover"},
                {"clip": "humo.mp4", "inPoint": 0, "outPoint": 2},
            ]},
            folder=self.folder,
            library=self.clips,
            force=True,
        )
        self.assertEqual(seq["scenes"][0]["fit"], "cover")
        self.assertEqual(seq["scenes"][1]["fit"], "contain")
        loaded, _ = load_sequence(self.folder, library=self.clips)
        self.assertEqual(loaded["scenes"][0]["fit"], "cover")
        self.assertEqual(loaded["scenes"][1]["fit"], "contain")

    def test_save_writes_bak(self):
        save_sequence(
            {"scenes": [{"clip": "rio.mp4", "inPoint": 0, "outPoint": 2}]},
            folder=self.folder, library=self.clips, force=True,
        )
        save_sequence(
            {"scenes": [{"clip": "humo.mp4", "inPoint": 0, "outPoint": 3}]},
            folder=self.folder, library=self.clips, force=True,
        )
        bak = json.loads((self.folder / "sequence.bak.json").read_text(encoding="utf-8"))
        self.assertEqual(bak["scenes"][0]["clip"], "rio.mp4")
        cur = json.loads((self.folder / "sequence.json").read_text(encoding="utf-8"))
        self.assertEqual(cur["scenes"][0]["clip"], "humo.mp4")

    def test_empty_does_not_wipe(self):
        save_sequence(
            {"scenes": [{"clip": "rio.mp4", "inPoint": 0, "outPoint": 5}]},
            folder=self.folder, library=self.clips, force=True,
        )
        seq, src = save_sequence({"scenes": []}, folder=self.folder, library=self.clips, force=False)
        self.assertEqual(src, "protected")
        self.assertEqual(seq["scenes"][0]["clip"], "rio.mp4")

    def test_rev_stamps_and_rejects_stale_tab(self):
        first, src = save_sequence(
            {"script": "casas.\n[pausa 3s]\nEL RIO", "scenes": [
                {"clip": "rio.mp4", "inPoint": 0, "outPoint": 5},
            ]},
            folder=self.folder, library=self.clips, force=True,
        )
        self.assertEqual(src, "sequence.json")
        self.assertEqual(first["rev"], 1)
        stale, src = save_sequence(
            {"rev": 0, "script": "casas. EL RIO", "scenes": [
                {"clip": "rio.mp4", "inPoint": 0, "outPoint": 5},
            ]},
            folder=self.folder, library=self.clips, force=False,
        )
        self.assertEqual(src, "stale")
        self.assertIn("[pausa 3s]", stale["script"])
        self.assertEqual(stale["rev"], 1)
        ok, src = save_sequence(
            {"rev": 1, "script": "casas.\n[pausa 3s]\nEL RIO NACE", "scenes": [
                {"clip": "rio.mp4", "inPoint": 0, "outPoint": 5},
            ]},
            folder=self.folder, library=self.clips, force=False,
        )
        self.assertEqual(src, "sequence.json")
        self.assertEqual(ok["rev"], 2)
        self.assertIn("NACE", ok["script"])

    def test_existing_id_survives_save_load_roundtrip(self):
        seq, src = save_sequence(
            {"scenes": [
                {"clip": "rio.mp4", "id": "keep-this-id", "inPoint": 0, "outPoint": 3},
            ]},
            folder=self.folder,
            library=self.clips,
            force=True,
        )
        self.assertEqual(src, "sequence.json")
        self.assertEqual(seq["scenes"][0]["id"], "keep-this-id")
        loaded, _ = load_sequence(self.folder, library=self.clips)
        self.assertEqual(loaded["scenes"][0]["id"], "keep-this-id")

    def _nine_scenes(self):
        return [
            {"clip": "rio.mp4", "inPoint": 0, "outPoint": 2, "id": "s%02d" % i}
            for i in range(9)
        ]

    def test_save_sequence_shrink_9_to_2_without_force(self):
        first, _ = save_sequence(
            {"scenes": self._nine_scenes()},
            folder=self.folder, library=self.clips, force=True,
        )
        path = self.folder / "sequence.json"
        before = path.read_bytes()
        self.assertEqual(SHRINK_MIN_SCENES, 4)
        self.assertEqual(SHRINK_RATIO, 0.5)
        out, src = save_sequence(
            {"rev": first["rev"], "scenes": [
                {"clip": "rio.mp4", "inPoint": 0, "outPoint": 2, "id": "s00"},
                {"clip": "humo.mp4", "inPoint": 0, "outPoint": 2, "id": "s01"},
            ]},
            folder=self.folder, library=self.clips, force=False,
        )
        self.assertEqual(src, "shrink")
        self.assertEqual(len(out["scenes"]), 9)
        self.assertEqual(path.read_bytes(), before)

    def test_save_sequence_allows_9_to_2_with_force(self):
        first, _ = save_sequence(
            {"scenes": self._nine_scenes()},
            folder=self.folder, library=self.clips, force=True,
        )
        out, src = save_sequence(
            {"rev": first["rev"], "scenes": [
                {"clip": "rio.mp4", "inPoint": 0, "outPoint": 2, "id": "s00"},
                {"clip": "humo.mp4", "inPoint": 0, "outPoint": 2, "id": "s01"},
            ]},
            folder=self.folder, library=self.clips, force=True,
        )
        self.assertEqual(src, "sequence.json")
        self.assertEqual(len(out["scenes"]), 2)

    def test_save_sequence_allows_9_to_8_without_force(self):
        first, _ = save_sequence(
            {"scenes": self._nine_scenes()},
            folder=self.folder, library=self.clips, force=True,
        )
        eight = self._nine_scenes()[:8]
        out, src = save_sequence(
            {"rev": first["rev"], "scenes": eight},
            folder=self.folder, library=self.clips, force=False,
        )
        self.assertEqual(src, "sequence.json")
        self.assertEqual(len(out["scenes"]), 8)

    def test_save_sequence_allows_3_to_1_below_min(self):
        first, _ = save_sequence(
            {"scenes": [
                {"clip": "rio.mp4", "inPoint": 0, "outPoint": 2, "id": "a"},
                {"clip": "humo.mp4", "inPoint": 0, "outPoint": 2, "id": "b"},
                {"clip": "rio.mp4", "inPoint": 1, "outPoint": 3, "id": "c"},
            ]},
            folder=self.folder, library=self.clips, force=True,
        )
        out, src = save_sequence(
            {"rev": first["rev"], "scenes": [
                {"clip": "rio.mp4", "inPoint": 0, "outPoint": 2, "id": "a"},
            ]},
            folder=self.folder, library=self.clips, force=False,
        )
        self.assertEqual(src, "sequence.json")
        self.assertEqual(len(out["scenes"]), 1)

    def test_api_sequence_returns_shrink_from_to(self):
        from fastapi.testclient import TestClient
        from main import app
        first, _ = save_sequence(
            {"scenes": self._nine_scenes()},
            folder=self.folder, library=self.clips, force=True,
        )

        def _save(raw, force=False, folder=None, library=None, **_k):
            return save_sequence(
                raw, folder=self.folder, library=self.clips, force=force,
            )

        with patch("main.save_sequence", _save):
            client = TestClient(app)
            res = client.post("/api/sequence", json={
                "rev": first["rev"],
                "scenes": [
                    {"clip": "rio.mp4", "inPoint": 0, "outPoint": 2, "id": "s00"},
                    {"clip": "humo.mp4", "inPoint": 0, "outPoint": 2, "id": "s01"},
                ],
            })
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()
        self.assertFalse(data.get("ok"))
        self.assertTrue(data.get("shrink"))
        self.assertEqual(data.get("from"), 9)
        self.assertEqual(data.get("to"), 2)
        self.assertEqual(data.get("source"), "shrink")
        self.assertEqual(len((data.get("sequence") or {}).get("scenes") or []), 9)

    def test_force_overrides_stale_rev(self):
        save_sequence(
            {"script": "old", "scenes": [{"clip": "rio.mp4", "inPoint": 0, "outPoint": 2}]},
            folder=self.folder, library=self.clips, force=True,
        )
        seq, src = save_sequence(
            {"rev": 0, "script": "new", "scenes": [{"clip": "humo.mp4", "inPoint": 0, "outPoint": 3}]},
            folder=self.folder, library=self.clips, force=True,
        )
        self.assertEqual(src, "sequence.json")
        self.assertEqual(seq["script"], "new")
        self.assertEqual(seq["rev"], 2)

    def test_save_sequence_lock_blocks_concurrent_writer(self):
        import video_create
        lock = getattr(video_create, "_save_sequence_lock", None)
        self.assertIsNotNone(lock)
        self.assertTrue(hasattr(lock, "acquire") and hasattr(lock, "release"))
        save_sequence(
            {"scenes": [{"clip": "rio.mp4", "inPoint": 0, "outPoint": 3}]},
            folder=self.folder, library=self.clips, force=True,
        )
        started = threading.Event()
        finished = threading.Event()

        def other():
            started.set()
            save_sequence(
                {"scenes": [{"clip": "humo.mp4", "inPoint": 0, "outPoint": 3}]},
                folder=self.folder, library=self.clips, force=True,
            )
            finished.set()

        lock.acquire()
        try:
            t = threading.Thread(target=other)
            t.start()
            self.assertTrue(started.wait(1.0))
            time.sleep(0.15)
            self.assertFalse(finished.is_set())
        finally:
            lock.release()
        t.join(2.0)
        self.assertTrue(finished.is_set())
        loaded, _ = load_sequence(self.folder, library=self.clips)
        self.assertEqual(loaded["scenes"][0]["clip"], "humo.mp4")

    def test_save_sequence_writes_via_tmp_then_replace(self):
        orig_write = Path.write_text
        orig_replace = Path.replace
        writes = []
        replaces = []

        def spy_write(self, data, encoding=None, errors=None, newline=None):
            writes.append(self.name)
            return orig_write(self, data, encoding=encoding, errors=errors)

        def spy_replace(self, target):
            replaces.append((self.name, Path(target).name if not isinstance(target, str) else Path(target).name))
            return orig_replace(self, target)

        with patch.object(Path, "write_text", spy_write):
            with patch.object(Path, "replace", spy_replace):
                save_sequence(
                    {"scenes": [{"clip": "rio.mp4", "inPoint": 0, "outPoint": 4}]},
                    folder=self.folder, library=self.clips, force=True,
                )
        self.assertTrue(any(n.endswith(".tmp") for n in writes))
        self.assertFalse(any(n == "sequence.json" for n in writes))
        self.assertTrue(any(dst == "sequence.json" for _src, dst in replaces))
        seq_path = self.folder / "sequence.json"
        self.assertTrue(seq_path.is_file())
        data = json.loads(seq_path.read_text(encoding="utf-8"))
        self.assertEqual(data["scenes"][0]["clip"], "rio.mp4")
        self.assertFalse((self.folder / "sequence.json.tmp").exists())

    def test_load_migrates_old_assemble_file(self):
        payload = {
            "savedAt": 9,
            "plan": {"title": "Old", "scenes": [
                {"clip": "rio.mp4", "inPoint": 2, "outPoint": 7},
            ]},
        }
        (self.folder / "video-creation.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        seq, source = load_sequence(self.folder, library=self.clips)
        self.assertTrue(source.startswith("migrated:"))
        self.assertEqual(seq["scenes"][0]["inPoint"], 2)
        self.assertEqual(seq["scenes"][0]["outPoint"], 7)

    def test_every_write_goes_through_sanitize(self):
        seq = sanitize_sequence(
            {"scenes": [{"clip": "rio.mp4", "grade": {"sat": 99}, "inPoint": 0, "outPoint": 3}]},
            library=self.clips,
        )
        self.assertEqual(seq["scenes"][0]["grade"]["sat"], 3.0)

    def test_roundtrip_keeps_capoff_trim_cues_and_vo(self):
        scenes = [
            {"clip": "rio.mp4", "inPoint": 0, "outPoint": 4, "capOff": False}
            for _ in range(7)
        ]
        scenes.append({
            "clip": "rio.mp4",
            "inPoint": 0,
            "outPoint": 1.5,
            "capOff": True,
            "voRate": 1.0,
        })
        payload = {
            "plan": {
                "title": "Docu",
                "script": "guion completo del río",
                "scenes": scenes,
            },
            "captions": {
                "on": True,
                "source": "narration",
                "style": "modern",
                "cues": [{"start": 0, "end": 2.4, "text": "el río baja turbio"}],
            },
            "vo": {
                "url": SEQUENCE_VO_URL,
                "sig": "full\x1fguion completo del río\x1fdocumentary",
                "duration": 9.25,
                "follow": True,
            },
            "currentTime": 4.2,
        }
        src = Path(self.folder) / "unit_vo_src.mp3"
        src.write_bytes(b"\xff" * 400)
        stored = persist_sequence_vo(src, folder=self.folder)
        self.assertIsNotNone(stored)
        self.assertEqual(stored, sequence_vo_path(self.folder))
        self.assertGreater(stored.stat().st_size, 200)
        dumped = json.dumps(payload)
        self.assertNotIn("\xff" * 20, dumped)

        with patch("tts_audio._edge_synthesize") as mock_edge:
            seq, src_name = save_sequence(
                payload, folder=self.folder, library=self.clips, force=True
            )
            loaded, source = load_sequence(self.folder, library=self.clips)
            mock_edge.assert_not_called()
        self.assertEqual(src_name, "sequence.json")
        self.assertEqual(source, "sequence.json")
        self.assertTrue(loaded["scenes"][7]["capOff"])
        self.assertEqual(loaded["scenes"][7]["outPoint"], 1.5)
        cues = (loaded.get("ui") or {}).get("captions") or {}
        self.assertEqual(cues.get("source"), "narration")
        self.assertEqual(cues.get("style"), "modern")
        self.assertEqual(cues["cues"][0]["text"], "el río baja turbio")
        vo = (loaded.get("ui") or {}).get("vo") or {}
        self.assertEqual(vo.get("url"), SEQUENCE_VO_URL)
        self.assertIn("guion completo del río", vo.get("sig") or "")
        self.assertAlmostEqual(vo.get("duration") or 0, 9.25)
        self.assertTrue(vo.get("follow"))
        self.assertEqual(loaded["ui"].get("currentTime"), 4.2)

        flipped = json.loads(json.dumps(payload))
        flipped["plan"]["scenes"][7]["capOff"] = False
        seq2 = sanitize_sequence(flipped, library=self.clips)
        self.assertEqual(
            seq2["ui"]["vo"]["sig"],
            loaded["ui"]["vo"]["sig"],
        )
        self.assertFalse(seq2["scenes"][7]["capOff"])

    def test_assemble_lock_post_get_and_edit_omits_capoff(self):
        scenes = [
            {"clip": "rio.mp4", "inPoint": 0, "outPoint": 4, "capOff": False}
            for _ in range(8)
        ]
        scenes[1]["outPoint"] = 1.2
        scenes[7]["capOff"] = True
        sig = "full\x1fguion B editado\x1fdocumentary"
        payload = {
            "plan": {
                "title": "Docu",
                "script": "guion B editado",
                "scenes": scenes,
            },
            "captions": {
                "on": True,
                "source": "narration",
                "style": "modern",
                "cues": [{"start": 0, "end": 1.5, "text": "el río"}],
            },
            "vo": {
                "url": SEQUENCE_VO_URL,
                "sig": sig,
                "duration": 8.0,
                "follow": True,
            },
        }
        save_sequence(payload, folder=self.folder, library=self.clips, force=True)
        loaded, source = load_sequence(self.folder, library=self.clips)
        self.assertEqual(source, "sequence.json")
        self.assertTrue(loaded["scenes"][7]["capOff"])
        self.assertEqual(loaded["script"].strip(), "guion B editado")
        self.assertEqual(loaded["scenes"][1]["outPoint"], 1.2)
        self.assertEqual(loaded["ui"]["vo"]["sig"], sig)

        clips = []
        for s in loaded["scenes"]:
            clips.append({
                "src": "/clips/" + s["clip"],
                "name": s["clip"],
                "inPoint": s["inPoint"],
                "outPoint": s["outPoint"],
                "speed": s.get("speed") or 1,
            })
        save_sequence(
            {"name": "Corte", "clips": clips},
            folder=self.folder,
            library=self.clips,
            force=True,
        )
        loaded2, _ = load_sequence(self.folder, library=self.clips)
        self.assertTrue(loaded2["scenes"][7]["capOff"])
        self.assertEqual(loaded2["script"].strip(), "guion B editado")
        self.assertEqual(loaded2["scenes"][1]["outPoint"], 1.2)
        self.assertEqual(loaded2["ui"]["vo"]["sig"], sig)
        self.assertEqual(loaded2["ui"]["captions"]["cues"][0]["text"], "el río")

    def test_capoff_survives_caption_retime_save(self):
        scenes = [
            {"clip": "rio.mp4", "inPoint": 0, "outPoint": 4, "capOff": False},
            {"clip": "humo.mp4", "inPoint": 0, "outPoint": 4, "capOff": True},
            {"clip": "rio.mp4", "inPoint": 0, "outPoint": 4, "capOff": False},
        ]
        payload = {
            "plan": {"title": "T", "script": "guion", "scenes": scenes},
            "captions": {
                "on": True,
                "source": "narration",
                "cues": [{"start": 0, "end": 12, "text": "todo el río"}],
            },
        }
        save_sequence(payload, folder=self.folder, library=self.clips, force=True)
        loaded, _ = load_sequence(self.folder, library=self.clips)
        self.assertTrue(loaded["scenes"][1]["capOff"])
        self.assertFalse(loaded["scenes"][0]["capOff"])
        payload2 = {
            "plan": {
                "title": "T",
                "script": "guion",
                "scenes": loaded["scenes"],
            },
            "captions": {
                "on": True,
                "source": "narration",
                "cues": [
                    {"start": 0, "end": 3.5, "text": "el"},
                    {"start": 3.5, "end": 8.0, "text": "río"},
                    {"start": 8.0, "end": 12, "text": "baja"},
                ],
            },
        }
        save_sequence(payload2, folder=self.folder, library=self.clips, force=True)
        loaded2, _ = load_sequence(self.folder, library=self.clips)
        self.assertTrue(loaded2["scenes"][1]["capOff"])
        self.assertFalse(loaded2["scenes"][0]["capOff"])
        self.assertFalse(loaded2["scenes"][2]["capOff"])
        hole_a, hole_b = _scene_span(loaded2["scenes"][1])
        texts = caption_export_texts(
            loaded2["ui"]["captions"]["cues"], loaded2["scenes"], "lower-third"
        )
        self.assertTrue(texts)
        for row in texts:
            mid = (row["start"] + row["end"]) / 2.0
            self.assertFalse(hole_a <= mid < hole_b)
            self.assertTrue(row["end"] <= hole_a + 0.001 or row["start"] >= hole_b - 0.001)

    def test_vo_follow_defaults_on_when_sig_present(self):
        seq = sanitize_sequence(
            {
                "plan": {
                    "title": "T",
                    "script": "guion",
                    "scenes": [{"clip": "rio.mp4", "inPoint": 0, "outPoint": 3}],
                },
                "vo": {"url": SEQUENCE_VO_URL, "sig": "abc", "duration": 4.0},
            },
            library=self.clips,
        )
        self.assertTrue(seq["ui"]["vo"]["follow"])
        seq_off = sanitize_sequence(
            {
                "plan": {
                    "title": "T",
                    "script": "guion",
                    "scenes": [{"clip": "rio.mp4", "inPoint": 0, "outPoint": 3}],
                },
                "vo": {
                    "url": SEQUENCE_VO_URL,
                    "sig": "abc",
                    "duration": 4.0,
                    "follow": False,
                },
            },
            library=self.clips,
        )
        self.assertFalse(seq_off["ui"]["vo"]["follow"])

    def test_vo_follow_defaults_on_when_guion_or_file(self):
        seq = sanitize_sequence(
            {
                "plan": {
                    "title": "T",
                    "script": "el río baja turbio",
                    "scenes": [{"clip": "rio.mp4", "inPoint": 0, "outPoint": 3}],
                },
                "vo": {"url": SEQUENCE_VO_URL},
            },
            library=self.clips,
        )
        self.assertTrue(seq["ui"]["vo"]["follow"])
        src = self.folder / "unit_vo_src.mp3"
        src.write_bytes(b"\xff" * 400)
        stored = persist_sequence_vo(src, folder=self.folder)
        self.assertIsNotNone(stored)
        payload = {
            "plan": {
                "title": "T",
                "script": "",
                "scenes": [{"clip": "rio.mp4", "inPoint": 0, "outPoint": 3}],
            },
            "vo": {"url": SEQUENCE_VO_URL, "sig": "", "duration": 0},
        }
        save_sequence(payload, folder=self.folder, library=self.clips, force=True)
        loaded, _ = load_sequence(self.folder, library=self.clips)
        self.assertTrue(loaded["ui"]["vo"]["follow"])

    def test_resolve_vo_source_uses_autosave_file(self):
        src = self.folder / "unit_vo_src.mp3"
        src.write_bytes(b"\xff" * 400)
        stored = persist_sequence_vo(src, folder=self.folder)
        self.assertIsNotNone(stored)
        found = resolve_vo_source(SEQUENCE_VO_URL, folder=self.folder)
        self.assertEqual(found, sequence_vo_path(self.folder))
        found2 = resolve_vo_source("/api/sequence/vo?t=1", folder=self.folder)
        self.assertEqual(found2, sequence_vo_path(self.folder))
        self.assertIsNone(resolve_vo_source("/uploads/missing_vo.mp3", folder=self.folder))

    def test_sanitize_keeps_narration_cues_and_source(self):
        payload = {
            "plan": {
                "title": "T",
                "script": "el río baja turbio",
                "scenes": [
                    {"clip": "rio.mp4", "inPoint": 0, "outPoint": 3, "capOff": True},
                    {"clip": "humo.mp4", "inPoint": 0, "outPoint": 3, "capOff": False},
                ],
            },
            "captions": {
                "on": False,
                "source": "narration",
                "cues": [
                    {"start": 0.12, "end": 1.8, "text": "el río baja"},
                    {"start": 1.8, "end": 3.4, "text": "turbio"},
                ],
            },
            "vo": {
                "url": SEQUENCE_VO_URL,
                "sig": "full\x1fel río baja turbio",
                "duration": 3.4,
                "follow": True,
            },
        }
        save_sequence(payload, folder=self.folder, library=self.clips, force=True)
        loaded, _ = load_sequence(self.folder, library=self.clips)
        self.assertTrue(loaded["scenes"][0]["capOff"])
        self.assertFalse(loaded["scenes"][1]["capOff"])
        cap = loaded["ui"]["captions"]
        self.assertEqual(cap["source"], "narration")
        self.assertFalse(cap["on"])
        self.assertAlmostEqual(cap["cues"][0]["start"], 0.12)
        self.assertEqual(cap["cues"][0]["text"], "el río baja")
        self.assertEqual(loaded["ui"]["vo"]["url"], SEQUENCE_VO_URL)

    def test_vo_follow_uncheck_survives_empty_sig_merge(self):
        payload = {
            "plan": {
                "title": "T",
                "script": "guion",
                "scenes": [{"clip": "rio.mp4", "inPoint": 0, "outPoint": 3}],
            },
            "vo": {
                "url": SEQUENCE_VO_URL,
                "sig": "full\x1fguion",
                "duration": 4.0,
                "follow": True,
            },
        }
        save_sequence(payload, folder=self.folder, library=self.clips, force=True)
        payload2 = {
            "plan": payload["plan"],
            "vo": {
                "url": SEQUENCE_VO_URL,
                "sig": "",
                "duration": 4.0,
                "follow": False,
            },
        }
        save_sequence(payload2, folder=self.folder, library=self.clips, force=True)
        loaded, _ = load_sequence(self.folder, library=self.clips)
        self.assertFalse(loaded["ui"]["vo"]["follow"])
        self.assertIn("guion", loaded["ui"]["vo"]["sig"])

    def test_ui_music_sanitize_and_persist(self):
        src = self.folder / "unit_song.mp3"
        src.write_bytes(b"\xff" * 400)
        stored = persist_sequence_music(src, folder=self.folder)
        self.assertIsNotNone(stored)
        self.assertEqual(stored, sequence_music_path(self.folder))
        self.assertGreater(stored.stat().st_size, 200)
        payload = {
            "plan": {
                "title": "T",
                "script": "guion",
                "scenes": [{"clip": "rio.mp4", "inPoint": 0, "outPoint": 3}],
            },
            "music": {
                "url": SEQUENCE_MUSIC_URL,
                "name": "tema.wav",
                "volume": 0.25,
                "mute": False,
            },
        }
        dumped = json.dumps(payload)
        self.assertNotIn("\xff" * 20, dumped)
        save_sequence(payload, folder=self.folder, library=self.clips, force=True)
        loaded, _ = load_sequence(self.folder, library=self.clips)
        music = loaded["ui"]["music"]
        self.assertEqual(music["url"], SEQUENCE_MUSIC_URL)
        self.assertEqual(music["name"], "tema.wav")
        self.assertAlmostEqual(music["volume"], 0.25)
        self.assertFalse(music["mute"])
        seq_off = sanitize_sequence(
            {
                "plan": {
                    "title": "T",
                    "script": "guion",
                    "scenes": [{"clip": "rio.mp4", "inPoint": 0, "outPoint": 3}],
                },
                "music": {"name": "x.mp3", "volume": 1.8, "mute": 1},
            },
            library=self.clips,
            folder=self.folder,
        )
        self.assertEqual(seq_off["ui"]["music"]["url"], SEQUENCE_MUSIC_URL)
        self.assertEqual(seq_off["ui"]["music"]["volume"], 1.0)
        self.assertTrue(seq_off["ui"]["music"]["mute"])
        payload2 = {
            "plan": payload["plan"],
            "music": {"url": SEQUENCE_MUSIC_URL, "name": "", "volume": 0.25, "mute": False},
        }
        clear_sequence_music(folder=self.folder)
        save_sequence(payload2, folder=self.folder, library=self.clips, force=True)
        loaded2, _ = load_sequence(self.folder, library=self.clips)
        self.assertEqual(loaded2["ui"]["music"]["name"], "")
        self.assertFalse(sequence_music_path(self.folder).is_file())


class ApiRouteTests(unittest.TestCase):
    def test_image_routes_registered(self):
        from main import app
        paths = {getattr(r, "path", None) for r in app.routes}
        self.assertIn("/api/images/search", paths)
        self.assertIn("/api/images/import", paths)
        self.assertIn("/api/stock/search", paths)
        self.assertIn("/api/sequence", paths)
        self.assertIn("/api/sequence/vo", paths)
        self.assertIn("/api/sequence/music", paths)
        self.assertIn("/api/sequence/scene", paths)
        self.assertIn("/api/clips/publish", paths)
        self.assertIn("/api/clips/replace-audio", paths)
        self.assertIn("/api/clips/replace-audio/preview", paths)
        self.assertIn("/api/send-to-create", paths)
        self.assertIn("/api/clips/peaks", paths)
        self.assertIn("/api/clips/poster", paths)
        self.assertIn("/api/clips/proxy", paths)
        self.assertIn("/api/eleven/status", paths)
        self.assertIn("/api/tts/status", paths)
        self.assertIn("/api/search", paths)
        self.assertIn("/api/projects", paths)
        self.assertIn("/api/projects/new", paths)
        self.assertIn("/api/projects/switch", paths)

    def test_search_transcript_uses_to_thread(self):
        from fastapi.testclient import TestClient
        from main import app
        names = []

        async def fake_to_thread(fn, *args, **kwargs):
            names.append(getattr(fn, "__name__", ""))
            if getattr(fn, "__name__", "") == "search_footage_pack":
                return ([{
                    "video_id": "abc12345678",
                    "title": "Río",
                    "url": "https://www.youtube.com/watch?v=abc12345678",
                    "thumbnail": "",
                    "duration": "1:00",
                    "channel": "doc",
                    "platform": "youtube",
                }], [])
            return []

        with patch("main.asyncio.to_thread", side_effect=fake_to_thread):
            with patch("main.load_cookie_map", return_value={}):
                client = TestClient(app)
                res = client.post(
                    "/api/search",
                    json={"query": "rio", "platform": "youtube", "max_results": 1},
                )
        self.assertEqual(res.status_code, 200)
        self.assertIn("search_footage_pack", names)
        self.assertIn("get_transcript_with_timestamps", names)

    def test_cors_allowlist_is_local_only(self):
        from fastapi.testclient import TestClient
        from main import app
        from starlette.middleware.cors import CORSMiddleware
        found = None
        for m in app.user_middleware:
            if getattr(m, "cls", None) is CORSMiddleware:
                found = m
                break
        self.assertIsNotNone(found)
        origins = list((found.kwargs or {}).get("allow_origins") or [])
        self.assertNotIn("*", origins)
        self.assertEqual(
            set(origins),
            {"http://localhost:8000", "http://127.0.0.1:8000"},
        )
        self.assertTrue((found.kwargs or {}).get("allow_credentials"))
        client = TestClient(app)
        evil = client.options("/api/health", headers={
            "Origin": "https://evil.example",
            "Access-Control-Request-Method": "GET",
        })
        self.assertNotEqual(evil.headers.get("access-control-allow-origin"), "https://evil.example")
        local = client.options("/api/health", headers={
            "Origin": "http://127.0.0.1:8000",
            "Access-Control-Request-Method": "GET",
        })
        self.assertEqual(local.headers.get("access-control-allow-origin"), "http://127.0.0.1:8000")

    def test_uvicorn_binds_localhost(self):
        import ast
        src = Path(__file__).resolve().parent.joinpath("main.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        hosts = []
        for node in tree.body:
            if not isinstance(node, ast.If):
                continue
            for call in ast.walk(node):
                if not isinstance(call, ast.Call):
                    continue
                func = call.func
                if not (isinstance(func, ast.Attribute) and func.attr == "run"):
                    continue
                for kw in call.keywords:
                    if kw.arg == "host" and isinstance(kw.value, ast.Constant):
                        hosts.append(kw.value.value)
        self.assertEqual(hosts, ["127.0.0.1"])

    def test_search_oserror_22_still_returns_200(self):
        from fastapi.testclient import TestClient
        from main import app
        yt_hit = {
            "video_id": "dQw4w9wg",
            "title": "Río sucio",
            "url": "https://www.youtube.com/watch?v=dQw4w9wg",
            "thumbnail": "",
            "duration": "1:00",
            "channel": "doc",
            "platform": "youtube",
        }
        with patch("search_platforms.search_instagram", side_effect=OSError(22, "Invalid argument")):
            with patch("search_platforms.search_tiktok", return_value=[]):
                with patch("search_platforms.search_dailymotion", side_effect=OSError(22, "Invalid argument")):
                    with patch("search_platforms.search_commons_video", return_value=[]):
                        with patch("search_platforms.search_archive_video", return_value=[]):
                            with patch("search_platforms.search_youtube_videos", return_value=[yt_hit]):
                                with patch("main.get_transcript_with_timestamps", return_value=None):
                                    client = TestClient(app)
                                    res = client.post(
                                        "/api/search",
                                        json={"query": "rio sucio", "platform": "all", "max_results": 4},
                                    )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data.get("results"))
        self.assertEqual(data["results"][0]["title"], "Río sucio")
        self.assertTrue(data.get("warning"))
        self.assertIn("22", str(data.get("warning")) or "")

    def test_default_all_skips_tiktok_instagram(self):
        from search_platforms import search_footage_pack
        yt_hit = {
            "video_id": "abc11111111",
            "title": "Río sucio en Panamá",
            "url": "https://www.youtube.com/watch?v=abc11111111",
            "thumbnail": "",
            "duration": "2:00",
            "channel": "doc",
            "platform": "youtube",
        }
        with patch("search_platforms.search_tiktok") as tt:
            with patch("search_platforms.search_instagram") as ig:
                with patch("search_platforms.search_dailymotion", return_value=[]):
                    with patch("search_platforms.search_commons_video", return_value=[]):
                        with patch("search_platforms.search_archive_video", return_value=[]):
                            with patch("search_platforms.search_youtube_videos", return_value=[yt_hit]):
                                items, _warn = search_footage_pack("rio sucio panama", "all", 4)
        tt.assert_not_called()
        ig.assert_not_called()
        self.assertTrue(items)
        self.assertEqual(items[0]["title"], "Río sucio en Panamá")

    def test_sanitize_search_query_strips_win_chars(self):
        from search_youtube import sanitize_search_query, is_netscape_cookiefile
        self.assertEqual(sanitize_search_query('rio:sucio/file*'), "rio sucio file")
        with tempfile.TemporaryDirectory() as d:
            sqlite = Path(d) / "cookies.txt"
            sqlite.write_bytes(b"SQLite format 3\x00more")
            self.assertFalse(is_netscape_cookiefile(sqlite))
            ns = Path(d) / "ns.txt"
            ns.write_text("# Netscape HTTP Cookie File\n.instagram.com\tTRUE\t/\tTRUE\t9\tsessionid\tabc\n", encoding="utf-8")
            self.assertTrue(is_netscape_cookiefile(ns))


class DailymotionTiktokSearchTests(unittest.TestCase):
    def test_dailymotion_skips_unresolvable_no_query_title(self):
        from search_platforms import search_dailymotion
        q = "agua sucia Azuero"
        with patch("search_platforms._ydl_playlist", return_value=[]):
            with patch(
                "search_platforms._ddg_links",
                return_value=["https://www.dailymotion.com/video/xabc123"],
            ):
                with patch("search_platforms._ydl_one", return_value=None):
                    got = search_dailymotion(q, 6)
        items = got[0] if isinstance(got, tuple) else got
        self.assertEqual(items, [])
        self.assertFalse(any((it.get("title") or "") == q for it in items))

    def test_tiktok_skips_unresolvable_no_query_title(self):
        from search_platforms import search_tiktok
        q = "agua sucia Azuero"
        with patch("search_platforms._ydl_playlist", return_value=[]):
            with patch(
                "search_platforms._ddg_links",
                return_value=["https://www.tiktok.com/@x/video/1234567890123456789"],
            ):
                with patch("search_platforms._ydl_one", return_value=None):
                    got = search_tiktok(q, 6)
        items = got[0] if isinstance(got, tuple) else got
        self.assertEqual(items, [])
        self.assertFalse(any((it.get("title") or "") == q for it in items))

    def test_ddg_links_site_operator_uses_bare_domain(self):
        from search_platforms import _ddg_links
        captured = {}

        def fake_post(url, data=None, **kwargs):
            captured["q"] = (data or {}).get("q")
            resp = MagicMock()
            resp.text = ""
            return resp

        with patch("search_platforms.httpx.post", side_effect=fake_post):
            _ddg_links("agua sucia Azuero", "dailymotion.com/video", 4)
        q = captured.get("q") or ""
        self.assertTrue(q.startswith("site:dailymotion.com "), q)
        self.assertNotIn("site:dailymotion.com/video", q)


class StillFromImageTests(unittest.TestCase):
    def test_missing_file(self):
        ok, msg, path = still_from_image(Path("no_such_image_clearview.jpg"))
        self.assertFalse(ok)
        self.assertIsNone(path)
        self.assertTrue(msg)

    def test_classify_upload_name(self):
        self.assertEqual(classify_upload_name("foto.JPG"), "image")
        self.assertEqual(classify_upload_name("rio.jpg"), "image")
        self.assertEqual(classify_upload_name("scan.BMP"), "image")
        self.assertEqual(classify_upload_name("a.webp"), "image")
        self.assertEqual(classify_upload_name("x.jfif"), "image")
        self.assertEqual(classify_upload_name("a.tif"), "image")
        self.assertEqual(classify_upload_name("a.tiff"), "image")
        self.assertEqual(classify_upload_name("clip.mp4"), "video")
        self.assertEqual(classify_upload_name("tape.mkv"), "video")
        self.assertEqual(classify_upload_name("tema.mp3"), "audio")
        self.assertEqual(classify_upload_name("loop.WAV"), "audio")
        self.assertEqual(classify_upload_name("bed.m4a"), "audio")
        self.assertEqual(classify_upload_name("sting.aac"), "audio")
        self.assertEqual(classify_upload_name("song", "audio/mpeg"), "audio")
        self.assertEqual(classify_upload_name("rio", "image/jpeg"), "image")
        self.assertIsNone(classify_upload_name("notes.pdf"))
        self.assertIsNone(classify_upload_name("noext"))

    def test_import_local_image_missing(self):
        ok, msg, path = import_local_image(Path("no_such_drop.jpg"), "unit")
        self.assertFalse(ok)
        self.assertIsNone(path)
        self.assertTrue(msg)

    def test_import_local_image_accepts_20_seconds(self):
        if not check_ffmpeg():
            self.skipTest("ffmpeg not installed")
        from video_tools import _run
        dest = None
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "hold20.png"
            cmd = [
                ffmpeg_bin(), "-y",
                "-f", "lavfi", "-i", "color=c=red:s=32x32:d=0.04",
                str(src),
            ]
            r = _run(cmd)
            if r.returncode != 0 or not src.exists() or src.stat().st_size < 80:
                self.skipTest("could not write png fixture")
            ok, msg, dest = import_local_image(src, "hold20", 20)
        try:
            self.assertTrue(ok, msg)
            self.assertIsNotNone(dest)
            got = probe_duration(dest)
            self.assertIsNotNone(got)
            self.assertAlmostEqual(float(got), 20.0, delta=0.4)
        finally:
            if dest is not None and dest.exists():
                dest.unlink()

    def test_import_local_image_tiny_png(self):
        png = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01"
            b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        dest = None
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "unit_drop.png"
            src.write_bytes(png)
            ok, msg, dest = import_local_image(src, "unit_drop")
        try:
            if ok:
                self.assertTrue(dest.exists())
                self.assertTrue(dest.name.startswith("img_"))
                self.assertEqual(dest.suffix.lower(), ".mp4")
                self.assertEqual(dest.parent.resolve(), CLIPS_DIR.resolve())
            else:
                self.assertTrue(msg)
        finally:
            if dest is not None and dest.exists():
                dest.unlink()


class TtsDispatcherTests(unittest.TestCase):
    def test_style_voice_map(self):
        self.assertEqual(pick_edge_voice("warm"), "es-PA-MargaritaNeural")
        self.assertEqual(pick_edge_voice("documentary"), "es-PA-RobertoNeural")
        self.assertEqual(pick_edge_voice("energetic"), "es-MX-JorgeNeural")
        self.assertEqual(pick_edge_voice("ad"), "es-MX-DaliaNeural")
        self.assertEqual(pick_edge_voice("calm"), "es-PA-MargaritaNeural")
        self.assertEqual(
            pick_edge_voice("documentary", "es-US-AlonsoNeural"),
            "es-US-AlonsoNeural",
        )
        self.assertTrue(edge_style_opts("documentary")["rate"].startswith("-"))
        self.assertTrue(edge_style_opts("energetic")["rate"].startswith("+"))
        self.assertTrue(edge_style_opts("calm")["rate"].startswith("-"))
        self.assertEqual(combined_edge_rate("documentary", 1.0), "-12%")
        self.assertEqual(combined_edge_rate("energetic", 1.0), "+14%")
        self.assertEqual(pick_edge_voice("warm", ""), "es-PA-MargaritaNeural")
        self.assertEqual(pick_edge_voice("warm", None), "es-PA-MargaritaNeural")

    def test_sanitize_voice_helpers(self):
        self.assertEqual(sanitize_engine("azure"), "edge")
        self.assertEqual(sanitize_engine("elevenlabs"), "eleven")
        self.assertEqual(sanitize_voice_name("es-PA-RobertoNeural"), "es-PA-RobertoNeural")
        self.assertEqual(sanitize_voice_name("TotallyFakeNeural"), "")
        self.assertEqual(sanitize_vo_fx("booth"), "booth")
        self.assertEqual(sanitize_vo_fx("spaceship"), "none")
        self.assertEqual(sanitize_vo_rate(1.3), 1.3)
        self.assertEqual(sanitize_vo_rate("nope"), 1.0)
        self.assertEqual(sanitize_vo_rate(9), 1.5)
        self.assertEqual(sanitize_vo_rate(0.1), 0.5)

    def test_rate_formatter(self):
        self.assertEqual(edge_rate_percent(1), "+0%")
        self.assertEqual(edge_rate_percent(1.0), "+0%")
        self.assertEqual(edge_rate_percent(1.25), "+25%")
        self.assertEqual(edge_rate_percent(0.5), "-50%")
        self.assertEqual(edge_rate_percent(1.5), "+50%")
        self.assertEqual(edge_rate_percent(1.3), "+30%")

    def test_sanitize_keeps_voice_name_and_fx(self):
        plan = _sanitize_plan(
            {"scenes": [{
                "clip": "a.mp4",
                "voiceName": "es-PA-RobertoNeural",
                "voFx": "booth",
                "voiceEngine": "edge",
                "voRate": 1.3,
            }]},
            lib("a.mp4"),
            "youtube",
        )
        s = plan["scenes"][0]
        self.assertEqual(s["voiceName"], "es-PA-RobertoNeural")
        self.assertEqual(s["voFx"], "booth")
        self.assertEqual(s["voiceEngine"], "edge")
        self.assertAlmostEqual(s["voRate"], 1.3)

    def test_sanitize_strips_junk_voice_fields(self):
        plan = _sanitize_plan(
            {"scenes": [{
                "clip": "a.mp4",
                "voiceName": "TotallyFakeNeural",
                "voFx": "spaceship",
                "voiceEngine": "azure",
                "voRate": "fast",
            }]},
            lib("a.mp4"),
            "youtube",
        )
        s = plan["scenes"][0]
        self.assertEqual(s.get("voiceName") or "", "")
        self.assertEqual(s["voFx"], "none")
        self.assertEqual(s["voiceEngine"], "edge")
        self.assertEqual(s["voRate"], 1.0)

    @patch("tts_audio.eleven_key", return_value=None)
    @patch("tts_audio._edge_synthesize", return_value=(True, "ok"))
    def test_dispatcher_uses_edge_when_no_key(self, mock_edge, _key):
        dest = Path("vo_unit_mock.mp3")
        ok, msg, *rest = synthesize_voiceover("hola río", dest, "documentary")
        self.assertTrue(ok)
        self.assertTrue(mock_edge.called)
        args, kwargs = mock_edge.call_args
        self.assertEqual(args[2], "es-PA-RobertoNeural")
        self.assertEqual(args[3], "-12%")
        self.assertEqual(args[4], "+0%")

    @patch("tts_audio.eleven_key", return_value=None)
    @patch("tts_audio._edge_synthesize", return_value=(True, "ok"))
    def test_dispatcher_user_rate_overrides_style(self, mock_edge, _key):
        ok, _msg, *_rest = synthesize_voiceover(
            "hola", Path("vo_unit_rate.mp3"), "documentary", vo_rate=1.3,
        )
        self.assertTrue(ok)
        args, _kwargs = mock_edge.call_args
        self.assertEqual(args[2], "es-PA-RobertoNeural")
        self.assertEqual(args[3], combined_edge_rate("documentary", 1.3))
        self.assertEqual(args[3], "+14%")

    @patch("tts_audio.eleven_key", return_value=None)
    @patch("tts_audio._edge_synthesize", return_value=(True, "ok"))
    def test_explicit_voice_keeps_style_rate_volume(self, mock_edge, _key):
        ok, _msg, *_rest = synthesize_voiceover(
            "hola",
            Path("vo_unit_named.mp3"),
            "energetic",
            voice_name="es-US-AlonsoNeural",
        )
        self.assertTrue(ok)
        args, _kwargs = mock_edge.call_args
        self.assertEqual(args[2], "es-US-AlonsoNeural")
        self.assertEqual(args[3], "+14%")
        self.assertEqual(args[4], "+4%")

    @patch("tts_audio.synthesize_elevenlabs")
    @patch("tts_audio.eleven_key", return_value=None)
    @patch("tts_audio._edge_synthesize", return_value=(True, "ok"))
    def test_eleven_without_key_falls_back_to_edge(self, mock_edge, _key, mock_el):
        ok, _msg, *_rest = synthesize_voiceover(
            "hola", Path("vo_unit_mock2.mp3"), "warm", engine="eleven",
        )
        self.assertTrue(ok)
        self.assertTrue(mock_edge.called)
        mock_el.assert_not_called()

    def test_split_twelve_paragraphs_to_twelve_scenes(self):
        paras = ["Párrafo %d del documental sobre el río." % i for i in range(1, 13)]
        text = "\n\n".join(paras)
        parts = split_script_to_scenes(text, [3.0] * 12)
        self.assertEqual(len(parts), 12)
        for i, p in enumerate(parts):
            self.assertIn("Párrafo %d" % (i + 1), p)

    def test_five_k_script_is_more_than_one_chunk(self):
        sentence = "El río baja turbio por la cuenca y nadie responde. "
        text = sentence * 200
        self.assertGreater(len(text), 5000)
        chunks = split_tts_chunks(text)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c) <= 3000 for c in chunks))
        self.assertGreater(sum(len(c) for c in chunks), 5000)

    def test_word_boundary_ticks_to_seconds(self):
        w = word_from_boundary({
            "type": "WordBoundary",
            "offset": 2_500_000,
            "duration": 2_000_000,
            "text": "río",
        })
        self.assertEqual(w["text"], "río")
        self.assertAlmostEqual(w["start"], 0.25)
        self.assertAlmostEqual(w["end"], 0.45)
        self.assertIsNone(word_from_boundary({"type": "audio", "data": b"x"}))

    def test_edge_stream_returns_words(self):
        class FakeComm:
            def __init__(self, text, voice, **kwargs):
                self.text = text
            async def stream(self):
                yield {"type": "audio", "data": b"\xff" * 250}
                t = 0.0
                for w in str(self.text).split():
                    yield {
                        "type": "WordBoundary",
                        "offset": int(round(t * 10_000_000)),
                        "duration": int(round(0.25 * 10_000_000)),
                        "text": w,
                    }
                    t += 0.25

        with tempfile.TemporaryDirectory() as d:
            dest = Path(d) / "vo_stream_unit.mp3"
            with patch("edge_tts.Communicate", FakeComm):
                ok, msg, words = _edge_synthesize(
                    "hola río", dest, "es-PA-RobertoNeural"
                )
            self.assertTrue(ok, msg)
            self.assertEqual([w["text"] for w in words], ["hola", "río"])
            self.assertAlmostEqual(words[0]["start"], 0.0)
            self.assertAlmostEqual(words[1]["start"], 0.25)

    def test_chunk_words_offset_by_previous_duration(self):
        def one(chunk, path):
            Path(path).write_bytes(b"x" * 300)
            if str(chunk).startswith("A"):
                return True, str(path), [{"text": "uno", "start": 0.0, "end": 0.4}]
            return True, str(path), [{"text": "dos", "start": 0.1, "end": 0.5}]

        with tempfile.TemporaryDirectory() as d:
            dest = Path(d) / "vo_chunk_unit.mp3"
            with patch("tts_audio._audio_duration_seconds", side_effect=[2.0, 3.0]):
                with patch("tts_audio._concat_audio_files", return_value=(True, "ok")):
                    ok, _msg, words = _synth_chunks_then_concat(
                        ["AAA text", "BBB text"], dest, one
                    )
            self.assertTrue(ok)
            self.assertEqual([w["text"] for w in words], ["uno", "dos"])
            self.assertAlmostEqual(words[0]["start"], 0.0)
            self.assertAlmostEqual(words[1]["start"], 2.1)
            shifted = shift_words([{"text": "x", "start": 0.5, "end": 0.8}], 4.0)
            self.assertAlmostEqual(shifted[0]["start"], 4.5)

    def test_pausa_marker_offsets_later_words(self):
        text = "casas. [pausa 2s] EL RIO LA VILLA"
        parts = split_pausa_parts(text)
        self.assertEqual(len(parts), 2)
        self.assertIn("casas.", parts[0][0])
        self.assertAlmostEqual(parts[0][1], 2.0)
        self.assertIn("EL RIO LA VILLA", parts[1][0])
        self.assertAlmostEqual(parts[1][1], 0.0)
        self.assertEqual(split_pausa_parts("hola [pausa] adiós")[0][1], 2.0)
        self.assertNotIn("pausa", strip_pausa_markers(text).lower())
        self.assertNotIn("[", strip_pausa_markers(text))

        def one(chunk, path):
            Path(path).write_bytes(b"x" * 300)
            if "casas" in str(chunk).lower():
                return True, str(path), [{"text": "casas", "start": 0.0, "end": 0.5}]
            return True, str(path), [
                {"text": "EL", "start": 0.0, "end": 0.2},
                {"text": "RIO", "start": 0.2, "end": 0.5},
            ]

        silences = []

        def fake_silence(path, seconds, **_kwargs):
            silences.append(seconds)
            Path(path).write_bytes(b"x" * 300)
            return True, str(path)

        with tempfile.TemporaryDirectory() as d:
            dest = Path(d) / "vo_pausa_unit.mp3"
            with patch("tts_audio._audio_duration_seconds", return_value=1.0):
                with patch("tts_audio._concat_audio_files", return_value=(True, "ok")):
                    with patch("tts_audio._make_silence_mp3", side_effect=fake_silence):
                        ok, msg, words = _synth_script_then_concat(text, dest, one)
        self.assertTrue(ok, msg)
        self.assertEqual(silences, [2.0])
        joined = " ".join(w["text"] for w in words).lower()
        self.assertNotIn("pausa", joined)
        self.assertEqual(words[0]["text"], "casas")
        later = [w for w in words if w["text"] == "EL"]
        self.assertTrue(later)
        self.assertAlmostEqual(later[0]["start"], 3.0, places=1)
        cues = words_to_cues(words, 32)
        self.assertEqual(cues[0]["text"], "casas")
        self.assertLessEqual(cues[0]["end"], 0.6)
        self.assertGreaterEqual(cues[1]["start"], 2.9)

    def test_eleven_pausa_two_tuple_and_silence_matches_rate(self):
        if not check_ffmpeg():
            self.skipTest("ffmpeg not installed")
        from video_tools import ffmpeg_bin, _run, get_video_info
        import tts_audio

        def fake_eleven(chunk, path, style=None, voice_id=None):
            ff = ffmpeg_bin()
            dest = Path(path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                ff, "-y",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=0.35",
                "-ar", "44100", "-ac", "1",
                "-c:a", "libmp3lame", "-q:a", "4",
                str(dest),
            ]
            r = _run(cmd)
            self.assertEqual(r.returncode, 0, r.stderr)
            return True, str(dest)

        captured = {}
        orig_silence = tts_audio._make_silence_mp3

        def spy_silence(dest, seconds, *args, **kwargs):
            captured["sample_rate"] = kwargs.get("sample_rate")
            captured["channel_layout"] = kwargs.get("channel_layout")
            ok, msg = orig_silence(dest, seconds, *args, **kwargs)
            info = get_video_info(Path(dest))
            for st in info.get("streams") or []:
                if st.get("codec_type") == "audio":
                    captured["probed_rate"] = int(float(st.get("sample_rate") or 0))
                    captured["probed_layout"] = str(st.get("channel_layout") or "")
                    break
            return ok, msg

        with tempfile.TemporaryDirectory() as d:
            dest = Path(d) / "vo_eleven_pausa.mp3"
            with patch("tts_audio.eleven_key", return_value="test-key"):
                with patch("tts_audio.synthesize_elevenlabs", side_effect=fake_eleven):
                    with patch("tts_audio._make_silence_mp3", side_effect=spy_silence):
                        try:
                            ok, msg, words = synthesize_voiceover(
                                "hola [pausa 2s] adiós",
                                dest,
                                "documentary",
                                engine="eleven",
                            )
                        except ValueError as e:
                            self.fail("eleven pausa unpack crashed: %s" % e)
        self.assertTrue(ok, msg)
        self.assertEqual(words, [])
        self.assertEqual(captured.get("sample_rate"), 44100)
        self.assertEqual(captured.get("probed_rate"), 44100)

    def test_leading_pausa_silence_matches_spoken_rate(self):
        if not check_ffmpeg():
            self.skipTest("ffmpeg not installed")
        from video_tools import ffmpeg_bin, _run, get_video_info
        import tts_audio

        def fake_eleven(chunk, path, style=None, voice_id=None):
            ff = ffmpeg_bin()
            dest = Path(path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                ff, "-y",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=0.35",
                "-ar", "44100", "-ac", "1",
                "-c:a", "libmp3lame", "-q:a", "4",
                str(dest),
            ]
            r = _run(cmd)
            self.assertEqual(r.returncode, 0, r.stderr)
            return True, str(dest)

        captured = {}
        orig_silence = tts_audio._make_silence_mp3

        def spy_silence(dest, seconds, *args, **kwargs):
            captured["sample_rate"] = kwargs.get("sample_rate")
            ok, msg = orig_silence(dest, seconds, *args, **kwargs)
            info = get_video_info(Path(dest))
            for st in info.get("streams") or []:
                if st.get("codec_type") == "audio":
                    captured["probed_rate"] = int(float(st.get("sample_rate") or 0))
                    break
            return ok, msg

        with tempfile.TemporaryDirectory() as d:
            dest = Path(d) / "vo_lead_pausa.mp3"
            with patch("tts_audio.eleven_key", return_value="test-key"):
                with patch("tts_audio.synthesize_elevenlabs", side_effect=fake_eleven):
                    with patch("tts_audio._make_silence_mp3", side_effect=spy_silence):
                        ok, msg, words = synthesize_voiceover(
                            "[pausa 2s] hola",
                            dest,
                            "documentary",
                            engine="eleven",
                        )
        self.assertTrue(ok, msg)
        self.assertEqual(captured.get("sample_rate"), 44100)
        self.assertEqual(captured.get("probed_rate"), 44100)

    def test_synth_script_leading_pausa_uses_spoken_rate(self):
        """Leading [pausa] must probe the spoken part (ElevenLabs 44100), not 24000."""
        def one(chunk, path):
            Path(path).write_bytes(b"x" * 300)
            return True, str(path), [{"text": "Hola", "start": 0.0, "end": 0.4}]

        captured = {}
        probed = []

        def fake_probe(path):
            probed.append(Path(path).name)
            return 44100, "mono"

        def spy_silence(path, seconds, sample_rate=24000, channel_layout="mono"):
            captured["sample_rate"] = sample_rate
            captured["channel_layout"] = channel_layout
            captured["seconds"] = seconds
            Path(path).write_bytes(b"x" * 300)
            return True, str(path)

        concat_names = []

        def fake_concat(parts, dest):
            concat_names.extend([Path(p).name for p in parts])
            Path(dest).write_bytes(b"x" * 300)
            return True, str(dest)

        with tempfile.TemporaryDirectory() as d:
            dest = Path(d) / "lead_pausa_unit.mp3"
            with patch("tts_audio._probe_audio_format", side_effect=fake_probe):
                with patch("tts_audio._audio_duration_seconds", return_value=0.4):
                    with patch("tts_audio._concat_audio_files", side_effect=fake_concat):
                        with patch("tts_audio._make_silence_mp3", side_effect=spy_silence):
                            ok, msg, words = _synth_script_then_concat(
                                "[pausa 2s] Hola", dest, one
                            )
        self.assertTrue(ok, msg)
        self.assertEqual(captured.get("sample_rate"), 44100)
        self.assertNotEqual(captured.get("sample_rate"), 24000)
        self.assertAlmostEqual(captured.get("seconds") or 0, 2.0)
        self.assertTrue(probed)
        self.assertTrue(any(".__t" in n for n in probed))
        self.assertFalse(any(".__p" in n for n in probed))
        self.assertTrue(concat_names)
        self.assertTrue(concat_names[0].find(".__p") >= 0, concat_names)
        self.assertTrue(concat_names[1].find(".__t") >= 0, concat_names)
        self.assertEqual(words[0]["text"], "Hola")
        self.assertAlmostEqual(words[0]["start"], 2.0, places=1)

    def test_sanitize_keeps_script(self):
        guion = "Uno.\n\nDos.\n\nTres." * 10
        plan = _sanitize_plan(
            {"scenes": [{"clip": "a.mp4"}], "script": guion},
            lib("a.mp4"),
            "youtube",
        )
        self.assertEqual(plan.get("script"), guion.strip())
        too_long = "x" * 60000
        plan2 = _sanitize_plan(
            {"scenes": [{"clip": "a.mp4"}], "script": too_long},
            lib("a.mp4"),
            "youtube",
        )
        self.assertEqual(len(plan2.get("script") or ""), 50000)


def _write_tone_wav(path: Path, seconds: float = 0.4, rate: int = 8000, hz: float = 440.0) -> None:
    n = max(64, int(seconds * rate))
    with wave.open(str(path), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = bytearray()
        for i in range(n):
            s = int(12000 * math.sin(2 * math.pi * hz * i / rate))
            frames.extend(struct.pack("<h", s))
        w.writeframes(bytes(frames))


class PeaksHelperTests(unittest.TestCase):
    def test_missing_file_empty(self):
        data = load_clip_peaks(Path("no_such_clearview_clip_peaks.mp4"))
        self.assertFalse(data["hasAudio"])
        self.assertEqual(data["peaks"], [])
        self.assertEqual(data["duration"], 0)

    def test_compute_missing_never_raises(self):
        data = compute_clip_peaks(Path("missing_peaks_unit.wav"))
        self.assertEqual(data["peaks"], [])
        self.assertFalse(data["hasAudio"])

    def test_pack_peaks_from_pcm(self):
        pcm = struct.pack("<hhhhhh", 0, 16383, -32767, 100, 0, 8000)
        peaks = _pack_peaks(pcm, 3)
        self.assertEqual(len(peaks), 3)
        self.assertTrue(all(0 <= p <= 1 for p in peaks))
        self.assertGreater(max(peaks), 0.9)

    def test_empty_peaks_shape(self):
        data = empty_peaks(2.5)
        self.assertEqual(data["peaks"], [])
        self.assertFalse(data["hasAudio"])
        self.assertEqual(data["duration"], 2.5)

    def test_tiny_wav_peaks_and_cache(self):
        if not check_ffmpeg():
            self.skipTest("ffmpeg not installed")
        with tempfile.TemporaryDirectory() as d:
            wav = Path(d) / "tone_unit.wav"
            _write_tone_wav(wav)
            data = load_clip_peaks(wav)
            self.assertTrue(data["hasAudio"])
            self.assertGreaterEqual(len(data["peaks"]), 8)
            self.assertTrue(all(isinstance(p, float) and 0 <= p <= 1 for p in data["peaks"]))
            self.assertGreater(max(data["peaks"]), 0.1)
            cache = peaks_sidecar(wav)
            self.assertTrue(cache.is_file())
            planted = {
                "v": 1,
                "peaks": [0.42, 0.41],
                "hasAudio": True,
                "duration": 0.4,
                "mtime": data["mtime"],
                "size": data["size"],
            }
            cache.write_text(json.dumps(planted), encoding="utf-8")
            again = load_clip_peaks(wav)
            self.assertEqual(again["peaks"], [0.42, 0.41])
            self.assertTrue(again["hasAudio"])

    def test_silent_video_no_audio_lane(self):
        if not check_ffmpeg():
            self.skipTest("ffmpeg not installed")
        ff = ffmpeg_bin()
        with tempfile.TemporaryDirectory() as d:
            dest = Path(d) / "silent_unit.mp4"
            from video_tools import _run
            cmd = [
                ff, "-y", "-f", "lavfi", "-i", "color=c=black:s=32x32:d=0.4:r=10",
                "-an", "-c:v", "mpeg4", str(dest),
            ]
            r = _run(cmd)
            if r.returncode != 0 or not dest.exists():
                self.skipTest("could not mux tiny mp4")
            data = compute_clip_peaks(dest)
            self.assertFalse(data["hasAudio"])
            self.assertEqual(data["peaks"], [])


class PreviewPerfTests(unittest.TestCase):
    def test_poster_writes_jpeg_from_tiny_mp4(self):
        if not check_ffmpeg():
            self.skipTest("ffmpeg not installed")
        ff = ffmpeg_bin()
        with tempfile.TemporaryDirectory() as d:
            mp4 = Path(d) / "unit_poster.mp4"
            from video_tools import _run
            cmd = [
                ff, "-y", "-f", "lavfi", "-i", "color=c=blue:s=64x48:d=0.4:r=10",
                "-an", "-c:v", "mpeg4", str(mp4),
            ]
            r = _run(cmd)
            if r.returncode != 0 or not mp4.exists():
                self.skipTest("could not mux tiny mp4")
            dest = ensure_clip_poster(mp4, 0.0)
            self.assertIsNotNone(dest)
            self.assertTrue(dest.is_file())
            self.assertGreater(dest.stat().st_size, 100)
            self.assertTrue(dest.name.endswith(".jpg"))
            self.assertEqual(dest, poster_sidecar(mp4, 0.0))
            again = ensure_clip_poster(mp4, 0.0)
            self.assertEqual(again, dest)

    def test_proxy_skips_under_size_cutoff(self):
        self.assertGreater(PROXY_MIN_BYTES, 1_000_000)
        with tempfile.TemporaryDirectory() as d:
            small = Path(d) / "tiny.mp4"
            small.write_bytes(b"not-a-real-video" * 20)
            self.assertLess(small.stat().st_size, PROXY_MIN_BYTES)
            self.assertFalse(proxy_needed(small))
            status, dest = ensure_clip_proxy(small)
            self.assertEqual(status, "skipped")
            self.assertIsNone(dest)

    def test_proxy_encode_has_no_180s_fuse_and_drops_failed_partial(self):
        if not check_ffmpeg():
            self.skipTest("ffmpeg not installed")
        from video_tools import PROXIES_DIR, ensure_clip_proxy
        PROXIES_DIR.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as d:
            big = Path(d) / "hevc_interview_unit.mp4"
            with big.open("wb") as fh:
                fh.seek((15 * 1024 * 1024) + 64)
                fh.write(b"\0")
            dest = PROXIES_DIR / big.name
            tmp = dest.with_name(dest.name + ".tmp.mp4")

            def boom(cmd, **kwargs):
                self.assertNotIn("timeout", kwargs)
                Path(cmd[-1]).parent.mkdir(parents=True, exist_ok=True)
                Path(cmd[-1]).write_bytes(b"x" * 4000)
                raise subprocess.TimeoutExpired(cmd, 180)

            def reject(cmd, **kwargs):
                self.assertNotIn("timeout", kwargs)
                Path(cmd[-1]).write_bytes(b"x" * 4000)
                return subprocess.CompletedProcess(cmd, 1)

            try:
                with patch("video_tools.subprocess.run", side_effect=boom):
                    status, out = ensure_clip_proxy(big)
                self.assertEqual(status, "failed")
                self.assertIsNone(out)
                self.assertFalse(tmp.exists())
                self.assertFalse(dest.exists())
                with patch("video_tools.subprocess.run", side_effect=reject):
                    status, out = ensure_clip_proxy(big)
                self.assertEqual(status, "failed")
                self.assertFalse(dest.exists())
                self.assertFalse(tmp.exists())

                def ok(cmd, **kwargs):
                    self.assertNotIn("timeout", kwargs)
                    Path(cmd[-1]).write_bytes(b"x" * 4000)
                    return subprocess.CompletedProcess(cmd, 0)

                with patch("video_tools.subprocess.run", side_effect=ok):
                    status, out = ensure_clip_proxy(big)
                self.assertEqual(status, "ready")
                self.assertTrue(dest.is_file())
                self.assertGreater(dest.stat().st_size, 2000)
                self.assertFalse(tmp.exists())
            finally:
                tmp.unlink(missing_ok=True)
                dest.unlink(missing_ok=True)


class ExportPathTests(unittest.TestCase):
    def test_normalize_segment_speed_keeps_out_duration(self):
        if not check_ffmpeg():
            self.skipTest("ffmpeg not installed")
        ff = ffmpeg_bin()
        from video_tools import _run
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "src_speed.mp4"
            cmd = [
                ff, "-y",
                "-f", "lavfi", "-i", "color=c=green:s=160x90:d=4:r=15",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
                "-c:v", "mpeg4", "-c:a", "aac", "-shortest",
                str(src),
            ]
            r = _run(cmd)
            if r.returncode != 0 or not src.exists() or src.stat().st_size < 1000:
                self.skipTest("could not mux speed fixture")
            cases = ((1.0, 1.0), (2.0, 1.0), (0.5, 2.0))
            for speed, out_dur in cases:
                dest = Path(d) / ("out_%.1fx.mp4" % speed)
                ok, msg = _normalize_segment(
                    src, 0.0, out_dur, dest, 160, 90, {"speed": speed},
                )
                self.assertTrue(ok, "speed %s: %s" % (speed, msg))
                got = probe_duration(dest)
                self.assertIsNotNone(got)
                self.assertAlmostEqual(
                    float(got), out_dur, delta=0.15,
                    msg="speed %s expected ~%ss got %s" % (speed, out_dur, got),
                )

    def test_freeze_still_hold_exports_about_20s(self):
        if not check_ffmpeg():
            self.skipTest("ffmpeg not installed")
        ff = ffmpeg_bin()
        from video_tools import _run
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "still4.mp4"
            cmd = [
                ff, "-y",
                "-f", "lavfi", "-i", "color=c=blue:s=160x90:d=4:r=15",
                "-c:v", "mpeg4",
                str(src),
            ]
            r = _run(cmd)
            if r.returncode != 0 or not src.exists() or src.stat().st_size < 400:
                self.skipTest("could not mux still fixture")
            dest = Path(d) / "hold20.mp4"
            ok, msg = _normalize_segment(
                src, 0.0, 20.0, dest, 160, 90, {"freeze": True},
            )
            self.assertTrue(ok, msg)
            got = probe_duration(dest)
            self.assertIsNotNone(got)
            self.assertAlmostEqual(float(got), 20.0, delta=0.5)

    def test_normalize_cover_uses_increase_crop(self):
        captured = []

        def fake_run(cmd):
            captured.append(list(cmd))
            class R:
                returncode = 1
                stderr = "fail"
            return R()

        with patch("video_tools.get_video_info", return_value={
            "streams": [{"codec_type": "video", "width": 1080, "height": 1920}],
        }):
            with patch("video_tools._has_video", return_value=True):
                with patch("video_tools.ffmpeg_bin", return_value="ffmpeg"):
                    with patch("video_tools._run", side_effect=fake_run):
                        ok, _msg = _normalize_segment(
                            Path("vert.mp4"), 0.0, 1.0, Path("out.mp4"),
                            1920, 1080, {"fit": "cover"},
                        )
        self.assertFalse(ok)
        self.assertTrue(captured)
        vf = captured[0][captured[0].index("-vf") + 1]
        self.assertIn("force_original_aspect_ratio=increase", vf)
        self.assertIn("crop=1920:1080", vf)
        self.assertNotIn("force_original_aspect_ratio=decrease", vf)
        self.assertNotIn("pad=1920:1080", vf)

    def test_export_reuses_stored_vo_only_when_sig_matches(self):
        script = "el río baja turbio"
        lead = {
            "voiceover": True,
            "narration": script,
            "voiceStyle": "documentary",
            "voiceName": "",
            "voiceEngine": "edge",
            "voFx": "none",
            "voRate": 1,
        }
        sig = _export_vo_signature(script, lead)
        self.assertTrue(sig.startswith("full\x1f"))
        self.assertEqual(sig, sig[:8000])
        self.assertIn(script, sig)
        long_script = "x" * 9000
        self.assertEqual(len(_export_vo_signature(long_script, lead)), 8000)
        clips = [
            {"clip": "a.mp4", "narration": ""},
            {"clip": "b.mp4", "voiceover": True, "narration": script, "voiceStyle": "warm"},
        ]
        self.assertIs(_export_vo_lead(clips), clips[1])
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            auto = root / "autosave"
            auto.mkdir()
            stored = auto / "vo.mp3"
            stored.write_bytes(b"S" * 400)
            (auto / "sequence.json").write_text(
                json.dumps({"ui": {"vo": {"sig": sig, "url": "/api/sequence/vo"}}}),
                encoding="utf-8",
            )
            dest = root / "out.mp3"
            with patch("video_tools.STORAGE_DIR", root):
                with patch("tts_audio.synthesize_voiceover") as syn:
                    syn.return_value = (True, "synth", [])
                    self.assertTrue(_maybe_use_stored_vo(script, lead, dest))
                    self.assertEqual(dest.read_bytes()[:1], b"S")
                    syn.assert_not_called()
                    dest.unlink()
                    self.assertFalse(_maybe_use_stored_vo(script + " CAMBIO", lead, dest))
                    self.assertFalse(dest.exists())
                    syn.assert_not_called()


class CaptionSceneExportTests(unittest.TestCase):
    def test_omits_cue_on_capoff_scene(self):
        scenes = [
            {"timelineStart": 0, "duration": 5, "capOff": True, "capPos": "top"},
            {"timelineStart": 5, "duration": 5, "capOff": False, "capPos": "center"},
        ]
        cues = [
            {"text": "hidden still", "start": 1.0, "end": 3.0},
            {"text": "next scene", "start": 6.0, "end": 8.0},
            {"content": "also hidden", "start": 4.5, "end": 4.9},
        ]
        out = caption_export_texts(cues, scenes, "lower-third")
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["content"], "next scene")
        self.assertEqual(out[0]["position"], "center")
        inherit = caption_export_texts(
            [{"text": "global pos", "start": 6.2, "end": 7.0}],
            [
                {"timelineStart": 0, "duration": 5, "capOff": True},
                {"timelineStart": 5, "duration": 5, "capOff": False, "capPos": ""},
            ],
            "lower-third",
        )
        self.assertEqual(len(inherit), 1)
        self.assertEqual(inherit[0]["position"], "lower-third")

    def test_capoff_omits_only_that_window(self):
        scenes = [
            {"timelineStart": 0, "duration": 5, "capOff": False, "capPos": ""},
            {"timelineStart": 5, "duration": 5, "capOff": True, "capPos": ""},
            {"timelineStart": 10, "duration": 5, "capOff": False, "capPos": "top"},
        ]
        cues = [
            {"text": "scene7", "start": 1.0, "end": 3.0},
            {"text": "scene8", "start": 6.0, "end": 8.0},
            {"text": "scene9", "start": 11.0, "end": 13.0},
            {"text": "span78", "start": 3.0, "end": 7.0},
            {"text": "span89", "start": 8.0, "end": 12.0},
        ]
        out = caption_export_texts(cues, scenes, "lower-third")
        by = {}
        for row in out:
            by.setdefault(row["content"], []).append(row)
        self.assertEqual(len(by.get("scene7") or []), 1)
        self.assertNotIn("scene8", by)
        self.assertEqual(len(by.get("scene9") or []), 1)
        self.assertEqual(by["scene9"][0]["position"], "top")
        self.assertEqual(len(by.get("span78") or []), 1)
        self.assertAlmostEqual(by["span78"][0]["start"], 3.0)
        self.assertAlmostEqual(by["span78"][0]["end"], 5.0)
        self.assertEqual(len(by.get("span89") or []), 1)
        self.assertAlmostEqual(by["span89"][0]["start"], 10.0)
        self.assertAlmostEqual(by["span89"][0]["end"], 12.0)

    def test_vo_cues_cover_capoff_scene_export_keeps_later_starts(self):
        words = []
        t = 0.0
        n = 0
        while t < 20:
            n += 1
            words.append({"text": "w%d" % n, "start": round(t, 3), "end": round(t + 0.4, 3)})
            t += 0.5
        cues = words_to_cues(words, 32)
        cover3 = [c for c in cues if c["start"] < 15 and c["end"] > 10]
        self.assertTrue(cover3)
        scene4 = [c for c in cues if c["start"] >= 15]
        self.assertTrue(scene4)
        start4 = scene4[0]["start"]
        scenes = [
            {"timelineStart": 0, "duration": 5, "capOff": False},
            {"timelineStart": 5, "duration": 5, "capOff": False},
            {"timelineStart": 10, "duration": 5, "capOff": True},
            {"timelineStart": 15, "duration": 5, "capOff": False},
        ]
        out = caption_export_texts(cues, scenes, "lower-third")
        for row in out:
            mid = (row["start"] + row["end"]) / 2.0
            self.assertFalse(10 <= mid < 15)
        after4 = [c for c in out if c["start"] >= 15]
        self.assertTrue(after4)
        self.assertAlmostEqual(after4[0]["start"], start4)
        vo_cues = cues_from_untimed_script("uno dos tres cuatro cinco seis siete ocho", 8.0)
        self.assertTrue(vo_cues)
        self.assertLessEqual(vo_cues[-1]["end"], 8.05)


class ProjectClipIsolationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.clips = lib(("rio.mp4", 12), ("humo.mp4", 8), ("mapa.mp4", 4))
        self.patcher = patch("video_create.BASE_DIR", self.base)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        self.tmp.cleanup()

    def _folder(self, pid):
        d = self.base / "storage" / "projects" / pid
        d.mkdir(parents=True, exist_ok=True)
        return d

    def test_clips_from_scenes_unique(self):
        scenes = [
            {"clip": "rio.mp4"},
            {"src": "/clips/rio.mp4"},
            {"clip": "humo.mp4"},
            {"clip": ""},
        ]
        self.assertEqual(_clips_from_scenes(scenes), ["rio.mp4", "humo.mp4"])

    def test_seed_from_each_project_sequence_scenes(self):
        save_sequence(
            {"title": "A", "scenes": [
                {"clip": "rio.mp4", "inPoint": 0, "outPoint": 2},
                {"clip": "mapa.mp4", "inPoint": 0, "outPoint": 2},
            ]},
            folder=self._folder("rio-1"),
            library=self.clips,
            force=True,
        )
        save_sequence(
            {"title": "B", "scenes": [
                {"clip": "humo.mp4", "inPoint": 0, "outPoint": 2},
            ]},
            folder=self._folder("rio-2"),
            library=self.clips,
            force=True,
        )
        idx_path = self.base / "storage" / "projects" / "index.json"
        idx_path.write_text(
            '{"activeId": "rio-1", "projects": ['
            '{"id": "rio-1", "name": "Rio 1", "updatedAt": 1},'
            '{"id": "rio-2", "name": "Rio 2", "updatedAt": 1}'
            "]}",
            encoding="utf-8",
        )
        idx = ensure_project_clip_membership()
        by = {p["id"]: p["clips"] for p in idx["projects"]}
        self.assertEqual(by["rio-1"], ["rio.mp4", "mapa.mp4"])
        self.assertEqual(by["rio-2"], ["humo.mp4"])
        self.assertNotIn("humo.mp4", by["rio-1"])
        self.assertNotIn("rio.mp4", by["rio-2"])

    def test_new_project_empty_and_keeps_existing(self):
        save_sequence(
            {"title": "A", "scenes": [{"clip": "rio.mp4", "inPoint": 0, "outPoint": 2}]},
            folder=self._folder("rio-1"),
            library=self.clips,
            force=True,
        )
        (self.base / "storage" / "projects" / "index.json").write_text(
            '{"activeId": "rio-1", "projects": [{"id": "rio-1", "name": "Rio 1", "updatedAt": 1}]}',
            encoding="utf-8",
        )
        ensure_project_clip_membership()
        out = create_project("Demo Nuevo")
        self.assertEqual(out["activeId"], "demo-nuevo")
        self.assertEqual(out["project"]["clips"], [])
        self.assertEqual(out["sequence"]["scenes"], [])
        idx = load_projects_index()
        ids = [p["id"] for p in idx["projects"]]
        self.assertIn("rio-1", ids)
        self.assertIn("demo-nuevo", ids)
        rio = next(p for p in idx["projects"] if p["id"] == "rio-1")
        self.assertEqual(rio["clips"], ["rio.mp4"])
        self.assertEqual(active_project_clip_names(), [])

    def test_attach_only_active_and_switch_empty(self):
        create_project("Alpha")
        attach_clip_to_active_project("rio.mp4")
        attach_clip_to_active_project("/clips/humo.mp4")
        self.assertEqual(active_project_clip_names(), ["rio.mp4", "humo.mp4"])
        create_project("Beta")
        self.assertEqual(active_project_clip_names(), [])
        attach_clip_to_active_project("mapa.mp4")
        self.assertEqual(active_project_clip_names(), ["mapa.mp4"])
        switched = switch_project("alpha")
        self.assertTrue(switched["ok"])
        self.assertEqual(switched["activeId"], "alpha")
        self.assertEqual(active_project_clip_names(), ["rio.mp4", "humo.mp4"])
        self.assertNotIn("mapa.mp4", active_project_clip_names())
        live, _ = load_sequence()
        self.assertEqual(live.get("scenes") or [], [])

    def test_create_rejects_blank_name(self):
        with self.assertRaises(ValueError):
            create_project("  ")


class FootageRankTests(unittest.TestCase):
    def test_spanish_english_tokens(self):
        from search_rank import meaningful_tokens, expand_tokens, expand_search_query, fold
        toks = meaningful_tokens("río panama contaminación")
        self.assertIn("rio", toks)
        self.assertIn("panama", toks)
        self.assertIn("contaminacion", toks)
        self.assertNotIn("stock", toks)
        self.assertNotIn("footage", toks)
        exp = expand_tokens(toks)
        self.assertIn("river", exp)
        self.assertIn("pollution", exp)
        q = expand_search_query("río sucio")
        self.assertIn("río", q)
        self.assertIn("river", fold(q))
        self.assertIn("sucio", q)

    def test_zero_token_dropped(self):
        from search_rank import filter_and_rank
        items = [
            {"video_id": "1", "title": "Honda civic swap K24a stock 233hp", "url": "https://ig/1", "platform": "instagram"},
            {"video_id": "2", "title": "Contaminación de los Ríos en Panamá - Documental", "url": "https://yt/2", "duration": "6:00", "platform": "youtube"},
            {"video_id": "3", "title": "RIVER PLATE AL DÍA", "url": "https://ig/3", "platform": "instagram"},
            {"video_id": "4", "title": "Un espejo puede transformar por completo un espacio.", "url": "https://ig/4", "platform": "instagram"},
        ]
        kept, dropped, raw = filter_and_rank(items, "río panama contaminación", 8)
        titles = [k["title"] for k in kept]
        self.assertEqual(raw, 4)
        self.assertGreaterEqual(dropped, 2)
        self.assertTrue(any("Contaminación" in t for t in titles))
        self.assertFalse(any("Honda" in t for t in titles))
        self.assertFalse(any("espejo" in t.lower() for t in titles))

    def test_english_synonym_matches_spanish_title(self):
        from search_rank import filter_and_rank
        items = [
            {"video_id": "a", "title": "River pollution in Panama b-roll", "url": "https://yt/a", "duration": "0:45", "platform": "youtube"},
            {"video_id": "b", "title": "Cooking pasta carbonara", "url": "https://yt/b", "duration": "0:30", "platform": "youtube"},
        ]
        kept, dropped, raw = filter_and_rank(items, "río contaminación panamá", 8)
        self.assertEqual(len(kept), 1)
        self.assertIn("River pollution", kept[0]["title"])
        self.assertEqual(dropped, 1)

    def test_long_vlog_ranks_below_short_doc(self):
        from search_rank import filter_and_rank
        items = [
            {"video_id": "long", "title": "Contaminación de ríos en Panamá live stream 3 hours", "url": "https://yt/l", "duration": "3:00:00", "platform": "youtube"},
            {"video_id": "short", "title": "Contaminación de ríos en Panamá b-roll", "url": "https://yt/s", "duration": "0:42", "platform": "youtube"},
        ]
        kept, dropped, raw = filter_and_rank(items, "río contaminación panamá", 8)
        self.assertEqual(len(kept), 2)
        self.assertEqual(kept[0]["video_id"], "short")

    def test_stopwords_stock_footage_do_not_save_a_miss(self):
        from search_rank import filter_and_rank
        items = [
            {"video_id": "x", "title": "Free stock footage 4K video clip", "url": "https://yt/x", "platform": "youtube"},
        ]
        kept, dropped, raw = filter_and_rank(items, "río contaminación stock footage", 8)
        self.assertEqual(kept, [])
        self.assertEqual(dropped, 1)

    def test_agua_sucia_azuero_ranks_above_fish_tank(self):
        from search_rank import (
            expand_tokens,
            filter_and_rank,
            meaningful_tokens,
            relevance_score,
            token_hit_count,
        )
        items = [
            {
                "video_id": "azuero",
                "title": "Contaminación del río en Azuero, agua sucia",
                "url": "https://yt/a",
                "duration": "0:42",
                "platform": "youtube",
            },
            {
                "video_id": "tank",
                "title": "How to clean your fish tank water",
                "url": "https://yt/b",
                "platform": "youtube",
            },
        ]
        orig = meaningful_tokens("agua sucia Azuero")
        exp = expand_tokens(orig)
        self.assertEqual(token_hit_count(items[0], orig), 3)
        self.assertEqual(token_hit_count(items[1], orig), 0)
        # Scoring now uses understand_query (subject > place). Water/pollution
        # must beat a place-only cattle hit; do not pin the old 21.8 constant.
        self.assertGreater(
            relevance_score(items[0], orig, exp),
            relevance_score(items[1], orig, exp),
        )
        kept, dropped, raw = filter_and_rank(items, "agua sucia Azuero", 8)
        self.assertEqual(raw, 2)
        self.assertEqual([k["video_id"] for k in kept], ["azuero"])

    def test_filter_and_rank_never_returns_below_min_score(self):
        from search_rank import (
            MIN_SCORE,
            expand_tokens,
            filter_and_rank,
            meaningful_tokens,
            relevance_score,
            understand_query,
        )
        mk = lambda t, d, c, i: {
            "title": t, "platform": "youtube", "duration_sec": d,
            "channel": c, "url": "https://y/" + i, "video_id": i,
        }
        items = [
            mk("River Plate 3-0 Boca Juniors", 600, "ESPN", "a"),
            mk("Amazon River Documentary 4K", 2400, "NatGeo", "b"),
            mk("Relaxing River Sounds 10 Hours", 36000, "Sleep", "c"),
            mk("Water Park Fails Compilation", 300, "Fun", "d"),
            mk("Rio La Villa Herrera Panama sequia", 240, "TVN", "e"),
            mk("How to clean your fish tank water", 420, "Pets", "f"),
            mk("Mississippi River flooding drone", 180, "News", "g"),
            mk("Minecraft river base build", 900, "Gaming", "h"),
        ]
        for q in ("agua sucia Azuero", "Rio La Villa Herrera Panama"):
            orig = meaningful_tokens(q)
            exp = expand_tokens(orig)
            kept, dropped, raw = filter_and_rank(items, q, 8)
            self.assertEqual(raw, 8)
            u = understand_query(q)
            for it in kept:
                self.assertGreaterEqual(
                    relevance_score(it, u), MIN_SCORE, it["title"],
                )
            titles = " ".join(k["title"] for k in kept).lower()
            if q == "agua sucia Azuero":
                self.assertNotIn("fish tank", titles)
                self.assertNotIn("water park", titles)
                self.assertEqual(kept, [])
            else:
                self.assertNotIn("minecraft", titles)
                self.assertNotIn("river plate", titles)
                self.assertNotIn("amazon", titles)
                self.assertNotIn("relaxing river", titles)
                self.assertTrue(any("la villa" in k["title"].lower() for k in kept))

    def test_closeup_humo_honest_empty_without_smoke(self):
        from search_rank import filter_and_rank
        items = [
            {"video_id": "a", "title": "How to close your account", "url": "https://yt/a", "platform": "youtube"},
            {"video_id": "b", "title": "Whats up compilation", "url": "https://yt/b", "platform": "youtube"},
            {"video_id": "c", "title": "Free stock footage 4K", "url": "https://yt/c", "platform": "youtube"},
        ]
        kept, dropped, raw = filter_and_rank(items, "close-up humo", 8)
        self.assertEqual(kept, [])
        self.assertEqual(raw, 3)
        self.assertEqual(dropped, 3)

    def test_closeup_humo_matches_english_smoke(self):
        from search_rank import filter_and_rank
        items = [
            {"video_id": "smoke", "title": "Industrial smoke stacks", "url": "https://yt/s", "duration": "0:20", "platform": "youtube"},
            {"video_id": "close", "title": "How to close your account", "url": "https://yt/c", "platform": "youtube"},
        ]
        kept, dropped, raw = filter_and_rank(items, "close-up humo", 8)
        self.assertEqual([k["video_id"] for k in kept], ["smoke"])
        self.assertEqual(dropped, 1)

    def test_stem_table(self):
        from search_rank import stem
        pairs = {
            "canaverales": "canaveral",
            "rios": "rio",
            "aguas": "agua",
            "quemas": "quema",
            "playas": "playa",
            "sucias": "sucia",
            "contaminacion": "contamina",
            "contaminaciones": "contamina",
            "contaminado": "contamin",
            "contaminada": "contamin",
            "vertimiento": "verti",
            "quemando": "quem",
            "basura": "basura",
            "humo": "humo",
            "rio": "rio",
            "agua": "agua",
            "mar": "mar",
            "ganado": "ganado",
            "sediento": "sediento",
            "fuego": "fuego",
        }
        for src, want in pairs.items():
            self.assertEqual(stem(src), want, src)
        for w in ("contaminacion", "contaminado", "contaminantes"):
            self.assertTrue(stem(w).startswith("contamin"), w)

    def test_activate_topics_agua_and_quema(self):
        from search_rank import activate_topics, meaningful_tokens, stem
        def stems(q):
            return [stem(t) for t in meaningful_tokens(q)]
        self.assertIn("agua_contaminada", activate_topics(stems("agua sucia")))
        self.assertIn("quema_agricola", activate_topics(stems("quema de canaverales")))

    def test_agua_sucia_azuero_not_cattle_top(self):
        from search_rank import MIN_SCORE, filter_and_rank, relevance_score, understand_query
        items = [
            {"video_id": "cattle", "title": "Canaveral y ganado en Azuero", "url": "https://yt/c", "duration": "0:42", "platform": "youtube"},
            {"video_id": "water", "title": "Contaminación del río en Azuero, agua sucia", "url": "https://yt/w", "duration": "0:42", "platform": "youtube"},
            {"video_id": "cane", "title": "Quema de cañaverales en Azuero", "url": "https://yt/q", "duration": "1:10", "platform": "youtube"},
        ]
        kept, dropped, raw = filter_and_rank(items, "agua sucia Azuero", 8)
        self.assertEqual(raw, 3)
        self.assertTrue(kept)
        self.assertEqual(kept[0]["video_id"], "water")
        self.assertNotEqual(kept[0]["video_id"], "cattle")
        u = understand_query("agua sucia Azuero")
        self.assertLess(relevance_score(items[0], u), MIN_SCORE)

    def test_quema_canaverales_has_en_expansion(self):
        from search_rank import understand_query
        u = understand_query("quema de canaverales")
        blob = " ".join(u["expansion_en"] + u["search_strings"]).lower()
        self.assertTrue(u["expansion_en"] or u["search_strings"])
        self.assertTrue("sugarcane" in blob or "burning" in blob)

    def test_empty_query_no_topic_returns_empty(self):
        from search_rank import filter_and_rank
        items = [
            {"video_id": "x", "title": "Free stock footage 4K video clip", "url": "https://yt/x", "platform": "youtube"},
        ]
        kept, dropped, raw = filter_and_rank(items, "zzzzqxqxqx", 8)
        self.assertEqual(kept, [])
        self.assertEqual(raw, 1)

    def test_llm_timeout_none_ranking_still_works(self):
        from search_rank import filter_and_rank, llm_query_terms
        with patch("search_rank._MEM_CACHE", OrderedDict()):
            with patch("search_rank._xai_key_for_search", return_value="k"):
                with patch("openai.OpenAI") as ctor:
                    client = MagicMock()
                    client.chat.completions.create.side_effect = TimeoutError("t")
                    ctor.return_value = client
                    self.assertIsNone(llm_query_terms("quema de canaverales"))
        items = [
            {"video_id": "w", "title": "Contaminación del río, agua sucia", "url": "https://yt/w", "duration": "0:40", "platform": "youtube"},
            {"video_id": "c", "title": "Cooking pasta carbonara", "url": "https://yt/c", "duration": "0:30", "platform": "youtube"},
        ]
        kept, dropped, raw = filter_and_rank(items, "agua sucia Azuero", 8)
        self.assertEqual(raw, 2)
        self.assertEqual(kept[0]["video_id"], "w")

    def test_llm_cache_second_query_one_call(self):
        from search_rank import llm_query_terms
        payload = {
            "topic": "quema_agricola",
            "terms_es": ["quema", "humo"],
            "terms_en": ["sugarcane", "burning"],
            "visuals": ["field"],
        }
        client = MagicMock()
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = json.dumps(payload)
        client.chat.completions.create.return_value = resp
        with tempfile.TemporaryDirectory() as d:
            cache = Path(d) / "search-terms-cache.json"
            with patch("search_rank.SEARCH_TERMS_CACHE", cache):
                with patch("search_rank._MEM_CACHE", OrderedDict()):
                    with patch("search_rank._xai_key_for_search", return_value="k"):
                        with patch("openai.OpenAI", return_value=client):
                            a = llm_query_terms("quema de canaverales")
                            b = llm_query_terms("quema de canaverales")
        self.assertEqual(a.get("terms_en"), ["sugarcane", "burning"])
        self.assertEqual(b.get("terms_en"), a.get("terms_en"))
        self.assertEqual(client.chat.completions.create.call_count, 1)


class FrontendWriterContractTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parent.parent
        self.editor = (root / "frontend" / "editor" / "index.html").read_text(encoding="utf-8")
        self.create = (root / "frontend" / "create" / "index.html").read_text(encoding="utf-8")

    def test_editor_is_not_a_sequence_writer(self):
        """Stage B: Edit is single-clip. Assemble is the only whole-document writer.
        The old idle-writer mirror (saveInFlight / POST /api/sequence) is obsolete by design."""
        self.assertNotIn('fetch("/api/sequence",', self.editor)
        self.assertIn('fetch("/api/clips/publish"', self.editor)
        self.assertIn('fetch("/api/sequence/scene"', self.editor)
        self.assertIn("if (saveInFlight) { saveQueued = true; return true; }", self.create)
        self.assertIn("URL.revokeObjectURL", self.editor)
        self.assertIn("URL.revokeObjectURL", self.create)
        self.assertIn("Editar este clip", self.create)
        self.assertIn("/editor?scene=", self.create)
        self.assertIn("Vas a pasar de", self.create)
        self.assertIn("payload.force = true", self.create)

    def test_editor_scene_query_loads_in_out_grade(self):
        self.assertIn('params.get("scene")', self.editor)
        self.assertIn("function sceneToClip(s)", self.editor)
        self.assertIn("const inPoint = Number(s.inPoint) || 0", self.editor)
        self.assertIn("const outPoint = Number(s.outPoint) || inPoint + 1", self.editor)
        self.assertIn("grade: normalizeGrade(s.grade)", self.editor)
        self.assertIn("loadSceneFromSequence", self.editor)

    def test_editor_clip_query_hides_apply_scene(self):
        self.assertIn('id="applySceneBtn"', self.editor)
        self.assertIn("applySceneBtn", self.editor)
        self.assertIn('params.get("clip")', self.editor)
        self.assertIn("syncApplyButton", self.editor)
        self.assertIn('if (sceneId) btn.classList.remove("hidden");', self.editor)
        self.assertIn('else btn.classList.add("hidden");', self.editor)

    def test_editor_localstorage_draft_is_one_clip(self):
        self.assertIn('clip: draft', self.editor)
        self.assertIn("serializeClip(project.clip)", self.editor)
        self.assertNotIn("clips: clips", self.editor)
        self.assertIn('localStorage.setItem(SAVE_KEY, JSON.stringify(payload))', self.editor)

    def test_clip_filename_empty_without_extension(self):
        self.assertIn("function clipFileName(clip)", self.editor)
        self.assertIn("function mediaFileNameFromSrc(path)", self.editor)
        self.assertIn("if (!name || !/\\.[a-z0-9]{2,5}$/i.test(name)) return \"\";", self.editor)
        self.assertEqual(self._js_media_file_name("/clips/Screen Recording 2026-08-29 175720"), "")
        self.assertEqual(
            self._js_media_file_name("/clips/Screen_Recording_2026-08-29_175720.mp4"),
            "Screen_Recording_2026-08-29_175720.mp4",
        )

    def test_kick_proxy_no_retry_on_404(self):
        start = self.editor.find("function kickProxyName(")
        self.assertGreater(start, 0)
        end = self.editor.find("function kickProxyClip(", start)
        body = self.editor[start:end]
        self.assertIn('if (!r.ok) { previewProxy[name] = ""; return null; }', body)
        self.assertIn('if (!r2.ok) { previewProxy[name] = ""; return null; }', body)
        self.assertIn("if (attempt >= 2) { previewProxy[name] = \"\"; return; }", body)
        after_get_404 = body.split("if (!r.ok)")[1].split(".then(d =>")[0]
        self.assertNotIn("setTimeout", after_get_404)
        after_post_404 = body.split("if (!r2.ok)")[1].split(".then(d2 =>")[0]
        self.assertNotIn("setTimeout", after_post_404)
        self.assertLessEqual(body.count("setTimeout"), 2)

    def test_editor_timeline_is_source_selection(self):
        """Single-clip Edit: bar is the whole source; handles mark in/out."""
        src = self.editor
        self.assertIn("function layoutSourceSelection(", src)
        self.assertIn("const srcPx = sourceMax(clip) * pxPerSec;", src)
        self.assertIn('el.style.width = srcPx + "px";', src)
        self.assertIn('left.style.left = (clip.inPoint * pxPerSec) + "px";', src)
        self.assertIn('right.style.left = (clip.outPoint * pxPerSec) + "px";', src)
        self.assertIn('class="clip-dim head"', src)
        self.assertIn('class="clip-sel"', src)
        self.assertIn('class="clip-dim tail"', src)
        self.assertIn('class="audio-dim head"', src)
        self.assertIn('class="audio-sel"', src)
        start = src.find("function startEdgeDrag(")
        self.assertGreater(start, 0)
        end = src.find('el.addEventListener("mousedown"', start)
        body = src[start:end]
        self.assertIn("pushUndo()", body)
        self.assertIn("interacting = true", body)
        self.assertIn("clip.inPoint = Math.max(0, Math.min(origIn + dt, origOut - MIN_CLIP));", body)
        self.assertIn("clip.outPoint = Math.max(origIn + MIN_CLIP, Math.min(maxOut, origOut + dt));", body)
        left_branch = body.split('else if (side === "left")')[1].split("} else {")[0]
        self.assertIn("clip.inPoint =", left_branch)
        self.assertNotIn("clip.outPoint =", left_branch)
        right_branch = body.split("} else {")[1].split("applyClipBounds")[0]
        self.assertIn("clip.outPoint =", right_branch)
        self.assertNotIn("clip.inPoint =", right_branch)
        self.assertNotIn("el.style.width = (clip.duration * pxPerSec)", body)
        self.assertIn("layoutSourceSelection(el, clip)", body)
        self.assertIn(
            "currentTime = side === \"left\" ? clip.timelineStart : clip.timelineStart + clip.duration - 0.04;",
            body,
        )
        rt = src[src.find("function renderTimeline("):src.find("function attachClipInteractions(")]
        self.assertIn("const srcMax = project.clip ? sourceMax(project.clip) : 0;", rt)
        self.assertIn("const srcPx = srcMax * pxPerSec;", rt)
        self.assertNotIn("clip.duration * pxPerSec", rt)
        self.assertNotIn("project.duration * pxPerSec", rt)

    def test_assemble_still_hold_and_fit(self):
        src = self.create
        self.assertIn("function isStillScene(", src)
        self.assertIn("/^img_/i.test(name)", src)
        self.assertIn("function sceneSetStillHold(", src)
        self.assertIn("s.freeze = true;", src)
        self.assertIn("s.duration = Math.round(d * 100) / 100;", src)
        self.assertIn("Math.max(0.4, Math.min(120, Number(sec) || 0.4))", src)
        self.assertIn("function applyFitToAll()", src)
        self.assertIn("Aplicar a todas las escenas", src)
        self.assertIn("Contain (barras negras)", src)
        self.assertIn("Cover (rellenar)", src)
        apply = src[src.find("function applyFitToAll()"):src.find("function sceneSetDuration(")]
        self.assertIn('(plan.scenes || []).forEach(function (s) { s.fit = fit; });', apply)
        self.assertIn("rebuild();", apply)
        self.assertNotIn("fetch(\"/api/sequence/scene\"", apply)
        self.assertEqual(apply.count("lockSave"), 0)
        self.assertIn('video.style.objectFit = sceneFit(s);', src)
        self.assertIn('id="stillHoldDur"', src)
        self.assertIn("function sceneDur(s)", src)
        self.assertIn("if (s.freeze)", src)

    def test_assemble_poster_until_proxy_and_mic_server_path(self):
        src = self.create
        self.assertIn('id="previewPoster"', src)
        self.assertIn("function showStagePoster(", src)
        kick = src[src.find("function kickProxy("):src.find("function posterUrl(")]
        self.assertIn("syncPreview(currentTime, isPlaying)", kick)
        preview = src[src.find("function syncPreview("):src.find("function updateHud(")]
        self.assertIn("showStagePoster(", preview)
        self.assertIn("proxyResolved(", preview)
        tools = src[src.find('id="sceneTools"'):src.find('id="editThisClip"')]
        self.assertIn('id="sceneMicBox"', tools)
        self.assertLess(tools.find('id="sceneMicBox"'), tools.find("sceneSplit()"))
        self.assertIn("Usar recording.2026-10-01-102704.caf", src)
        self.assertIn("storage/uploads/recording.2026-10-01-102704.caf", src)
        self.assertIn("Música de fondo", src)
        tag_at = src.find('<input id="sceneMicOffset"')
        self.assertGreater(tag_at, 0)
        tag = src[tag_at:src.find(">", tag_at)]
        self.assertNotIn("value=", tag)
        mic = src[src.find("async function saveSceneMic("):src.find("function sceneSplit(")]
        self.assertIn('fetch("/api/clips/replace-audio"', mic)
        self.assertIn('fd.append("audio_path", serverPath)', mic)
        self.assertIn("syncPreview(currentTime, false)", mic)
        self.assertIn("puede tardar", mic)
        self.assertNotIn('fetch("/uploads/', mic)
        self.assertNotIn(".blob()", mic)
        self.assertNotIn("/api/sequence/music", mic)
        self.assertNotIn("replaceAudio", mic)

    def test_assemble_drag_trim_matches_edit(self):
        src = self.create
        self.assertIn("function layoutSceneTrim(", src)
        self.assertIn("function startSceneEdgeDrag(", src)
        self.assertIn('class="clip-dim head"', src)
        self.assertIn('class="clip-sel"', src)
        self.assertIn('class="clip-dim tail"', src)
        self.assertIn('class="handle left"', src)
        self.assertIn('class="handle right"', src)
        self.assertIn('title="Inicio"', src)
        self.assertIn('title="Final"', src)
        self.assertIn("Inicio recortado.", src)
        self.assertIn("Final recortado. Si el tirador no avanza, no queda más video en ese lado.", src)
        start = src.find("function startSceneEdgeDrag(")
        self.assertGreater(start, 0)
        body = src[start:src.find("function sceneTrim(", start)]
        self.assertIn("s.inPoint = Math.max(0, Math.min(origIn + dt, origOut - 0.2));", body)
        self.assertIn("s.outPoint = Math.max(origIn + 0.2, Math.min(maxOut, origOut + dt));", body)
        left_branch = body.split('else if (side === "left")')[1].split("} else {")[0]
        self.assertIn("s.inPoint =", left_branch)
        self.assertNotIn("s.outPoint =", left_branch)
        right_branch = body.split("} else {")[1].split("layoutSceneTrim")[0]
        self.assertIn("s.outPoint =", right_branch)
        self.assertNotIn("s.inPoint =", right_branch)
        self.assertNotIn("replace-audio", body)
        self.assertNotIn("replaceAudio", body)
        self.assertNotIn("music.mp3", body)
        self.assertNotIn("/api/sequence/music", body)

    def test_clip_picker_uses_api_url_not_title(self):
        start = self.editor.find("async function fillClipPicker()")
        self.assertGreater(start, 0)
        end = self.editor.find("function pickLibraryClip(", start)
        body = self.editor[start:end]
        self.assertIn("encodeURIComponent(c.url)", body)
        self.assertNotIn("encodeURIComponent(c.title", body)
        self.assertIn("escapeHtml(c.title || c.name)", body)
        self.assertIn("function loadLibraryClip(url)", self.editor)

    @staticmethod
    def _js_media_file_name(path):
        import re as _re
        raw = str(path or "").split("?")[0]
        m = _re.search(r"/(?:clips|uploads)/([^/]+)$", raw, _re.I) or _re.search(r"([^/\\]+)$", raw)
        name = m.group(1) if m else ""
        if not name or not _re.search(r"\.[a-z0-9]{2,5}$", name, _re.I):
            return ""
        return name


class DirectorChatHistoryTests(unittest.TestCase):
    def _fake_grok(self):
        client = MagicMock()
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = json.dumps({"reply": "ok"})
        client.chat.completions.create.return_value = resp
        return client

    def _chat(self, history, client):
        with patch("video_create._xai_key", return_value="test-key"):
            with patch("video_create.list_library_clips", return_value=[]):
                with patch("video_create.active_project_clip_names", return_value=None):
                    with patch("video_create.apply_director_command", return_value=None):
                        with patch("openai.OpenAI", return_value=client):
                            return chat_edit_plan(
                                "qué opinas del ritmo?",
                                history=history,
                                current_plan={},
                            )

    def _forwarded_hist(self, client):
        kwargs = client.chat.completions.create.call_args.kwargs
        messages = kwargs["messages"]
        return messages[1:-1]

    def test_empty_history_does_not_raise(self):
        client = self._fake_grok()
        out = self._chat([], client)
        self.assertEqual(out.get("reply"), "ok")
        self.assertEqual(self._forwarded_hist(client), [])

    def test_three_turns_does_not_raise(self):
        client = self._fake_grok()
        history = [
            {"role": "user", "content": "uno"},
            {"role": "assistant", "content": "dos"},
            {"role": "user", "content": "tres"},
        ]
        out = self._chat(history, client)
        self.assertEqual(out.get("reply"), "ok")
        hist = self._forwarded_hist(client)
        self.assertEqual(len(hist), 3)
        self.assertEqual([h["content"] for h in hist], ["uno", "dos", "tres"])

    def test_twenty_turns_forwards_last_sixteen(self):
        client = self._fake_grok()
        history = [
            {
                "role": "user" if i % 2 == 0 else "assistant",
                "content": "turn-%d" % i,
            }
            for i in range(20)
        ]
        out = self._chat(history, client)
        self.assertEqual(out.get("reply"), "ok")
        hist = self._forwarded_hist(client)
        self.assertEqual(len(hist), 16)
        self.assertEqual(
            [h["content"] for h in hist],
            ["turn-%d" % i for i in range(4, 20)],
        )


class CaptionAssOpacityTests(unittest.TestCase):
    def _dialogue(self, ass: str) -> str:
        for line in ass.splitlines():
            if line.startswith("Dialogue:"):
                return line
        self.fail("no Dialogue line in ASS")

    def test_opacity_fully_opaque_emits_1a_00_and_6digit_1c(self):
        ass = _texts_to_ass([{
            "content": "hola",
            "start": 0,
            "end": 2,
            "opacity": 1.0,
            "color": "ffffff",
        }])
        d = self._dialogue(ass)
        self.assertEqual(_ass_alpha(1.0), "&H00&")
        self.assertIn("\\1a&H00&", d)
        self.assertRegex(d, r"\\1c&H[0-9A-Fa-f]{6}&")
        self.assertNotRegex(d, r"\\1c&H[0-9A-Fa-f]{8}&")

    def test_opacity_point_four_emits_1a_99(self):
        ass = _texts_to_ass([{
            "content": "hola",
            "start": 0,
            "end": 2,
            "opacity": 0.4,
            "color": "ffffff",
        }])
        d = self._dialogue(ass)
        self.assertEqual(_ass_alpha(0.4), "&H99&")
        self.assertIn("\\1a&H99&", d)
        self.assertRegex(d, r"\\1c&H[0-9A-Fa-f]{6}&")
        self.assertNotRegex(d, r"\\1c&H[0-9A-Fa-f]{8}&")

    def test_highlight_box_opacity_emits_3a_and_6digit_3c(self):
        for bop, aa in ((1.0, "00"), (0.4, "99")):
            ass = _texts_to_ass([{
                "content": "hola",
                "start": 0,
                "end": 2,
                "opacity": 1.0,
                "color": "ffffff",
                "highlight": "ffff00",
                "boxOpacity": bop,
            }])
            d = self._dialogue(ass)
            self.assertIn("\\3a&H%s&" % aa, d)
            self.assertRegex(d, r"\\3c&H[0-9A-Fa-f]{6}&")
            self.assertNotRegex(d, r"\\3c&H[0-9A-Fa-f]{8}&")


class ClipPublishAndScenePatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name)
        self.clips = lib(("rio.mp4", 12), ("humo.mp4", 8), ("mapa.mp4", 6))
        self.media = self.folder / "media"
        self.media.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def _dump(self, scene):
        return json.dumps(scene, sort_keys=True, ensure_ascii=False)

    def _seed(self):
        seq, _ = save_sequence(
            {
                "title": "Doc",
                "scenes": [
                    {"id": "s-a", "clip": "rio.mp4", "inPoint": 0, "outPoint": 3, "text": "uno"},
                    {"id": "s-b", "clip": "humo.mp4", "inPoint": 1, "outPoint": 4, "text": "dos"},
                    {"id": "s-c", "clip": "mapa.mp4", "inPoint": 0, "outPoint": 2, "text": "tres"},
                ],
            },
            folder=self.folder,
            library=self.clips,
            force=True,
        )
        return seq

    def _scene_patches(self):
        def _load(*_a, **_k):
            return load_sequence(folder=self.folder, library=self.clips)

        def _save(raw, force=False, folder=None, library=None, **_k):
            return save_sequence(
                raw, folder=self.folder, library=self.clips, force=force,
            )

        return _load, _save

    def test_publish_new_file_leaves_sequence_bytes(self):
        if not check_ffmpeg():
            self.skipTest("ffmpeg not installed")
        from fastapi.testclient import TestClient
        from main import app
        from video_create import sequence_paths
        from video_tools import _run, ffmpeg_bin
        ff = ffmpeg_bin()
        src = self.media / "src_pub.mp4"
        cmd = [
            ff, "-y",
            "-f", "lavfi", "-i", "color=c=blue:s=160x90:d=2:r=15",
            "-c:v", "mpeg4", "-an", str(src),
        ]
        r = _run(cmd)
        if r.returncode != 0 or not src.exists() or src.stat().st_size < 500:
            self.skipTest("could not mux publish fixture")
        seq_path, _bak = sequence_paths()
        before = seq_path.read_bytes() if seq_path.is_file() else b""
        out_dir = self.media / "clips_out"
        out_dir.mkdir()
        with patch("main._resolve_media_src", return_value=src):
            with patch("video_tools.CLIPS_DIR", out_dir):
                with patch("main.attach_clip_to_active_project", return_value=[]):
                    with patch("main.save_sequence") as no_save:
                        no_save.side_effect = AssertionError("publish must not write sequence")
                        client = TestClient(app)
                        res = client.post("/api/clips/publish", json={
                            "src": "/clips/src_pub.mp4",
                            "start": 0,
                            "end": 1,
                            "output_name": "pub_edit",
                            "extras": {
                                "speed": 1,
                                "volume": 1,
                                "muted": True,
                                "rotation": 0,
                                "filter": "none",
                                "fadeIn": False,
                                "fadeOut": False,
                                "freeze": False,
                            },
                        })
        after = seq_path.read_bytes() if seq_path.is_file() else b""
        self.assertEqual(before, after, "publish must leave sequence.json byte-identical")
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()
        self.assertTrue(data.get("ok"))
        self.assertTrue(data.get("filename"))
        self.assertTrue(str(data.get("path") or "").startswith("/clips/"))
        baked = out_dir / data["filename"]
        self.assertTrue(baked.is_file(), data)
        self.assertGreater(baked.stat().st_size, 100)
        self.assertNotEqual(baked.resolve(), src.resolve())
        self.assertIsNotNone(data.get("duration"))

    def test_scene_patch_only_named_scene(self):
        from fastapi.testclient import TestClient
        from main import app
        seq = self._seed()
        others_before = {
            s["id"]: self._dump(s) for s in seq["scenes"] if s.get("id") != "s-b"
        }
        _load, _save = self._scene_patches()
        with patch("main.load_sequence", _load), patch("main.save_sequence", _save):
            client = TestClient(app)
            res = client.post("/api/sequence/scene", json={
                "id": "s-b",
                "props": {"text": "humo editado", "muted": True},
            })
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()
        self.assertTrue(data.get("ok"))
        by_id = {s["id"]: s for s in data["sequence"]["scenes"]}
        self.assertEqual(by_id["s-b"]["text"], "humo editado")
        self.assertTrue(by_id["s-b"]["muted"])
        for sid, blob in others_before.items():
            self.assertEqual(self._dump(by_id[sid]), blob, sid)

    def test_scene_patch_rejects_scenes_array(self):
        from fastapi.testclient import TestClient
        from main import app
        seq = self._seed()
        _load, _save = self._scene_patches()
        with patch("main.load_sequence", _load), patch("main.save_sequence", _save):
            client = TestClient(app)
            res = client.post("/api/sequence/scene", json={
                "id": "s-a",
                "props": {"text": "x"},
                "scenes": [{"id": "s-a", "text": "hijack"}],
            })
        self.assertEqual(res.status_code, 400, res.text)
        loaded, _ = load_sequence(folder=self.folder, library=self.clips)
        self.assertEqual(loaded["rev"], seq["rev"])
        self.assertEqual(loaded["scenes"][0]["text"], "uno")

    def test_scene_patch_fit_survives(self):
        from fastapi.testclient import TestClient
        from main import app
        seq = self._seed()
        _load, _save = self._scene_patches()
        with patch("main.load_sequence", _load), patch("main.save_sequence", _save):
            client = TestClient(app)
            res = client.post("/api/sequence/scene", json={
                "id": "s-b",
                "props": {"fit": "cover"},
            })
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()
        by_id = {s["id"]: s for s in data["sequence"]["scenes"]}
        self.assertEqual(by_id["s-b"]["fit"], "cover")
        self.assertEqual(by_id["s-a"]["fit"], "contain")

    def test_scene_patch_unknown_prop_400(self):
        from fastapi.testclient import TestClient
        from main import app
        seq = self._seed()
        _load, _save = self._scene_patches()
        with patch("main.load_sequence", _load), patch("main.save_sequence", _save):
            client = TestClient(app)
            res = client.post("/api/sequence/scene", json={
                "id": "s-a",
                "props": {"narration": "nope"},
            })
        self.assertEqual(res.status_code, 400, res.text)
        loaded, _ = load_sequence(folder=self.folder, library=self.clips)
        self.assertEqual(loaded["rev"], seq["rev"])

    def test_scene_patch_bad_id_404(self):
        from fastapi.testclient import TestClient
        from main import app
        seq = self._seed()
        _load, _save = self._scene_patches()
        with patch("main.load_sequence", _load), patch("main.save_sequence", _save):
            client = TestClient(app)
            res = client.post("/api/sequence/scene", json={
                "id": "does-not-exist",
                "props": {"text": "x"},
            })
        self.assertEqual(res.status_code, 404, res.text)
        loaded, _ = load_sequence(folder=self.folder, library=self.clips)
        self.assertEqual(loaded["rev"], seq["rev"])

    def test_scene_patch_rev_increments_once_per_call(self):
        from fastapi.testclient import TestClient
        from main import app
        seq = self._seed()
        r0 = seq["rev"]
        _load, _save = self._scene_patches()
        with patch("main.load_sequence", _load), patch("main.save_sequence", _save):
            client = TestClient(app)
            a = client.post("/api/sequence/scene", json={
                "id": "s-a", "props": {"text": "a1"},
            })
            b = client.post("/api/sequence/scene", json={
                "id": "s-a", "props": {"text": "a2"},
            })
        self.assertEqual(a.status_code, 200, a.text)
        self.assertEqual(b.status_code, 200, b.text)
        self.assertEqual(a.json()["rev"], r0 + 1)
        self.assertEqual(b.json()["rev"], r0 + 2)
        loaded, _ = load_sequence(folder=self.folder, library=self.clips)
        self.assertEqual(loaded["rev"], r0 + 2)
        self.assertEqual(loaded["scenes"][0]["text"], "a2")


def _rms_windows(path, win=0.1, rate=8000):
    """Per-window RMS of the file's audio. Used to see offset and dropped camera tone."""
    from video_tools import _run_bytes, ffmpeg_bin
    ff = ffmpeg_bin()
    cmd = [
        ff, "-hide_banner", "-loglevel", "error",
        "-i", str(path),
        "-vn", "-ac", "1", "-ar", str(rate),
        "-f", "s16le", "pipe:1",
    ]
    result = _run_bytes(cmd, timeout=30)
    pcm = result.stdout or b""
    if len(pcm) % 2:
        pcm = pcm[:-1]
    samples = array.array("h")
    samples.frombytes(pcm)
    n = max(1, int(win * rate))
    out = []
    for i in range(0, len(samples) - n + 1, n):
        chunk = samples[i:i + n]
        acc = sum(s * s for s in chunk) / len(chunk)
        out.append(acc ** 0.5)
    return out


def _first_loud(windows, thresh=400.0):
    for i, v in enumerate(windows):
        if v > thresh:
            return i
    return None


class ReplaceMicAudioTests(unittest.TestCase):
    """Podcast mic onto a Library video. New clip only. Bed and Sequence stay put."""

    def setUp(self):
        if not check_ffmpeg():
            self.skipTest("ffmpeg not installed")
        self.tmp = tempfile.TemporaryDirectory()
        self.clips = Path(self.tmp.name) / "clips"
        self.exports = Path(self.tmp.name) / "exports_tmp"
        self.clips.mkdir()
        self.exports.mkdir()
        self.media = Path(self.tmp.name) / "media"
        self.media.mkdir()
        from video_create import sequence_music_path, sequence_paths
        self.seq_path, _bak = sequence_paths()
        self.music_path = sequence_music_path()
        self.seq_existed = self.seq_path.is_file()
        self.music_existed = self.music_path.is_file()
        self.seq_before = self.seq_path.read_bytes() if self.seq_existed else None
        self.music_before = self.music_path.read_bytes() if self.music_existed else None
        self.bak_path = self.seq_path.with_name("sequence.bak.json")
        self.bak_existed = self.bak_path.is_file()
        self.bak_before = self.bak_path.read_bytes() if self.bak_existed else None
        if not self.music_existed:
            self.music_path.parent.mkdir(parents=True, exist_ok=True)
            self.music_path.write_bytes(b"\xff\xfb" + b"\x11" * 480)
            self.music_before = self.music_path.read_bytes()
        if not self.seq_existed:
            self.seq_path.parent.mkdir(parents=True, exist_ok=True)
            self.seq_path.write_text('{"version":1,"rev":1,"scenes":[]}', encoding="utf-8")
            self.seq_before = self.seq_path.read_bytes()

    def tearDown(self):
        try:
            if self.seq_existed:
                self.seq_path.write_bytes(self.seq_before)
            elif self.seq_path.is_file():
                self.seq_path.unlink()
            if self.bak_existed:
                self.bak_path.write_bytes(self.bak_before)
            elif self.bak_path.is_file():
                self.bak_path.unlink()
            if self.music_existed:
                self.music_path.write_bytes(self.music_before)
            elif self.music_path.is_file():
                self.music_path.unlink()
        except Exception:
            pass
        self.tmp.cleanup()

    def _sha(self, path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _ff(self, cmd):
        from video_tools import _run
        return _run(cmd)

    def _video(self, dest: Path):
        ff = ffmpeg_bin()
        cmd = [
            ff, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "color=c=blue:s=160x90:d=2:r=15",
            "-f", "lavfi", "-i", "sine=frequency=1000:duration=2:sample_rate=48000",
            "-c:v", "mpeg4", "-c:a", "aac", "-shortest", str(dest),
        ]
        r = self._ff(cmd)
        if r.returncode != 0 or not dest.is_file() or dest.stat().st_size < 500:
            self.skipTest("could not mux camera fixture")

    def _mic(self, dest: Path, head=0.4, tone=1.6):
        ff = ffmpeg_bin()
        cmd = [
            ff, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", f"anullsrc=r=48000:cl=mono:d={head}",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={tone}:sample_rate=48000",
            "-filter_complex", "[0:a][1:a]concat=n=2:v=0:a=1",
            "-c:a", "pcm_s16le", str(dest),
        ]
        r = self._ff(cmd)
        if r.returncode != 0 or not dest.is_file():
            self.skipTest("could not mux mic fixture")

    def _client(self):
        from fastapi.testclient import TestClient
        from main import app
        return TestClient(app)

    def _dirs(self):
        return (
            patch("main.CLIPS_DIR", self.clips),
            patch("main.EXPORTS_TMP", self.exports),
        )

    def _post(self, client, clip_id, offset, audio_path, label=None, preview=False):
        url = "/api/clips/replace-audio/preview" if preview else "/api/clips/replace-audio"
        data = {"clip_id": clip_id}
        if offset is not None:
            data["offset_ms"] = offset
        if label is not None:
            data["label"] = label
        with audio_path.open("rb") as fh:
            files = {"audio": (audio_path.name, fh, "audio/wav")}
            return client.post(url, data=data, files=files)

    def _assert_bed_and_sequence(self):
        self.assertEqual(self.seq_path.read_bytes(), self.seq_before)
        self.assertEqual(self.music_path.read_bytes(), self.music_before)
        self.assertTrue(self.music_path.is_file())
        self.assertEqual(self.music_path.name, "music.mp3")

    def test_new_clip_source_unchanged_sequence_and_music_untouched(self):
        src = self.clips / "pixel_interview.mp4"
        mic = self.media / "ocenaudio.wav"
        self._video(src)
        self._mic(mic)
        src_hash = self._sha(src)
        src_mtime = src.stat().st_mtime_ns
        mic_hash = self._sha(mic)
        attached = []

        def _attach(name):
            attached.append(name)
            return [name]

        with self._dirs()[0], self._dirs()[1]:
            with patch("main.attach_clip_to_active_project", side_effect=_attach):
                with patch("main.save_sequence", side_effect=AssertionError("sequence write")):
                    with patch("main.persist_sequence_music", side_effect=AssertionError("bed music")):
                        client = self._client()
                        res = self._post(client, "pixel_interview.mp4", "0", mic, label="entrevista_mic")
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()
        self.assertTrue(data.get("ok"))
        self.assertFalse(data.get("preview"))
        self.assertEqual(data.get("offset_ms"), 0)
        self.assertEqual(data.get("source"), "pixel_interview.mp4")
        self.assertTrue(str(data.get("path") or "").startswith("/clips/"))
        baked = self.clips / data["filename"]
        self.assertTrue(baked.is_file(), data)
        self.assertNotEqual(baked.resolve(), src.resolve())
        self.assertEqual(baked.suffix.lower(), ".mp4")
        self.assertTrue(baked.name.startswith("entrevista_mic"))
        self.assertEqual(self._sha(src), src_hash)
        self.assertEqual(src.stat().st_mtime_ns, src_mtime)
        self.assertEqual(self._sha(mic), mic_hash)
        self.assertEqual(attached, [baked.name])
        from video_tools import get_video_info
        info = get_video_info(baked)
        kinds = [s.get("codec_type") for s in info.get("streams") or []]
        self.assertEqual(kinds.count("video"), 1)
        self.assertEqual(kinds.count("audio"), 1)
        self.assertAlmostEqual(probe_duration(baked) or 0, probe_duration(src) or 0, delta=0.15)
        with self._dirs()[0]:
            client = self._client()
            listed = client.get("/api/clips")
        names = [c["name"] for c in listed.json().get("clips") or []]
        self.assertIn(baked.name, names)
        self.assertIn("pixel_interview.mp4", names)
        self._assert_bed_and_sequence()

    def test_offset_required(self):
        src = self.clips / "pixel_interview.mp4"
        mic = self.media / "ocenaudio.wav"
        self._video(src)
        self._mic(mic)
        before = {p.name for p in self.clips.iterdir()}
        with self._dirs()[0], self._dirs()[1]:
            client = self._client()
            missing = self._post(client, src.name, None, mic)
            blank = self._post(client, src.name, "  ", mic)
            frac = self._post(client, src.name, "1.5", mic)
            word = self._post(client, src.name, "auto", mic)
        for res in (missing, blank, frac, word):
            self.assertEqual(res.status_code, 400, res.text)
        self.assertEqual({p.name for p in self.clips.iterdir()}, before)
        self._assert_bed_and_sequence()

    def test_positive_and_negative_offset(self):
        src = self.clips / "pixel_interview.mp4"
        mic = self.media / "ocenaudio.wav"
        self._video(src)
        self._mic(mic)
        with self._dirs()[0], self._dirs()[1]:
            with patch("main.attach_clip_to_active_project", return_value=[]):
                client = self._client()
                pos = self._post(client, src.name, "500", mic, label="mic_plus")
                neg = self._post(client, src.name, "-300", mic, label="mic_minus")
        self.assertEqual(pos.status_code, 200, pos.text)
        self.assertEqual(neg.status_code, 200, neg.text)
        self.assertEqual(pos.json().get("offset_ms"), 500)
        self.assertEqual(neg.json().get("offset_ms"), -300)
        plus = _first_loud(_rms_windows(self.clips / pos.json()["filename"]))
        minus = _first_loud(_rms_windows(self.clips / neg.json()["filename"]))
        self.assertIsNotNone(plus)
        self.assertIsNotNone(minus)
        self.assertAlmostEqual(plus, 9, delta=1)
        self.assertAlmostEqual(minus, 1, delta=1)
        self.assertGreater(plus, minus)
        self._assert_bed_and_sequence()

    def test_camera_audio_discarded(self):
        src = self.clips / "pixel_interview.mp4"
        mic = self.media / "ocenaudio.wav"
        self._video(src)
        self._mic(mic)
        cam = _rms_windows(src)
        self.assertGreater(sum(cam[:3]) / 3.0, 500)
        with self._dirs()[0], self._dirs()[1]:
            with patch("main.attach_clip_to_active_project", return_value=[]):
                client = self._client()
                res = self._post(client, src.name, "0", mic, label="mic_only")
        self.assertEqual(res.status_code, 200, res.text)
        out = _rms_windows(self.clips / res.json()["filename"])
        self.assertLess(sum(out[:3]) / 3.0, 80)
        loud = _first_loud(out)
        self.assertIsNotNone(loud)
        self.assertAlmostEqual(loud, 4, delta=1)
        from video_tools import get_video_info
        info = get_video_info(self.clips / res.json()["filename"])
        audio = [s for s in info.get("streams") or [] if s.get("codec_type") == "audio"]
        self.assertEqual(len(audio), 1)

    def test_preview_stays_in_exports_tmp(self):
        src = self.clips / "pixel_interview.mp4"
        mic = self.media / "ocenaudio.wav"
        self._video(src)
        self._mic(mic)
        with self._dirs()[0], self._dirs()[1]:
            with patch("main.attach_clip_to_active_project", side_effect=AssertionError("preview registers")):
                client = self._client()
                res = self._post(client, src.name, "-120", mic, preview=True)
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()
        self.assertTrue(data.get("preview"))
        self.assertEqual(data.get("offset_ms"), -120)
        self.assertFalse(str(data.get("url") or "").startswith("/clips/"))
        preview = self.exports / data["filename"]
        self.assertTrue(preview.is_file(), data)
        self.assertEqual(preview.parent.resolve(), self.exports.resolve())
        names = [p.name for p in self.clips.iterdir() if p.is_file()]
        self.assertEqual(names, ["pixel_interview.mp4"])
        self._assert_bed_and_sequence()

    def test_unknown_clip_404(self):
        mic = self.media / "ocenaudio.wav"
        self._mic(mic)
        with self._dirs()[0], self._dirs()[1]:
            client = self._client()
            res = self._post(client, "no-such-interview.mp4", "0", mic)
        self.assertEqual(res.status_code, 404, res.text)
        self.assertEqual(list(self.clips.iterdir()), [])
        self.assertEqual(list(self.exports.iterdir()), [])
        self._assert_bed_and_sequence()

    def test_caf_and_ui_contract(self):
        src = self.clips / "pixel_interview.mp4"
        wav = self.media / "ocenaudio.wav"
        caf = self.media / "ocenaudio.caf"
        self._video(src)
        self._mic(wav)
        ff = ffmpeg_bin()
        r = self._ff([
            ff, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(wav), "-c:a", "pcm_s16le", str(caf),
        ])
        if r.returncode != 0 or not caf.is_file():
            self.skipTest("could not write caf")
        with self._dirs()[0], self._dirs()[1]:
            with patch("main.attach_clip_to_active_project", return_value=[]):
                client = self._client()
                with caf.open("rb") as fh:
                    res = client.post(
                        "/api/clips/replace-audio",
                        data={"clip_id": src.name, "offset_ms": "0", "label": "desde_caf"},
                        files={"audio": ("toma.caf", fh, "audio/x-caf")},
                    )
        self.assertEqual(res.status_code, 200, res.text)
        self.assertTrue((self.clips / res.json()["filename"]).is_file())
        root = Path(__file__).resolve().parent.parent
        library = (root / "frontend" / "index.html").read_text(encoding="utf-8")
        editor = (root / "frontend" / "editor" / "index.html").read_text(encoding="utf-8")
        assemble = (root / "frontend" / "create" / "index.html").read_text(encoding="utf-8")
        for page in (library, editor):
            self.assertIn("Audio de micrófono", page)
            self.assertIn("Desfase (ms)", page)
            self.assertIn(">Probar<", page)
            self.assertIn("Guardar clip nuevo", page)
            self.assertIn('fetch("/api/clips/replace-audio"', page)
            self.assertIn('fetch("/api/clips/replace-audio/preview"', page)
        self.assertIn("Audio de micrófono", assemble)
        self.assertIn("Desfase (ms)", assemble)
        self.assertIn("Guardar en esta escena", assemble)
        self.assertIn(
            "Crea un clip nuevo en Library y lo pone en esta escena. El original no se toca. No es la música de fondo.",
            assemble,
        )
        self.assertIn("Música de fondo", assemble)
        self.assertIn("storage/uploads/recording.2026-10-01-102704.caf", assemble)
        self.assertIn('fetch("/api/clips/replace-audio"', assemble)
        self.assertNotIn("replaceAudio", assemble)
        mic_fn = assemble[assemble.find("async function saveSceneMic("):assemble.find("function sceneSplit(")]
        self.assertIn('fetch("/api/clips/replace-audio"', mic_fn)
        self.assertNotIn("/api/sequence/music", mic_fn)
        self.assertNotIn("replaceAudio", mic_fn)
        self._assert_bed_and_sequence()

    def test_audio_path_reads_uploads_and_offset_stays_required(self):
        src = self.clips / "pixel_interview.mp4"
        wav = self.media / "ocenaudio.wav"
        uploads = Path(self.tmp.name) / "uploads"
        uploads.mkdir()
        caf = uploads / "recording.2026-10-01-102704.caf"
        self._video(src)
        self._mic(wav)
        ff = ffmpeg_bin()
        r = self._ff([
            ff, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(wav), "-c:a", "pcm_s16le", str(caf),
        ])
        if r.returncode != 0 or not caf.is_file():
            self.skipTest("could not write caf")
        caf_hash = self._sha(caf)
        src_hash = self._sha(src)
        before = {p.name for p in self.clips.iterdir()}
        with self._dirs()[0], self._dirs()[1], patch("main.UPLOADS_DIR", uploads):
            with patch("main.attach_clip_to_active_project", return_value=[]):
                with patch("main.persist_sequence_music", side_effect=AssertionError("bed music")):
                    client = self._client()
                    missing = client.post("/api/clips/replace-audio", data={
                        "clip_id": src.name,
                        "audio_path": "storage/uploads/recording.2026-10-01-102704.caf",
                    })
                    bed = client.post("/api/clips/replace-audio", data={
                        "clip_id": src.name,
                        "offset_ms": "0",
                        "audio_path": "storage/autosave/music.mp3",
                    })
                    baked = client.post("/api/clips/replace-audio", data={
                        "clip_id": src.name,
                        "offset_ms": "0",
                        "audio_path": "/uploads/recording.2026-10-01-102704.caf",
                    })
        self.assertEqual(missing.status_code, 400, missing.text)
        self.assertEqual(bed.status_code, 400, bed.text)
        self.assertEqual(baked.status_code, 200, baked.text)
        data = baked.json()
        self.assertEqual(data.get("offset_ms"), 0)
        self.assertEqual(data.get("source"), src.name)
        out = self.clips / data["filename"]
        self.assertTrue(out.is_file())
        self.assertNotEqual(out.name, src.name)
        self.assertEqual({p.name for p in self.clips.iterdir()}, before | {out.name})
        self.assertEqual(self._sha(src), src_hash)
        self.assertEqual(self._sha(caf), caf_hash)
        self._assert_bed_and_sequence()

    def test_scene_ref_save_after_bake_rejects_stale_rev(self):
        src = self.clips / "pixel_interview.mp4"
        mic = self.media / "ocenaudio.wav"
        self._video(src)
        self._mic(mic)
        src_hash = self._sha(src)
        from video_create import load_sequence, save_sequence
        lib = [
            {"name": "pixel_interview.mp4", "duration": 2.0},
            {"name": "otro.mp4", "duration": 2.0},
        ]
        save_sequence({
            "title": "Entrevista",
            "format": "youtube",
            "rev": 1,
            "scenes": [
                {"id": "s-a", "clip": "pixel_interview.mp4", "inPoint": 0, "outPoint": 2},
                {"id": "s-b", "clip": "otro.mp4", "inPoint": 0, "outPoint": 2, "text": "sigue"},
            ],
        }, library=lib, force=True)
        seeded = self.seq_path.read_bytes()
        music_now = self.music_path.read_bytes()
        with self._dirs()[0], self._dirs()[1]:
            with patch("main.attach_clip_to_active_project", return_value=[]):
                with patch("main.persist_sequence_music", side_effect=AssertionError("bed music")):
                    client = self._client()
                    baked_res = self._post(client, src.name, "0", mic, label="escena_mic")
        self.assertEqual(baked_res.status_code, 200, baked_res.text)
        baked_name = baked_res.json()["filename"]
        self.assertEqual(self.seq_path.read_bytes(), seeded)
        self.assertEqual(self._sha(src), src_hash)
        self.assertEqual(self.music_path.read_bytes(), music_now)
        seq, _src = load_sequence()
        self.assertEqual(seq["scenes"][0]["clip"], "pixel_interview.mp4")
        self.assertEqual(seq["scenes"][1]["clip"], "otro.mp4")
        self.assertNotIn("replaceAudio", seq["scenes"][0])
        used_rev = seq["rev"]
        seq["scenes"][0]["clip"] = baked_name
        seq["scenes"][0]["src"] = "/clips/" + baked_name
        seq["scenes"][0]["url"] = "/clips/" + baked_name
        lib2 = lib + [{"name": baked_name, "duration": 2.0}]
        with patch("video_create.list_library_clips", return_value=lib2):
            client = self._client()
            saved = client.post("/api/sequence", json=seq)
        self.assertEqual(saved.status_code, 200, saved.text)
        body = saved.json()
        self.assertTrue(body.get("ok"), body)
        scenes = body["sequence"]["scenes"]
        self.assertEqual(scenes[0]["id"], "s-a")
        self.assertEqual(scenes[0]["clip"], baked_name)
        self.assertEqual(scenes[1]["id"], "s-b")
        self.assertEqual(scenes[1]["clip"], "otro.mp4")
        self.assertEqual(scenes[1].get("text"), "sigue")
        self.assertNotIn("replaceAudio", scenes[0])
        self.assertNotIn("replaceAudio", scenes[1])
        self.assertEqual(self._sha(src), src_hash)
        self.assertEqual(self.music_path.read_bytes(), music_now)
        disk_after = self.seq_path.read_bytes()
        self.assertNotEqual(disk_after, seeded)
        stale_body = json.loads(json.dumps(body["sequence"]))
        stale_body["rev"] = used_rev
        stale_body["scenes"][0]["clip"] = "should-not-land.mp4"
        stale_body["scenes"][1]["clip"] = "should-not-land.mp4"
        with patch("video_create.list_library_clips", return_value=lib2):
            client = self._client()
            stale = client.post("/api/sequence", json=stale_body)
        self.assertEqual(stale.status_code, 200, stale.text)
        stale_data = stale.json()
        self.assertFalse(stale_data.get("ok"))
        self.assertTrue(stale_data.get("stale"))
        self.assertEqual(stale_data["source"], "stale")
        self.assertEqual(self.seq_path.read_bytes(), disk_after)
        disk_seq, _disk_src = load_sequence()
        self.assertEqual(disk_seq["scenes"][0]["clip"], baked_name)
        self.assertEqual(disk_seq["scenes"][1]["clip"], "otro.mp4")
        self.assertEqual(self._sha(src), src_hash)
        self.assertEqual(self.music_path.read_bytes(), music_now)


if __name__ == "__main__":
    unittest.main()



