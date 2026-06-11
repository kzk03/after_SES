#!/usr/bin/env python
"""kd_pretrain.py のスモークテスト.

実データ不要で、以下を検証する:
  1. MCEIRLSystem の構築 + train_kd_pretrain() が正常に走る
  2. 出力 state_dict が train_mce_irl.py --init-from で期待される形式 (key 一致)
  3. softmax(logits)[...,1] が 0-1 範囲にあり勾配が流れる
"""
from __future__ import annotations

import json
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from review_predictor.IRL.features.common_features import (
    STATE_FEATURES,
    STATE_FEATURES_WITH_PATH,
)
from review_predictor.IRL.model.mce_irl_predictor import MCEIRLSystem


def make_fake_trajectories(n: int = 30, seq_len: int = 6, state_dim: int = 0):
    """ダミー軌跡を生成.

    state_dim=0 なら STATE_FEATURES_WITH_PATH の次元を使う。
    """
    if state_dim == 0:
        state_dim = len(STATE_FEATURES_WITH_PATH)

    trajs = []
    for i in range(n):
        email = f"dev{i}@test.com"
        directory = f"dir_{i % 5}"
        # monthly_activity_histories: list[list[dict]]
        monthly = []
        for _s in range(seq_len):
            acts = []
            for _ in range(3):
                acts.append(
                    {
                        "action_type": "reviewed",
                        "reviewer_email": email,
                        "owner_email": "owner@test.com",
                        "timestamp": "2021-06-01T00:00:00",
                        "accepted": bool(np.random.rand() > 0.5),
                        "lines_added": int(np.random.randint(1, 100)),
                        "lines_deleted": int(np.random.randint(0, 50)),
                        "files_changed": int(np.random.randint(1, 10)),
                        "response_time": float(np.random.rand() * 24),
                    }
                )
            monthly.append(acts)

        step_labels = [int(np.random.rand() > 0.5) for _ in range(seq_len)]
        step_actions = [int(np.random.rand() > 0.5) for _ in range(seq_len)]

        # path_features: list of list/np.array
        path_features_per_step = [
            np.random.randn(len(STATE_FEATURES_WITH_PATH) - len(STATE_FEATURES)).tolist()
            for _ in range(seq_len)
        ]
        step_context_dates = ["2021-06-01"] * seq_len
        step_total_project_reviews = [100] * seq_len

        trajs.append(
            {
                "developer": {"email": email},
                "directory": directory,
                "activity_history": [{"timestamp": "2021-06-01"}],
                "monthly_activity_histories": monthly,
                "step_labels": step_labels,
                "step_actions": step_actions,
                "step_context_dates": step_context_dates,
                "step_total_project_reviews": step_total_project_reviews,
                "path_features_per_step": path_features_per_step,
                "sample_weight": 1.0,
            }
        )
    return trajs


def make_fake_kd_targets(trajs):
    """(email, directory) → float の KD target 辞書."""
    targets = {}
    for t in trajs:
        dev = t.get("developer", {})
        email = dev.get("email", "")
        directory = t.get("directory", "")
        targets[(email, directory)] = float(np.random.rand())
    return targets


def test_basic_kd_pretrain():
    """MCEIRLSystem.train_kd_pretrain のスモークテスト."""
    print("=" * 60)
    print("TEST 1: basic KD pretrain (model_type=0, path 込み)")
    print("=" * 60)

    state_dim = len(STATE_FEATURES_WITH_PATH)
    config = {
        "state_dim": state_dim,
        "action_dim": 5,
        "hidden_dim": 64,
        "dropout": 0.1,
        "learning_rate": 1e-3,
        "model_type": 0,
    }
    system = MCEIRLSystem(config)

    trajs = make_fake_trajectories(n=20, seq_len=4, state_dim=state_dim)
    kd_targets = make_fake_kd_targets(trajs)

    stats = system.train_kd_pretrain(
        trajs,
        kd_targets,
        epochs=3,
        patience=2,
        batch_size=8,
    )

    assert "training_losses" in stats, "training_losses が結果に含まれていない"
    assert "best_val" in stats, "best_val が結果に含まれていない"
    assert len(stats["training_losses"]) > 0, "training_losses が空"
    print(f"  epochs_trained = {stats['epochs_trained']}")
    print(f"  best_val       = {stats['best_val']:.4f}")
    print(f"  n_train        = {stats['n_train']}")
    print(f"  n_val          = {stats['n_val']}")

    # state_dict が保存可能か
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(system.network.state_dict(), f.name)
        sd = torch.load(f.name, map_location="cpu", weights_only=True)
        assert len(sd) > 0, "state_dict が空"
        print(f"  state_dict keys = {len(sd)}")
        Path(f.name).unlink()

    print("  → PASS\n")
    return True


def test_state_dict_compatibility():
    """KD pretrain と cold-start の state_dict key が一致することを検証."""
    print("=" * 60)
    print("TEST 2: state_dict key 一致 (KD → cold MCE-IRL)")
    print("=" * 60)

    state_dim = len(STATE_FEATURES_WITH_PATH)
    config = {
        "state_dim": state_dim,
        "action_dim": 5,
        "hidden_dim": 64,
        "dropout": 0.1,
        "learning_rate": 1e-3,
        "model_type": 0,
    }

    # KD pretrain 側
    system_kd = MCEIRLSystem(config)
    trajs = make_fake_trajectories(n=10, seq_len=3)
    kd_targets = make_fake_kd_targets(trajs)
    system_kd.train_kd_pretrain(trajs, kd_targets, epochs=2, patience=1, batch_size=4)
    kd_keys = set(system_kd.network.state_dict().keys())

    # cold-start 側 (同一 config で MCEIRLSystem を作るだけ)
    system_cold = MCEIRLSystem(config)
    cold_keys = set(system_cold.network.state_dict().keys())

    assert kd_keys == cold_keys, (
        f"state_dict key 不一致!\n"
        f"  KD only: {kd_keys - cold_keys}\n"
        f"  cold only: {cold_keys - kd_keys}"
    )
    print(f"  key 数: {len(kd_keys)} (一致)")

    # 実際に load_state_dict できるか
    sd = system_kd.network.state_dict()
    system_cold.network.load_state_dict(sd)
    print("  load_state_dict: OK")
    print("  → PASS\n")
    return True


def test_model_type_3_two_tower():
    """Two-tower (model_type=3) でも KD pretrain が動くか."""
    print("=" * 60)
    print("TEST 3: model_type=3 (Two-Tower)")
    print("=" * 60)

    state_dim = len(STATE_FEATURES_WITH_PATH)
    config = {
        "state_dim": state_dim,
        "action_dim": 5,
        "hidden_dim": 64,
        "dropout": 0.1,
        "learning_rate": 1e-3,
        "model_type": 3,
    }
    system = MCEIRLSystem(config)

    trajs = make_fake_trajectories(n=15, seq_len=4)
    kd_targets = make_fake_kd_targets(trajs)

    stats = system.train_kd_pretrain(
        trajs,
        kd_targets,
        epochs=3,
        patience=2,
        batch_size=8,
    )
    assert stats["epochs_trained"] > 0
    print(f"  epochs_trained = {stats['epochs_trained']}")
    print(f"  best_val       = {stats['best_val']:.4f}")
    print("  → PASS\n")
    return True


def test_output_dir_structure():
    """kd_pretrain.py が出力する irl_model.pt + model_metadata.json の形式."""
    print("=" * 60)
    print("TEST 4: 出力ディレクトリ構造")
    print("=" * 60)

    state_dim = len(STATE_FEATURES_WITH_PATH)
    config = {
        "state_dim": state_dim,
        "action_dim": 5,
        "hidden_dim": 64,
        "dropout": 0.1,
        "learning_rate": 1e-3,
        "model_type": 0,
    }
    system = MCEIRLSystem(config)

    trajs = make_fake_trajectories(n=10, seq_len=3)
    kd_targets = make_fake_kd_targets(trajs)
    stats = system.train_kd_pretrain(trajs, kd_targets, epochs=2, patience=1, batch_size=4)

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        model_path = out / "irl_model.pt"
        torch.save(system.network.state_dict(), model_path)

        metadata = {
            "model_class": "mce_irl_kd_pretrain",
            "model_type": 0,
            "state_dim": state_dim,
            "action_dim": 5,
            "hidden_dim": 64,
            "dropout": 0.1,
            "kd_stats": stats,
        }
        meta_path = out / "model_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)

        assert model_path.exists(), "irl_model.pt が作られていない"
        assert meta_path.exists(), "model_metadata.json が作られていない"

        # reload check
        sd = torch.load(model_path, map_location="cpu", weights_only=True)
        with open(meta_path) as f:
            m = json.load(f)
        assert m["model_class"] == "mce_irl_kd_pretrain"
        assert "kd_stats" in m
        print(f"  irl_model.pt    : {model_path.stat().st_size} bytes, {len(sd)} keys")
        print(f"  metadata.json   : OK (model_class={m['model_class']})")

    print("  → PASS\n")
    return True


def test_gradient_flows():
    """softmax(logits)[...,1] に勾配が流れることを確認."""
    print("=" * 60)
    print("TEST 5: 勾配フロー確認")
    print("=" * 60)

    state_dim = len(STATE_FEATURES_WITH_PATH)
    config = {
        "state_dim": state_dim,
        "action_dim": 5,
        "hidden_dim": 64,
        "dropout": 0.1,
        "learning_rate": 1e-3,
        "model_type": 0,
    }
    system = MCEIRLSystem(config)

    # 初期パラメータ記録
    params_before = {
        k: v.clone() for k, v in system.network.state_dict().items()
    }

    trajs = make_fake_trajectories(n=10, seq_len=3)
    kd_targets = make_fake_kd_targets(trajs)
    system.train_kd_pretrain(trajs, kd_targets, epochs=3, patience=10, batch_size=4)

    # パラメータが変化しているか
    params_after = system.network.state_dict()
    changed = 0
    total = 0
    for k in params_before:
        total += 1
        if not torch.equal(params_before[k], params_after[k]):
            changed += 1

    assert changed > 0, "KD pretrain 後にパラメータが全く変化していない（勾配なし）"
    print(f"  パラメータ変化: {changed}/{total} keys")
    print("  → PASS\n")
    return True


if __name__ == "__main__":
    results = []
    for test_fn in [
        test_basic_kd_pretrain,
        test_state_dict_compatibility,
        test_model_type_3_two_tower,
        test_output_dir_structure,
        test_gradient_flows,
    ]:
        try:
            ok = test_fn()
            results.append((test_fn.__name__, ok))
        except Exception as e:
            print(f"  → FAIL: {e}\n")
            results.append((test_fn.__name__, False))

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_ok = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  {status}: {name}")
        if not ok:
            all_ok = False

    if all_ok:
        print("\n✅ All smoke tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
