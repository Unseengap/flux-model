"""FLX Serialization — .flx save/load format.

Serializes the complete FLXNano model to a directory structure:

mymodel.flx/
├── manifest.yaml
├── shared_trunk/weights.bin
├── thalamic_router/weights.bin
├── cortices/{domain}/{stratum}/weights.bin + deltas/
├── bridges/{src}_{tgt}.yaml + weights.bin
├── state_hub/ (working_memory, episodes, thermal, activation history)
├── meta_generator/weights.bin
└── thermal_estimator/weights.bin
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

from .bridges import CrossCorticalBridge, build_bridges
from .delta import DeltaMetadata, DeltaStack, FLXDelta
from .memory import EpisodicBuffer, EpisodicCompressor, MemoryController
from .meta_gen import MetaDeltaGenerator
from .model import FLXNano
from .router import ThalamicRouter
from .thermal import ThermalEstimator


def _hash_state_dict(state_dict: dict) -> str:
    """Compute a hash of a state dict for integrity checking."""
    h = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        h.update(key.encode())
        h.update(state_dict[key].cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def save_flx(
    model: FLXNano,
    path: str | Path,
    episodic_buffer: EpisodicBuffer | None = None,
    activation_history: dict | None = None,
) -> None:
    """Save a complete FLXNano model to .flx directory format.

    Args:
        model: The FLXNano model to save.
        path: Directory path (typically ending in .flx).
        episodic_buffer: Episodic memory buffer to persist.
        activation_history: Cortex activation history to persist.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # --- Shared Trunk ---
    trunk_dir = path / "shared_trunk"
    trunk_dir.mkdir(exist_ok=True)
    torch.save(model.shared_trunk.state_dict(), trunk_dir / "weights.bin")

    # --- Thalamic Router ---
    if model.thalamic_router is not None:
        router_dir = path / "thalamic_router"
        router_dir.mkdir(exist_ok=True)
        torch.save(model.thalamic_router.state_dict(), router_dir / "weights.bin")

    # --- Domain Cortices ---
    cortices_dir = path / "cortices"
    cortices_dir.mkdir(exist_ok=True)

    for name, cortex in model.cortices.items():
        cortex_dir = cortices_dir / name
        cortex_dir.mkdir(exist_ok=True)

        # Cortex metadata
        meta = {
            "domain": name,
            "strata": list(cortex.strata.keys()),
            "d_model": model.d_model,
        }

        # Cortex-level weights (difficulty_gate etc.)
        cortex_level_state = {
            k: v for k, v in cortex.state_dict().items()
            if not any(k.startswith(f"strata.{s}.") for s in cortex.strata.keys())
        }
        torch.save(cortex_level_state, cortex_dir / "cortex_weights.bin")

        # Save each stratum
        for stratum_name, stratum in cortex.strata.items():
            stratum_dir = cortex_dir / stratum_name
            stratum_dir.mkdir(exist_ok=True)

            # Stratum base weights (excluding delta_stack)
            stratum_state = {
                k: v for k, v in stratum.state_dict().items()
                if not k.startswith("delta_stack.")
            }
            torch.save(stratum_state, stratum_dir / "weights.bin")

            # Delta stack
            deltas_dir = stratum_dir / "deltas"
            deltas_dir.mkdir(exist_ok=True)

            for idx, delta in enumerate(stratum.delta_stack.deltas):
                delta_name = f"d{idx:03d}"
                torch.save(delta.state_dict(), deltas_dir / f"{delta_name}.bin")

                delta_meta = {
                    "name": delta.metadata.name or delta_name,
                    "source": delta.metadata.source,
                    "target_cortex": delta.metadata.target_cortex or name,
                    "target_stratum": delta.metadata.target_stratum or stratum_name,
                    "rank": delta.rank,
                    "d_in": delta.d_in,
                    "d_out": delta.d_out,
                    "thermal_threshold": delta.thermal_threshold,
                    "confidence": delta.confidence.item(),
                    "scale": delta.scale,
                    "created_at": delta.metadata.created_at or "",
                    "description": delta.metadata.description,
                }
                with open(deltas_dir / f"{delta_name}.yaml", "w") as f:
                    yaml.dump(delta_meta, f, default_flow_style=False)

            meta[f"{stratum_name}_delta_count"] = len(stratum.delta_stack)

        with open(cortex_dir / "meta.yaml", "w") as f:
            yaml.dump(meta, f, default_flow_style=False)

    # --- Bridges ---
    if model.bridges is not None:
        bridges_dir = path / "bridges"
        bridges_dir.mkdir(exist_ok=True)
        for key, bridge in model.bridges.items():
            bridge_data = {
                "source_cortex": bridge.source_cortex,
                "target_cortex": bridge.target_cortex,
                "tau_min": bridge.tau_min,
                "tau_max": bridge.tau_max,
                "bandwidth": bridge.bandwidth.item(),
                "compatibility": bridge.compatibility.item(),
            }
            with open(bridges_dir / f"{key}.yaml", "w") as f:
                yaml.dump(bridge_data, f, default_flow_style=False)
            torch.save(bridge.state_dict(), bridges_dir / f"{key}_weights.bin")

    # --- Cortex Merger + Decoder ---
    torch.save(model.cortex_merger.state_dict(), path / "cortex_merger_weights.bin")
    torch.save(model.decoder.state_dict(), path / "decoder_weights.bin")

    # --- Thermal Estimator ---
    if model.thermal_estimator is not None:
        thermal_dir = path / "thermal_estimator"
        thermal_dir.mkdir(exist_ok=True)
        torch.save(model.thermal_estimator.state_dict(), thermal_dir / "weights.bin")

    # --- Memory Controller ---
    if model.memory_controller is not None:
        memory_dir = path / "memory_controller"
        memory_dir.mkdir(exist_ok=True)
        torch.save(model.memory_controller.state_dict(), memory_dir / "weights.bin")

    # --- Meta Generator ---
    if model.meta_generator is not None:
        meta_dir = path / "meta_generator"
        meta_dir.mkdir(exist_ok=True)
        torch.save(model.meta_generator.state_dict(), meta_dir / "weights.bin")

    # --- State Hub ---
    state_dir = path / "state_hub"
    state_dir.mkdir(exist_ok=True)

    # Thermal history
    if model.thermal_estimator is not None:
        thermal_state = {
            "history": model.thermal_estimator.get_history(),
        }
        with open(state_dir / "thermal.json", "w") as f:
            json.dump(thermal_state, f, indent=2)

    # Episodic buffer
    if episodic_buffer is not None and len(episodic_buffer) > 0:
        episodes = torch.stack(episodic_buffer.get_all())
        torch.save(episodes, state_dir / "episode_buffer.bin")

    # Activation history
    if activation_history is not None:
        with open(state_dir / "cortex_activation_history.json", "w") as f:
            json.dump(activation_history, f, indent=2)

    # --- Manifest ---
    trunk_hash = _hash_state_dict(model.shared_trunk.state_dict())
    total_deltas = sum(
        len(stratum.delta_stack)
        for cortex in model.cortices.values()
        for stratum in cortex.strata.values()
    )

    # Infer hyperparams from model structure
    trunk_layers = model.shared_trunk.trunk_layers.num_layers
    nhead = model.shared_trunk.trunk_layers.layers[0].self_attn.num_heads
    vocab_size = model.shared_trunk.token_embedding.num_embeddings
    max_seq_len = model.shared_trunk.position_embedding.num_embeddings
    dim_feedforward = model.shared_trunk.trunk_layers.layers[0].linear1.out_features
    layers_per_stratum = next(iter(next(iter(model.cortices.values())).strata.values())).layers.num_layers

    manifest = {
        "version": "0.1.0",
        "creation_date": datetime.now(timezone.utc).isoformat(),
        "base_model_hash": trunk_hash,
        "cortex_registry": list(model.cortices.keys()),
        "shared_trunk": {
            "d_model": model.d_model,
            "trunk_layers": trunk_layers,
            "nhead": nhead,
            "vocab_size": vocab_size,
            "max_seq_len": max_seq_len,
            "dim_feedforward": dim_feedforward,
        },
        "layers_per_stratum": layers_per_stratum,
        "delta_count": total_deltas,
        "delta_rank": model.delta_rank,
        "delta_capacity": model.cortices[list(model.cortices.keys())[0]].strata[list(next(iter(model.cortices.values())).strata.keys())[0]].delta_stack.capacity,
        "has_router": model.thalamic_router is not None,
        "has_thermal": model.thermal_estimator is not None,
        "has_bridges": model.bridges is not None,
        "has_memory": model.memory_controller is not None,
        "has_meta_gen": model.meta_generator is not None,
    }
    if model.memory_controller is not None:
        manifest["episode_dim"] = model.memory_controller.query_head.out_features
    with open(path / "manifest.yaml", "w") as f:
        yaml.dump(manifest, f, default_flow_style=False)


def load_flx(
    path: str | Path,
    device: str = "cpu",
) -> tuple[FLXNano, EpisodicBuffer | None, dict | None]:
    """Load a complete FLXNano model from .flx directory format.

    Args:
        path: Path to .flx directory.
        device: Target device for tensors.

    Returns:
        (model, episodic_buffer, activation_history)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"FLX model not found: {path}")

    # --- Read Manifest ---
    with open(path / "manifest.yaml") as f:
        manifest = yaml.safe_load(f)

    cortex_names = manifest["cortex_registry"]
    trunk_cfg = manifest["shared_trunk"]
    d_model = trunk_cfg["d_model"]
    delta_rank = manifest.get("delta_rank", 32)

    # --- Create Model ---
    model = FLXNano(
        vocab_size=trunk_cfg.get("vocab_size", 32000),
        d_model=d_model,
        nhead=trunk_cfg.get("nhead", 8),
        trunk_layers=trunk_cfg.get("trunk_layers", 6),
        layers_per_stratum=manifest.get("layers_per_stratum", 2),
        cortex_names=cortex_names,
        delta_rank=delta_rank,
        delta_capacity=manifest.get("delta_capacity", 3),
        max_seq_len=trunk_cfg.get("max_seq_len", 2048),
        dim_feedforward=trunk_cfg.get("dim_feedforward", 2048),
    )

    # --- Load Shared Trunk ---
    trunk_state = torch.load(
        path / "shared_trunk" / "weights.bin",
        map_location=device,
        weights_only=True,
    )
    model.shared_trunk.load_state_dict(trunk_state)

    # --- Load Cortex Merger + Decoder ---
    merger_path = path / "cortex_merger_weights.bin"
    if merger_path.exists():
        model.cortex_merger.load_state_dict(
            torch.load(merger_path, map_location=device, weights_only=True)
        )
    decoder_path = path / "decoder_weights.bin"
    if decoder_path.exists():
        model.decoder.load_state_dict(
            torch.load(decoder_path, map_location=device, weights_only=True)
        )

    # --- Load Cortices + Deltas ---
    cortices_dir = path / "cortices"
    for name in cortex_names:
        cortex_dir = cortices_dir / name
        if not cortex_dir.exists():
            continue

        cortex = model.cortices[name]

        # Load cortex-level weights (difficulty_gate etc.)
        cortex_weights_path = cortex_dir / "cortex_weights.bin"
        if cortex_weights_path.exists():
            cortex.load_state_dict(
                torch.load(cortex_weights_path, map_location=device, weights_only=True),
                strict=False,
            )

        for stratum_name in cortex.stratum_names():
            stratum_dir = cortex_dir / stratum_name
            if not stratum_dir.exists():
                continue

            stratum = cortex.strata[stratum_name]
            weights_path = stratum_dir / "weights.bin"
            if weights_path.exists():
                stratum.load_state_dict(
                    torch.load(weights_path, map_location=device, weights_only=True),
                    strict=False,
                )

            # Load deltas
            deltas_dir = stratum_dir / "deltas"
            if deltas_dir.exists():
                for delta_file in sorted(deltas_dir.glob("*.bin")):
                    delta_name = delta_file.stem
                    meta_file = deltas_dir / f"{delta_name}.yaml"

                    if meta_file.exists():
                        with open(meta_file) as f:
                            dmeta = yaml.safe_load(f)
                    else:
                        dmeta = {}

                    delta = FLXDelta(
                        d_in=dmeta.get("d_in", d_model),
                        d_out=dmeta.get("d_out", d_model),
                        rank=dmeta.get("rank", delta_rank),
                        thermal_threshold=dmeta.get("thermal_threshold", 0.0),
                        confidence=dmeta.get("confidence", 1.0),
                        scale=dmeta.get("scale", None),
                    )
                    delta.load_state_dict(
                        torch.load(delta_file, map_location=device, weights_only=True)
                    )
                    delta.metadata = DeltaMetadata(
                        name=dmeta.get("name", delta_name),
                        source=dmeta.get("source", ""),
                        target_cortex=dmeta.get("target_cortex", name),
                        target_stratum=dmeta.get("target_stratum", stratum_name),
                        created_at=dmeta.get("created_at", ""),
                        description=dmeta.get("description", ""),
                    )
                    stratum.delta_stack.push(delta)

    # --- Load Router ---
    router_dir = path / "thalamic_router"
    if manifest.get("has_router") and router_dir.exists():
        router = ThalamicRouter(d_model=d_model, cortex_names=cortex_names)
        router.load_state_dict(
            torch.load(router_dir / "weights.bin", map_location=device, weights_only=True)
        )
        model.attach_router(router)

    # --- Load Bridges ---
    bridges_dir = path / "bridges"
    if manifest.get("has_bridges") and bridges_dir.exists():
        bridges = build_bridges(cortex_names, d_model=d_model)
        for key, bridge in bridges.items():
            weights_path = bridges_dir / f"{key}_weights.bin"
            if weights_path.exists():
                bridge.load_state_dict(
                    torch.load(weights_path, map_location=device, weights_only=True)
                )
        model.attach_bridges(bridges)

    # --- Load Thermal Estimator ---
    thermal_dir = path / "thermal_estimator"
    if manifest.get("has_thermal") and thermal_dir.exists():
        thermal = ThermalEstimator(d_model=d_model)
        thermal.load_state_dict(
            torch.load(thermal_dir / "weights.bin", map_location=device, weights_only=True)
        )
        # Restore thermal history
        thermal_state_path = path / "state_hub" / "thermal.json"
        if thermal_state_path.exists():
            with open(thermal_state_path) as f:
                thermal_state = json.load(f)
            thermal.set_history(thermal_state.get("history", []))
        model.attach_thermal(thermal)

    # --- Load Memory Controller ---
    memory_dir = path / "memory_controller"
    if manifest.get("has_memory") and memory_dir.exists():
        episode_dim = manifest.get("episode_dim", 256)
        mem_ctrl = MemoryController(d_model=d_model, episode_dim=episode_dim)
        mem_ctrl.load_state_dict(
            torch.load(memory_dir / "weights.bin", map_location=device, weights_only=True)
        )
        model.attach_memory(mem_ctrl)

    # --- Load Meta Generator ---
    meta_dir = path / "meta_generator"
    if manifest.get("has_meta_gen") and meta_dir.exists():
        meta_gen = MetaDeltaGenerator(
            d_model=d_model, delta_rank=delta_rank, num_cortices=len(cortex_names)
        )
        meta_gen.load_state_dict(
            torch.load(meta_dir / "weights.bin", map_location=device, weights_only=True)
        )
        model.attach_meta_generator(meta_gen)

    # --- Load State Hub ---
    episodic_buffer = None
    state_dir = path / "state_hub"
    episode_path = state_dir / "episode_buffer.bin"
    if episode_path.exists():
        episodic_buffer = EpisodicBuffer()
        episodes = torch.load(episode_path, map_location=device, weights_only=True)
        for ep in episodes:
            episodic_buffer.add(ep)

    activation_history = None
    ah_path = state_dir / "cortex_activation_history.json"
    if ah_path.exists():
        with open(ah_path) as f:
            activation_history = json.load(f)

    model.to(device)
    return model, episodic_buffer, activation_history
