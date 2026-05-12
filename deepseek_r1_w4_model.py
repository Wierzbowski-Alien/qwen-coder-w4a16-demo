"""W4A16 model loader and Decoder API for DeepSeek-R1-Distill-Qwen-7B."""

import struct
import torch
import ctypes

NUM_LAYERS        = 28
HIDDEN_SIZE       = 3584
INTERMEDIATE_SIZE = 18944
VOCAB_SIZE        = 152064
MAX_SEQ_LEN       = 4096

FA_NUM_Q_HEADS    = 28
FA_NUM_KV_HEADS   = 4
FA_HEAD_DIM       = 128
FA_Q_SIZE         = FA_NUM_Q_HEADS * FA_HEAD_DIM   # 3584
FA_KV_SIZE        = FA_NUM_KV_HEADS * FA_HEAD_DIM  # 512

LM_NUM_BLOCKS     = 112

WEIGHTS_FILE      = "deepseek_r1_w4a16.pt"

_decode        = None
_decode_nc     = None
_prefill       = None
_verify        = None
_verify_nc     = None
_vf_fi_embed      = None
_vf_fi_layer_pre  = None
_vf_fi_layer_post = None
_vf_fi_lm_head    = None


def _load_ops():
    global _decode, _decode_nc, _prefill, _verify, _verify_nc
    global _vf_fi_embed, _vf_fi_layer_pre, _vf_fi_layer_post, _vf_fi_lm_head
    if _decode is None:
        import deepseek_r1_w4a16_C
        _decode     = torch.ops.deepseek_r1_w4a16_C.decode_w4a16
        _prefill    = torch.ops.deepseek_r1_w4a16_C.prefill_w4a16
        _verify     = torch.ops.deepseek_r1_w4a16_C.verify_w4a16
        _verify_nc  = torch.ops.deepseek_r1_w4a16_C.verify_w4a16_nc
        try:
            _decode_nc  = torch.ops.deepseek_r1_w4a16_C.decode_w4a16_nc
        except AttributeError:
            _decode_nc = None
        try:
            _vf_fi_embed      = torch.ops.deepseek_r1_w4a16_C.vf_fi_embed
            _vf_fi_layer_pre  = torch.ops.deepseek_r1_w4a16_C.vf_fi_layer_pre
            _vf_fi_layer_post = torch.ops.deepseek_r1_w4a16_C.vf_fi_layer_post
            _vf_fi_lm_head    = torch.ops.deepseek_r1_w4a16_C.vf_fi_lm_head
        except AttributeError:
            pass


def load_weights(weights_file=WEIGHTS_FILE, verbose=True,
                 tokenizer_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"):
    """
    Load W4A16 quantized weights from disk.
    Supports both DeepSeek-R1 and Qwen2.5-Coder (same 7B architecture).
    """
    import os
    if not os.path.exists(weights_file):
        raise FileNotFoundError(
            f"{weights_file} introuvable. "
            f"Lance d'abord: python deepseek_r1_w4_quantize.py"
        )
    if verbose:
        size_gb = os.path.getsize(weights_file) / 1e9
        print(f"Chargement {weights_file} ({size_gb:.2f} GB)...")

    data = torch.load(weights_file, map_location="cpu", weights_only=True)

    from transformers import AutoTokenizer
    if verbose:
        print(f"Chargement tokenizer: {tokenizer_name}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    return data, tokenizer


def _compute_imma_scales(w_packed: torch.Tensor, group_scales: torch.Tensor):
    """
    Precompute IMMA scale factors for on-the-fly INT4→INT8 conversion in the kernel.

    Each packed nibble n (in [-8, 7]) is converted to INT8 via:
        w_i8 = clamp(round(n * imma_sf[n, k//128]), -127, 127)

    where imma_sf = group_scale * 127 / row_max

    Returns:
        imma_sf:   float32 [N, K//128] — combined scale factor per group
        row_scale: float32 [N]         — epilogue rescaling factor = row_max / 127
    """
    N, K_half = w_packed.shape
    K = K_half * 2

    lo_u = w_packed.to(torch.int32) & 0x0F
    hi_u = w_packed.to(torch.int32) >> 4
    lo = torch.where(lo_u >= 8, lo_u - 16, lo_u).to(torch.float32)
    hi = torch.where(hi_u >= 8, hi_u - 16, hi_u).to(torch.float32)
    w_fp = torch.stack([lo, hi], dim=2).reshape(N, K)
    scales_exp = group_scales.repeat_interleave(128, dim=1)
    w_fp = w_fp * scales_exp

    row_max = w_fp.abs().amax(dim=1, keepdim=True).clamp(min=1e-8).squeeze(1)
    row_scale = row_max / 127.0

    # imma_sf[g] = group_scale[g] * 127 / row_max for converting nibble → INT8
    row_max_exp = row_max.unsqueeze(1).expand(N, K // 128)
    imma_sf = group_scales * (127.0 / row_max_exp)

    return imma_sf.contiguous(), row_scale.contiguous()


def _move_layer_to_cuda(ld: dict, skip_imma: bool = False) -> dict:
    """Move layer tensors to CUDA and optionally precompute IMMA scale factors."""
    out = {}
    if not skip_imma:
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"]:
            w4_key = f"{proj}_w4"
            gs_key = f"{proj}_scale"
            if w4_key in ld and gs_key in ld:
                imma_sf, row_s = _compute_imma_scales(ld[w4_key], ld[gs_key])
                ld[f"{proj}_imma_sf"] = imma_sf
                ld[f"{proj}_row_scale"] = row_s
    for k, v in ld.items():
        out[k] = v.cuda()
    return out


def _pack_layer_weights(layers_cpu: list, imma_dummy_ptr: int = 0) -> torch.Tensor:
    """
    Pack 33 device pointers per layer into a uint8 blob.

    Layout:
      [0..18]  Original 19-ptr (packed INT4 + group scales)
      [19..32] IMMA scale factors + per-row scales (7 projections × 2)
               Set to imma_dummy_ptr when GEMM-style (K-tiled) weights are used.

    Pointer order (33 total):
      [0]  input_layernorm_weight     BF16
      [1]  q_proj_w4                  uint8
      [2]  q_proj_scale               F32
      [3]  q_proj_bias                BF16
      [4]  k_proj_w4                  uint8
      [5]  k_proj_scale               F32
      [6]  k_proj_bias                BF16
      [7]  v_proj_w4                  uint8
      [8]  v_proj_scale               F32
      [9]  v_proj_bias                BF16
      [10] o_proj_w4                  uint8
      [11] o_proj_scale               F32
      [12] post_attn_layernorm_weight BF16
      [13] gate_proj_w4               uint8
      [14] gate_proj_scale            F32
      [15] up_proj_w4                 uint8
      [16] up_proj_scale              F32
      [17] down_proj_w4               uint8
      [18] down_proj_scale            F32
      [19] q_proj_imma_sf             F32  (dummy if K-tiled)
      [20] q_proj_row_scale           F32  (dummy if K-tiled)
      [21] k_proj_imma_sf             F32  (dummy if K-tiled)
      [22] k_proj_row_scale           F32  (dummy if K-tiled)
      [23] v_proj_imma_sf             F32  (dummy if K-tiled)
      [24] v_proj_row_scale           F32  (dummy if K-tiled)
      [25] o_proj_imma_sf             F32  (dummy if K-tiled)
      [26] o_proj_row_scale           F32  (dummy if K-tiled)
      [27] gate_proj_imma_sf          F32  (dummy if K-tiled)
      [28] gate_proj_row_scale        F32  (dummy if K-tiled)
      [29] up_proj_imma_sf            F32  (dummy if K-tiled)
      [30] up_proj_row_scale          F32  (dummy if K-tiled)
      [31] down_proj_imma_sf          F32  (dummy if K-tiled)
      [32] down_proj_row_scale        F32  (dummy if K-tiled)
    """
    NUM_PTRS   = 33
    PTR_SIZE   = 8
    STRUCT_SZ  = NUM_PTRS * PTR_SIZE

    buf = bytearray(NUM_LAYERS * STRUCT_SZ)
    for i, ld in enumerate(layers_cpu):
        off = i * STRUCT_SZ
        ordered = [
            ld["input_layernorm_weight"],
            ld["q_proj_w4"],   ld["q_proj_scale"],   ld["q_proj_bias"],
            ld["k_proj_w4"],   ld["k_proj_scale"],   ld["k_proj_bias"],
            ld["v_proj_w4"],   ld["v_proj_scale"],   ld["v_proj_bias"],
            ld["o_proj_w4"],   ld["o_proj_scale"],
            ld["post_attn_layernorm_weight"],
            ld["gate_proj_w4"], ld["gate_proj_scale"],
            ld["up_proj_w4"],   ld["up_proj_scale"],
            ld["down_proj_w4"], ld["down_proj_scale"],
        ]
        imma_keys = [
            "q_proj_imma_sf",    "q_proj_row_scale",
            "k_proj_imma_sf",    "k_proj_row_scale",
            "v_proj_imma_sf",    "v_proj_row_scale",
            "o_proj_imma_sf",    "o_proj_row_scale",
            "gate_proj_imma_sf", "gate_proj_row_scale",
            "up_proj_imma_sf",   "up_proj_row_scale",
            "down_proj_imma_sf", "down_proj_row_scale",
        ]
        for k in imma_keys:
            if k in ld:
                ordered.append(ld[k])
        # Fill remaining slots with null pointers (K-tiled/GEMM path has no IMMA data)
        while len(ordered) < NUM_PTRS:
            ordered.append(None)

        for j, t in enumerate(ordered):
            ptr_val = t.data_ptr() if t is not None else 0
            struct.pack_into("Q", buf, off + j * PTR_SIZE, ptr_val)

    return torch.frombuffer(buf, dtype=torch.uint8).cuda()


class Decoder:
    """Stateful W4A16 decoder for DeepSeek-R1-Distill-Qwen-7B."""

    def __init__(self, weights_data=None, tokenizer=None,
                 weights_file=WEIGHTS_FILE, verbose=True,
                 pf_seq_len: int = MAX_SEQ_LEN,
                 kv_seq_len: int = MAX_SEQ_LEN,
                 vf_seq_len: int = 16,
                 weight_format: str = None,
                 tokenizer_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"):
        _load_ops()

        if weights_data is None:
            weights_data, tokenizer = load_weights(weights_file, verbose=verbose,
                                                   tokenizer_name=tokenizer_name)

        # Auto-detect from weights metadata if not explicitly specified
        if weight_format is None:
            fmt = weights_data.get("weight_format", "")
            weight_format = "k_tiled" if (fmt.startswith("k_tiled") or fmt.startswith("block_interleaved")) else "row_major"

        self._weight_format = weight_format
        self._use_gemm = 1 if weight_format == "k_tiled" else 0
        self.tokenizer = tokenizer
        self._position = 0
        self._data     = weights_data

        self._embed_weight      = weights_data["embed_weight"].cuda()
        self._final_norm_weight = weights_data["final_norm_weight"].cuda()
        self._lm_head_weight    = weights_data["lm_head_weight"].cuda()

        # INT8 lm_head quantization (per-row symmetric, 2x BW reduction)
        _lm_w = self._lm_head_weight.float()
        self._lm_head_scales = (_lm_w.abs().max(dim=1).values / 127.0).cuda()
        self._lm_head_weight_int8 = (
            (_lm_w / self._lm_head_scales.unsqueeze(1)).round().clamp(-128, 127)
            .to(torch.int8).cuda()
        )
        # All active paths use INT8 for k_tiled; free BF16 (1040 MB)
        if weight_format == "k_tiled":
            del self._lm_head_weight
            self._lm_head_weight = self._lm_head_weight_int8

        if verbose:
            print("Déplacement des couches vers CUDA...")
        skip_imma = (weight_format == "k_tiled")
        self._layers_cuda = [_move_layer_to_cuda(ld, skip_imma=skip_imma) for ld in weights_data["layers"]]
        self._layer_weights_packed = _pack_layer_weights(self._layers_cuda)
        # Pinned host copy — stable address for per-layer FI C functions (h_layers_ptr)
        self._layer_weights_packed_cpu = self._layer_weights_packed.cpu().pin_memory()

        if verbose:
            vram = torch.cuda.memory_allocated() / 1e9
            print(f"Poids chargés — VRAM: {vram:.2f} GB")

        bf16 = dict(dtype=torch.bfloat16, device="cuda")
        f32  = dict(dtype=torch.float32,  device="cuda")
        i32  = dict(dtype=torch.int32,    device="cuda")
        u8   = dict(dtype=torch.uint8,    device="cuda")
        u32  = dict(dtype=torch.uint32,   device="cuda")

        self._kv_seq_len = kv_seq_len
        self._fa_k_cache = torch.zeros(NUM_LAYERS, FA_NUM_KV_HEADS, kv_seq_len, FA_HEAD_DIM, **bf16)
        self._fa_v_cache = torch.zeros_like(self._fa_k_cache)

        self._hidden      = torch.empty(HIDDEN_SIZE,       **bf16)
        self._activations = torch.empty(HIDDEN_SIZE,       **f32)
        self._residual    = torch.empty(HIDDEN_SIZE,       **bf16)
        self._q_scratch   = torch.empty(FA_Q_SIZE,         **f32)
        self._kv_scratch  = torch.empty(FA_KV_SIZE * 2,    **f32)
        self._attn_out    = torch.empty(FA_Q_SIZE,         **f32)
        self._mlp_inter   = torch.empty(INTERMEDIATE_SIZE, **f32)
        self._normalized  = torch.empty(HIDDEN_SIZE,       **f32)
        # Partial sums buffer for K-split matvecs (worst case: gate+up, SPLIT=2, 18944 → ~303KB)
        self._partial     = torch.empty(2 * 1024 * 1024,     **u8)  # 2 MB (SPLIT≤7 gate+up: 2*7*18944*4 = 1.01 MB)

        self._barrier_counter    = torch.zeros(1,              **u32)
        self._barrier_generation = torch.zeros(1,              **u32)
        self._block_max_vals     = torch.empty(LM_NUM_BLOCKS,  **f32)
        self._block_max_idxs     = torch.empty(LM_NUM_BLOCKS,  **i32)
        self._lm_sync_counter    = torch.zeros(1,              **u32)
        self._out_token          = torch.empty(1,              **i32)
        self._nc_params          = torch.zeros(2,               **i32)  # {token_id, position}

        PF_S = pf_seq_len
        self._pf_seq_len = PF_S
        self._pf_hidden    = torch.empty(PF_S, HIDDEN_SIZE,       **bf16)
        self._pf_residual  = torch.empty(PF_S, HIDDEN_SIZE,       **bf16)
        self._pf_normalized = torch.empty(PF_S, HIDDEN_SIZE,      **bf16)
        self._pf_proj_buf  = torch.empty(PF_S, INTERMEDIATE_SIZE, **bf16)
        self._pf_proj_buf2 = torch.empty(PF_S, INTERMEDIATE_SIZE, **bf16)
        self._pf_attn_buf  = torch.empty(PF_S, FA_KV_SIZE,        **bf16)
        self._pf_mlp_buf   = torch.empty(PF_S, INTERMEDIATE_SIZE, **bf16)
        self._pf_final_normed    = torch.empty(HIDDEN_SIZE,    **bf16)
        self._pf_hidden_bf16_out = torch.empty(HIDDEN_SIZE,    **bf16)
        self._pf_lm_bmv  = torch.empty(512,  **f32)
        self._pf_lm_bmi  = torch.empty(512,  **i32)

        # Verify pass buffers (spec_verify_w4)
        VF_S = vf_seq_len
        self._vf_hidden      = torch.empty(VF_S, HIDDEN_SIZE,       **bf16)
        self._vf_residual    = torch.empty(VF_S, HIDDEN_SIZE,       **bf16)
        self._vf_normalized  = torch.empty(VF_S, HIDDEN_SIZE,       **bf16)
        self._vf_proj_buf    = torch.empty(VF_S, INTERMEDIATE_SIZE, **bf16)
        self._vf_proj_buf2   = torch.empty(VF_S, INTERMEDIATE_SIZE, **bf16)
        self._vf_attn_buf    = torch.empty(VF_S, FA_KV_SIZE,        **bf16)
        self._vf_mlp_buf     = torch.empty(VF_S, INTERMEDIATE_SIZE, **bf16)
        self._vf_final_normed = torch.empty(VF_S, HIDDEN_SIZE,      **bf16)
        self._vf_lm_bmv  = torch.empty(50, **f32)   # max S=50 supported by kernel
        self._vf_lm_bmi  = torch.empty(50, **i32)
        self._vf_out_tokens = torch.empty(VF_S, **i32)

        # Partial buffers for chunked Flash Decoding attention (Phase 7)
        _max_chunks = (kv_seq_len + 31) // 32  # VA_CHUNK_SIZE = 32
        self._vf_partial_max = torch.empty(VF_S, FA_NUM_Q_HEADS, _max_chunks, **f32)
        self._vf_partial_sum = torch.empty(VF_S, FA_NUM_Q_HEADS, _max_chunks, **f32)
        self._vf_partial_v   = torch.empty(VF_S, FA_NUM_Q_HEADS, _max_chunks, FA_HEAD_DIM, **f32)

        # Q buffer for FlashInfer verify path — [VF_S, Q_HEADS, HEAD_DIM]
        self._vf_fi_q = torch.empty(VF_S, FA_NUM_Q_HEADS, FA_HEAD_DIM, **bf16)

        # Debug buffers for per-layer hidden state capture
        self._debug_hidden    = torch.zeros(NUM_LAYERS, HIDDEN_SIZE,       **bf16)
        self._debug_q_nc        = torch.zeros(FA_Q_SIZE,                     **f32)
        self._debug_kv_nc       = torch.zeros(2 * FA_KV_SIZE,               **f32)
        self._debug_attn_nc     = torch.zeros(FA_Q_SIZE,                     **f32)
        self._debug_post_attn_nc = torch.zeros(HIDDEN_SIZE,                  **bf16)
        self._debug_ffn_nc      = torch.zeros(128,                           **f32)
        self._vf_debug_hidden = torch.zeros(NUM_LAYERS * VF_S, HIDDEN_SIZE, **bf16)
        self._vf_debug_q      = torch.zeros(VF_S, FA_Q_SIZE,               **bf16)
        self._vf_debug_norm   = torch.zeros(VF_S, HIDDEN_SIZE,             **bf16)
        self._vf_debug_post_attn = torch.zeros(VF_S, HIDDEN_SIZE,          **bf16)
        self._vf_debug_k      = torch.zeros(VF_S, FA_KV_SIZE,               **bf16)
        self._vf_debug_v      = torch.zeros(VF_S, FA_KV_SIZE,               **bf16)
        self._vf_debug_attn   = torch.zeros(VF_S, FA_Q_SIZE,                **bf16)
        self._vf_debug_ffn    = torch.zeros(128,                            **bf16)

        # Persistent device buffers for verify CUDA graphs
        self._vf_token_ids_d    = torch.zeros(VF_S, **i32)       # updated before each replay
        self._vf_d_cache_offset = torch.zeros(1,    **i32)       # updated before each replay
        self._vf_graphs: dict   = {}                              # S → CUDAGraph (old path)
        self._vf_nc_graphs: dict = {}                             # S → CUDAGraph (NC path)
        self._vf_seq_len        = VF_S

        # CUDA Graph for NC decode (captured on first use)
        self._nc_graph = None

    def _capture_nc_graph(self):
        """Capture the NC decode kernel sequence into a CUDA Graph."""
        # Warmup run to allocate all transient memory
        self._nc_params[0] = 1
        self._nc_params[1] = 0
        _decode_nc(
            self._out_token,
            self._embed_weight, self._layer_weights_packed,
            self._final_norm_weight, self._lm_head_weight, self._lm_head_scales,
            self._fa_k_cache, self._fa_v_cache,
            self._hidden, self._activations, self._residual,
            self._q_scratch, self._kv_scratch, self._attn_out,
            self._mlp_inter, self._normalized,
            self._partial,
            self._barrier_counter, self._barrier_generation,
            self._block_max_vals, self._block_max_idxs,
            self._lm_sync_counter,
            self._nc_params, self._kv_seq_len,
        )
        torch.cuda.synchronize()

        # Capture
        self._nc_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._nc_graph):
            _decode_nc(
                self._out_token,
                self._embed_weight, self._layer_weights_packed,
                self._final_norm_weight, self._lm_head_weight, self._lm_head_scales,
                self._fa_k_cache, self._fa_v_cache,
                self._hidden, self._activations, self._residual,
                self._q_scratch, self._kv_scratch, self._attn_out,
                self._mlp_inter, self._normalized,
                self._barrier_counter, self._barrier_generation,
                self._block_max_vals, self._block_max_idxs,
                self._lm_sync_counter,
                self._nc_params, self._kv_seq_len,
            )
        torch.cuda.synchronize()

    def step_nc_graph(self, token_id: int) -> int:
        if self._nc_graph is None:
            self._capture_nc_graph()
        self._nc_params[0] = token_id
        self._nc_params[1] = self._position
        self._nc_graph.replay()
        self._position += 1
        return self._out_token.item()

    def step(self, token_id: int) -> int:
        # Cooperative decode matvecs are not yet ported to K-tiled.
        # Fall back to NC decode path for K-tiled weights.
        if self._use_gemm:
            return self.step_nc(token_id)
        _decode(
            self._out_token, token_id,
            self._embed_weight, self._layer_weights_packed,
            self._final_norm_weight, self._lm_head_weight, self._lm_head_scales,
            self._fa_k_cache, self._fa_v_cache,
            self._hidden, self._activations, self._residual,
            self._q_scratch, self._kv_scratch, self._attn_out,
            self._mlp_inter, self._normalized,
            self._partial,
            self._barrier_counter, self._barrier_generation,
            self._block_max_vals, self._block_max_idxs,
            self._lm_sync_counter,
            self._position, self._kv_seq_len,
        )
        self._position += 1
        return self._out_token.item()

    def step_nc(self, token_id: int) -> int:
        self._nc_params[0] = token_id
        self._nc_params[1] = self._position
        _decode_nc(
            self._out_token,
            self._embed_weight, self._layer_weights_packed,
            self._final_norm_weight, self._lm_head_weight, self._lm_head_scales,
            self._fa_k_cache, self._fa_v_cache,
            self._hidden, self._activations, self._residual,
            self._q_scratch, self._kv_scratch, self._attn_out,
            self._mlp_inter, self._normalized,
            self._partial,
            self._barrier_counter, self._barrier_generation,
            self._block_max_vals, self._block_max_idxs,
            self._lm_sync_counter,
            self._nc_params, self._debug_hidden,
            self._debug_q_nc, self._debug_kv_nc,
            self._debug_attn_nc, self._debug_post_attn_nc,
            self._debug_ffn_nc, self._kv_seq_len,
        )
        self._position += 1
        return self._out_token.item()

    def prefill(self, token_ids: list[int]) -> int:
        if len(token_ids) > self._kv_seq_len:
            raise ValueError(f"Prompt trop long ({len(token_ids)} > {self._kv_seq_len})")
        # Use step-by-step decode for prefill when using K-tiled weights.
        # The cuBLAS-based batch prefill (pf_dequant + GEMM) produces slightly
        # different KV cache entries than the custom NC matvecs, causing token
        # mismatches for prompts longer than 3 tokens.
        if self._use_gemm:
            self._position = 0
            for tid in token_ids:
                self.step_nc(tid)
            return self._out_token.item()
        ids = torch.tensor(token_ids, dtype=torch.int32, device="cuda")
        _prefill(
            self._out_token, ids,
            self._embed_weight, self._layer_weights_packed,
            self._final_norm_weight, self._lm_head_weight, self._lm_head_scales,
            self._fa_k_cache, self._fa_v_cache,
            self._kv_seq_len,
            self._pf_hidden, self._pf_residual, self._pf_normalized,
            self._pf_proj_buf, self._pf_proj_buf2,
            self._pf_attn_buf, self._pf_mlp_buf,
            self._pf_final_normed, self._pf_hidden_bf16_out,
            self._pf_lm_bmv, self._pf_lm_bmi,
        )
        self._position = len(token_ids)
        self._hidden.copy_(self._pf_hidden_bf16_out)
        return self._out_token.item()

    def _do_verify_call(self, S: int):
        """Execute one verify pass using pre-allocated device buffers.
        Reads token IDs from _vf_token_ids_d[:S] and cache_offset from _vf_d_cache_offset.
        Safe inside torch.cuda.graph() context (no dynamic allocations, stable data_ptr)."""
        _verify(
            self._vf_out_tokens[:S],
            self._vf_token_ids_d[:S],   # slice preserves data_ptr() == base, size=S for C binding
            self._vf_d_cache_offset,
            self._embed_weight, self._layer_weights_packed,
            self._final_norm_weight, self._lm_head_weight, self._lm_head_scales,
            self._fa_k_cache, self._fa_v_cache,
            self._kv_seq_len,
            self._vf_hidden[:S], self._vf_residual[:S], self._vf_normalized[:S],
            self._vf_proj_buf[:S], self._vf_proj_buf2[:S],
            self._vf_attn_buf[:S], self._vf_mlp_buf[:S],
            self._vf_final_normed[:S],
            self._vf_lm_bmv, self._vf_lm_bmi,
            self._vf_partial_max, self._vf_partial_sum, self._vf_partial_v,
        )

    def _do_verify_nc_call(self, S: int):
        """Execute one NC-style verify pass (Option C3, no per-tile __syncthreads)."""
        _verify_nc(
            self._vf_out_tokens[:S],
            self._vf_token_ids_d[:S],
            self._vf_d_cache_offset,
            self._embed_weight, self._layer_weights_packed,
            self._final_norm_weight, self._lm_head_weight, self._lm_head_scales,
            self._fa_k_cache, self._fa_v_cache,
            self._kv_seq_len,
            self._vf_hidden[:S], self._vf_residual[:S], self._vf_normalized[:S],
            self._vf_proj_buf[:S], self._vf_proj_buf2[:S],
            self._vf_attn_buf[:S], self._vf_mlp_buf[:S],
            self._vf_final_normed[:S],
            self._vf_lm_bmv, self._vf_lm_bmi,
            self._vf_partial_max, self._vf_partial_sum, self._vf_partial_v,
            self._vf_debug_hidden, self._vf_debug_q, self._vf_debug_norm,
            self._vf_debug_post_attn, self._vf_debug_k, self._vf_debug_v,
            self._vf_debug_attn, self._vf_debug_ffn,
            self._use_gemm,
        )

    @staticmethod
    def _gqa_attn(q, k, v, S, cache_offset):
        """Manual GQA causal attention (no cuDNN/FlashInfer needed).

        q : [S, Q_HEADS, HEAD_DIM]  BF16  (RoPE already applied)
        k : [KV_HEADS, max_seq, HEAD_DIM] BF16  HND layout (layer slice from fa_k_cache)
        v : same
        Returns [S, Q_HEADS, HEAD_DIM] BF16
        """
        GQA = FA_NUM_Q_HEADS // FA_NUM_KV_HEADS  # 7
        N   = cache_offset + S
        scale = FA_HEAD_DIM ** -0.5

        # Expand KV from 4 to 28 heads: [28, N, HEAD_DIM]
        k_exp = k[:, :N, :].repeat_interleave(GQA, dim=0).float()  # [28, N, 128]
        v_exp = v[:, :N, :].repeat_interleave(GQA, dim=0).float()  # [28, N, 128]
        q_f   = q.float()  # [S, 28, 128]

        # scores: [S, 28, N]
        scores = torch.einsum('shd,hnd->shn', q_f, k_exp) * scale

        # Causal mask: token i (position cache_offset+i) sees k[0..cache_offset+i]
        mask = torch.full((S, N), float('-inf'), device=q.device, dtype=torch.float32)
        for i in range(S):
            mask[i, :cache_offset + i + 1] = 0.0

        scores = scores + mask.unsqueeze(1)          # [S, 28, N]
        attn   = scores.softmax(dim=-1)              # [S, 28, N]

        # Output: [S, 28, 128]
        out = torch.einsum('shn,hnd->shd', attn, v_exp).to(torch.bfloat16)
        return out

    def _do_verify_fi_call(self, S: int, cache_offset: int):
        """Python layer loop verify — GQA attention in PyTorch, matvecs via custom CUDA."""
        kv_len = cache_offset + S
        h_ptr  = self._layer_weights_packed_cpu.data_ptr()

        _vf_fi_embed(self._vf_token_ids_d[:S], S,
                     self._embed_weight, self._vf_hidden[:S])

        for li in range(NUM_LAYERS):
            _vf_fi_layer_pre(
                li, S,
                self._vf_d_cache_offset,
                h_ptr,
                self._vf_hidden[:S],
                self._vf_residual[:S],
                self._vf_normalized[:S],
                self._vf_proj_buf2[:S],     # K scratch [S, FA_KV_SIZE]
                self._vf_attn_buf[:S],      # V scratch [S, FA_KV_SIZE]
                self._vf_fi_q[:S],          # Q output  [S, Q_HEADS, HEAD_DIM]
                self._fa_k_cache,
                self._fa_v_cache,
                self._kv_seq_len,
                self._use_gemm,
            )

            # GQA causal attention (RoPE already applied in pre-layer)
            attn_out = self._gqa_attn(
                self._vf_fi_q[:S],
                self._fa_k_cache[li],   # [KV_HEADS, max_seq, HEAD_DIM] HND
                self._fa_v_cache[li],
                S, cache_offset,
            )  # [S, Q_HEADS, HEAD_DIM] BF16

            # Write attn_out into _vf_fi_q (backed by VF_S*Q_SIZE elements).
            # nc_vf_fused_residual_kernel<SM> reads S_MAX rows (SM >= S) so
            # the act buffer must have >= SM*Q_SIZE elements — passing a fresh
            # S-row tensor would OOB. _vf_fi_q has VF_S >= SM rows, safe.
            self._vf_fi_q[:S].copy_(attn_out)

            _vf_fi_layer_post(
                li, S,
                h_ptr,
                self._vf_fi_q,          # VF_S × Q_SIZE backing, S_MAX rows safe
                self._vf_hidden[:S],
                self._vf_residual[:S],
                self._vf_normalized[:S],
                self._vf_mlp_buf[:S],
                self._use_gemm,
            )

        _vf_fi_lm_head(
            S,
            self._vf_hidden[:S],
            self._final_norm_weight,
            self._lm_head_weight, self._lm_head_scales,
            self._vf_final_normed[:S],
            self._vf_lm_bmv,
            self._vf_lm_bmi,
            self._vf_out_tokens[:S],
        )

    def _capture_vf_graph(self, S: int):
        """Capture a CUDA graph for verify(S).  Two warmup runs first to trigger
        the one-shot D2H cudaMemcpy inside launch_verify_w4a16 (static guard)."""
        assert S <= self._vf_seq_len, f"S={S} > vf_seq_len={self._vf_seq_len}"
        # Use an offset near the END of the KV buffer so the dummy writes don't
        # corrupt prompt positions (0..prompt_len-1) written by the preceding prefill.
        safe_offset = max(0, self._kv_seq_len - S - 1)
        self._vf_token_ids_d[:S].fill_(1)
        self._vf_d_cache_offset[0] = safe_offset
        # Two warmups: first triggers the static D2H copy (not graph-safe), second is clean.
        self._do_verify_call(S)
        torch.cuda.synchronize()
        self._do_verify_call(S)
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._do_verify_call(S)
        self._vf_graphs[S] = g

    def _capture_vf_nc_graph(self, S: int):
        """Capture a CUDA graph for NC-style verify(S)."""
        assert S <= self._vf_seq_len, f"S={S} > vf_seq_len={self._vf_seq_len}"
        safe_offset = max(0, self._kv_seq_len - S - 1)
        self._vf_token_ids_d[:S].fill_(1)
        self._vf_d_cache_offset[0] = safe_offset
        # Two warmups to trigger the D2H layer-weight copy before graph capture.
        self._do_verify_nc_call(S)
        torch.cuda.synchronize()
        self._do_verify_nc_call(S)
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._do_verify_nc_call(S)
        self._vf_nc_graphs[S] = g

    def verify_k(self, token_ids: list[int], cache_offset: int,
                 use_graph: bool = True, use_nc: bool = False,
                 use_fi: bool = False) -> list[int]:
        """
        Run S=len(token_ids) tokens through the verifier starting at cache_offset.
        Returns S predicted tokens (one per input position).
        use_fi=True: FlashInfer attention (fastest, no CUDA graph).
        use_nc=True: NC-style path (Option C3, required for K-tiled weights).
        use_graph=True (default): CUDA graph replay (incompatible with use_fi).
        Note: K-tiled weights always use NC/FI path.
        """
        S = len(token_ids)
        self._vf_token_ids_d[:S].copy_(
            torch.tensor(token_ids, dtype=torch.int32))
        self._vf_d_cache_offset[0] = cache_offset

        if use_fi and _vf_fi_embed is not None:
            self._do_verify_fi_call(S, cache_offset)
        elif use_nc or self._use_gemm:
            if use_graph:
                if S not in self._vf_nc_graphs:
                    self._capture_vf_nc_graph(S)
                self._vf_nc_graphs[S].replay()
            else:
                self._do_verify_nc_call(S)
        else:
            if use_graph:
                if S not in self._vf_graphs:
                    self._capture_vf_graph(S)
                self._vf_graphs[S].replay()
            else:
                self._do_verify_call(S)

        return self._vf_out_tokens[:S].tolist()

    def reset(self):
        self._position = 0
        self._fa_k_cache.zero_()
        self._fa_v_cache.zero_()

    def generate(self, prompt: str, max_tokens: int = 512,
                 thinking: bool = False) -> str:
        self.reset()

        if thinking:
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = [
                {"role": "system", "content": "/no_think"},
                {"role": "user", "content": prompt},
            ]

        chat_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        ids = self.tokenizer.encode(chat_text, add_special_tokens=False)

        if thinking:
            think_id = self.tokenizer.convert_tokens_to_ids("<think>")
            if think_id != self.tokenizer.unk_token_id:
                ids = ids + [think_id]

        seq_len = self._kv_seq_len
        if len(ids) >= seq_len - max_tokens:
            ids = ids[-(seq_len - max_tokens - 1):]

        if len(ids) > 1:
            next_id = self.prefill(ids)
        else:
            next_id = self.step(ids[0])

        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        eos_ids = {self.tokenizer.eos_token_id, im_end_id}
        out = []
        for _ in range(max_tokens):
            if next_id in eos_ids:
                break
            out.append(next_id)
            next_id = self.step(next_id)

        import re
        decoded = self.tokenizer.decode(out, skip_special_tokens=True)
        # Fix GPT-2 byte-fallback: Ġ(U+0120)→space, Ċ(U+010A)→\n, etc.
        decoded = ''.join(
            chr(ord(c) & 0xFF) if 0x100 <= ord(c) < 0x200 else c
            for c in decoded
        )
        decoded = re.sub(r'<think>.*?</think>', '', decoded, flags=re.DOTALL)
        decoded = decoded.replace('</think>', '').strip()
        return decoded

    def generate_stream(self, prompt: str, max_tokens: int = 512,
                        thinking: bool = False):
        """Generator yielding progressively decoded text for SSE streaming."""
        self.reset()

        if thinking:
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = [
                {"role": "system", "content": "/no_think"},
                {"role": "user", "content": prompt},
            ]

        chat_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        ids = self.tokenizer.encode(chat_text, add_special_tokens=False)

        if thinking:
            think_id = self.tokenizer.convert_tokens_to_ids("<think>")
            if think_id != self.tokenizer.unk_token_id:
                ids = ids + [think_id]

        seq_len = self._kv_seq_len
        if len(ids) >= seq_len - max_tokens:
            ids = ids[-(seq_len - max_tokens - 1):]

        if len(ids) > 1:
            next_id = self.prefill(ids)
        else:
            next_id = self.step(ids[0])

        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        eos_ids = {self.tokenizer.eos_token_id, im_end_id}
        out = []
        for _ in range(max_tokens):
            if next_id in eos_ids:
                break
            out.append(next_id)

            # Progressive decode with byte-fallback fix
            decoded = self.tokenizer.decode(out, skip_special_tokens=True)
            decoded = ''.join(
                chr(ord(c) & 0xFF) if 0x100 <= ord(c) < 0x200 else c
                for c in decoded
            )
            yield decoded

            next_id = self.step(next_id)
