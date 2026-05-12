"""Qwen 2.5 Coder W4A16 — demo web interactive with SSE streaming."""
import sys, os, time, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify, Response, render_template
import torch
import deepseek_r1_w4_model as m

app = Flask(__name__)
decoder = None
current_context = 4096

WEIGHTS_FILE = "qwen2.5_coder_7b_w4a16_bi.pt"
TOKENIZER_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
MODEL_DISPLAY_NAME = "Qwen 2.5 Coder 7B"

VALID_CONTEXTS = {4096, 8192, 16384}


def get_decoder():
    global decoder
    if decoder is None:
        if not os.path.exists(WEIGHTS_FILE):
            raise FileNotFoundError(
                f"Poids introuvables : {WEIGHTS_FILE}\n"
                "Lance d'abord : python deepseek_r1_w4_quantize.py\n"
                "Puis : python reformat_weights_k_tiled.py --block-interleaved"
            )
        decoder = m.Decoder(weights_file=WEIGHTS_FILE, verbose=False,
                          tokenizer_name=TOKENIZER_NAME,
                          pf_seq_len=current_context,
                          kv_seq_len=current_context)
    return decoder


@app.route("/")
def index():
    return render_template("index.html", context_length=current_context)


@app.route("/context")
def get_context():
    return jsonify({"context_length": current_context})


@app.route("/set_context/<int:value>", methods=["POST"])
def set_context(value):
    global current_context
    if value not in VALID_CONTEXTS:
        return jsonify({"error": f"Contexte invalide: {value}. Valide: {sorted(VALID_CONTEXTS)}"}), 400
    current_context = value
    # Schedule restart after returning the response
    def restart():
        import subprocess
        time.sleep(0.5)
        script = os.path.abspath(__file__)
        subprocess.Popen([sys.executable, script, "--context", str(value)],
                        stdout=sys.stdout, stderr=sys.stderr,
                        start_new_session=True)
        os._exit(0)
    import threading
    threading.Thread(target=restart, daemon=True).start()
    return jsonify({"status": "ok", "context_length": value, "message": "Redemarrage..."})


@app.route("/stream", methods=["POST"])
def stream():
    data = request.get_json()
    prompt = data.get("prompt", "").strip()
    max_tokens = int(data.get("max_tokens", 100))
    max_tokens = min(max(max_tokens, 1), 1024)

    if not prompt:
        return jsonify({"error": "Prompt vide"}), 400

    d = get_decoder()

    def generate():
        t_start = time.perf_counter()
        t0 = None  # decode-only timer (starts after prefill)
        token_count = 0
        ttft = 0.0
        try:
            for text in d.generate_stream(prompt, max_tokens=max_tokens):
                if t0 is None:
                    t0 = time.perf_counter()
                    ttft = round((t0 - t_start) * 1000)  # ms
                token_count += 1
                yield f"data: {json.dumps({'text': text, 'n': token_count})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return

        elapsed = time.perf_counter() - t0 if t0 else 0
        dec_tok_s = round((token_count - 1) / elapsed, 1) if token_count > 1 and elapsed > 0 else 0
        yield f"data: {json.dumps({'done': True, 'tokens': token_count, 'decode_tok_s': dec_tok_s, 'ttft_ms': ttft})}\n\n"

    return Response(generate(), mimetype="text/event-stream",
                    headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"})


@app.route("/health")
def health():
    try:
        d = get_decoder()
        return jsonify({"status": "ok", "gpu": torch.cuda.get_device_name(0)})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen Coder W4A16 Demo Server")
    parser.add_argument("--context", type=int, default=4096,
                       choices=[4096, 8192, 16384],
                       help="Longueur de contexte maximale (defaut: 4096)")
    parser.add_argument("--port", type=int, default=8080,
                       help="Port HTTP (defaut: 8080)")
    args = parser.parse_args()

    current_context = args.context

    print(f"Chargement du modele (contexte: {current_context} tokens)...")
    print(f"KV cache: ~{current_context * 28 * 4 * 128 * 2 * 2 / 1e9:.1f} GB")
    get_decoder()
    print(f"Modele pret. GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM utilisee: {torch.cuda.memory_allocated()/1e9:.1f} GB")
    print(f"\nServeur demarre → http://localhost:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False)
