"""DeepSeek-R1 W4A16 — demo web interactive with SSE streaming."""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify, Response, render_template
import torch
import deepseek_r1_w4_model as m

app = Flask(__name__)
decoder = None

WEIGHTS_FILE = "qwen2.5_coder_7b_w4a16_bi.pt"
TOKENIZER_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
MODEL_DISPLAY_NAME = "Qwen 2.5 Coder 7B"


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
                          tokenizer_name=TOKENIZER_NAME)
    return decoder


@app.route("/")
def index():
    return render_template("index.html")


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
        t0 = time.perf_counter()
        token_count = 0
        try:
            for text in d.generate_stream(prompt, max_tokens=max_tokens):
                token_count += 1
                yield f"data: {json.dumps({'text': text, 'n': token_count})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return

        elapsed = time.perf_counter() - t0
        yield f"data: {json.dumps({'done': True, 'time_s': round(elapsed, 2), 'tokens': token_count, 'tok_s': round(token_count / elapsed, 1) if elapsed > 0 else 0})}\n\n"

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
    print("Chargement du modele...")
    get_decoder()
    print(f"Modele pret. GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM utilisee: {torch.cuda.memory_allocated()/1e9:.1f} GB")
    print("\nServeur demarre → http://localhost:8080")
    app.run(host="0.0.0.0", port=8080, debug=False)
