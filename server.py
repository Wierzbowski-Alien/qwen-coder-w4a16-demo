"""DeepSeek-R1 W4A16 — demo web interactive."""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify, render_template
import torch
import deepseek_r1_w4_model as m

app = Flask(__name__)
decoder = None

WEIGHTS_FILE = "deepseek_r1_w4a16_bi.pt"


def get_decoder():
    global decoder
    if decoder is None:
        if not os.path.exists(WEIGHTS_FILE):
            raise FileNotFoundError(
                f"Poids introuvables : {WEIGHTS_FILE}\n"
                "Lance d'abord : python deepseek_r1_w4_quantize.py\n"
                "Puis : python reformat_weights_k_tiled.py --block-interleaved"
            )
        decoder = m.Decoder(weights_file=WEIGHTS_FILE, verbose=False)
    return decoder


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    prompt = data.get("prompt", "").strip()
    max_tokens = int(data.get("max_tokens", 100))
    max_tokens = min(max(max_tokens, 1), 1024)

    if not prompt:
        return jsonify({"error": "Prompt vide"}), 400

    try:
        d = get_decoder()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        output = d.generate(prompt, max_tokens=max_tokens)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        return jsonify({
            "response": output,
            "time_s": round(elapsed, 2),
            "tokens": max_tokens,
            "tok_s": round(max_tokens / elapsed, 1),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    try:
        d = get_decoder()
        return jsonify({"status": "ok", "gpu": torch.cuda.get_device_name(0)})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    print("Chargement du modèle...")
    get_decoder()
    print(f"Modèle prêt. GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM utilisée: {torch.cuda.memory_allocated()/1e9:.1f} GB")
    print("\nServeur démarré → http://localhost:8080")
    app.run(host="0.0.0.0", port=8080, debug=False)
