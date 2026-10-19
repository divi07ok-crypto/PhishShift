
import os
import json
import csv
import io
import logging
import threading
import webbrowser
from datetime import datetime

import joblib
from flask import Flask, render_template, request, Response

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "models", "phishshift_model.pkl")
HISTORY_FILE = os.path.join(BASE_DIR, "scan_history.json")

MAX_HISTORY = 500
MAX_MESSAGE_LENGTH = 10000
UNCERTAIN_THRESHOLD = 80.0

SUSPICIOUS_PHRASES = [
    "urgent",
    "verify your account",
    "account will be blocked",
    "click this link",
    "click here",
    "password",
    "otp",
    "bank account",
    "prize",
    "winner",
    "suspended",
    "immediately",
    "confirm your details",
    "limited time",
    "login now",
    "update your account",
    "share your otp",
    "reactivate your account",
    "personal information",
    "claim your reward",
]

# Load the trained model
model = None
vectorizer = None
MODEL_ERROR = None

try:
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_FILE}"
        )

    saved = joblib.load(MODEL_FILE)

    if isinstance(saved, dict):
        model = saved.get("model")
        vectorizer = saved.get("vectorizer")
    else:
        model = saved

    if model is None:
        raise ValueError("No model found in the model file.")

except Exception as exc:
    MODEL_ERROR = str(exc)
    app.logger.exception("Model loading failed")


def load_history():
    """Load scan history safely."""
    if not os.path.exists(HISTORY_FILE):
        return []

    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)

        return data if isinstance(data, list) else []

    except (OSError, json.JSONDecodeError):
        app.logger.exception("Could not read scan history")
        return []


def save_history(history):
    """Save scan history safely."""
    temp_path = HISTORY_FILE + ".tmp"

    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(
            history[:MAX_HISTORY],
            file,
            indent=4,
            ensure_ascii=False
        )
        file.flush()
        os.fsync(file.fileno())

    os.replace(temp_path, HISTORY_FILE)


def scan_message(message):
    """Classify a message using ML and suspicious phrase rules."""
    if model is None:
        raise RuntimeError(
            f"Model loading failed: {MODEL_ERROR}"
        )

    # Support either a complete sklearn Pipeline
    # or a separate vectorizer and classifier.
    if hasattr(model, "steps"):
        features = [message]
    else:
        if vectorizer is None:
            raise RuntimeError(
                "Vectorizer is missing. Check the saved model file."
            )

        features = vectorizer.transform([message])

    prediction = model.predict(features)[0]
    predicted_label = str(prediction).strip().lower()

    confidence = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        confidence = round(
            float(max(probabilities)) * 100, 2
        )

    # Map the model label to a readable result.
    if predicted_label in (
        "1", "spam", "phishing", "phish", "malicious"
    ):
        result = "Phishing"

    elif predicted_label in (
        "0", "ham", "legitimate", "safe", "legit"
    ):
        result = "Legitimate"

    else:
        result = str(prediction)

    # Find suspicious phrases in the message.
    lower_message = message.lower()

    cues = [
        phrase
        for phrase in SUSPICIOUS_PHRASES
        if phrase in lower_message
    ]

    # Mark low-confidence predictions as Uncertain.
    uncertain = (
        confidence is not None
        and confidence < UNCERTAIN_THRESHOLD
    )

    if uncertain:
        result = "Uncertain"

    return result, confidence, cues, uncertain


def calculate_stats(history):
    """Calculate dashboard totals from scan history."""
    phishing = sum(
        1 for item in history
        if item.get("result") == "Phishing"
    )

    legitimate = sum(
        1 for item in history
        if item.get("result") == "Legitimate"
    )

    uncertain = sum(
        1 for item in history
        if item.get("result") == "Uncertain"
        or item.get("uncertain", False)
    )

    return {
        "total_scans": len(history),
        "phishing_count": phishing,
        "legitimate_count": legitimate,
        "uncertain_count": uncertain,
    }


@app.before_request
def log_request():
    print(
        f"REQUEST: {request.method} {request.path}",
        flush=True
    )


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    confidence = None
    cues = []
    uncertain = False
    message = ""
    warning = None

    if request.method == "POST":
        print("SCAN FORM SUBMITTED", flush=True)

        message = request.form.get("message", "").strip()

        if not message:
            warning = "Please enter a message to analyse."

        elif len(message) > MAX_MESSAGE_LENGTH:
            warning = (
                "Message is too long. Maximum length is "
                f"{MAX_MESSAGE_LENGTH} characters."
            )

        else:
            try:
                result, confidence, cues, uncertain = scan_message(
                    message
                )

                scan_record = {
                    "time": datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "message": message,
                    "result": result,
                    "confidence": confidence,
                    "cues": cues,
                    "uncertain": uncertain,
                }

                history = load_history()
                history.insert(0, scan_record)
                save_history(history)

                print(
                    f"HISTORY SAVED: "
                    f"{len(history[:MAX_HISTORY])} scans",
                    flush=True
                )

            except Exception as exc:
                app.logger.exception("Scan failed")
                warning = (
                    "Could not complete the scan. "
                    "Check the server terminal for details."
                )

    history = load_history()

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        cues=cues,
        uncertain=uncertain,
        message=message,
        warning=warning,
        history=history[:20],
        stats=calculate_stats(history),
    )


@app.route("/download-report")
def download_report():
    """Download the saved scan history as CSV."""
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow([
        "Time",
        "Message",
        "Result",
        "Confidence",
        "Suspicious Cues",
        "Uncertain",
    ])

    for item in load_history():
        writer.writerow([
            item.get("time", ""),
            item.get("message", ""),
            item.get("result", ""),
            item.get("confidence", ""),
            ", ".join(item.get("cues", [])),
            item.get("uncertain", False),
        ])

    response = Response(
        output.getvalue(),
        mimetype="text/csv"
    )

    response.headers["Content-Disposition"] = (
        "attachment; filename=phishshift_scan_report.csv"
    )

    return response


if __name__ == "__main__":
    if not os.path.exists(HISTORY_FILE):
        save_history([])

    print("PhishShift starting...", flush=True)
    print(f"Project folder: {BASE_DIR}", flush=True)
    print(f"Model file: {MODEL_FILE}", flush=True)
    print(f"History file: {HISTORY_FILE}", flush=True)

    if MODEL_ERROR is None:
        print("Model loaded successfully.", flush=True)
    else:
        print(f"MODEL ERROR: {MODEL_ERROR}", flush=True)

    threading.Timer(
        1.5,
        lambda: webbrowser.open("http://127.0.0.1:5000")
    ).start()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False
    )
