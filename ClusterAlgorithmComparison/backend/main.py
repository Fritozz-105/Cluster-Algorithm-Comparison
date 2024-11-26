from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app, origins="*")

@app.route("/api/books", methods=["GET"])
def greeting():
    return jsonify(
        {
            "books": [
                "To Kill a Mocking Bird",
                "Hamlet",
                "Brave New World",
                "1984",
                "The Great Gatsby"
            ]
        }
    )

if __name__ == "__main__":
    app.run(debug=True, port=5000)
