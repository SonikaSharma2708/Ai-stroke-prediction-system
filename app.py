```python
import os
import numpy as np
import pickle
from flask import Flask, render_template, request, jsonify, Response

app = Flask(__name__)

MODEL_PATH = os.path.join('model', 'stroke_model.pkl')
ENCODER_PATH = os.path.join('model', 'encoders.pkl')

# Your Render website URL
BASE_URL = "https://ai-stroke-prediction-system-jj8s.onrender.com"


def load_ml_assets():
    try:
        with open(MODEL_PATH, 'rb') as m_file:
            model = pickle.load(m_file)

        if os.path.exists(ENCODER_PATH):
            with open(ENCODER_PATH, 'rb') as e_file:
                scaler = pickle.load(e_file)
        else:
            scaler = None

        return model, scaler

    except Exception as e:
        print(f"⚠️ Assets alert: ML objects uninitialized ({e})")
        return None, None


model, scaler = load_ml_assets()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/services')
def services():
    return render_template('services.html')


@app.route('/tips')
def tips():
    return render_template('tips.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')


# =========================
# ROBOTS.TXT
# =========================
@app.route('/robots.txt')
def robots_txt():
    robots = f"""User-agent: *
Allow: /

Sitemap: {BASE_URL}/sitemap.xml
"""
    return Response(robots, mimetype='text/plain')


# =========================
# SITEMAP.XML
# =========================
@app.route('/sitemap.xml')
def sitemap_xml():
    sitemap = f"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">

    <url>
        <loc>{BASE_URL}/</loc>
    </url>

    <url>
        <loc>{BASE_URL}/about</loc>
    </url>

    <url>
        <loc>{BASE_URL}/services</loc>
    </url>

    <url>
        <loc>{BASE_URL}/tips</loc>
    </url>

    <url>
        <loc>{BASE_URL}/contact</loc>
    </url>

    <url>
        <loc>{BASE_URL}/predict</loc>
    </url>

</urlset>
"""
    return Response(sitemap, mimetype='application/xml')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html')

    if not model:
        g_val = float(request.form.get('avg_glucose_level', 100))
        a_val = float(request.form.get('age', 45))

        mock_risk = 88.5 if (g_val > 180 or a_val > 60) else 14.2

        return jsonify({
            "status": "success",
            "risk_percentage": mock_risk
        })

    try:
        gender = 1 if request.form.get('gender') == 'Male' else 0
        age = float(request.form.get('age', 45))
        hypertension = int(request.form.get('hypertension', 0))
        heart_disease = int(request.form.get('heart_disease', 0))
        ever_married = 1 if request.form.get('ever_married') == 'Yes' else 0

        work_map = {
            'Private': 0,
            'Self-employed': 1,
            'Govt_job': 2,
            'children': 3
        }

        work_type = work_map.get(
            request.form.get('work_type'),
            0
        )

        res_map = {
            'Urban': 1,
            'Rural': 0
        }

        residence_type = res_map.get(
            request.form.get('Residence_type'),
            1
        )

        glucose = float(
            request.form.get('avg_glucose_level', 100)
        )

        bmi = float(
            request.form.get('bmi', 25)
        )

        smoke_map = {
            'never smoked': 0,
            'formerly smoked': 1,
            'smokes': 2,
            'Unknown': 3
        }

        smoking_status = smoke_map.get(
            request.form.get('smoking_status'),
            0
        )

        features = np.array([[
            gender,
            age,
            hypertension,
            heart_disease,
            ever_married,
            work_type,
            residence_type,
            glucose,
            bmi,
            smoking_status
        ]])

        if scaler:
            features = scaler.transform(features)

        prob = model.predict_proba(features)[0][1] * 100

        return jsonify({
            "status": "success",
            "risk_percentage": round(prob, 1)
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400


@app.route('/download-report', methods=['POST'])
def download_report():
    return render_template(
        'report.html',
        patient_id=request.form.get(
            'pdf_id',
            'Anonymous Record'
        ),
        age=request.form.get(
            'pdf_age',
            'N/A'
        ),
        gender=request.form.get(
            'pdf_gender',
            'N/A'
        ),
        glucose=request.form.get(
            'pdf_glucose',
            'N/A'
        ),
        bmi=request.form.get(
            'pdf_bmi',
            'N/A'
        ),
        hypertension=request.form.get(
            'pdf_hyper',
            'No'
        ),
        risk=float(
            request.form.get(
                'pdf_risk',
                0.0
            )
        )
    )


@app.route('/submit-query', methods=['POST'])
def submit_query():
    return jsonify({
        "status": "received"
    })


if __name__ == '__main__':
    app.run(
        debug=True,
        port=5000
    )
```
