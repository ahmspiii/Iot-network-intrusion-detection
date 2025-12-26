from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

#   
MODEL_PATH = r"D:\data grad\wataiData\csv\CICIoT2023\output\rf final high recall\rf_model_final.pkl"
model = joblib.load(MODEL_PATH)

#   Label Encoder  
ENCODER_PATH = r"D:\data grad\wataiData\csv\CICIoT2023\output\rf final high recall\label_encoder2.pkl"
label_encoder = joblib.load(ENCODER_PATH)

#   features
X_columns = [
    'flow_duration', 'Header_Length', 'Protocol Type', 'Duration',
    'Rate', 'Srate', 'Drate', 'fin_flag_number', 'syn_flag_number',
    'rst_flag_number', 'psh_flag_number', 'ack_flag_number',
    'ece_flag_number', 'cwr_flag_number', 'ack_count',
    'syn_count', 'fin_count', 'urg_count', 'rst_count', 
    'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC', 'TCP',
    'UDP', 'DHCP', 'ARP', 'ICMP', 'IPv', 'LLC', 'Tot sum', 'Min',
    'Max', 'AVG', 'Std', 'Tot size', 'IAT', 'Number', 'Magnitue',
    'Radius', 'Covariance', 'Variance', 'Weight'
]

@app.route("/predict", methods=["POST"])
def predict():
    try:
        #  request data
        data = request.get_json(force=True)

        #  features check
        missing = [col for col in X_columns if col not in data]
        if missing:
            return jsonify({
                "error": f"Missing features: {missing}"
            }), 400

        # columns arrangement
        input_data = [data[col] for col in X_columns]
        input_array = np.array(input_data).reshape(1, -1)

        # prediction 
        prediction = model.predict(input_array)

        #  encoder if exists
        if label_encoder:
            prediction_label = label_encoder.inverse_transform(prediction)
        else:
            prediction_label = prediction

        return jsonify({
            "prediction": prediction_label.tolist(),
            "status": "success"
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "failed"
        }), 500

if __name__ == "__main__":
    
    app.run(port=5000)

