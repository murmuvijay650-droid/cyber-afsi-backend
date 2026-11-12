from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64

app = Flask(__name__)

# Base64 स्ट्रिंग को OpenCV इमेज में बदलने का फंक्शन
def base64_to_image(base64_string):
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        img_data = base64.b64decode(base64_string)
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
        return img
    except Exception as e:
        print("Error decoding image:", e)
        return None

# दो इमेजेस को मैच करने का फंक्शन
def match_fingerprints(img1, img2):
    if img1 is None or img2 is None:
        return 0.0

    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return 0.0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = [m for m in matches if m.distance < 50]

    if len(des1) > 0 and len(des2) > 0:
        score = (len(good_matches) / min(len(des1), len(des2))) * 100
        score = min(max(score, 0.0), 100.0)
        return round(score, 2)
    return 0.0

@app.route('/match', methods=['POST'])
def match_api():
    data = request.json
    if not data or 'query_img' not in data or 'suspect_fingers' not in data:
        return jsonify({"error": "Invalid data format"}), 400

    query_img = base64_to_image(data['query_img'])
    suspect_fingers = data['suspect_fingers']
    best_score = 0.0
    
    for finger_base64 in suspect_fingers:
        if finger_base64:
            suspect_img = base64_to_image(finger_base64)
            score = match_fingerprints(query_img, suspect_img)
            if score > best_score:
                best_score = score

    return jsonify({"match_score": best_score})

if __name__ == '__main__':
    # Render के लिए पोर्ट 10000 ज़रूरी है
    app.run(host='0.0.0.0', port=10000)
