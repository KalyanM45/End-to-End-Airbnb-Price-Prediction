from flask import Flask, render_template, request
from PIL import Image
import pickle

app = Flask(__name__)

# Load the model
model = pickle.load(open("catboostalgo.pkl", "rb"))

# Define the home route
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        input_propertytype = int(request.form["propertytype"])
        input_roomtype = int(request.form["roomtype"])
        input_bedrooms = int(request.form["bedrooms"])
        input_beds = int(request.form["beds"])
        input_amenties = int(request.form["amenties"])
        input_accommodates = int(request.form["accommodates"])
        input_bathrooms = int(request.form["bathrooms"])
        input_bedtype = int(request.form["bedtype"])
        input_canceltype = int(request.form["canceltype"])
        input_clean = int(request.form["clean"])
        input_city = int(request.form["city"])
        input_dp = int(request.form["dp"])
        input_verify = int(request.form["verify"])
        input_hostresponse = int(request.form["hostresponse"])
        input_instbook = int(request.form["instbook"])
        input_lat = float(request.form["lat"])
        input_long = float(request.form["long"])
        input_review = int(request.form["review"])
        input_overallreview = int(request.form["overallreview"])

        # Make a prediction
        prediction = model.predict([[input_propertytype, input_roomtype, input_amenties, input_accommodates, input_bathrooms,
                                      input_bedtype, input_canceltype, input_clean, input_city, input_dp, input_verify,
                                      input_hostresponse, input_instbook, input_lat, input_long, input_review,
                                      input_overallreview, input_bedrooms, input_beds]])

        return str(prediction[0])

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True,host="0.0.0.0",port=5000)
