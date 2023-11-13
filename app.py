from flask import Flask, request, render_template
from src.Airbnb.pipelines.Prediction_Pipeline import CustomData, PredictPipeline

app = Flask(__name__)

# Define the home route
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        data = CustomData(
            property_type=request.form.get("propertytype"),
            room_type=request.form.get("roomtype"),
            bedrooms=int(request.form.get("bedrooms")),
            beds=int(request.form.get("beds")),
            amenities=int(request.form.get("amenties")),
            accommodates=int(request.form.get("accommodates")),
            bathrooms=float(request.form.get("bathrooms")),
            bed_type=request.form.get("bedtype"),
            cancellation_policy=request.form.get("canceltype"),
            cleaning_fee=float(request.form.get("clean")),
            city=request.form.get("city"),
            host_has_profile_pic=request.form.get("dp"),
            host_identity_verified=request.form.get("verify"),
            host_response_rate=request.form.get("hostresponse"),
            instant_bookable=request.form.get("instbook"),
            latitude=float(request.form.get("lat")),
            longitude=float(request.form.get("long")),
            number_of_reviews=int(request.form.get("review")),
            review_scores_rating=float(request.form.get("overallreview"))
        )

        final_data = data.get_data_as_dataframe()

        predict_pipeline = PredictPipeline()

        pred = predict_pipeline.predict(final_data)

        result = round(pred[0], 2)

        return render_template("result.html", final_result=result)

# Execution begins
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
