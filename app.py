from flask import Flask, request, render_template
from src.Airbnb.pipelines.Prediction_Pipeline import CustomData, PredictPipeline

app = Flask(__name__)

# Define the home route
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            # Validate and convert form data to CustomData object
            data = CustomData(
                property_type=request.form.get("propertytype"),
                room_type=request.form.get("roomtype"),
                amenities=(request.form.get("amenties")),
                accommodates=(request.form.get("accommodates")),
                bathrooms=(request.form.get("bathrooms")),
                bed_type=request.form.get("bedtype"),
                cancellation_policy=request.form.get("canceltype"),
                cleaning_fee=(request.form.get("clean")),
                city=request.form.get("city"),
                host_has_profile_pic=request.form.get("dp"),
                host_identity_verified=request.form.get("verify"),
                host_response_rate=request.form.get("hostresponse"),
                instant_bookable=request.form.get("instbook"),
                latitude=(request.form.get("lat")),
                longitude=(request.form.get("long")),
                number_of_reviews=(request.form.get("review")),
                review_scores_rating=(request.form.get("overallreview")),
                bedrooms=(request.form.get("bedrooms")),
                beds=(request.form.get("beds"))
            )

            final_data = data.get_data_as_dataframe()

            # Make prediction
            predict_pipeline = PredictPipeline()
            pred = predict_pipeline.predict(final_data)
            result = round(pred[0], 2)
            return render_template("index.html", final_result=result)

        except Exception as e:
            # Handle exceptions gracefully
            error_message = f"Error during prediction: {str(e)}"
            return render_template("error.html", error_message=error_message)

    else:
        # Render the initial page
        return render_template("index.html")

# Execution begins
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
