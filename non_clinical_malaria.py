from flask import Flask, render_template, request, jsonify
import geopandas as gpd
import joblib
import folium
from sklearn.preprocessing import StandardScaler
from folium import Choropleth

app = Flask(__name__)

# Paths to model, scaler, and shapefile
model_path = "./Models/Non Clinical/models/random_forest.joblib"
scaler_path = "./Models/Non Clinical/models/scaler_rf.joblib"
data_path = "./ward/Wards.shp"

# Load the trained model and scaler
trained_model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Load the shapefile (GeoDataFrame)
prediction_data = gpd.read_file(data_path)

# Extract features for prediction
features = ['Rainfall', 'LST', 'Relative_H']
X_pred = prediction_data[features]

# Check and handle missing values
if X_pred.isnull().any().any():
    X_pred = X_pred.fillna(X_pred.mean())

# Scale the prediction data
X_pred_scaled = scaler.transform(X_pred)

# Make predictions
predictions = trained_model.predict(X_pred_scaled)
prediction_data['pred_cases'] = predictions.astype(int)

@app.route('/')
def index():
    return render_template('non_clinical_index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Create a Folium map dynamically centered
        center = prediction_data.geometry.centroid.unary_union.centroid
        m = folium.Map(location=[center.y, center.x], zoom_start=6)

        # Add choropleth layer for visualizing predicted cases
        Choropleth(
            geo_data=prediction_data,
            data=prediction_data,
            columns=['wardcode', 'pred_cases'],
            key_on='feature.properties.wardcode',
            fill_color='YlOrRd',
            fill_opacity=1.0,
            line_opacity=0.1,
            legend_name='Predicted Cases'
        ).add_to(m)

        # Add a hover popup (tooltip) for each ward
        for _, row in prediction_data.iterrows():
            # Create a GeoJson object for each ward and add a tooltip that shows predicted cases, LGA, and Ward
            folium.GeoJson(
                row.geometry,
                tooltip=folium.Tooltip(
                    f"Predicted Cases: {row['pred_cases']}<br>"
                    f"Ward: {row['wardname']}<br>"
                    f"LGA: {row['lganame']}"
                ),  # Show predicted cases, Ward, and LGA on hover
            ).add_to(m)

        # Render the map to HTML
        map_html = m._repr_html_()
        return render_template('non_clinical_index.html', map_html=map_html)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(port=5002, debug=True)
