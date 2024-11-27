import openrouteservice
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Initialize OpenRouteService client with your API key
ORS_API_KEY = "5b3ce3597851110001cf6248780e38ed197e40da8323a1132ca9ae0e"
client = openrouteservice.Client(key=ORS_API_KEY)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get-route', methods=['POST'])
def get_route():
    try:
        # Extract source and destination from the request
        data = request.json
        source = data.get('source')
        destination = data.get('destination')

        # Determine if input is a coordinate pair
        def parse_coordinates(input_value):
            if isinstance(input_value, list) and len(input_value) == 2:
                # Assume input_value is [longitude, latitude]
                return input_value
            else:
                # Perform geocoding if input is not coordinates
                return client.pelias_search(text=input_value)['features'][0]['geometry']['coordinates']

        # Parse source and destination inputs
        source_coords = parse_coordinates(source)
        destination_coords = parse_coordinates(destination)

        # Fetch road-based route between source and destination
        route = client.directions(
            coordinates=[source_coords, destination_coords],
            profile='driving-car',
            format='geojson'
        )

        # Extract route geometry
        route_geometry = route['features'][0]['geometry']['coordinates']

        # Return the route data
        return jsonify({'status': 'success', 'route': route_geometry})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'status': 'error', 'message': 'Failed to fetch the route. Please check your input.'}), 500

if __name__ == '__main__':
    app.run(debug=True)
