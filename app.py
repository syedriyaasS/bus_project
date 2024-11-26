from flask import Flask, render_template, request, jsonify
import json

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get-route', methods=['POST'])
def get_route():
    data = request.json
    source = data.get('source')
    destination = data.get('destination')
    time_of_day = data.get('time_of_day')

    # Log received data for debugging
    print(f"Source: {source}, Destination: {destination}, Time: {time_of_day}")

    # Example static route data (replace with actual route calculation logic)
    route_info = [
        {"lat": 28.7041, "lng": 77.1025},  # Source (Delhi)
        {"lat": 28.5355, "lng": 77.3910}   # Destination (Noida)
    ]

    # Return the route information as JSON
    return jsonify({'status': 'success', 'route': route_info})

if __name__ == '__main__':
    app.run(debug=True)
