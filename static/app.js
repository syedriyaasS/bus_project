document.getElementById('route-form').addEventListener('submit', async function (event) {
    event.preventDefault();
    const source = document.getElementById('source').value;
    const destination = document.getElementById('destination').value;
    const time_of_day = document.getElementById('time').value;

    const response = await fetch('/get-route', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ source, destination, time_of_day })
    });

    const result = await response.json();
    const routeInfoDiv = document.getElementById('route-info');

    if (result.status === 'success') {
        routeInfoDiv.textContent = `Route: ${JSON.stringify(result.route)}`;

        // Debug: Log the route to check if it contains valid coordinates
        console.log(result.route);

        // Initialize the map
        const map = L.map('map').setView([28.7041, 77.1025], 13); // Set center (initial) coordinates to Delhi

        // Add tile layer (OpenStreetMap tiles)
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        }).addTo(map);

        // Plot route markers (assuming result.route contains coordinates)
        const route = result.route; // Array of route coordinates
        if (Array.isArray(route) && route.length > 0) {
            const sourceCoordinates = route[0];  // Assuming the first coordinate is the source
            const destinationCoordinates = route[route.length - 1];  // Last coordinate as destination

            // Add markers for source and destination
            L.marker(sourceCoordinates).addTo(map)
                .bindPopup("Source")
                .openPopup();

            L.marker(destinationCoordinates).addTo(map)
                .bindPopup("Destination")
                .openPopup();

            // Draw a polyline (path between source and destination)
            const latlngs = route.map(coord => [coord.lat, coord.lng]); // Assuming route contains {lat, lng} objects
            L.polyline(latlngs, { color: 'blue' }).addTo(map);
            map.fitBounds(latlngs); // Automatically zoom to fit the route
        }
    } else {
        routeInfoDiv.textContent = `Error: ${result.message}`;
    }
});
