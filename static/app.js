document.getElementById('route-form').addEventListener('submit', async function (event) {
    event.preventDefault();

    const source = document.getElementById('source').value;
    const destination = document.getElementById('destination').value;

    try {
        const response = await fetch('/get-route', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ source, destination })
        });

        const result = await response.json();
        const routeInfoDiv = document.getElementById('route-info');
        const mapContainer = document.getElementById('map');
        mapContainer.innerHTML = ''; // Clear the map for a new route

        if (result.status === 'success') {
            routeInfoDiv.textContent = 'Route successfully retrieved!';
            const route = result.route;

            // Initialize Leaflet map
            const map = L.map('map').setView([route[0][1], route[0][0]], 13);

            // Add OpenStreetMap tiles
            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                attribution: '&copy; OpenStreetMap contributors'
            }).addTo(map);

            // Convert route to LatLng array for Leaflet
            const latlngs = route.map(coord => [coord[1], coord[0]]);

            // Add the polyline to the map
            L.polyline(latlngs, { color: 'blue', weight: 5 }).addTo(map);

            // Fit map bounds to the route
            map.fitBounds(L.polyline(latlngs).getBounds());

            // Add markers for source and destination
            L.marker(latlngs[0]).addTo(map).bindPopup('Source').openPopup();
            L.marker(latlngs[latlngs.length - 1]).addTo(map).bindPopup('Destination').openPopup();
        } else {
            routeInfoDiv.textContent = `Error: ${result.message}`;
        }
    } catch (error) {
        console.error('Error fetching the route:', error);
        document.getElementById('route-info').textContent = 'Failed to fetch the route. Please try again.';
    }
});
