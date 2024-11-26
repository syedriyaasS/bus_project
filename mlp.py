import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

class TrafficManagementSystem:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def load_and_preprocess_data(self, csv_file):
        # Load the dataset
        self.df = pd.read_csv(csv_file)
        
        # Identify categorical columns
        categorical_columns = ['source', 'destination', 'time_of_day']
        for i in range(1, 13):  # For stops 1-12
            categorical_columns.extend([
                f'stop{i}',
                f'stop{i}_traffic',
                f'stop{i}_weather',
                f'stop{i}_road_condition'
            ])
        
        # Create a copy of the dataframe for processing
        processed_df = self.df.copy()
        
        # Encode categorical variables
        for column in categorical_columns:
            if column in processed_df.columns:
                self.label_encoders[column] = LabelEncoder()
                # Handle NaN values before encoding
                processed_df[column] = processed_df[column].fillna('None')
                processed_df[column] = self.label_encoders[column].fit_transform(processed_df[column])
        
        # Convert boolean columns
        for i in range(1, 13):
            alt_route_col = f'stop{i}_alt_route'
            if alt_route_col in processed_df.columns:
                processed_df[alt_route_col] = processed_df[alt_route_col].astype(int)
        
        # Separate features and targets
        self.target_columns = ['travel_time_minutes', 'alt_travel_time_minutes']
        self.feature_columns = [col for col in processed_df.columns 
                              if col not in self.target_columns 
                              and not col.endswith('_passenger_demand_in')
                              and not col.endswith('_passenger_demand_out')]
        
        self.features = processed_df[self.feature_columns]
        self.targets = processed_df[self.target_columns]
        
        # Scale features
        self.features_scaled = self.scaler.fit_transform(self.features)
        
        # Store column names for later use
        self.feature_names = self.features.columns
        
    def build_model(self):
        # Define the neural network architecture
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu', input_shape=(self.features_scaled.shape[1],)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(2)  # Output layer for main and alternate route times
        ])
        
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
    def train_model(self, epochs=40):
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            self.features_scaled, self.targets, test_size=0.2, random_state=42
        )
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=32,
            verbose=1
        )
        return history

    def evaluate_model(self):
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            self.features_scaled, self.targets, test_size=0.2, random_state=42
        )
        
        # Evaluate the model
        loss, mae = self.model.evaluate(X_test, y_test, verbose=1)
        print(f"\nModel Evaluation Results:")
        print(f"Mean Squared Error (Loss): {loss:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")
        return loss, mae
    
    def prepare_input_data(self, source, destination, time_of_day):
        # Create a dictionary with the same structure as the training data
        input_data = {}
        
        # Get a sample row for structure
        sample_row = self.df.iloc[0]
        
        # Fill in the known values
        for col in self.feature_names:
            if col == 'source':
                input_data[col] = source
            elif col == 'destination':
                input_data[col] = destination
            elif col == 'time_of_day':
                input_data[col] = time_of_day
            elif col.endswith('_alt_route'):
                input_data[col] = 0  # Default to False for alternate routes
            else:
                # Use the sample values for other features
                input_data[col] = sample_row[col]
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Encode categorical variables
        for column, encoder in self.label_encoders.items():
            if column in input_df.columns:
                try:
                    input_df[column] = encoder.transform(input_df[column])
                except ValueError:
                    # If value not in training data, use the first category
                    input_df[column] = encoder.transform([encoder.classes_[0]])
        
        # Scale the features
        input_scaled = self.scaler.transform(input_df)
        
        return input_scaled
    
    def get_route_stops(self, source, destination):
        # Get the stops for a route from the dataset
        route_data = self.df[
            (self.df['source'] == source) & 
            (self.df['destination'] == destination)
        ].iloc[0]
        
        stops = []
        for i in range(1, 13):  # For stops 1 through 12
            stop_name = route_data.get(f'stop{i}')
            if isinstance(stop_name, str) and stop_name.strip():
                stop_info = {
                    'name': stop_name,
                    'traffic': route_data.get(f'stop{i}_traffic', 'Unknown'),
                    'weather': route_data.get(f'stop{i}_weather', 'Unknown'),
                    'road_condition': route_data.get(f'stop{i}_road_condition', 'Unknown'),
                    'passenger_demand': {
                        'in': route_data.get(f'stop{i}_passenger_demand_in', 0),
                        'out': route_data.get(f'stop{i}_passenger_demand_out', 0)
                    }
                }
                stops.append(stop_info)
        return stops
    
    def suggest_route(self, source, destination, time_of_day):
        # Prepare input data
        input_scaled = self.prepare_input_data(source, destination, time_of_day)
        
        # Make prediction
        prediction = self.model.predict(input_scaled)
        main_route_time, alt_route_time = prediction[0]
        
        # Get route details from sample data
        try:
            route_data = self.df[
                (self.df['source'] == source) & 
                (self.df['destination'] == destination)
            ].iloc[0]
            
            main_stops = self.get_route_stops(source, destination)
            
            return {
                'main_route': {
                    'time': max(0, main_route_time),
                    'distance': route_data['distance_km'],
                    'stops': main_stops
                },
                'alternate_route': {
                    'time': max(0, alt_route_time),
                    'distance': route_data['alt_distance_km']
                }
            }
        except IndexError:
            return None

def print_route_details(route_info):
    if route_info is None:
        print("No route found for the specified source and destination.")
        return
    
    print("\n=== ROUTE SUGGESTIONS ===")
    print("\nMAIN ROUTE:")
    print(f"Estimated Distance: {route_info['main_route']['distance']:.2f} km")
    print(f"Estimated Time: {route_info['main_route']['time']:.0f} minutes")
    
    print("\nStops along the main route:")
    print("-" * 50)
    for stop in route_info['main_route']['stops']:
        print(f"\nStop: {stop['name']}")
        print(f"Traffic Condition: {stop['traffic']}")
        print(f"Weather: {stop['weather']}")
        print(f"Road Condition: {stop['road_condition']}")
        print(f"Passenger Demand - In: {stop['passenger_demand']['in']}, Out: {stop['passenger_demand']['out']}")
    
    print("\nALTERNATE ROUTE:")
    print(f"Estimated Distance: {route_info['alternate_route']['distance']:.2f} km")
    print(f"Estimated Time: {route_info['alternate_route']['time']:.0f} minutes")
    print("\nRecommendation:")
    if route_info['main_route']['time'] <= route_info['alternate_route']['time']:
        print("→ Take the main route")
    else:
        print("→ Take the alternate route")

def calculate_accuracy(y_true, y_pred, threshold=10):
    """
    Calculate accuracy as the percentage of predictions that are within a given threshold.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    within_threshold = np.abs(y_true - y_pred) <= threshold
    accuracy = np.mean(within_threshold) * 100
    return accuracy

def evaluate_model_with_accuracy(self):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        self.features_scaled, self.targets, test_size=0.2, random_state=42
    )
    
    # Evaluate the model
    predictions = self.model.predict(X_test)
    loss, mae = self.model.evaluate(X_test, y_test, verbose=1)
    
    # Calculate accuracy
    accuracies = []
    for i in range(y_test.shape[1]):  # For each target column
        acc = calculate_accuracy(y_test.iloc[:, i], predictions[:, i])
        accuracies.append(acc)
    
    print(f"\nModel Evaluation Results:")
    print(f"Mean Squared Error (Loss): {loss:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Accuracy for travel_time_minutes: {accuracies[0]:.2f}%")
    print(f"Accuracy for alt_travel_time_minutes: {accuracies[1]:.2f}%")
    
    return loss, mae, accuracies

# Add the method to the TrafficManagementSystem class
TrafficManagementSystem.evaluate_model_with_accuracy = evaluate_model_with_accuracy

# Update the main function to use the new evaluation method
def main():
    # Initialize the system
    tms = TrafficManagementSystem()
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    tms.load_and_preprocess_data('traffic.csv')
    
    # Build and train the model
    print("Building and training the model...")
    tms.build_model()
    tms.train_model(epochs=50)  # Reduced epochs for demonstration purposes
    
    # Evaluate the model with accuracy
    print("\nEvaluating the model with accuracy...")
    tms.evaluate_model_with_accuracy()
    
    while True:
        print("\n=== TRAFFIC MANAGEMENT SYSTEM ===")
        time_of_day = input("\nEnter time of day (format HH:MM, e.g., 09:11): ")
        source = input("Enter source location: ")
        destination = input("Enter destination location: ")
        
        # Get and display route suggestions
        print("\nCalculating routes...")
        routes = tms.suggest_route(source, destination, time_of_day)
        print_route_details(routes)
        
        # Ask if user wants to check another route
        again = input("\nWould you like to check another route? (yes/no): ")
        if again.lower() != 'yes':
            break
    
    print("\nThank you for using the Traffic Management System!")

if __name__ == "__main__":
    main()
