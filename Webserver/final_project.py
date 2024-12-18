import pandas as pd
from flask import Flask, render_template, request, jsonify
import pycountry
import plotly.express as px

app = Flask(__name__, static_folder='static')

data = pd.read_csv("processed_data.csv")

@app.route('/')
def index():
    return render_template('index.html')

def get_country_flag_url(country_name):
    try:
        country_code = pycountry.countries.lookup(country_name).alpha_2.lower()
        return f"https://flagcdn.com/w80/{country_code}.png"
    except LookupError:
        return ""

@app.route('/country', methods=['GET'])
def get_country_data():
    country_name = request.args.get('country')
    if not country_name:
        return jsonify({"error": "Country name is required"}), 400

    country_data = data[data['Country or region'] == country_name]
    if country_data.empty:
        return jsonify({"error": "Country not found"}), 404

    result = country_data.to_dict(orient='records')[0]

    response = {
        "Country": result['Country or region'],
        "Overall Rank": result['Overall rank'],
        "Score": {
            "value": result['Score'],
            "percentile": result['Score_percentile']
        },
        "GDP per capita": {
            "value": result['GDP per capita'],
            "percentile": result['GDP per capita_percentile']
        },
        "Social support": {
            "value": result['Social support'],
            "percentile": result['Social support_percentile']
        },
        "Healthy life expectancy": {
            "value": result['Healthy life expectancy'],
            "percentile": result['Healthy life expectancy_percentile']
        },
        "Freedom to make life choices": {
            "value": result['Freedom to make life choices'],
            "percentile": result['Freedom to make life choices_percentile']
        },
        "Generosity": {
            "value": result['Generosity'],
            "percentile": result['Generosity_percentile']
        },
        "Perceptions of corruption": {
            "value": result['Perceptions of corruption'],
            "percentile": result['Perceptions of corruption_percentile']
        },
        "Flag": get_country_flag_url(country_name)
    }
    return jsonify(response)

@app.route('/interactive-map', methods=['GET'])
def interactive_map():
    fig = px.choropleth(
        data_frame=data,
        locations='Country or region',
        locationmode='country names',
        color='Score',
        hover_data={'Overall rank': True, 'Score': True},
        hover_name='Country or region',
        title='World Happiness Rankings 2019',
        color_continuous_scale=['red', 'yellow', 'green']
    )
    fig.update_geos(projection_type="natural earth")
    
    return fig.to_json()

@app.route('/overall-trends', methods=['GET'])
def overall_trends():
    relevant_columns = [
        'GDP per capita', 'Social support',
        'Healthy life expectancy', 'Freedom to make life choices',
        'Generosity', 'Perceptions of corruption'
    ]
    correlation_results = data[relevant_columns + ['Overall rank']].corr()['Overall rank']
    sorted_correlations = correlation_results.drop('Overall rank').sort_values()

    return render_template('trends.html', correlations=sorted_correlations.to_dict())

@app.route('/scatterplot', methods=['GET'])
def scatterplot():
    category = request.args.get('category')
    
    valid_categories = [
        'GDP per capita', 'Social support', 'Healthy life expectancy',
        'Freedom to make life choices', 'Generosity', 'Perceptions of corruption'
    ]
    if category not in valid_categories:
        return jsonify({"error": f"Invalid category. Choose from {', '.join(valid_categories)}"}), 400

    fig = px.scatter(
        data,
        x=category,
        y='Overall rank',
        hover_name='Country or region',
        title=f'Scatterplot of {category} vs Overall Rank',
        labels={'Overall rank': 'Happiness Rank', category: category},
    )
    fig.update_traces(marker=dict(size=10, color='blue', line=dict(width=1, color='DarkSlateGrey')))
    fig.update_layout(title_x=0.5)
    
    return fig.to_json()

from geopy.distance import geodesic
import pandas as pd

coordinates = pd.read_csv('countries.csv')
processed_data = pd.read_csv('processed_data.csv')
coordinates = coordinates[coordinates['name'].isin(processed_data['Country or region'])]


@app.route('/knn-analysis', methods=['GET', 'POST'])
def knn_analysis():
    country_name = request.args.get('country', '')
    k = int(request.args.get('k', 5))

    if not country_name or country_name not in processed_data['Country or region'].values:
        return render_template('knn.html', error="Please enter a valid country name.", neighbors=None, country=None, k=k)

    input_coords_row = coordinates[coordinates['name'] == country_name]
    if input_coords_row.empty:
        return render_template('knn.html', error=f"Coordinates not found for {country_name}", neighbors=None, country=None, k=k)

    input_coords = (input_coords_row.iloc[0]['latitude'], input_coords_row.iloc[0]['longitude'])

    distances = []
    for _, row in coordinates.iterrows():
        if row['name'] != country_name:
            coords = (row['latitude'], row['longitude'])
            distance = geodesic(input_coords, coords).kilometers
            distances.append((row['name'], distance))

    distances = sorted(distances, key=lambda x: x[1])
    nearest_neighbors = distances[:k]

    neighbors = []
    for country, _ in nearest_neighbors:
        score = processed_data.loc[processed_data['Country or region'] == country, 'Score'].values[0]
        neighbors.append({"name": country, "score": round(score, 2)})

    predicted_score = sum(n['score'] for n in neighbors) / len(neighbors)
    actual_score = processed_data.loc[processed_data['Country or region'] == country_name, 'Score'].values[0]

    return render_template('knn.html', country=country_name, neighbors=neighbors, predicted_score=round(predicted_score, 2), actual_score=round(actual_score, 2), k=k)

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

@app.route('/kmeans-clustering', methods=['GET', 'POST'])
def kmeans_clustering():
    k = request.args.get('k', default=3, type=int)
    k = min(k, 10)

    map_data = processed_data[['Country or region', 'Score']].copy()

    coordinates_with_scores = pd.merge(
        map_data, 
        coordinates[['name', 'latitude', 'longitude']], 
        left_on='Country or region', 
        right_on='name'
    )

    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(
        coordinates_with_scores[['Score', 'latitude', 'longitude']]
    )

    kmeans = KMeans(n_clusters=k, random_state=42)
    coordinates_with_scores['Cluster'] = kmeans.fit_predict(standardized_data)

    cluster_avg_scores = coordinates_with_scores.groupby('Cluster')['Score'].mean().sort_values(ascending=False)
    cluster_order = cluster_avg_scores.index.tolist()


    cluster_colors = [
        "blue", "green", "red", "purple", "orange", 
        "pink", "brown", "cyan", "magenta", "gray"
    ]
    color_map = {cluster: cluster_colors[i] for i, cluster in enumerate(cluster_order)}

    coordinates_with_scores['Cluster_Sorted'] = coordinates_with_scores['Cluster'].map(
        {cluster: i for i, cluster in enumerate(cluster_order)}
    )

    legend_html = "<ul style='list-style: none; padding: 0;'>"
    for i, cluster in enumerate(cluster_order):
        avg_score = cluster_avg_scores[cluster]
        legend_html += f"<li style='color: {color_map[cluster]};'><strong>Cluster {i+1}: {avg_score:.2f}</strong></li>"
    legend_html += "</ul>"

    coordinates_with_scores['Cluster_Color'] = coordinates_with_scores['Cluster_Sorted'].map(color_map)
    coordinates_with_scores['Cluster_Display'] = coordinates_with_scores['Cluster_Sorted'] + 1


    fig = px.choropleth(
        coordinates_with_scores,
        locations='Country or region',
        locationmode='country names',
        color='Cluster_Display',
        hover_name='Country or region',
        hover_data={'Score': True, 'Cluster_Display': True},
        title=f'K-Means Clustering of Happiness Scores (k={k})'
    )

    fig.update_layout(coloraxis_showscale=False)
    fig.update_geos(projection_type="natural earth")
    fig.update_layout(showlegend=False)

    graph_html = fig.to_html(full_html=False)
    return render_template('kmeans.html', graph_html=graph_html, legend_html=legend_html, k=k)




# API Endpoints

import geopy.distance

@app.route('/api/knn-analysis', methods=['GET'])
def api_knn_analysis():
    country_name = request.args.get('country')
    k = request.args.get('k', type=int, default=3)

    if not country_name:
        return jsonify({"error": "Country name is required"}), 400

    if country_name not in processed_data['Country or region'].values:
        return jsonify({"error": f"Country '{country_name}' not found"}), 404

    if k <= 0:
        return jsonify({"error": "k must be a positive integer"}), 400

    input_coords_row = coordinates[coordinates['name'] == country_name]
    if input_coords_row.empty:
        return jsonify({"error": f"Coordinates for '{country_name}' not found"}), 404
    input_coords = (input_coords_row.iloc[0]['latitude'], input_coords_row.iloc[0]['longitude'])

    coordinates['Distance'] = coordinates.apply(
        lambda row: geopy.distance.distance(input_coords, (row['latitude'], row['longitude'])).km, axis=1
    )

    nearest_neighbors = coordinates.nsmallest(k + 1, 'Distance')
    nearest_neighbors = nearest_neighbors[nearest_neighbors['name'] != country_name].head(k)

    nearest_neighbors = pd.merge(nearest_neighbors, processed_data, left_on='name', right_on='Country or region')
    predicted_rank = nearest_neighbors['Score'].mean()

    input_country_score = processed_data.loc[
        processed_data['Country or region'] == country_name, 'Score'
    ].values[0]

    response = {
        "input_country": country_name,
        "input_country_score": input_country_score,
        "predicted_rank": round(predicted_rank, 2),
        "nearest_neighbors": nearest_neighbors[['name', 'Score', 'Distance']].to_dict(orient='records')
    }
    return jsonify(response)

@app.route('/api/kmeans-clustering', methods=['GET'])
def api_kmeans_clustering():
    k = request.args.get('k', type=int, default=3)
    k = min(k, 10)

    if k <= 1:
        return jsonify({"error": "k must be greater than 1"}), 400

    map_data = processed_data[['Country or region', 'Score']].copy()

    coordinates_with_scores = pd.merge(
        map_data,
        coordinates[['name', 'latitude', 'longitude']],
        left_on='Country or region',
        right_on='name'
    )

    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(
        coordinates_with_scores[['Score', 'latitude', 'longitude']]
    )

    kmeans = KMeans(n_clusters=k, random_state=42)
    coordinates_with_scores['Cluster'] = kmeans.fit_predict(standardized_data)

    cluster_means = coordinates_with_scores.groupby('Cluster')['Score'].mean().sort_values(ascending=False)

    response = {
        "clusters": coordinates_with_scores[['Country or region', 'Cluster']].to_dict(orient='records'),
        "cluster_means": cluster_means.to_dict()
    }
    return jsonify(response)

@app.route('/api/country-data', methods=['GET'])
def api_country_data():
    country_name = request.args.get('country')

    if not country_name:
        return jsonify({"error": "Country name is required"}), 400

    country_data = processed_data[processed_data['Country or region'] == country_name]

    if country_data.empty:
        return jsonify({"error": f"Country '{country_name}' not found"}), 404

    result = country_data.iloc[0].to_dict()

    response = {
        "Country": result['Country or region'],
        "Rank": result['Overall rank'],
        "Score": result['Score'],
        "GDP per capita": result['GDP per capita'],
        "Social support": result['Social support'],
        "Healthy life expectancy": result['Healthy life expectancy'],
        "Freedom to make life choices": result['Freedom to make life choices'],
        "Generosity": result['Generosity'],
        "Perceptions of corruption": result['Perceptions of corruption']
    }

    return jsonify(response)


# Example API Queries

# http://127.0.0.1:5000/api/knn-analysis?country=Germany&k=4

# http://127.0.0.1:5000/api/kmeans-clustering?k=5

# http://127.0.0.1:5000/api/country-data?country=Finland


if __name__ == '__main__':
    app.run(debug=True)