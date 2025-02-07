from PIL import Image
import numpy as np
import requests
from io import BytesIO
import plotly.express as px

import os
import numpy as np
import keras
from keras import layers
import tensorflow as tf

from scipy.ndimage import center_of_mass, shift

from dash import Dash, dcc, html
import dash_bootstrap_components as dbc

from dash import Input, Output, callback
import json
import numpy as np

mnist_model = tf.keras.models.load_model('my_model.keras')

def predict_digit(pixel_array):
    processed_array = center_image(pixel_array).reshape(1, 28, 28, 1) / 255
    return mnist_model.predict(processed_array).argmax()


def center_image(image):
    cy, cx = center_of_mass(image)
    shift_y = 14 - cy
    shift_x = 14 - cx
    centered_image = shift(image, shift=(shift_y, shift_x), mode='constant', cval=0)
    return centered_image



image_path = "https://raw.githubusercontent.com/adidror005/canvas/refs/heads/main/background_image_28.png"
response = requests.get(image_path)
response.raise_for_status()

img = Image.open(BytesIO(response.content))
img = np.array(img)

fig = px.imshow(img,height=420)

fig.update_layout(
    dragmode="drawopenpath",
    newshape=dict(line_color='red', line_width=40)  # Set color to red and width to 5
)


app = Dash()
app.layout = html.Div([
    html.H1("Draw a Digit!", style={'text-align': 'center'}),
    dcc.Graph(
        id="graph-picture",
        figure=fig,
        config={"modeBarButtonsToAdd": ['eraseshape']}
    ),
    html.H1(id="annotations-data", style={'text-align': 'center'}),
    dbc.Button("Reset", id="reset-button", color="primary", style={'text-align': 'center'})
])


@callback(
    Output("annotations-data", "children"),
    Input("graph-picture", "relayoutData"),
    prevent_initial_call=True
)
def on_new_annotation(relayout_data):
    if "shapes" in relayout_data:
        shapes = relayout_data["shapes"]
        path_data_list = [shape['path'] for shape in shapes]
        pixel_array = path_to_pixel_matrix(path_data_list)
        pred = predict_digit(pixel_array)
        return f"You drew a {pred}"
    return "No drawing yet!"

import numpy as np
import re
from PIL import Image, ImageDraw

def path_to_pixel_matrix(path_data_list):
    return np.sum(
            [single_path_to_pixel_matrix(path, 28, 28) for path in path_data_list], axis=0
        )

def single_path_to_pixel_matrix(path_data, width, height):
    """Converts an SVG path data string to a pixel matrix (binary image).

    Args:
        path_data: The SVG path data string.
        width: The width of the output image.
        height: The height of the output image.

    Returns:
        A NumPy array representing the pixel matrix (0 for background, 1 for path).
    """

    # Create a blank image
    img = Image.new("L", (width, height), color=0)  # "L" mode for grayscale
    draw = ImageDraw.Draw(img)

    # Split the path data into commands and coordinates
    commands = re.findall(r"[MLHVCSQTAZ]", path_data)
    coordinates = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", path_data)
    coordinates = [float(coord) for coord in coordinates]

    # Initialize the points array
    points = []
    current_point = (0, 0)

    # Iterate through commands and coordinates to extract points
    coord_index = 0
    for command in commands:
        if command in "ML":  # Moveto or Lineto
            num_coords = 2
            x, y = coordinates[coord_index : coord_index + num_coords]
            points.append((x, y))
            current_point = (x, y)
            coord_index += num_coords
        # Add handling for other commands if needed

    # Draw the path on the image
    if points:
        draw.line(points, fill=255, width=3)  # Draw line with width 3

    # Convert the image to a NumPy array
    pixel_matrix = np.array(img)

    return pixel_matrix

@app.callback(
    Output("graph-picture", "figure"),
    Input("reset-button", "n_clicks"),
    prevent_initial_call=True,
)
def reset_graph(n_clicks):
    fig = px.imshow(img, height=420)
    fig = px.imshow(img,height=420)
    fig.update_layout(
        dragmode="drawopenpath",
        newshape=dict(line_color='red', line_width=45)  # Set color to red and width to 5
    )
    return fig

if __name__ == "__main__":
    app.run(debug=False)
