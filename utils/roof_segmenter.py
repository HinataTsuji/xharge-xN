import os
import streamlit as st
from roboflow import Roboflow
from PIL import Image

def segment_roof_facets(image: Image.Image, confidence_threshold: int = 35) -> list[dict]:
    """
    Sends an image to Proglint's roof_ridge-segmentation model on Roboflow.
    Returns structured sub-plane facets for the PV optimization engine.
    """
    # 1. Gracefully handle missing environment API keys
    if "ROBOFLOW_API_KEY" not in st.secrets:
        st.error("Missing Secret Key! Please add ROBOFLOW_API_KEY inside your .streamlit/secrets.toml file.")
        return []

    api_key = st.secrets.get("ROBOFLOW_API_KEY")
    if not api_key:
        st.error("ROBOFLOW_API_KEY is empty. Please set a valid Roboflow API key in .streamlit/secrets.toml.")
        return []

    # 2. Connect to the Roboflow Client
    try:
        rf = Roboflow(api_key=api_key)
    except Exception as e:
        st.error(f"Failed to initialize Roboflow client: {e}")
        return []
    
    # 3. Reference the specific model route you selected
    try:
        workspace = rf.workspace("proglint-asacd")
        project = workspace.project("roof_ridge-segmentation")
        preferred_version = 4
        version = project.version(preferred_version)
        model = version.model

        if model is None:
            versions = sorted(project.versions(), key=lambda item: int(item.version), reverse=True)
            for candidate_version in versions:
                if candidate_version.model is not None:
                    version = candidate_version
                    model = candidate_version.model
                    st.info(
                        f"Roboflow version {preferred_version} has no trained model yet. Using version {candidate_version.version} instead."
                    )
                    break
    except Exception as e:
        st.error(f"Failed to load Roboflow model: {e}")
        return []

    if model is None:
        st.error(
            "Roboflow returned no model object. Check the workspace name, project name, and version number, or train a version with a model."
        )
        return []

    # 4. Save file payload temporarily for the SDK stream wrapper
    temp_filename = "temp_segment_target.jpg"
    image.convert("RGB").save(temp_filename)
    
    try:
        # Run inference (adjust confidence threshold depending on your image qualities)
        response = model.predict(temp_filename, confidence=confidence_threshold).json()
    except Exception as e:
        st.error(f"Roboflow API inference error: {e}")
        return []
    finally:
        # Cleanup file path immediately from runtime context
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

    parsed_facets = []
    predictions = response.get("predictions", [])

    def extract_polygon_points(prediction: dict) -> list[tuple[float, float]]:
        """Normalize common Roboflow polygon response shapes into (x, y) tuples."""
        if isinstance(prediction.get("polygon"), list):
            polygon = []
            for point in prediction["polygon"]:
                if isinstance(point, dict) and "x" in point and "y" in point:
                    polygon.append((point["x"], point["y"]))
                elif isinstance(point, (list, tuple)) and len(point) >= 2:
                    polygon.append((point[0], point[1]))
            if polygon:
                return polygon

        if isinstance(prediction.get("points"), list):
            polygon = []
            for point in prediction["points"]:
                if isinstance(point, dict) and "x" in point and "y" in point:
                    polygon.append((point["x"], point["y"]))
            if polygon:
                return polygon

        if isinstance(prediction.get("x"), list) and isinstance(prediction.get("y"), list):
            return list(zip(prediction["x"], prediction["y"]))

        if all(key in prediction for key in ("x", "y", "width", "height")):
            x = prediction["x"]
            y = prediction["y"]
            width = prediction["width"]
            height = prediction["height"]
            return [
                (x - width / 2, y - height / 2),
                (x + width / 2, y - height / 2),
                (x + width / 2, y + height / 2),
                (x - width / 2, y + height / 2),
            ]

        return []

    # 5. Restructure API point streams into standard (x, y) coordinate arrays
    for pred in predictions:
        polygon_points = extract_polygon_points(pred)

        # Keep only structurally sound boundaries (polygons with 3+ vertices)
        if len(polygon_points) >= 3:
            parsed_facets.append({
                "class": pred.get("class", "unknown_facet"),
                "confidence": pred.get("confidence", 0.0),
                "polygon": polygon_points,
            })

    return parsed_facets