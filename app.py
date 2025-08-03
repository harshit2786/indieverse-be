from beam import Image, asgi, QueueDepthAutoscaler, Volume
from pydantic import BaseModel
import os
from typing import List, Dict, Any

class ImageRequest(BaseModel):
    file: str
    session_id: str

class PointRequest(BaseModel):
    x: int
    y: int
    session_id: str

class ColorMaskRequest(BaseModel):
    session_id: str
    mask_indices: List[int]
    color: List[int] 

def init_models():
    import torch
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2
    
    # Download the checkpoint file if it doesn't exist
    checkpoint_url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
    checkpoint_path = "/tmp/sam2.1_hiera_large.pt"
    
    if not os.path.exists(checkpoint_path):
        import requests
        print("Downloading SAM2 checkpoint...")
        response = requests.get(checkpoint_url)
        with open(checkpoint_path, "wb") as f:
            f.write(response.content)
        print("Download completed!")
    
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    sam2 = build_sam2(
        model_cfg, 
        checkpoint_path,  # Use the downloaded file
        device="cuda" if torch.cuda.is_available() else "cpu", 
        apply_postprocessing=False
    )

    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=64,           # More sampling points (default is usually 32)
        points_per_batch=128,         # Process more points per batch
        pred_iou_thresh=0.7,          # Lower threshold for mask quality
        stability_score_thresh=0.92,   # Higher stability requirement
        stability_score_offset=0.7,   # Stability offset
        crop_n_layers=1,              # Use image crops for better small object detection
        box_nms_thresh=0.7,           # Non-maximum suppression threshold
        crop_n_points_downscale_factor=2,  # Downscale factor for crop points
        min_mask_region_area=25.0,    # Minimum area for masks (smaller = more detailed)
        use_m2m=True,                 # Use mask-to-mask refinement
    )
    return mask_generator

@asgi(
    name="sam2-api",
    autoscaler=QueueDepthAutoscaler(min_containers=1, max_containers=1),
    image=Image(python_version="python3.10")
        .add_commands([
            "apt-get update -y",
            "apt-get install -y wget",
            # Install OpenGL and related libraries
            "apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1",
            # Alternative: Install minimal OpenGL libraries
            "apt-get install -y libgl1-mesa-dev",
        ])
        .add_python_packages([
            "fastapi",
            "pydantic", 
            "torch",
            "transformers",
            "numpy",
            "pillow",
            "requests",
            "git+https://github.com/facebookresearch/sam2.git",
            "opencv-python-headless",  # Use headless version instead
        ]),
    on_start=init_models,
    memory=4096,
    gpu="RTX4090"
)
def handler(context):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    import base64
    import io
    import numpy as np
    from PIL import Image
    from typing import List, Dict, Any
    import cv2
    import logging
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    app = FastAPI()
    masks_generated: Dict[str, Any] = {}
    
    # Validate mask generator from context
    try:
        mask_generator = context.on_start_value
        if mask_generator is None:
            logger.error("Mask generator not found in context")
            raise Exception("Mask generator initialization failed")
    except Exception as e:
        logger.error(f"Failed to get mask generator from context: {str(e)}")
        raise HTTPException(status_code=500, detail="Service initialization failed")
    
    @app.post("/generate-masks")
    async def predict(image_request: ImageRequest):
        try:
            # Validate request
            if not image_request.file:
                raise HTTPException(status_code=400, detail="No image file provided")
            if not image_request.session_id:
                raise HTTPException(status_code=400, detail="Session ID is required")
            
            base64_image = image_request.file
            session_id = image_request.session_id
            
            # Remove data URL prefix if present
            if "," in base64_image:
                base64_image = base64_image.split(",")[1]
            
            # Decode base64 image
            try:
                image_data = base64.b64decode(base64_image)
            except Exception as e:
                logger.error(f"Failed to decode base64 image: {str(e)}")
                raise HTTPException(status_code=400, detail="Invalid base64 image data")
            
            # Open and convert image
            try:
                image = Image.open(io.BytesIO(image_data)).convert("RGB")
                image_array = np.array(image)
            except Exception as e:
                logger.error(f"Failed to process image: {str(e)}")
                raise HTTPException(status_code=400, detail="Invalid image format")
            
            # Validate image dimensions
            if image_array.shape[0] == 0 or image_array.shape[1] == 0:
                raise HTTPException(status_code=400, detail="Image has invalid dimensions")
            
            # Generate masks
            try:
                masks = mask_generator.generate(image_array)
                logger.info(f"Generated {len(masks)} masks for session {session_id}")
            except Exception as e:
                logger.error(f"Mask generation failed: {str(e)}")
                raise HTTPException(status_code=500, detail="Mask generation failed")
            
            # Store masks for session
            try:
                masks_generated[session_id] = (masks, image_array)
            except Exception as e:
                logger.error(f"Failed to store masks for session {session_id}: {str(e)}")
                raise HTTPException(status_code=500, detail="Failed to store session data")
            
            # Serialize masks
            serializable_masks = []
            try:
                for i, mask in enumerate(masks):
                    try:
                        # Validate mask structure
                        if 'segmentation' not in mask:
                            logger.warning(f"Mask {i} missing segmentation data")
                            continue
                        
                        # Compress the segmentation mask
                        segmentation = mask['segmentation'].astype(np.uint8) * 255
                        success, buffer = cv2.imencode('.png', segmentation)
                        
                        if not success:
                            logger.warning(f"Failed to encode mask {i}")
                            continue
                        
                        mask_b64 = base64.b64encode(buffer).decode('utf-8')
                        
                        serializable_masks.append({
                            "segmentation": mask_b64,
                            "area": float(mask.get('area', 0)),
                            "bbox": [float(x) for x in mask.get('bbox', [0, 0, 0, 0])],
                            "point_coords": mask.get('point_coords', [])
                        })
                    except Exception as e:
                        logger.warning(f"Failed to process mask {i}: {str(e)}")
                        continue
                        
            except Exception as e:
                logger.error(f"Failed to serialize masks: {str(e)}")
                raise HTTPException(status_code=500, detail="Failed to process masks")
            
            if not serializable_masks:
                raise HTTPException(status_code=500, detail="No valid masks generated")
            
            return {
                "width": int(image_array.shape[1]),
                "height": int(image_array.shape[0]),
                "masks": serializable_masks
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in predict endpoint: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error")
    
    @app.post("/get-mask-at-point")
    async def point_request(point_request: PointRequest):
        try:
            # Validate request
            if not point_request.session_id:
                raise HTTPException(status_code=400, detail="Session ID is required")
            
            session_id = point_request.session_id
            
            # Check if session exists
            if session_id not in masks_generated:
                logger.warning(f"Session {session_id} not found")
                raise HTTPException(status_code=404, detail="Session not found")
            
            try:
                session_data = masks_generated.get(session_id)
                if not session_data or len(session_data) < 2:
                    raise HTTPException(status_code=404, detail="Invalid session data")
                
                masks, image_array = session_data
            except Exception as e:
                logger.error(f"Failed to retrieve session data: {str(e)}")
                raise HTTPException(status_code=500, detail="Failed to retrieve session data")
            
            # Validate point coordinates
            if point_request.x < 0 or point_request.y < 0:
                raise HTTPException(status_code=400, detail="Point coordinates must be non-negative")
            
            if (point_request.y >= image_array.shape[0] or 
                point_request.x >= image_array.shape[1]):
                raise HTTPException(status_code=400, detail="Point coordinates are outside image bounds")
            
            # Find masks at point
            selected_masks = []
            try:
                point = (point_request.x, point_request.y)
                
                for i, mask in enumerate(masks):
                    try:
                        if 'segmentation' not in mask:
                            continue
                        
                        segmentation = mask['segmentation']
                        if (point[1] < segmentation.shape[0] and 
                            point[0] < segmentation.shape[1] and
                            segmentation[point[1], point[0]]):
                            selected_masks.append(i)
                    except Exception as e:
                        logger.warning(f"Error checking mask {i} at point: {str(e)}")
                        continue
                        
            except Exception as e:
                logger.error(f"Failed to find masks at point: {str(e)}")
                raise HTTPException(status_code=500, detail="Failed to process point query")
            
            return {"mask_indices": selected_masks}
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in point_request endpoint: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error")
    
    @app.post("/apply-colors")
    async def color_mask_request(request: ColorMaskRequest):
        try:
            # Validate request
            if not request.session_id:
                raise HTTPException(status_code=400, detail="Session ID is required")
            if not request.mask_indices:
                raise HTTPException(status_code=400, detail="No mask indices provided")
            if not request.color or len(request.color) != 3:
                raise HTTPException(status_code=400, detail="Color must be RGB array with 3 values")
            
            # Validate color values
            for color_val in request.color:
                if not (0 <= color_val <= 255):
                    raise HTTPException(status_code=400, detail="Color values must be between 0 and 255")
            
            # Check if session exists
            if request.session_id not in masks_generated:
                logger.warning(f"Session {request.session_id} not found")
                raise HTTPException(status_code=404, detail="Session not found")
            
            try:
                session_data = masks_generated.get(request.session_id)
                if not session_data or len(session_data) < 2:
                    raise HTTPException(status_code=404, detail="Invalid session data")
                
                masks, image_array = session_data
            except Exception as e:
                logger.error(f"Failed to retrieve session data: {str(e)}")
                raise HTTPException(status_code=500, detail="Failed to retrieve session data")
            
            # Create output image
            try:
                output = image_array.copy()
            except Exception as e:
                logger.error(f"Failed to copy image array: {str(e)}")
                raise HTTPException(status_code=500, detail="Failed to process image")
            
            # Apply colors to selected masks
            applied_count = 0
            try:
                for idx in request.mask_indices:
                    try:
                        # Validate mask index
                        if not isinstance(idx, int) or idx < 0:
                            logger.warning(f"Invalid mask index: {idx}")
                            continue
                        
                        if idx >= len(masks):
                            logger.warning(f"Mask index {idx} out of range (max: {len(masks)-1})")
                            continue
                        
                        mask_data = masks[idx]
                        if 'segmentation' not in mask_data:
                            logger.warning(f"Mask {idx} missing segmentation data")
                            continue
                        
                        mask = mask_data['segmentation']
                        
                        # Apply color with blending
                        output[mask] = (
                            0.3 * output[mask] +  # Keep some original color
                            0.7 * np.array(request.color)  # Apply new color
                        ).astype(np.uint8)
                        
                        applied_count += 1
                        
                    except Exception as e:
                        logger.warning(f"Failed to apply color to mask {idx}: {str(e)}")
                        continue
                        
            except Exception as e:
                logger.error(f"Failed to apply colors: {str(e)}")
                raise HTTPException(status_code=500, detail="Failed to apply colors")
            
            if applied_count == 0:
                raise HTTPException(status_code=400, detail="No valid masks were colored")
            
            # Convert to base64
            try:
                output_image = Image.fromarray(output)
                buffered = io.BytesIO()
                output_image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                
                logger.info(f"Successfully applied colors to {applied_count} masks")
                return {"colored_image": f"data:image/png;base64,{img_str}"}
                
            except Exception as e:
                logger.error(f"Failed to encode output image: {str(e)}")
                raise HTTPException(status_code=500, detail="Failed to encode output image")
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in color_mask_request endpoint: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error")

    # Health check endpoint
    @app.get("/health")
    async def health_check():
        try:
            return {"status": "healthy", "mask_generator": mask_generator is not None}
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Service unhealthy")

    return app
