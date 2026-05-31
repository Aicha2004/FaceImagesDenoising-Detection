import cv2
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
from datetime import datetime

class FastNoiseDetector:
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.noise_types = [
            'defocus_blur', 'fog', 'gaussian', 'gaussian_blur', 'jpeg',
            'low_light', 'mixed', 'motion_blur', 'poisson', 'rain',
            'salt_pepper', 'sensor_noise', 'shadow', 'speckle', 'stripe',
            'stripe_noise', 'zoom_blur'
        ]
        
        # Pre-defined colors for speed
        self.color_map = {noise: self._fast_color(i) for i, noise in enumerate(self.noise_types)}
    
    def _fast_color(self, index):
        """Fast color generation without HSV conversion"""
        colors = [
            (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0),
            (0, 128, 128), (128, 128, 0), (255, 20, 147), (0, 255, 127),
            (255, 99, 71), (75, 0, 130), (255, 215, 0), (0, 191, 255),
            (255, 105, 180)
        ]
        return colors[index % len(colors)]
    
    def process_single_image(self, image_path):
        """Process a single image - fast version"""
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        
        # Fast noise detection using simple variance
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Use simpler, faster method - local variance with small window
        kernel = np.ones((5, 5), np.float32) / 25
        mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        sqr_mean = cv2.filter2D(gray.astype(np.float32)**2, -1, kernel)
        variance = sqr_mean - mean**2
        
        # Threshold for noise detection
        thresh = np.mean(variance) + 0.7 * np.std(variance)
        noise_mask = (variance > thresh).astype(np.uint8) * 255
        
        # Fast morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        noise_mask = cv2.morphologyEx(noise_mask, cv2.MORPH_CLOSE, kernel)
        
        # Get bounding boxes
        contours, _ = cv2.findContours(noise_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        folder_name = image_path.parent.name.lower()
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= 200:  # Smaller minimum area
                x, y, w, h = cv2.boundingRect(contour)
                
                # Fast noise type classification
                noise_type = self._fast_classify(gray[y:y+h, x:x+w], folder_name)
                
                detections.append({
                    'bbox': [x, y, x+w, y+h],
                    'noise_type': noise_type
                })
        
        return img, detections
    
    def _fast_classify(self, region, folder_name):
        """Ultra-fast noise classification"""
        if region.size == 0:
            return 'unknown'
        
        # Simple statistics
        std = np.std(region)
        extreme = np.sum((region < 10) | (region > 245)) / region.size
        
        # Fast classification
        if extreme > 0.02:
            return 'salt_pepper'
        elif std > 40:
            return 'gaussian'
        elif 'blur' in folder_name:
            return 'defocus_blur' if 'defocus' in folder_name else 'gaussian_blur'
        elif 'motion' in folder_name:
            return 'motion_blur'
        elif 'speckle' in folder_name:
            return 'speckle'
        elif 'poisson' in folder_name:
            return 'poisson'
        elif 'jpeg' in folder_name:
            return 'jpeg'
        elif 'low' in folder_name:
            return 'low_light'
        elif 'salt' in folder_name:
            return 'salt_pepper'
        else:
            return 'sensor_noise' if std > 30 else 'mixed'
    
    def draw_coordinates_fast(self, image, detections):
        """Draw coordinates on image - fast version"""
        annotated = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            noise_type = det['noise_type']
            color = self.color_map.get(noise_type, (0, 255, 0))
            
            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Quick coordinate text
            coord_text = f"[{x1},{y1},{x2},{y2}]"
            
            # Use smaller font for speed
            cv2.putText(annotated, coord_text, (x1+2, y1+15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Add noise type
            cv2.putText(annotated, noise_type, (x1+2, y1+30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return annotated
    
    def process_batch(self, image_paths, output_dir, batch_size=10):
        """Process a batch of images"""
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i:i+batch_size]
            results = []
            
            for img_path in batch:
                result = self.process_single_image(img_path)
                if result is not None:
                    results.append((img_path, result[0], result[1]))
            
            # Save results
            for img_path, img, detections in results:
                if detections:
                    annotated = self.draw_coordinates_fast(img, detections)
                    output_path = output_dir / f"{img_path.stem}_detected.jpg"
                    cv2.imwrite(str(output_path), annotated)
            
            print(f"  Progress: {min(i+batch_size, len(image_paths))}/{len(image_paths)}")
    
    def process_all_fast(self, input_dir=None, output_dir=None):
        """Process all images with maximum speed"""
        
        if input_dir is None:
            input_dir = self.base_path / 'outputs' / 'denoised_pipeline_existing_metrics'
        
        if output_dir is None:
            output_dir = self.base_path / 'outputs' / 'fast_detected_noise'
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect all images first
        all_images = []
        for noise_folder in input_dir.iterdir():
            if noise_folder.is_dir() and not noise_folder.name.startswith('.'):
                images = list(noise_folder.glob('*.jpg')) + list(noise_folder.glob('*.png'))
                all_images.extend(images)
        
        print(f"📊 Found {len(all_images)} total images")
        
        # Process sequentially but fast
        print("🚀 Processing images...")
        start_time = datetime.now()
        
        # Process in batches
        self.process_batch(all_images, output_dir, batch_size=50)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        print(f"\n✅ COMPLETE!")
        print(f"  Processed {len(all_images)} images in {elapsed:.1f} seconds")
        print(f"  Average: {len(all_images)/elapsed:.1f} images/second")
        print(f"  Results saved to: {output_dir}")

# ============================================
# MAIN EXECUTION - FAST VERSION
# ============================================

if __name__ == "__main__":
    base_path = Path(r"C:\Users\DELL\Desktop\Projet_FaceDenoising")
    detector = FastNoiseDetector(base_path)
    
    # Process all images FAST
    detector.process_all_fast()